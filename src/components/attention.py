import math
import torch
from torch import Tensor
from torch import nn
from torch.nn import functional as F
from .rope import RotaryEmbedding
from .pos import RelativePositionBias


def chunked_torch_spda(
    q: torch.Tensor,  # (B_eff, H, Lq, D)
    k: torch.Tensor,  # (B_eff, H, Lk, D)
    v: torch.Tensor,  # (B_eff, H, Lk, D)
    attn_mask: torch.Tensor
    | None = None,  # broadcastable to (B_eff, 1|H, Lq, Lk) or (1, 1|H, Lq, Lk)
    *,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    max_bh: int = 60000,  # keep (chunk_B * H) <= max_bh to avoid CUDA grid limits (~65535)
) -> torch.Tensor:
    B_eff, H, Lq, D = q.shape
    _, _, Lk, _ = k.shape

    # Fast path if already under the limit
    if B_eff * H <= max_bh:
        return F.scaled_dot_product_attention(
            q.contiguous(),
            k.contiguous(),
            v.contiguous(),
            attn_mask=None if attn_mask is None else attn_mask.contiguous(),
            dropout_p=dropout_p,
            is_causal=is_causal,
        )

    # Otherwise, chunk along batch
    chunk_B = max(1, max_bh // max(H, 1))
    n_chunks = math.ceil(B_eff / chunk_B)

    outs = []
    for i in range(n_chunks):
        s = i * chunk_B
        e = min((i + 1) * chunk_B, B_eff)
        qi = q[s:e].contiguous()
        ki = k[s:e].contiguous()
        vi = v[s:e].contiguous()

        # Slice mask on batch dim only if it actually has that dim; otherwise keep as-is for broadcasting
        if attn_mask is None:
            mi = None
        else:
            if attn_mask.size(0) == B_eff:
                mi = attn_mask[s:e].contiguous()
            else:
                mi = attn_mask.contiguous()

        yi = F.scaled_dot_product_attention(
            qi, ki, vi, attn_mask=mi, dropout_p=dropout_p, is_causal=is_causal
        )
        outs.append(yi)

    return torch.cat(outs, dim=0)


class MultiHeadAttention(torch.nn.Module):
    def __init__(
        self,
        n_heads: int,
        d_model: int,
        source_seq_len: int | None = None,
        target_seq_len: int | None = None,
        rotary_embedding: RotaryEmbedding | None = None,
        qk_norm: bool = False,
        q_bias: bool = True,
        k_bias: bool = False,
        v_bias: bool = True,
        out_bias: bool = True,
        scale: float = 0.0,
        dropout: float = 0.1,
        is_causal: bool = False,
        flash: bool = True,
    ):
        super().__init__()
        # Model embedding is split across heads.
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.dropout = dropout
        self.is_causal = is_causal
        assert scale
        self.scale = scale
        self.rotary_embedding = rotary_embedding
        self.qk_norm = qk_norm
        if self.qk_norm:
            assert scale
        self.tau = nn.Parameter(torch.tensor(scale))
        # The projectons are scalled differently in the original whisper implementation
        self.q_proj = nn.Linear(d_model, d_model, bias=q_bias)
        self.k_proj = nn.Linear(d_model, d_model, bias=k_bias)
        self.v_proj = nn.Linear(d_model, d_model, bias=v_bias)
        # Output projection.
        self.out_proj = nn.Linear(d_model, d_model, bias=out_bias)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        self.flash = flash
        self.source_seq_len = source_seq_len
        self.target_seq_len = target_seq_len
        if self.is_causal:
            assert source_seq_len is not None
            assert target_seq_len is not None
            bias = torch.triu(
                torch.full(
                    (1, 1, target_seq_len, source_seq_len),
                    torch.finfo(self.q_proj.weight.dtype).min,
                ),
                diagonal=1,
            )
            self.register_buffer("bias", bias)

    def split_heads(self, x: torch.Tensor, B: int, T: int, D: int) -> torch.Tensor:
        """Split matrices into heads and reshape to have heads as child ranks."""
        return x.view(B, T, self.n_heads, D // self.n_heads).transpose(1, 2)

    def qkv_attention(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        attention_mask: Tensor | None,
    ) -> Tensor:
        """
        Args:
            q: Tensor[float] (B, nhead, T_q, D_head)
            k: Tensor[float] (B, nhead, T_kv, D_head)
            v: Tensor[float] (B, nhead, T_kv, D_head)
            attention_mask: Tensor[float] (B, 1, T_q, T_kv)
        """
        if self.flash and not self.qk_norm:
            y = chunked_torch_spda(
                q, k, v, attention_mask, dropout_p=self.dropout, is_causal=False
            )

        else:
            # (B, nhead, T_q, D_head) x (B, nhead, D_head, T_kv) -> (B, nhead, T_q, T_kv).
            qk = (q @ k.transpose(-2, -1)) * (self.tau if self.qk_norm else self.scale)
            # Add attention bias (input masking, causal masking, relative pos, etc).
            qk = qk + attention_mask if attention_mask is not None else qk
            attn = F.softmax(qk, dim=-1, dtype=torch.float32).type_as(qk)
            attn = self.attn_dropout(attn) if self.training else attn
            # (B, nhead, T, T) x (B, nhead, T, D_head) -> (B, nhead, T, D_head).
            y = attn @ v
        return y

    def forward(
        self,
        x: Tensor,
        xc: Tensor | None = None,
        kv_cache: dict[int, Tensor] | None = None,
        attention_mask: Tensor | None = None,
        seq_pos: Tensor | None = None,
        dynamic_rope_freqs: Tensor | None = None,
    ) -> tuple[Tensor]:
        B, T_q, D = x.size()  # Batch size, sequence length, model dimension.
        T_kv = xc.size(1) if xc is not None else T_q
        # Instantiate a 'dummy' kv cache to make the logic simpler.
        if kv_cache is None:
            kv_cache = {}

        k = kv_cache.get(hash(self.k_proj), None)
        v = kv_cache.get(hash(self.v_proj), None)
        T_cached = k.size(1) if k is not None else 0
        if k is None or k.size(1) != T_kv:
            k_new = self.k_proj((x if xc is None else xc)[:, T_cached:, :])
            v_new = self.v_proj((x if xc is None else xc)[:, T_cached:, :])
            kv_cache[self.k_proj] = (
                k_new if k is None else torch.concat([k, k_new], dim=1)
            )
            kv_cache[self.v_proj] = (
                v_new if v is None else torch.concat([v, v_new], dim=1)
            )

        q = self.q_proj(x)
        k = kv_cache[self.k_proj]
        v = kv_cache[self.v_proj]
        q = self.split_heads(q, B, T_q, D)
        k = self.split_heads(k, B, T_kv, D)
        v = self.split_heads(v, B, T_kv, D)
        if self.qk_norm:
            q = F.normalize(q, dim=-1)
            k = F.normalize(k, dim=-1)
        bias = (
            self.bias[:, :, T_cached : T_cached + T_q, :T_kv]
            if hasattr(self, "bias")
            else None
        )
        if attention_mask is not None:
            attention_mask = attention_mask[:, None, None, :].expand(
                B, self.n_heads, T_q, T_kv
            )
            if bias is not None:
                attention_mask = attention_mask + bias
        if self.rotary_embedding is not None:
            if seq_pos is not None:
                (
                    q,
                    k,
                ) = self.rotary_embedding.rotate_queries_and_keys_with_custom_seq_pos(
                    q, k, seq_pos
                )
            else:
                q = self.rotary_embedding.rotate_queries_or_keys(q, offset=T_cached)
                k = self.rotary_embedding.rotate_queries_or_keys(k)
        elif dynamic_rope_freqs is not None:
            q, k = RotaryEmbedding.rotate_quries_and_keys_with_dynamic_freqs(
                dynamic_rope_freqs, q, k
            )
        y = self.qkv_attention(
            q,
            k,
            v,
            bias if attention_mask is None else attention_mask,
        )
        # Flatten heads.
        y = y.transpose(1, 2).contiguous().view(B, T_q, D)
        y = self.out_proj(y)
        return self.resid_dropout(y) if self.training else y


class RelativePositionMultiHeadAttention(MultiHeadAttention):
    def __init__(
        self,
        n_heads: int,
        d_model: int,
        source_seq_len: int,
        target_seq_len: int,
        q_bias: bool = True,
        k_bias: bool = False,
        v_bias: bool = True,
        out_bias: bool = True,
        scale: float = 0.0,
        dropout: float = 0.1,
        is_causal: bool = False,
        flash: bool = True,
    ):
        super().__init__(
            n_heads=n_heads,
            d_model=d_model,
            source_seq_len=source_seq_len,
            target_seq_len=target_seq_len,
            q_bias=q_bias,
            k_bias=k_bias,
            v_bias=v_bias,
            out_bias=out_bias,
            scale=scale,
            dropout=dropout,
            is_causal=is_causal,
            flash=flash,
        )
        self.rp_bias = RelativePositionBias(
            bidirectional=not is_causal, n_heads=self.n_heads
        )
        assert (
            self.source_seq_len == self.target_seq_len
        ), "Relative position MHA can only be used in self-attention!"

        self.compute_bias()
        self.register_load_state_dict_post_hook(self.compute_bias)

    def compute_bias(self, *args, **kwargs):
        assert self.source_seq_len is not None
        assert self.target_seq_len is not None
        bias = self.rp_bias(self.target_seq_len, self.source_seq_len)
        if self.is_causal:
            bias = bias + torch.triu(
                torch.full(
                    (1, 1, self.target_seq_len, self.source_seq_len),
                    torch.finfo(self.rp_bias.relative_attention_bias.weight.dtype).min,
                ),
                diagonal=1,
            )
        self.register_buffer("bias", bias)
