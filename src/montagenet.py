from dataclasses import dataclass, field
import torch
from torch import einsum as einsum
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F
import mne
from rotary_embedding_torch import RotaryEmbedding
from einops import rearrange, repeat

from src.components.norm import RMSNorm
from src.components.attention import MultiHeadAttention
from src.components.activations import GEGLU
from utils.electrode_utils import physionet_64_montage


class TemporalAttentionBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_mlp: int,
        rotary_embedding: RotaryEmbedding | None = None,
        dropout: float = 0.0,
        scale_exponent: float = -0.25,
    ):
        super().__init__()
        self.attn_ln = RMSNorm(d_model)
        self.self_attn = MultiHeadAttention(
            n_heads,
            d_model,
            scale=(d_model // n_heads) ** scale_exponent,
            k_bias=True,
            rotary_embedding=rotary_embedding,
            dropout=dropout,
        )
        self.mlp_ln = RMSNorm(d_model)
        self.mlp = nn.Sequential(
            GEGLU(d_model, d_mlp, bias=False),
            nn.Linear(d_mlp, d_model, bias=False),
        )

    def forward(
        self,
        x: Tensor,
        kv_cache: dict[int, Tensor] | None = None,
    ) -> Tensor:
        x = x + self.self_attn(self.attn_ln(x), kv_cache=kv_cache)
        x = x + self.mlp(self.mlp_ln(x))
        return x


class PerceiverResamplerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_mlp: int,
        rotary_embedding: RotaryEmbedding | None,
        dropout: float = 0.0,
        scale_exponent: float = -0.25,
    ):
        super().__init__()
        self.cross_attn = MultiHeadAttention(
            n_heads,
            d_model,
            scale=(d_model // n_heads) ** scale_exponent,
            k_bias=True,
            rotary_embedding=rotary_embedding,
            dropout=dropout,
        )
        self.latents_ln = RMSNorm(d_model)
        self.source_ln = RMSNorm(d_model)
        self.mlp = nn.Sequential(
            GEGLU(d_model, d_mlp, bias=False),
            nn.Linear(d_mlp, d_model, bias=False),
        )
        self.mlp_ln = RMSNorm(d_model)

    def forward(
        self,
        latents: Tensor,
        source: Tensor,
        attention_mask: Tensor | None = None,
        kv_cache: dict[int, Tensor] | None = None,
    ) -> Tensor:
        latents = self.latents_ln(latents)
        source = self.source_ln(source)
        latents = latents + self.cross_attn(
            latents,
            torch.cat([source, latents], dim=1),
            attention_mask=attention_mask,
            kv_cache=kv_cache,
        )
        latents = latents + self.mlp(self.mlp_ln(latents))
        return latents


@dataclass
class EEGDataConfig:
    # N_channels x 3 tensor of electrode positions normalised to [-1, 1]
    channel_positions: Tensor = field(
        default_factory=lambda: torch.tensor(
            list(physionet_64_montage().get_positions()["ch_pos"].values())
        )
    )
    # Maximum sampling rate used.
    max_sr: int = 180
    # Minimum sampling rate used.
    min_sr: int = 180
    # lowest frequency that the embedding can capture.
    f_min: int = 5

    @property
    def max_channels(self):
        return self.channel_positions.shape[0]


@dataclass
class TaskConfig:
    key: str
    n_classes: int


class EEGPerceiverResampler(nn.Module):
    def __init__(
        self,
        data_config: EEGDataConfig,
        n_latents: int,
        d_model: int,
        n_heads: int,
        d_mlp: int,
        n_blocks: int,
        pool_latents: bool,
        dropout: float = 0.0,
        scale_exponent: float = -0.25,
    ):
        super().__init__()
        self.d_model = d_model
        self.register_buffer("channel_positions", data_config.channel_positions)
        self.n_latents = n_latents
        self.pool_latents = pool_latents
        self.query_latents = nn.Parameter(torch.randn(n_latents, d_model))
        emb_kernel_size = int(data_config.max_sr / data_config.f_min)
        emb_stride = int(data_config.max_sr / data_config.min_sr)
        self.embed = nn.Conv1d(
            in_channels=1,
            out_channels=d_model,
            kernel_size=emb_kernel_size,
            stride=emb_stride,
            padding="same",
        )
        self.embed_positions = nn.Linear(3, d_model)
        self.embed_l_out = lambda T: int((T - emb_kernel_size) / emb_stride + 1)
        rotary_embedding = RotaryEmbedding(dim=32, cache_max_seq_len=256)
        self.resampler_blocks = nn.ModuleList(
            [
                PerceiverResamplerBlock(
                    d_model,
                    n_heads,
                    d_mlp,
                    None,
                    dropout,
                    scale_exponent,
                )
                for _ in range(n_blocks)
            ]
        )
        self.temporal_attn_blocks = nn.ModuleList(
            [
                TemporalAttentionBlock(
                    d_model,
                    n_heads,
                    d_mlp,
                    rotary_embedding,
                    dropout,
                    scale_exponent,
                )
                for _ in range(n_blocks)
            ]
        )

    def forward(
        self,
        source: Tensor,
        attention_mask: Tensor | None = None,
        kv_cache: dict[int, Tensor] | None = None,
    ) -> Tensor:
        B, C, T = source.shape
        T_emb = T  # self.embed_l_out(T)
        # Regenerate positional embeddings.
        pos_emb = repeat(
            self.embed_positions(self.channel_positions),
            "C D -> B C D T_emb",
            B=B,
            T_emb=T_emb,
            D=self.d_model,
        )
        # Embed EEG data.
        source = rearrange(
            self.embed(rearrange(source, "B C T -> (B C) 1 T")),
            "(B C) D T_emb -> B C D T_emb",
            B=B,
            C=C,
            D=self.d_model,
        )
        source = rearrange(source + pos_emb, "B C D T_emb -> (B T_emb) C D")
        # source = rearrange(source, "B C D T_emb -> (B T_emb) C D")
        # Initialize query latents
        latents = repeat(
            self.query_latents,
            "L D -> (B T_emb) L D",
            B=B,
            T_emb=T_emb,
            D=self.d_model,
            L=self.n_latents,
        )

        for resampler_block, temporal_attn_block in zip(
            self.resampler_blocks, self.temporal_attn_blocks
        ):
            latents = resampler_block(
                latents,
                source,
                attention_mask=attention_mask,
                kv_cache=kv_cache,
            )
            latents = rearrange(
                latents,
                "(B T_emb) L D -> (B L) T_emb D",
                B=B,
                T_emb=T_emb,
                L=self.n_latents,
                D=self.d_model,
            )
            latents = temporal_attn_block(latents, kv_cache=kv_cache)
            latents = rearrange(
                latents,
                "(B L) T_emb D -> (B T_emb) L D",
                B=B,
                L=self.n_latents,
                T_emb=T_emb,
                D=self.d_model,
            )

        if self.pool_latents:
            # Mean pool across time.
            latents = rearrange(
                latents,
                "(B T_emb) L D -> B T_emb (L D)",
                B=B,
                T_emb=T_emb,
                D=self.d_model,
                L=self.n_latents,
            )
            return latents.mean(dim=1)  # Average across time dimension
        else:
            # Keep time dimension.
            return rearrange(
                latents,
                "(B T_emb) L D -> B (T_emb L) D",
                B=B,
                T_emb=T_emb,
                L=self.n_latents,
                D=self.d_model,
            )


@dataclass
class MontageNetConfig:
    n_latents: int
    d_model: int
    n_heads: int
    d_mlp: int
    dropout: float
    scale_exponent: float
    pool_latents: bool
    n_blocks: int
    n_classes: int
    tasks: list[TaskConfig]
    data_config: EEGDataConfig = field(default_factory=EEGDataConfig)


class MontageNet(nn.Module):
    def __init__(
        self,
        config: MontageNetConfig,
        rank: int,
        world_size: int,
    ):
        super().__init__()
        self.data_config = config.data_config
        self.n_latents = config.n_latents
        self.d_model = config.d_model
        self.encoder = EEGPerceiverResampler(
            config.data_config,
            config.n_latents,
            config.d_model,
            config.n_heads,
            config.d_mlp,
            config.n_blocks,
            config.pool_latents,
            config.dropout,
            config.scale_exponent,
        )
        # TODO: This will be slow, but we can optimise later.
        self.task_heads = nn.ModuleDict(
            {
                task.key: nn.Linear(config.d_model * config.n_latents, task.n_classes)
                if config.pool_latents
                else nn.Conv1d(config.d_model * config.n_latents, task.n_classes, 1)
                for task in config.tasks
            }
        )

    def forward(self, batch: dict[str, list[str] | Tensor]):
        task_keys, eeg, labels = (
            batch["tasks"],
            batch["input_features"],
            batch["labels"],
        )
        assert isinstance(labels, Tensor)
        assert isinstance(task_keys, list)
        latents = self.encoder(eeg)
        logits = torch.stack(
            [
                self.task_heads[task_key](latent)
                for task_key, latent in zip(task_keys, latents)
            ],
        )
        loss = torch.stack(
            [F.cross_entropy(logit, label) for logit, label in zip(logits, labels)]
        ).mean()
        return loss, logits, labels

    @property
    def module(self):
        """For interoperability with DDP"""
        return self

    def optim_groups(self, weight_decay: float = 1e-1) -> list[dict[str, str]]:
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(
            f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters"
        )
        print(
            f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters"
        )
        return optim_groups
