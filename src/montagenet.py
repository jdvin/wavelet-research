from dataclasses import dataclass, field
import torch
from torch import einsum as einsum
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F
import mne
from rotary_embedding_torch import RotaryEmbedding

from src.components.norm import RMSNorm
from src.components.attention import MultiHeadAttention
from src.components.activations import GEGLU


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
        self.source_ln = RMSNorm(d_model)
        self.latents_ln = RMSNorm(d_model)

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
            list(
                mne.channels.make_standard_montage("GSN-HydroCel-129")
                .get_positions()["ch_pos"]
                .values()
            )
        )
    )
    # Maximum sampling rate used.
    max_sr: int = 500
    # Minimum sampling rate used.
    min_sr: int = 500
    # lowest frequency that the embedding can capture.
    f_min: int = 5

    @property
    def max_channels(self):
        return self.channel_positions.shape[0]


class EEGPerceiverResampler(nn.Module):
    def __init__(
        self,
        data_config: EEGDataConfig,
        n_latents: int,
        d_model: int,
        n_heads: int,
        d_mlp: int,
        n_blocks: int,
        pool_latents: bool = False,
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
        rotary_embedding = RotaryEmbedding(dim=32, cache_max_seq_len=512)
        self.resampler_blocks = nn.ModuleList(
            [
                PerceiverResamplerBlock(
                    d_model,
                    n_heads,
                    d_mlp,
                    None,  # rotary_embedding,
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
        B, T, C = source.shape
        T_emb = T  # self.embed_l_out(T)
        pos_emb = (
            self.embed_positions(self.channel_positions)
            .unsqueeze(0)
            .unsqueeze(-1)
            .expand(B, C, self.d_model, T_emb)
        )
        source = self.embed(source.reshape(B * C, 1, T)).reshape(
            B, C, self.d_model, T_emb
        )
        source = (
            (source + pos_emb).permute(0, 3, 1, 2).reshape(B * T_emb, C, self.d_model)
        )
        # source = source.permute(0, 3, 1, 2).reshape(B * T_emb, C, self.d_model)
        latents = (
            self.query_latents.clone()
            .unsqueeze(0)
            .expand(B * T_emb, self.n_latents, self.d_model)
        )
        for block in self.resampler_blocks:
            latents = block(
                latents,
                source,
                attention_mask=attention_mask,
                kv_cache=kv_cache,
            )
        if self.pool_latents:
            return latents.reshape(B, T_emb, self.n_latents, self.d_model).mean(2)
        else:
            return latents.reshape(B, T_emb * self.n_latents, self.d_model)


@dataclass
class MontageNetConfig:
    data_config: EEGDataConfig = field(default_factory=EEGDataConfig)
    n_latents: int = 1
    d_model: int = 1024
    n_heads: int = 8
    d_mlp: int = 2048
    dropout: float = 0.0
    scale_exponent: float = -0.25
    pool_latents: bool = False
    n_blocks: int = 4
    n_classes: int = 3


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
            False,
            config.dropout,
            config.scale_exponent,
        )
        self.head = nn.Conv1d(config.d_model * config.n_latents, config.n_classes, 1)

    def forward(
        self,
        x: Tensor,
    ):
        B, T, C = x.shape
        latents = self.encoder(x).reshape(B, self.n_latents * self.d_model, T)
        return self.head(latents)

    def step(self, batch: dict[str, Tensor]):
        eeg, labels = batch["input_features"], batch["labels"]
        B, T = labels.shape
        logits = self(eeg).reshape(B * T, -1)
        labels = labels.reshape(B * T)
        loss = F.cross_entropy(logits, labels)
        return loss, logits, labels

    @property
    def module(self):
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
