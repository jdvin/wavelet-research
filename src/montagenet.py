from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable
from functools import partial
import torch
from torch import einsum as einsum
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F
from rotary_embedding_torch import RotaryEmbedding
from einops import rearrange, repeat
from math import gcd
from functools import reduce

from src.components.focal_loss import FocalLoss
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


class SpatioTemporalAttentionBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_mlp: int,
        n_latents: int,
        rotary_embedding: RotaryEmbedding | None,
        dropout: float = 0.0,
        scale_exponent: float = -0.25,
    ):
        super().__init__()
        self.resampler_block = PerceiverResamplerBlock(
            d_model,
            n_heads,
            d_mlp,
            None,
            dropout,
            scale_exponent,
        )
        self.temporal_attn_block = TemporalAttentionBlock(
            d_model,
            n_heads,
            d_mlp,
            rotary_embedding,
            dropout,
            scale_exponent,
        )
        self.n_latents = n_latents
        self.d_model = d_model

    def forward(
        self,
        latents: Tensor,
        source: Tensor,
        T_emb: int,
        attention_mask: Tensor | None = None,
        kv_cache: dict[int, Tensor] | None = None,
    ):
        latents = self.resampler_block(
            latents,
            source,
            attention_mask=attention_mask,
            kv_cache=kv_cache,
        )
        latents = rearrange(
            latents,
            "(B T_emb) L D -> (B L) T_emb D",
            T_emb=T_emb,
            L=self.n_latents,
            D=self.d_model,
        )
        latents = self.temporal_attn_block(latents, kv_cache=kv_cache)
        latents = rearrange(
            latents,
            "(B L) T_emb D -> (B T_emb) L D",
            L=self.n_latents,
            T_emb=T_emb,
            D=self.d_model,
        )
        return latents


class ReturnLatents(Enum):
    ALL = "all"
    MIDDLE = "middle"
    LAST = "last"
    MEAN_POOL = "mean_pool"


@dataclass
class DataConfig:
    # Number of channel for each data source.
    channel_counts: list[int]
    # Sampling rate for each data source.
    sampling_rates: list[int]
    # lowest frequency that the embedding should capture.
    f_min: int


@dataclass
class TaskConfig:
    key: str
    n_classes: int


class ComplexMorletBank(nn.Module):
    def __init__(self, d_model, f_lo, f_hi, K_sec):
        super().__init__()
        # Learn center freq and log-bandwidth per filter (Hz, seconds)
        self.f_c = nn.Parameter(torch.linspace(f_lo, f_hi, d_model))
        self.log_Q = nn.Parameter(torch.zeros(d_model))  # Q controls σ via bandwidth
        self.K_sec = K_sec
        self.d_model = d_model

    def kernel_for_sr(self, sr: int, device="cpu"):
        K = int(round(self.K_sec * sr)) | 1
        t = (torch.arange(K, device=device) - K // 2) / sr  # seconds
        f = F.softplus(self.f_c)  # >0 Hz
        Q = torch.exp(self.log_Q) + 1.0  # >1
        # bandwidth-to-sigma (approx; tweak as desired)
        sigma = Q / (2 * torch.pi * f + 1e-8)  # seconds

        # Complex Morlet: exp(j 2π f t) * exp(-t^2/(2σ^2)), zero-mean correction optional
        # Real and Imag parts (two output channels per filter)
        carrier = 2 * torch.pi * f[:, None] * t[None, :]
        gauss = torch.exp(-0.5 * (t[None, :] / (sigma[:, None] + 1e-8)) ** 2)
        k_r = torch.cos(carrier) * gauss  # (d_model, K)
        k_i = torch.sin(carrier) * gauss  # (d_model, K)

        # Window & normalize (optional; gauss already windows)
        k_r = k_r / (k_r.norm(dim=-1, keepdim=True) + 1e-8)
        k_i = k_i / (k_i.norm(dim=-1, keepdim=True) + 1e-8)

        # Stack as 2*d_model real filters: [real; imag]
        k = torch.stack([k_r, k_i], dim=1).reshape(2 * self.d_model, 1, K)
        return k.contiguous()  # (2*d_model, 1, K)


class ContinuousSignalEmbedder(nn.Module):
    def __init__(
        self,
        data_config: DataConfig,
        d_model: int,
    ):
        super().__init__()
        self.kernel_bank = ComplexMorletBank(
            d_model,
            data_config.f_min,
            data_config.max_sr // 2,
            data_config.K_sec,
        )


class SpatioTemporalPerceiverResampler(nn.Module):
    def __init__(
        self,
        data_config: DataConfig,
        n_latents: int,
        d_model: int,
        n_heads: int,
        d_mlp: int,
        n_blocks: int,
        return_latents: ReturnLatents,
        dropout: float = 0.0,
        scale_exponent: float = -0.25,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_latents = n_latents
        self.return_latents = return_latents
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
        self.blocks = nn.ModuleList(
            [
                SpatioTemporalAttentionBlock(
                    d_model,
                    n_heads,
                    d_mlp,
                    n_latents,
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
        channel_positions: Tensor,
        attention_mask: Tensor | None = None,
        kv_cache: dict[int, Tensor] | None = None,
    ) -> Tensor:
        B, C, T = source.shape
        T_emb = T  # self.embed_l_out(T)
        pos_emb = repeat(
            self.embed_positions(channel_positions),
            "B C D -> B C D T_emb",
            B=B,
            T_emb=T_emb,
            D=self.d_model,
        )
        # chan_embed = repeat(
        #     self.embed_channels.weight,
        #     "C D -> B C D T_emb",
        #     B=B,
        #     C=C,
        #     T_emb=T_emb,
        #     D=self.d_model,
        # )
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

        for block in self.blocks:
            latents = block(latents, source, T_emb, attention_mask=attention_mask)

        latents = rearrange(
            latents,
            "(B T_emb) L D -> B T_emb (L D)",
            B=B,
            T_emb=T_emb,
            D=self.d_model,
            L=self.n_latents,
        )
        if self.return_latents == ReturnLatents.MEAN_POOL:
            # Mean pool across time.
            return latents.mean(dim=1)  # Average across time dimension
        elif self.return_latents == ReturnLatents.ALL:
            return latents
        elif self.return_latents == ReturnLatents.LAST:
            return latents[:, -1, :]
        elif self.return_latents == ReturnLatents.MIDDLE:
            return latents[:, T_emb // 2, :]
        else:
            raise NotImplementedError(self.return_latents)


@dataclass
class MontageNetConfig:
    n_latents: int
    d_model: int
    n_heads: int
    d_mlp: int
    dropout: float
    scale_exponent: float
    return_latents: ReturnLatents
    n_blocks: int
    tasks: list[TaskConfig]
    data_config: DataConfig

    def __init__(
        self,
        n_latents,
        d_model,
        n_heads,
        d_mlp,
        dropout,
        scale_exponent,
        return_latents,
        n_blocks,
        tasks: list[dict[str, Any]],
        data_config: dict[str, Any],
    ):
        self.n_latents = n_latents
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_mlp = d_mlp
        self.dropout = dropout
        self.scale_exponent = scale_exponent
        self.n_blocks = n_blocks
        self.tasks = [TaskConfig(**task) for task in tasks]
        self.data_config = DataConfig(**data_config)
        self.return_latents = ReturnLatents(return_latents)


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
        self.encoder = SpatioTemporalPerceiverResampler(
            config.data_config,
            config.n_latents,
            config.d_model,
            config.n_heads,
            config.d_mlp,
            config.n_blocks,
            config.return_latents,
            config.dropout,
            config.scale_exponent,
        )
        # TODO: This will be slow, but we can optimise later.
        self.task_heads = nn.ModuleList(
            [
                nn.Linear(config.d_model * config.n_latents, task.n_classes)
                if config.return_latents != ReturnLatents.ALL
                else nn.Conv1d(config.d_model * config.n_latents, task.n_classes, 1)
                for task in config.tasks
            ]
        )

        self.loss = partial(F.cross_entropy, label_smoothing=0.1)

    def compute_difficulty(
        self, epsilon: float, speech_densities: Tensor, labels: Tensor
    ):
        return epsilon * (
            (1 - labels) * speech_densities + labels * (1 - speech_densities)
        )

    def forward(self, batch: dict[str, Tensor]):
        channel_positions, task_keys, channel_signals, labels = (
            batch["channel_positions"],
            batch["tasks"],
            batch["channel_signals"],
            batch["labels"],
        )
        latents = self.encoder(channel_signals, channel_positions)
        losses, logits = [], []
        for (
            task_key,
            latent,
            label,
        ) in zip(task_keys, latents, labels):
            logit = self.task_heads[int(task_key.item())](latent)
            loss = self.loss(logit.unsqueeze(0), label.unsqueeze(0))
            logits.append(logit)
            losses.append(loss)
        loss = torch.stack(losses).mean()
        logits = torch.stack(logits)
        return loss, logits, labels

    @property
    def module(self):
        """For interoperability with DDP"""
        return self
