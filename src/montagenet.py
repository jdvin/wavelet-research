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
        T: int,
        seq_pos: Tensor | None = None,
        spatial_attention_mask: Tensor | None = None,
        temporal_attention_mask: Tensor | None = None,
        kv_cache: dict[int, Tensor] | None = None,
    ):
        latents = self.resampler_block(
            latents,
            source,
            attention_mask=spatial_attention_mask,
            kv_cache=kv_cache,
        )
        latents = rearrange(
            latents,
            "(B T) L D -> (B L) T D",
            T=T,
            L=self.n_latents,
            D=self.d_model,
        )
        latents = self.temporal_attn_block(
            latents,
            kv_cache=kv_cache,
            attention_mask=temporal_attention_mask,
            seq_pos=seq_pos,
        )
        latents = rearrange(
            latents,
            "(B L) T D -> (B T) L D",
            L=self.n_latents,
            T=T,
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
    # highest frequency that the embedding should capture.
    f_max: int
    kernel_sec: float
    sequence_lenght_seconds: float

    def __post_init__(self):
        assert self.f_max // 2 < min(self.sampling_rates)


def lcm2(a: int, b: int) -> int:
    if a == 0 or b == 0:
        return 0
    return abs(a) // gcd(a, b) * abs(b)


def lcmN(*nums: int) -> int:
    return reduce(lcm2, nums, 1)


@dataclass
class TaskConfig:
    key: str
    n_classes: int


class ComplexMorletBank(nn.Module):
    def __init__(self, data_config: DataConfig, d_model: int):
        super().__init__()
        # Learn center freq and log-bandwidth per filter (Hz, seconds)
        self.f_c = nn.Parameter(
            torch.linspace(data_config.f_min, data_config.f_max, d_model)
        )
        self.log_Q = nn.Parameter(torch.zeros(d_model))  # Q controls σ via bandwidth
        self.K_sec = data_config.kernel_sec
        self.d_model = d_model
        self.kernel_banks = {}

    def kernel_for_sr(self, sr: int, device="cpu"):
        if kb := self.kernel_banks.get(sr):
            return kb
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
        self.kernel_banks[sr] = k.contiguous()  # (2*d_model, 1, K)
        return self.kernel_banks[sr]

    def __getitem__(self, sr):
        return self.kernel_for_sr(sr)


class ContinuousSignalEmbedder(nn.Module):
    def __init__(
        self,
        data_config: DataConfig,
        d_model: int,
    ):
        super().__init__()
        self.data_config = data_config
        self.kernel_bank_factory = ComplexMorletBank(data_config, d_model)
        self.sr_lcm = lcmN(*data_config.sampling_rates)
        self.sr_seq_positions = {
            sr: torch.linspace(
                0,
                int(data_config.sequence_lenght_seconds * self.sr_lcm),
                self.sr_lcm // sr,
            )
            for sr in data_config.sampling_rates
        }

    def stack_grouped_conv(self, X, k_list):
        """
        X: Tensor of shape (BC, T) each channel is a signal from an electrode.
        k_list: list of (d_model, 1, K_i)  (kernel bank per sample; centered)
        returns: (BC, d_model, T_max)
        """
        BC = X.shape[0]
        X = X.unsqueeze(0)  # (1, B, T_max)

        K_max = max(k.size(-1) for k in k_list)
        # center-pad every kernel bank to K_max
        K_pad = []
        for k in k_list:
            Ki = k.size(-1)
            left = (K_max - Ki) // 2
            right = K_max - Ki - left
            K_pad.append(F.pad(k, (left, right)))  # (d_model,1,K_max)
        W = torch.cat(K_pad, dim=0)  # (BC*d_model,1,K_max)

        # grouped conv: groups=BC, in_channels=BC, out_channels=BC*d_model
        Y = F.conv1d(X, W, padding=K_max // 2, groups=BC)  # (1, BC*d_model, T_max)
        Y = rearrange(Y, "1 (bc d) t -> bc d t", b=BC)  # (BC,d_model,T_max)
        return Y

    def forward(
        self, x: Tensor, indexes: list[int], srs: list[int], max_channels: int
    ) -> tuple[Tensor, Tensor]:
        """
        x: Input signal tensor of shape (BC, T) with channels folded into the batch dimension.
        indexes: LongTensor of shape (B,) with the indexes of the signal in the
            original batch dimension.
        srs: LongTensor of shape (B,) with the sampling rates of the signals.
        max_channels: Maximum number of channels for a sample in the microbatch.
        """
        kernel_banks_list, seq_positions = zip(
            *[(self.kernel_bank_factory[sr], self.sr_seq_positions[sr]) for sr in srs]
        )
        embs = self.stack_grouped_conv(x, kernel_banks_list)
        E = []
        for start, end in zip(indexes, indexes[1:] + [None]):
            # Slice out embedding of all channels for this training example and pad up to max channels.
            e = embs[start:end]
            e = F.pad(e, (0, 0, 0, max_channels - e.shape[1]))
            E.append(e)
        # norms?
        return torch.stack(E), torch.stack(seq_positions).to(x.device)


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
        rotary_embedding = RotaryEmbedding(dim=32, cache_max_seq_len=256)
        self.embedder = ContinuousSignalEmbedder(
            data_config,
            d_model,
        )
        self.embed_positions = nn.Linear(3, d_model)
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
        channel_mask: Tensor | None = None,
        samples_mask: Tensor | None = None,
    ) -> Tensor:
        """
        source: signals tensor of shape (batch, channels, time).
        channel_positions: tensor of shape (batch, channels, 3) with the position of each channel in the signal.
        channel_masks: boolean tensor of shape (batch, channels) with True for each channel that should be included in the embedding and spatial attention.
        samples_mask: boolean tensor of shape (batch, samples) with True for each sample that should be included in the embedding.
        """
        if channel_mask is None:
            channel_mask = torch.ones(
                source.size(0), source.size(1), dtype=torch.bool, device=source.device
            )

        B, C, T = source.shape
        pos_emb = repeat(
            self.embed_positions(channel_positions),
            "B C D -> B C D T",
            B=B,
            T=T,
            D=self.d_model,
        )
        # TODO: Is this slicing maneuver cheaper than just running the padded signals through the embedder?
        # Could I write a custom kernel that recognises the channel mask and only computes the relevant channels?
        signals = torch.stack(
            [source[i, : channel_mask[i], :] for i in range(source.size(0))]
        )
        embeddings, seq_positions = self.embedder(signals)
        source = rearrange(
            embeddings,
            "(B C) D T -> B C D T",
            B=B,
            C=C,
            D=self.d_model,
        )
        source = rearrange(source + pos_emb, "B C D T -> (B T) C D")
        # source = rearrange(source, "B C D T -> (B T) C D")
        # Initialize query latents
        latents = repeat(
            self.query_latents,
            "L D -> (B T) L D",
            B=B,
            T=T,
            D=self.d_model,
            L=self.n_latents,
        )

        for block in self.blocks:
            latents = block(
                latents,
                source,
                T,
                seq_pos=seq_positions,
                spatial_attention_mask=channel_mask,
                temporal_attention_mask=samples_mask,
            )

        latents = rearrange(
            latents,
            "(B T) L D -> B T (L D)",
            B=B,
            T=T,
            D=self.d_model,
            L=self.n_latents,
        )
        if self.return_latents == ReturnLatents.MEAN_POOL:
            # Mean pool across time, account for samples mask.
            if samples_mask is not None:
                mask_expanded = samples_mask.unsqueeze(-1)
                masked_latents = latents * mask_expanded
                summed = masked_latents.sum(dim=1)
                counts = samples_mask.sum(dim=1, keepdim=True).clamp(min=1)
                return summed / counts
            else:
                # No mask, simple mean pooling
                return latents.mean(dim=1)
        elif self.return_latents == ReturnLatents.ALL:
            return latents
        elif self.return_latents == ReturnLatents.LAST:
            return latents[:, -1, :]
        elif self.return_latents == ReturnLatents.MIDDLE:
            return latents[:, T // 2, :]
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
        (
            channel_signals,
            channel_positions,
            channel_mask,
            samples_mask,
            task_keys,
            labels,
        ) = (
            batch["channel_signals"],
            batch["channel_positions"],
            batch["channel_mask"],
            batch["samples_mask"],
            batch["tasks"],
            batch["labels"],
        )
        latents = self.encoder(
            channel_signals, channel_positions, channel_mask, samples_mask
        )
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
