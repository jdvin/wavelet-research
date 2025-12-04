from dataclasses import dataclass, field
from enum import Enum
import math
from typing import Any
from functools import partial
import torch
from torch import einsum as einsum
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F
from einops import rearrange, repeat
from math import gcd
from functools import reduce
from src.components.rope import RotaryEmbedding
from src.components.focal_loss import FocalLoss
from src.components.norm import RMSNorm
from src.components.attention import MultiHeadAttention
from src.components.activations import GEGLU


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
        attention_mask: Tensor | None = None,
        seq_pos: Tensor | None = None,
        kv_cache: dict[int, Tensor] | None = None,
    ) -> Tensor:
        x = x + self.self_attn(
            self.attn_ln(x),
            attention_mask=attention_mask,
            seq_pos=seq_pos,
            kv_cache=kv_cache,
        )
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
        dynamic_rope_freqs: Tensor | None = None,
        kv_cache: dict[int, Tensor] | None = None,
    ) -> Tensor:
        latents = self.latents_ln(latents)
        source = self.source_ln(source)
        latents = latents + self.cross_attn(
            latents,
            torch.cat([latents, source], dim=1),
            attention_mask=attention_mask,
            kv_cache=kv_cache,
            dynamic_rope_freqs=dynamic_rope_freqs,
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
        dynamic_rope_freqs: Tensor | None = None,
        spatial_attention_mask: Tensor | None = None,
        temporal_attention_mask: Tensor | None = None,
        kv_cache: dict[int, Tensor] | None = None,
    ):
        latents = self.resampler_block(
            latents,
            source,
            dynamic_rope_freqs=dynamic_rope_freqs,
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
    # lowest frequency that the embedding should capture.
    f_min: int
    # highest frequency that the embedding should capture.
    f_max: int
    kernel_sec: float
    sequence_length_seconds: float
    position_index_per_second: int


@dataclass
class TaskConfig:
    key: str
    index: int
    labels_map: dict[str, int]

    @property
    def n_classes(self) -> int:
        return len(self.labels_map)


class SpectrumGridKernelFactory(nn.Module):
    """
    Learn a spectrum on a shared Hz grid with spacing Δf = 1/kernel_sec.
    For each sr: build only the unaliased positive-half spectrum and irfft
    at native length K = round(kernel_sec * sr). Real time-domain kernels by construction.
    """

    def __init__(
        self,
        data_config: DataConfig,
        d_model: int,
    ):
        super().__init__()
        self.d_model = d_model
        self.kernel_sec = float(data_config.kernel_sec)
        # Shared Hz grid (positive half), independent of SR:
        # freqs = [0, Δf, 2Δf, ...] up to f_max
        self.df = 1.0 / self.kernel_sec
        n_bins = int(math.floor(data_config.f_max / self.df)) + 1  # include DC
        self.register_buffer(
            "freqs_hz", torch.arange(n_bins) * self.df, persistent=False
        )

        # Parameterize log-magnitude and phase on this grid
        self.logmag = nn.Parameter(torch.full((d_model, n_bins), -4.0))
        self.phase = nn.Parameter(torch.zeros(d_model, n_bins))

    def kernel_for_sr(self, sr: int, device=None, dtype=None):
        if device is None:
            device = self.logmag.device
        if dtype is None:
            dtype = self.logmag.dtype

        # Native kernel length & rfft size for this sr
        K = int(round(self.kernel_sec * int(sr))) | 1
        F = K // 2 + 1
        nyq = 0.5 * sr

        # Select only unaliased bins (<= nyquist) from the shared grid
        keep = self.freqs_hz <= nyq
        H_pos = torch.zeros(self.d_model, F, dtype=torch.complex64, device=device)

        # Indices on the shared grid we will copy from
        src = keep.nonzero(as_tuple=False).squeeze(-1)
        if src.numel() > 0:
            # Destination indices on the native rfft grid: they match in Hz because Δf = sr/K = 1/kernel_sec
            # (same Δf by design). Clip if shared grid is longer than native F.
            dst = torch.arange(min(src.numel(), F), device=device)
            mag = self.logmag[:, src].exp()  # (d_model, |src|)
            phs = self.phase[:, src]
            H_pos[:, dst] = mag[:, : dst.numel()] * torch.exp(
                1j * phs[:, : dst.numel()]
            )

        # Enforce real DC and Nyquist for Hermitian symmetry.
        H_pos[:, 0] = H_pos[:, 0].real
        if K % 2 == 0:
            H_pos[:, -1] = H_pos[:, -1].real

        # Time-domain kernel on native grid
        h = torch.fft.irfft(H_pos, n=K)  # (d_model, K), real
        h = h / (h.norm(dim=-1, keepdim=True) + 1e-8)  # stabilize scale
        return h.squeeze(0)  # (d_model, 1, K)


class ComplexMorletFactory(nn.Module):
    def __init__(self, data_config: DataConfig, d_model: int):
        super().__init__()
        # Learn center freq and log-bandwidth per filter (Hz, seconds)
        self.f_c = nn.Parameter(
            torch.linspace(data_config.f_min, data_config.f_max, d_model // 2)
        )
        self.log_Q = nn.Parameter(
            torch.zeros(d_model // 2)
        )  # Q controls σ via bandwidth
        self.K_sec = data_config.kernel_sec
        self.kernel_banks = {}
        self.d_model = d_model
        self.data_config = data_config

    def kernel_for_sr(
        self,
        sr: int,
    ):
        # if self.kernel_banks.get(sr) is not None:
        #     return self.kernel_banks[sr]
        K = int(round(self.K_sec * int(sr))) | 1
        t = (torch.arange(K, device=self.f_c.device) - K // 2) / sr  # seconds
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
        k = torch.stack([k_r, k_i], dim=1).reshape(self.d_model, K)
        return k.contiguous()  # (2*d_model, 1, K)


class MAGNetFactory(nn.Module):
    """
    1D MAGNet-style kernel generator.

    Each output channel c has K atoms:
        phi_{c,k}(t) = A_{c,k} * exp(-t^2 / (2*sigma_{c,k}^2)) * cos(2*pi*f_{c,k}*t + phi_{c,k})

    The kernel for channel c is sum_k phi_{c,k}(t), sampled on a grid t that
    depends on the requested sampling rate (sr) and kernel length (L_K).
    """

    def __init__(
        self,
        d_model: int,
        kernel_length_sec: float,
        n_atoms: int = 4,
        init_freq_range_hz=(1.0, 40.0),
    ):
        """
        Args:
            out_channels: number of output channels (C_out).
            n_atoms: number of Gabor atoms per output channel.
            kernel_size: default kernel length (L_K) if not overridden in kernel_for_sr.
            init_freq_range_hz: tuple (f_min, f_max) for random freq init in Hz.
            init_sigma_range_s: tuple (s_min, s_max) for random sigma init in seconds.
        """
        super().__init__()
        self.d_model = d_model
        self.n_atoms = n_atoms
        self.kernel_length_sec = kernel_length_sec

        # Parameters are per (channel, atom)
        self.log_amp = nn.Parameter(torch.zeros(d_model, n_atoms))
        self.log_sigma = nn.Parameter(torch.zeros(d_model, n_atoms))
        self.log_freq = nn.Parameter(torch.zeros(d_model, n_atoms))
        self.phase = nn.Parameter(torch.zeros(d_model, n_atoms))

        # Simple random initialization
        with torch.no_grad():
            # amplitudes near 1
            self.log_amp.uniform_(math.log(0.3), math.log(1.0))

            # sigmas (seconds)
            s_min, s_max = 0.1, kernel_length_sec
            self.log_sigma.uniform_(math.log(s_min), math.log(s_max))

            # frequencies (Hz)
            f_min, f_max = init_freq_range_hz
            self.log_freq.uniform_(math.log(f_min), math.log(f_max))

            # random phases in [-pi, pi]
            self.phase.uniform_(-math.pi, math.pi)

    def kernel_for_sr(
        self,
        sr: int,
    ) -> torch.Tensor:
        """
        Generate a MAGNet kernel matrix for a given sampling rate.

        Args:
            sr: sampling rate (Hz). Time step is dt = 1/sr.

        Returns:
            kernel: Tensor of shape (C_out, L_K),
                    suitable for conv1d as weight = kernel.unsqueeze(1)
                    (i.e. depthwise conv with C_in = 1).
        """
        device = self.log_amp.device
        L_K = round(self.kernel_length_sec * int(sr))

        # Time grid centered at 0 in *seconds*
        # t: (1, 1, L_K)
        center = L_K / 2.0
        t = (torch.arange(L_K, device=device) - center) / float(sr)
        t = t.view(1, 1, L_K)

        # Positive amplitudes, sigmas, frequencies
        amp = F.softplus(self.log_amp)  # (C, K)
        sigma = F.softplus(self.log_sigma) + 1e-6  # seconds, (C, K)
        freq = F.softplus(self.log_freq) + 1e-6  # Hz, (C, K)
        phase = self.phase  # (C, K)

        # Reshape for broadcasting: (C, K, 1)
        amp = amp.unsqueeze(-1)
        sigma = sigma.unsqueeze(-1)
        freq = freq.unsqueeze(-1)
        phase = phase.unsqueeze(-1)

        # Gaussian envelope: exp(-t^2 / (2*sigma^2))
        envelope = torch.exp(-0.5 * (t**2) / (sigma**2))  # (C, K, L_K) via broadcasting

        # Cosine wave: cos(2*pi*f*t + phase)
        wave = torch.cos(2 * math.pi * freq * t + phase)  # (C, K, L_K)

        # Atoms and sum over atoms
        atoms = amp * envelope * wave  # (C, K, L_K)
        return atoms.sum(dim=1)  # (C, L_K)


class KernelFactoryType(Enum):
    MORLET = "morlet"
    SPECTRUM = "spectrum"
    MAGNET = "magnet"


class ContinuousSignalEmbedder(nn.Module):
    def __init__(
        self,
        data_config: DataConfig,
        d_model: int,
        kernel_factory_type: KernelFactoryType = KernelFactoryType.SPECTRUM,
    ):
        super().__init__()
        self.data_config = data_config
        assert d_model % 2 == 0
        if kernel_factory_type == KernelFactoryType.MAGNET:
            self.kernel_bank_factory = MAGNetFactory(
                d_model,
                kernel_length_sec=data_config.kernel_sec,
            )
        elif kernel_factory_type == KernelFactoryType.MORLET:
            self.kernel_bank_factory = ComplexMorletFactory(
                data_config,
                d_model,
            )
        elif kernel_factory_type == KernelFactoryType.SPECTRUM:
            self.kernel_bank_factory = SpectrumGridKernelFactory(
                data_config,
                d_model,
            )
        else:
            raise NotImplementedError(kernel_factory_type)

        self.d_model = d_model
        self.out = nn.Linear(d_model, d_model)

    def stack_grouped_conv(self, X, k_list):
        """
        X: Tensor of shape (BC, T) each channel is a signal from an electrode.
        k_list: list of (d_model, 1, K_i)  (kernel bank per sample; centered)
        returns: (BC, d_model, T_max)

        NB: C variable per example.
        """
        BC, T = X.shape
        K_max = max(k.size(-1) for k in k_list)
        # center-pad every kernel bank to K_max
        K_pad = []
        for k in k_list:
            Ki = k.size(-1)
            left = (K_max - Ki) // 2
            right = K_max - Ki - left
            K_pad.append(F.pad(k, (left, right)))  # (d_model,1,K_max)
        W = rearrange(
            torch.cat(K_pad, dim=0).to(X.dtype),
            "BC D K -> (BC D) 1 K",  # (BC*d_model,1,K_max)
            BC=BC,
            D=self.d_model,
            K=K_max,
        )

        # grouped conv: groups=BC, in_channels=BC, out_channels=BC*d_model
        # Needs to be BC groups because we don't know C.
        # Query: Is this more efficiennt than just padding each input to max channels and then folding into batch dimension and doing a normal conv?
        Y = F.conv1d(X, W, padding="same", groups=BC)  # (1, BC*d_model, T_max)
        Y = rearrange(Y, "(BC D) T -> BC T D", BC=BC, D=self.d_model, T=Y.shape[-1])
        return Y

    def forward(
        self, x: Tensor, channel_counts: Tensor, srs: list[int]
    ) -> tuple[Tensor, Tensor]:
        """
        x: Input signal tensor of shape (BC, T) with channels folded into the batch dimension.
        indexes: LongTensor of shape (B,) with the indexes of the signal in the
            original batch dimension.
        srs: LongTensor of shape (B,) with the sampling rates of the signals.
        max_channels: Maximum number of channels for a sample in the microbatch.
        """
        indexes = [0] + channel_counts.cumsum(dim=0).tolist()[:-1]
        # NEED ONE KERNEL PER CHANNEL
        kernel_banks_list = [
            self.kernel_bank_factory.kernel_for_sr(sr).unsqueeze(0).expand(C, -1, -1)
            for C, sr in zip(channel_counts, srs)
        ]

        embs = self.stack_grouped_conv(x, kernel_banks_list)
        E = []
        for start, end in zip(indexes, indexes[1:] + [None]):
            # Slice out embedding of all channels for this training example and pad up to max channels.
            e = embs[start:end]
            e = F.pad(e, (0, 0, 0, 0, 0, int(channel_counts.max() - e.shape[0])))
            E.append(e)

        return self.out(torch.stack(E))


class SpatioTemporalPerceiverResampler(nn.Module):
    def __init__(
        self,
        data_config: DataConfig,
        n_latents: int,
        d_model: int,
        n_heads: int,
        d_mlp: int,
        n_blocks: int,
        rope_dim: int,
        return_latents: ReturnLatents,
        dropout: float = 0.0,
        scale_exponent: float = -0.25,
    ):
        super().__init__()
        self.d_model = d_model
        self.rope_dim = rope_dim
        self.n_latents = n_latents
        self.return_latents = return_latents
        self.query_latents = nn.Parameter(torch.randn(n_latents, d_model))
        temporal_rotary_embedding = RotaryEmbedding(
            dim=rope_dim, cache_max_seq_len=256, freqs_for="pixel"
        )
        self.embedder = ContinuousSignalEmbedder(
            data_config,
            d_model,
        )
        self.positions_to_freqs = nn.Sequential(
            GEGLU(3, d_model, bias=False), nn.Linear(d_model, rope_dim, bias=False)
        )
        self.blocks = nn.ModuleList([
            SpatioTemporalAttentionBlock(
                d_model,
                n_heads,
                d_mlp,
                n_latents,
                temporal_rotary_embedding,
                dropout,
                scale_exponent,
            )
            for _ in range(n_blocks)
        ])
        self.data_config = data_config

    def forward(
        self,
        source: Tensor,
        channel_positions: Tensor,
        sequence_positions: Tensor,
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

        # TODO: Is this slicing maneuver cheaper than just running the padded signals through the embedder?
        # Could I write a custom kernel that recognises the channel mask and only computes the relevant channels?
        channel_counts = channel_mask.sum(dim=1)
        # If there is no mask, assume that all samples have the same sampling rate.
        sampling_rates = (
            samples_mask.sum(dim=1) // self.data_config.sequence_length_seconds
            if samples_mask is not None
            else T // self.data_config.sequence_length_seconds
        )
        signals = torch.cat([
            source[i, :channel_count, :]
            for i, channel_count in enumerate(channel_counts)
        ])
        embeddings = self.embedder(signals, channel_counts, sampling_rates)
        # Perpare source for spatial attention.
        source = rearrange(
            embeddings,
            "B C T D -> (B T) C D",
            B=B,
            T=T,
            C=C,
            D=self.d_model,
        )
        # Initialize query latents
        latents = repeat(
            self.query_latents,
            "L D -> (B T) L D",
            B=B,
            T=T,
            D=self.d_model,
            L=self.n_latents,
        )
        # The source sequence is of size channels plus latents because the latents are added to the source
        CpL = C + self.n_latents
        # Add "True" for each query latent to the left of each channel mask so that they are attended to.
        # Expand each mask for the temporal dimsion.
        # Fold temporal dimension into batch dimension.
        channel_mask = rearrange(
            F.pad(channel_mask, (self.n_latents, 0), value=True)
            .unsqueeze(1)
            .expand(-1, T, -1),
            "B T CpL -> (B T) CpL",
            B=B,
            T=T,
            CpL=CpL,
        )

        # Pad the freqs with zeros vectors on the left for identity ropes to the query latents.
        # They do not need to be spatially embeded because they are already free parameters.
        pos_rope_freqs = rearrange(
            F.pad(
                self.positions_to_freqs(channel_positions), (0, 0, self.n_latents, 0)
            ),
            "B CpL Rd -> (B CpL) Rd",
            B=B,
            CpL=CpL,
            Rd=self.rope_dim,
        )
        for block in self.blocks:
            latents = block(
                latents,
                source,
                T,
                seq_pos=sequence_positions,
                dynamic_rope_freqs=pos_rope_freqs,
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
    rope_dim: int
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
        rope_dim,
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
        self.rope_dim = rope_dim
        self.n_blocks = n_blocks
        self.tasks = [TaskConfig(**task) for task in tasks]
        self.data_config = DataConfig(**data_config)
        self.return_latents = ReturnLatents(return_latents)

    @property
    def tasks_map(self) -> dict[str, int]:
        out = {}
        for i, task in enumerate(sorted(self.tasks, key=lambda task: task.index)):
            assert task.key not in out
            assert task.index == i
            out[task.key] = i
        return out

    @property
    def labels_map(self) -> dict[str, int]:
        out = {}
        for task in self.tasks:
            for label in task.labels_map:
                assert label not in out
                out[label] = task.labels_map[label]
        return out


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
            config.rope_dim,
            config.return_latents,
            config.dropout,
            config.scale_exponent,
        )
        # TODO: This will be slow, but we can optimise later.
        self.task_heads = nn.ModuleList([
            nn.Linear(config.d_model * config.n_latents, task.n_classes)
            if config.return_latents != ReturnLatents.ALL
            else nn.Conv1d(config.d_model * config.n_latents, task.n_classes, 1)
            for task in config.tasks
        ])

        self.loss = partial(F.cross_entropy, label_smoothing=0.1)

    def compute_difficulty(
        self, epsilon: float, speech_densities: Tensor, labels: Tensor
    ):
        return epsilon * (
            (1 - labels) * speech_densities + labels * (1 - speech_densities)
        )

    def forward(
        self,
        channel_signals: Tensor,
        channel_positions: Tensor,
        sequence_positions: Tensor,
        task_keys: Tensor,
        labels: Tensor,
        channel_mask: Tensor | None = None,
        samples_mask: Tensor | None = None,
    ):
        """
        channel_signals: Tensor of shape (batch, channels, time).
        channel_positions: Tensor of shape (batch, channels, 3) with the position of each channel in the signal.
        sequence_positions: Tensor of shape (batch, samples) with the index of each sample in the signal.
            NB: Because each sample has potentially a different sampling rate, this will vary per sample.
        task_keys: Tensor of shape (batch,) with the key of the task for each sample.
        labels: Tensor of shape (batch,) with the label for each sample.
        channel_masks: Boolean tensor of shape (batch, channels) with True for each channel that should be included in the embedding and spatial attention.
        samples_mask: Boolean tensor of shape (batch, samples) with True for each sample that should be included in the embedding.

        """
        latents = self.encoder(
            channel_signals,
            channel_positions,
            sequence_positions,
            channel_mask,
            samples_mask,
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
