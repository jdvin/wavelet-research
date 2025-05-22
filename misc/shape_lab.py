from dataclasses import dataclass
import torch
from torch import einsum as einsum
import torch.nn as nn
from torch import Tensor

from src.components.norm import RMSNorm
from src.components.attention import MultiHeadAttention
from src.components.activations import GEGLU


class PerceiverResamplerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_mlp: int,
        dropout: float = 0.0,
        scale_exponent: float = -0.25,
    ):
        super().__init__()

        self.cross_attn = MultiHeadAttention(
            n_heads,
            d_model,
            scale=(d_model // n_heads) ** scale_exponent,
            k_bias=True,
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
            source,
            attention_mask=attention_mask,
            kv_cache=kv_cache,
        )
        latents = latents + self.mlp(self.mlp_ln(latents))
        return latents


@dataclass
class EEGDataConfig:
    # Maximum number of channels used.
    max_channels: int = 64
    # Maximum sampling rate used.
    max_sr: int = 1000
    # Minimum sampling rate used.
    min_sr: int = 256
    # lowest frequency that the embedding can capture.
    f_min: int = 5


class EEGPerceiverResampler(nn.Module):
    def __init__(
        self,
        data_config: EEGDataConfig,
        n_latents: int,
        d_model: int,
        n_heads: int,
        d_mlp: int,
        n_blocks: int,
        dropout: float = 0.0,
        scale_exponent: float = -0.25,
    ):
        super().__init__()
        self.d_model = d_model
        self.query_latents = nn.Parameter(torch.randn(n_latents, d_model))
        emb_kernel_size = int(data_config.max_sr / data_config.f_min)
        emb_stride = int(data_config.max_sr / data_config.min_sr)
        self.embed = nn.Conv1d(
            in_channels=1,
            out_channels=d_model,
            kernel_size=emb_kernel_size,
            stride=emb_stride,
        )
        self.embed_l_out = lambda T: int((T - emb_kernel_size) / emb_stride + 1)
        self.resampler_blocks = nn.ModuleList(
            [
                PerceiverResamplerBlock(
                    d_model,
                    n_heads,
                    d_mlp,
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
        T_emb = self.embed_l_out(T)
        source = self.embed(source.reshape(B * C, 1, T)).reshape(
            B, C, self.d_model, T_emb
        )
        source = source.permute(0, 3, 1, 2).reshape(B * T_emb, C, self.d_model)
        latents = (
            self.query_latents.clone().unsqueeze(0).expand(B * T_emb, 1, self.d_model)
        )
        for block in self.resampler_blocks:
            latents = block(
                latents,
                source,
                attention_mask=attention_mask,
                kv_cache=kv_cache,
            )
        return latents.reshape(B, T_emb, self.d_model)


def create_mock_eeg(C, T, B):
    eeg = torch.randn(C, T)
    eegs = []
    for _ in range(B):
        # Create a random bit mask and mask out channels.
        mask = torch.randint(0, 2, size=(C,)).unsqueeze(1).expand(C, T).unsqueeze(0)
        eegs.append(eeg.clone() * mask)
    source = torch.cat(eegs, dim=0)
    return source


def main():
    L = 1
    T = 512
    D = 1024
    B = 4
    data_config = EEGDataConfig()
    source = create_mock_eeg(data_config.max_channels, T, B)
    perceiver = EEGPerceiverResampler(data_config, L, D, 8, D * 4, 4)
    out = perceiver(source)
    print(out.shape)


if __name__ == "__main__":
    main()
