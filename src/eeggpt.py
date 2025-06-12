from dataclasses import dataclass, field

import torch
from torch import nn


class EegGptEncoderConfig:
    patch_len: int = 50
    d_model: int = 1024
    n_heads: int = 8
    d_mlp: int = 2048
    dropout: float = 0.0
    scale_exponent: float = -0.25
    n_blocks: int = 4


class EegGptEncoder(nn.Module):
    def __init__(self, config: EegGptEncoderConfig, rank: int, world_size: int):
        super().__init__()
        self.patch_len = config.patch_len

        self.embed = nn.Linear(self.patch_len, config.d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        assert T % self.patch_len == 0
        N_P = T // self.patch_len
        x = x.transpose(1, 2).reshape(B, C, N_P, self.patch_len)
        x = self.embed(x)
