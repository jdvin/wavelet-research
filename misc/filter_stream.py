import argparse
from dataclasses import dataclass
from enum import Enum
import numpy as np
from scipy import signal


class FilterType(Enum):
    BANDPASS = "bandpass"
    BANDSTOP = "bandstop"


@dataclass(frozen=True)
class FilterConfig:
    bounds: tuple[float, float]
    btype: FilterType


class StreamingSOSFilter:
    def __init__(self, filter_configs: list[FilterConfig], fs: int, n_channels: int):
        self.sos = [
            signal.butter(4, fc.bounds, btype=fc.btype.value, fs=fs, output="sos")
            for fc in filter_configs
        ]
        zi_base = [signal.sosfilt_zi(sos_i) for sos_i in self.sos]  # (n_sections, 2)
        # add channel dimension
        self.zi = [
            np.repeat(zi_base[:, np.newaxis, :], n_channels, axis=-2)
            for zi_base in zi_base
        ]

    def process(self, x: np.ndarray) -> np.ndarray:
        # x shape: (n_channels, n_samples)
        y = x.copy()
        for i in range(len(self.sos)):
            y, self.zi[i] = signal.sosfilt(self.sos[i], y, zi=self.zi[i], axis=-1)
        return x


def per_channel_median_shift(x: np.ndarray) -> np.ndarray:
    """Normalize each channel of a tensor independently."""
    return x - np.median(x, axis=-1, keepdims=True)


def per_channel_normalize(x: np.ndarray) -> np.ndarray:
    return (x - x.mean(axis=-1, keepdims=True)) / x.std(axis=-1, keepdims=True)


def main(
    input_stub: str,
    output_stub: str,
    fs: int,
    n_channels: int,
    epoch_length_sec: float,
    epoch_overlap_sec: float,
) -> None:
    filter_configs = [
        FilterConfig((1, 60), FilterType.BANDPASS),
        FilterConfig((49.5, 50.5), FilterType.BANDSTOP),
    ]
    filter = StreamingSOSFilter(filter_configs, fs, n_channels)
    x = np.load(input_stub + "_eeg.npy")
    if x.shape[0] != n_channels:
        x = x.squeeze(0)
    y = np.load(input_stub + "_labels.npy")
    epoch_length_samples = int(round(epoch_length_sec * fs))
    epoch_overlap_samples = int(round(epoch_overlap_sec * fs))
    steps = np.arange(0, x.shape[-1], epoch_length_samples - epoch_overlap_samples)
    x_ = np.memmap(
        output_stub + "_eeg.npy",
        mode="w+",
        dtype=x.dtype,
        shape=(len(steps), n_channels, epoch_length_samples),
    )
    y_ = np.memmap(
        output_stub + "_labels.npy",
        mode="w+",
        dtype=y.dtype,
        shape=(len(y), 2),
    )
    for i in steps:
        x_slice = x[:, i : i + epoch_length_samples]
        y_slice = y.copy()
        x_slice = filter.process(x_slice)
        x_slice = per_channel_median_shift(x_slice)
        x_[i, ...] = x_slice
        y_[i, ...] = y_slice
