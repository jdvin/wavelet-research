import argparse
from dataclasses import dataclass
from enum import Enum

from loguru import logger
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
        return y


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
        FilterConfig((48.5, 51.5), FilterType.BANDSTOP),
    ]
    filter = StreamingSOSFilter(filter_configs, fs, n_channels)
    x = np.load(input_stub + "_eeg.npy")
    if x.shape[0] != n_channels:
        x = x.squeeze(0)
    y = np.load(input_stub + "_labels.npy")
    epoch_length_samples = int(round(epoch_length_sec * fs))
    epoch_overlap_samples = int(round(epoch_overlap_sec * fs))
    steps = np.arange(0, x.shape[-1], epoch_length_samples - epoch_overlap_samples)
    logger.info(
        f"Filtering {len(steps)} epochs with length {epoch_length_samples} and overlap {epoch_overlap_samples}."
    )
    x_ = np.lib.format.open_memmap(
        output_stub + "_eeg.npy",
        mode="w+",
        dtype=x.dtype,
        shape=(len(steps), n_channels, epoch_length_samples),
    )
    y_ = np.lib.format.open_memmap(
        output_stub + "_labels.npy",
        mode="w+",
        dtype=y.dtype,
        shape=(len(steps), 2),
    )
    for i_out, i_in in enumerate(steps):
        x_slice = x[:, i_in : i_in + epoch_length_samples]
        y_slice = y.copy()
        x_slice = filter.process(x_slice)
        x_slice = per_channel_normalize(x_slice)
        x_[i_out, ...] = x_slice
        y_[i_out, ...] = y_slice

    x_.flush()
    y_.flush()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filter EEG data.")
    parser.add_argument(
        "input_stub",
        type=str,
        help="Path stub to the EEG data to be filtered.",
    )
    parser.add_argument(
        "output_stub",
        type=str,
        help="Path stub to the filtered EEG data.",
    )
    parser.add_argument(
        "--fs",
        type=int,
        default=125,
        help="Sampling rate of the input EEG data.",
    )
    parser.add_argument(
        "--n-channels",
        type=int,
        default=16,
        help="Number of channels in the input EEG data.",
    )
    parser.add_argument(
        "--epoch-length-sec",
        type=float,
        default=5.0,
        help="Length of each epoch in seconds.",
    )
    parser.add_argument(
        "--epoch-overlap-sec",
        type=float,
        default=0.0,
        help="Overlap between epochs in seconds.",
    )
    args = parser.parse_args()
    main(
        args.input_stub,
        args.output_stub,
        args.fs,
        args.n_channels,
        args.epoch_length_sec,
        args.epoch_overlap_sec,
    )
