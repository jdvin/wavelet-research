from dataclasses import dataclass
from enum import Enum
import hashlib
from typing import Sequence

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F

from utils.electrode_utils import (
    EPOC14_CHANNELS,
    INSIGHT5_CHANNELS,
    LEMON_CHANNELS,
    NEUROTECHS_CHANNELS,
    PHYSIONET_64_CHANNELS,
    RESTING_METHODS_CHANNELS,
    ChannelMaskConfig,
)


# Task AND label dtype. 11 character string.
TASK_LABEL_DTYPE = np.dtype("<U32")


def _normalize_epochs(data: np.ndarray) -> np.ndarray:
    """Apply per-channel normalization along the last axis."""
    normalized = data.astype(np.float32, copy=True)
    mean = normalized.mean(axis=-1, keepdims=True)
    std = normalized.std(axis=-1, keepdims=True)
    np.maximum(std, 1e-6, out=std)
    normalized -= mean
    normalized /= std
    return normalized


def _hash_identifiers(values: Sequence[str]) -> str:
    if not values:
        return "none"
    joined = "||".join(sorted(values))
    return hashlib.sha1(joined.encode("utf-8")).hexdigest()[:16]


def get_spectrogram(signal: torch.Tensor, n_fft: int, hop_length: int):
    window = torch.hann_window(n_fft)
    signal = (signal - signal.mean()) / torch.sqrt(signal.var() + 1e-7)
    stft = torch.stft(signal, n_fft, hop_length, window=window, return_complex=True)
    # Freq 0 is not needed because the signal zero-centered.
    return stft[:, 1:, :-1].abs() ** 2


def mapped_label_ds_collate_fn(
    ds_samples: list[dict[str, torch.Tensor | int | np.memmap]],
) -> dict[str, torch.Tensor]:
    (
        channel_signals,
        channel_positons,
        sequence_positions,
        channel_masks,
        samples_masks,
        tasks,
        labels,
    ) = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )
    channel_counts, samples_counts = zip(*[
        (
            ds_sample["channel_signals"].shape[0],
            ds_sample["channel_signals"].shape[1],
        )
        for ds_sample in ds_samples
    ])
    max_channels = max(channel_counts)
    max_samples = max(samples_counts)
    for ds_sample in ds_samples:
        tasks.append(ds_sample["task_keys"])
        labels.append(ds_sample["labels"])

        cs = ds_sample["channel_signals"]
        cp = ds_sample["channel_positions"]
        sp = ds_sample["sequence_positions"]

        assert isinstance(cs, np.memmap)
        assert isinstance(cp, Tensor)
        assert isinstance(sp, Tensor)
        channel_mask = F.pad(
            torch.ones(cs.shape[0], dtype=torch.bool, device=cs.device),
            (0, max_channels - cs.shape[0]),
            value=False,
        )
        samples_mask = F.pad(
            torch.ones(cs.shape[1], dtype=torch.bool, device=cs.device),
            (0, max_samples - cs.shape[1]),
            value=False,
        )
        cs = F.pad(
            torch.tensor(cs),
            (0, max_samples - cs.shape[1], 0, max_channels - cs.shape[0]),
        )
        cp = F.pad(cp, (0, 0, 0, max_channels - cp.shape[0]))
        sp = F.pad(sp, (0, max_samples - sp.shape[0]))

        channel_signals.append(cs)
        channel_positons.append(cp)
        sequence_positions.append(sp)
        samples_masks.append(samples_mask)
        channel_masks.append(channel_mask)

    channel_signals_tensor = torch.stack(channel_signals)
    channel_positons_tensor = torch.stack(channel_positons)
    sequence_positions_tensor = torch.stack(sequence_positions)
    channel_masks_tensor = torch.stack(channel_masks).to(dtype=torch.bool)
    samples_masks_tensor = torch.stack(samples_masks).to(dtype=torch.bool)
    tasks_tensor = torch.tensor(tasks)
    labels_tensor = torch.tensor(labels)
    return {
        "channel_signals": channel_signals_tensor,
        "channel_positions": channel_positons_tensor,
        "sequence_positions": sequence_positions_tensor,
        "channel_mask": channel_masks_tensor,
        "samples_mask": samples_masks_tensor,
        "task_keys": tasks_tensor,
        "labels": labels_tensor,
    }


class Task(Enum):
    MOVE_EYES = "move_eyes"
    MOVE_LEFT_RIGHT_FIST = "move_left_right_fist"
    IMAGE_LEFT_RIGHT_FIST = "imag_left_right_fist"
    MOVE_FIST_FEET = "move_fist_feet"
    IMAG_FIST_FEET = "imag_fist_feet"
    # Hybrid tasks that force model to differentiate real and imagined movement.
    # Use by swapping into `EEG_MMI_SESSION_TO_TASK` below.
    MOVE_IMAGE_LEFT_RIGHT_FIST = "move_image_left_right_fist"
    MOVE_IMAGE_FIST_FEET = "move_image_fist_feet"
    MOVE_IMAGE_LEFT_RIGHT_FIST_BOTH_FIST_FEET = (
        "move_image_left_right_fist_both_fist_feet"
    )


class Annotation(Enum):
    EYES_OPEN = "eyes_open"
    EYES_CLOSED = "eyes_closed"
    REST = "rest"
    MOVE_LEFT_FIST = "move_left_fist"
    MOVE_RIGHT_FIST = "move_right_fist"
    MOVE_BOTH_FIST = "move_both_fist"
    MOVE_BOTH_FEET = "move_both_feet"
    IMAG_LEFT_FIST = "imag_left_fist"
    IMAG_RIGHT_FIST = "imag_right_fist"
    IMAG_BOTH_FIST = "imag_both_fist"
    IMAG_BOTH_FEET = "imag_both_feet"


class DataSource(Enum):
    EEG_MMI = "eeg_mmi"
    EMOTIVE_ALPHA = "emotive_alpha"
    LEMON_REST = "lemon_rest"
    NEUROTECHS = "neurotechs"
    RESTING_METHODS = "resting_methods"


class Headset(Enum):
    PHYSIONET_64 = "physionet_64"
    EMOTIV_INSIGHT_5 = "emotiv_insight_5"
    EMOTIV_EPOC_14 = "emotiv_epoc_14"
    LEMON_61 = "lemon_61"
    UNICORN_HYBRID_BLACK_8 = "unicorn_hybrid_black_8"
    BRAIN_ACTICHAMP_31 = "brain_actichamp_31"


HEADSET_TO_CHANNELS: dict[Headset, list[str]] = {
    Headset.EMOTIV_EPOC_14: EPOC14_CHANNELS,
    Headset.EMOTIV_INSIGHT_5: INSIGHT5_CHANNELS,
    Headset.LEMON_61: LEMON_CHANNELS,
    Headset.UNICORN_HYBRID_BLACK_8: NEUROTECHS_CHANNELS,
    Headset.BRAIN_ACTICHAMP_31: RESTING_METHODS_CHANNELS,
    Headset.PHYSIONET_64: PHYSIONET_64_CHANNELS,
}


@dataclass
class DataSplit:
    """Base class for data splits.

    Use to uniquely define a division of a particular dataset.
    Creates a unique code for the split."""

    split_name: str
    subjects: list[str]
    sessions: list[str]
    source_base_path: str
    output_path: str
    sampling_rate: int
    epoch_length_sec: float
    headset: Headset
    data_source: DataSource
    max_subjects: int = 0
    max_sessions: int = 0
    channel_mask_config: ChannelMaskConfig | None = None
    channel_mask: list[int] | None = None

    def __post_init__(self) -> None:
        if isinstance(self.data_source, str):
            self.data_source = DataSource(self.data_source)
        if isinstance(self.headset, str):
            self.headset = Headset(self.headset)

        if self.channel_mask_config is not None and isinstance(
            self.channel_mask_config, dict
        ):
            self.channel_mask_config = ChannelMaskConfig(**self.channel_mask_config)

        if not all(isinstance(subject, str) for subject in self.subjects):
            raise TypeError(
                "DataSplit subjects must be strings. Update configuration to provide normalized identifiers."
            )
        # Normalize whitespace without altering identifier semantics.
        self.subjects = [subject.strip() for subject in self.subjects]

        # Normalize session identifiers to strings.
        if not isinstance(self.sessions, list):
            raise TypeError("sessions must be provided as a list.")
        self.sessions = [str(session).strip() for session in self.sessions]

        if self.max_subjects:
            assert (
                len(self.subjects) <= self.max_subjects
            ), f"max_subjects={self.max_subjects}, subjects={len(self.subjects)}"
        if self.max_sessions:
            assert (
                len(self.sessions) <= self.max_sessions
            ), f"max_sessions={self.max_sessions}, sessions={len(self.sessions)}"

    def code(self) -> str:
        subjects_key = _hash_identifiers(self.subjects)
        session_tokens = [str(session).strip() for session in self.sessions]
        sessions_key = _hash_identifiers(session_tokens)
        return (
            f"ds-{self.data_source.value}--hs-{self.headset.value}"
            f"--sub-{subjects_key}--sess-{sessions_key}"
        )

    def __str__(self) -> str:
        return f"""DataSplit(
    split_name={self.split_name},
    subjects={self.subjects},
    sessions={self.sessions},
    data_source={self.data_source},
    headset={self.headset},
)
"""
