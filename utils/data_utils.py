from dataclasses import dataclass, field
from enum import Enum
import hashlib
import os
from typing import Callable, Any, Sequence
import json
import requests
import re
from pathlib import Path

import mne
from loguru import logger
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from torch import Tensor
from tqdm import tqdm
from numpy.lib.format import open_memmap
from pnpl.datasets import LibriBrainSpeech

from src.montagenet import DataConfig, MontageNetConfig, TaskConfig
from utils.torch_datasets import (
    LibriBrainSpeechDataset,
    MappedLabelDataset,
    EEGEyeNetDataset,
    MultiMappedLabelDataset,
)
from utils.electrode_utils import (
    EPOC14_CHANNELS,
    EPOC14_CHANNEL_POSITIONS,
    INSIGHT5_CHANNELS,
    INSIGHT5_CHANNEL_POSITIONS,
    LEMON_CHANNELS,
    LEMON_CHANNEL_POSITIONS,
    NEUROTECHS_CHANNELS,
    NEUROTECHS_CHANNEL_POSITIONS,
    PHYSIONET_64_CHANNELS,
    RESTING_METHODS_CHANNELS,
    RESTING_METHODS_CHANNEL_POSITIONS,
    PHYSIONET_64_CHANNEL_POSITIONS,
    ChannelMaskConfig,
    create_mask,
)
from utils.train_utils import TrainingConfig, load_yaml


class ValidationType(Enum):
    DEFAULT = "default"
    RANDOM = "random"
    SUBJECT = "subject"
    OBJECT = "object"


ELECTRODE_ORDER = np.array([
    "Fp1",
    "F3",
    "F7",
    "FT9",
    "FC5",
    "FC1",
    "C3",
    "T7",
    "TP9",
    "CP5",
    "CP1",
    "Pz",
    "P3",
    "P7",
    "O1",
    "Oz",
    "O2",
    "P4",
    "P8",
    "TP10",
    "CP6",
    "CP2",
    "Cz",
    "C4",
    "T8",
    "FT10",
    "FC6",
    "FC2",
    "F4",
    "F8",
    "Fp2",
    "AF7",
    "AF3",
    "AFz",
    "F1",
    "F5",
    "FT7",
    "FC3",
    "FCz",
    "C1",
    "C5",
    "TP7",
    "CP3",
    "P1",
    "P5",
    "PO7",
    "PO3",
    "POz",
    "PO4",
    "PO8",
    "P6",
    "P2",
    "CPz",
    "CP4",
    "TP8",
    "C6",
    "C2",
    "FC4",
    "FT8",
    "F6",
    "F2",
    "AF4",
    "AF8",
])
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


def extract_eeg_eye_net_ds(
    root_dir: str,
    labels_map: dict[str, int] = {
        "saccade": 0,
        "blink": 1,
        "fixation": 2,
    },
):
    return {
        "train": EEGEyeNetDataset(os.path.join(root_dir, "train"), labels_map),
        "val": EEGEyeNetDataset(os.path.join(root_dir, "val"), labels_map),
    }


def extract_things_100ms_ds(
    root_dir: str,
    subjects: list[int] | range,
    validation_type: ValidationType = ValidationType.DEFAULT,
    epoch_start: int = -200,
    epoch_end: int = 200,
    is_test_run: bool = False,
    reset_cache: bool = False,
) -> dict[str, np.memmap]:
    """
    Extract the EEG data from the raw files and save it to a memmap file.

    Args:
        ...
        is_test_run: If True, construct a testing dataset consisting of only `SESSION_PER_SUBJECT` epochs, for the given subjects. Have the `test` set be the same as the training set.
                    Useful for debugging.
        ...

    This is going to be a doozy.

    This may look unecessarily verbose, but the goal here is to be able to go straight from the
    the raw files as they were given to a dataset of desired structure.
    That way, if the structure changes, it can just be reflected here, instead of having to do preprocessing each time.

    Side Note: I have no idea why the dude set the dataset up like this, would it not have made so much more sense to just
    use the common indexing scheme of THINGS between the training and test sets to begin with?
    """
    ds_str = (
        "-".join([str(sub) for sub in subjects]) + str(epoch_start) + str(epoch_end)
    )
    if is_test_run:
        sessions_per_subject = 1
        session_epochs = {"train": 120, "test": 120}
        ds_str += "_test"
    else:
        sessions_per_subject = 4
        session_epochs = {"train": 16710, "test": 4080}

    epoch_length = epoch_end - epoch_start
    target_obj_onset_idx = -1 * epoch_start
    if validation_type != ValidationType.DEFAULT:
        raise NotImplementedError("Only default validation type is supported atm.")
    # Keys are coded: `{split_type}_img_concepts_THINGS`.
    # Values are coded arrays of strings each coded: `{index}_{object_id}`.
    # The index is offset by +1 relative to THINGS probably because `0` is used as padding in the stim channel.
    eeg_img_metadata = {
        key.split("_")[0]: [float(obj.split("_")[0]) - 1 for obj in value]
        for key, value in np.load(f"{root_dir}/image_metadata.npy", allow_pickle=True)
        .all()
        .items()
        if "THINGS" in key
    }

    training_file_path = f"{root_dir}/{ds_str}_train.npy"
    test_file_path = f"{root_dir}/{ds_str}_test.npy"
    cached = (
        os.path.exists(training_file_path) and os.path.exists(test_file_path)
    ) and not reset_cache
    logger.info(f"Using cached dataset: {cached}.")

    def split_shape(epochs_per_session: int) -> tuple[int, int, int]:
        return (
            len(subjects) * sessions_per_subject * epochs_per_session,
            len(ELECTRODE_ORDER)
            + 2,  # +2 for the stimulus channel and subject channel.
            epoch_end - epoch_start,
        )

    train_split_shape = split_shape(session_epochs["train"])
    test_split_shape = split_shape(session_epochs["test"])
    ds = {
        "train": np.memmap(
            dtype=np.float32,
            filename=training_file_path,
            mode="r" if cached else "w+",
            shape=train_split_shape,
        ),
        "test": np.memmap(
            dtype=np.float32,
            filename=test_file_path,
            mode="r" if cached else "w+",
            shape=test_split_shape,
        ),
    }
    if cached and not reset_cache:
        return ds
    total_rows = sum([
        session_epochs[split_type] * sessions_per_subject * len(subjects)
        for split_type in ds.keys()
    ])
    split_types = ["train", "test"] if not is_test_run else ["train"]
    pbar = tqdm(total=total_rows, desc="Extracting EEG Data.")
    n = 0
    for sub_i, sub in enumerate(subjects):
        for ses_i, ses in enumerate(range(1, sessions_per_subject + 1)):
            for split_type in split_types:
                path = os.path.join(
                    root_dir,
                    f"sub-{'0' if sub <= 9 else ''}{sub}",
                    f"ses-0{ses}",
                    f"raw_eeg_{split_type}.npy",
                )
                data = np.load(path, allow_pickle=True).all()
                stim_index = data["ch_types"].index("stim")
                ch_names = np.array([
                    name for name in data["ch_names"] if name != "stim"
                ])
                # Ensure the order of the electrode order is consistent.
                # This may be overkill but it is very important, so worth being sure about.
                _, ordered_electrode_indexes = np.where(
                    ELECTRODE_ORDER[:, None] == ch_names
                )
                # Get the true THINGS id...
                # TODO: Why do we do this first, why not as we are iterating through the epoch indexes?
                stims = np.array(
                    [
                        (
                            eeg_img_metadata[split_type][int(i) - 1]
                            if i not in {0.0, 99999.0}
                            else i
                        )
                        for i in data["raw_eeg_data"][stim_index, :]
                    ],
                    dtype=data["raw_eeg_data"].dtype,
                )
                # Get ordered electrode data.
                data = data["raw_eeg_data"][ordered_electrode_indexes, :]
                # Stack with stimulus data.
                data = np.vstack((stims, data))
                # Get the index of each stimulus onset.
                epoch_indexes = data[0, :].nonzero()[0]
                epoch_i = 0
                for epoch_loc in epoch_indexes:
                    target_obj = data[0, epoch_loc]
                    if target_obj == 99999.0:
                        continue
                    # Get the absolute index of the current epoch.
                    n = (
                        sub_i * sessions_per_subject * session_epochs[split_type]
                        + ses_i * session_epochs[split_type]
                        + epoch_i
                    )
                    # Slice the current epoch out of the data stream.
                    # rows x ch x time <- ch x time.
                    ds[split_type][n, 1:, :] = data[
                        :, epoch_loc + epoch_start : epoch_loc + epoch_end
                    ]
                    assert ds[split_type][n, 1:, :].sum() > 0
                    # Label the stimulus channel and subject at the start of the epoch.
                    ds[split_type][n, 0, :] = np.full(
                        shape=epoch_length, fill_value=sub_i
                    )
                    ds[split_type][n, 1, :] = np.full(
                        shape=epoch_length,
                        fill_value=target_obj,
                    )
                    pbar.update(1)
                    epoch_i += 1
                    if is_test_run and epoch_i >= session_epochs[split_type]:
                        break
    if is_test_run:
        ds["test"] = ds["train"].copy()
    assert all([isinstance(value, np.memmap) for value in ds.values()])
    return ds


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


def get_things_100ms_collate_fn(
    tokenizer: Any,
    stop_token_id: int,
    pad_token_id: int,
    get_spectrogram: bool,
    start_token_id_sequence: Tensor | None = None,
    n_fft: int | None = None,
    fft_hop_length: int | None = None,
    things_concepts_path: str = "data/things_concepts.csv",
) -> Callable[[list[np.memmap]], dict[str, torch.Tensor]]:
    # Load the map from object ID to word.
    things_concepts = pd.read_csv(things_concepts_path)
    tokenizer.pad_token_id = pad_token_id
    stop_token = tokenizer.decode([stop_token_id])
    if get_spectrogram:
        assert n_fft is not None and fft_hop_length is not None

    # Define transformation function with parameters.
    def collate_fn(
        samples: list[np.memmap],
    ) -> dict[str, torch.Tensor]:
        batch_size = len(samples)
        object_words = []
        object_ids = []
        subject_ids = []
        eeg_features = []
        for sample in samples:
            subject_ids.append(sample[0][0])
            object_id = sample[1][0]
            object_ids.append(object_id)
            object_words.append(things_concepts["Word"][object_id])
            # If `get_spectrogram, sample is of shape (N_C, NF, T).
            # else, sampel is of shape (N_C, T)
            eeg_features.append(
                get_spectrogram(torch.tensor(sample[2:, :]), n_fft, fft_hop_length)
                if get_spectrogram
                else torch.tensor(sample[2:, :])
            )
        # We are doing all of the special tokens manually because (1) We do not trust HF, and (2) we have more control.
        # Here we add the stop token manually because it will then be included in the _attended to_ region of the attenton mask.
        # Which is not the default behaviour if we have `add_special_tokens=False`, because then all added tokens are treated like padding (i.e., not attended to).
        objects = [
            " " + object_word.lower().strip() + stop_token
            for object_word in object_words
        ]
        tokenizer_out = tokenizer.batch_encode_plus(
            objects,
            padding=True,
            add_special_tokens=False,
            return_tensors="pt",
        )
        attention_mask = tokenizer_out["attention_mask"]
        input_ids = tokenizer_out["input_ids"]
        assert isinstance(attention_mask, torch.Tensor)
        assert isinstance(input_ids, torch.Tensor)
        if start_token_id_sequence is not None:
            start_sequences = torch.tile(start_token_id_sequence, (batch_size, 1))
            input_ids = torch.cat([start_sequences, input_ids], dim=1)
            attention_mask = torch.cat(
                [torch.ones_like(start_sequences), attention_mask], dim=1
            )

        assert isinstance(eeg_features, list)

        return {
            "input_features": torch.stack(eeg_features),
            "object_ids": torch.tensor(object_ids).to(torch.long),
            "subject_ids": torch.tensor(subject_ids).to(torch.long),
            "input_ids": input_ids,
            "decoder_attention_mask": attention_mask,
        }

    return collate_fn


class Task(Enum):
    MOVE_EYES = "move_eyes"
    MOVE_LEFT_RIGHT_FIST = "move_left_right_fist"
    IMAGE_LEFT_RIGHT_FIST = "imag_left_right_fist"
    MOVE_FIST_FEET = "move_fist_feet"
    IMAG_FIST_FEET = "imag_fist_feet"


EEG_MMI_SESSION_TO_TASK = {
    "R01": Task.MOVE_EYES,
    "R02": Task.MOVE_EYES,
    "R03": Task.MOVE_LEFT_RIGHT_FIST,
    "R04": Task.IMAGE_LEFT_RIGHT_FIST,
    "R05": Task.MOVE_FIST_FEET,
    "R06": Task.IMAG_FIST_FEET,
    "R07": Task.MOVE_LEFT_RIGHT_FIST,
    "R08": Task.IMAGE_LEFT_RIGHT_FIST,
    "R09": Task.MOVE_FIST_FEET,
    "R10": Task.IMAG_FIST_FEET,
    "R11": Task.MOVE_LEFT_RIGHT_FIST,
    "R12": Task.IMAGE_LEFT_RIGHT_FIST,
    "R13": Task.MOVE_FIST_FEET,
    "R14": Task.IMAG_FIST_FEET,
}


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


EEG_MMI_SESSION_ANNOTATION_CODE_MAP: dict[str, dict[str, Annotation]] = {
    "R01": {
        "T0": Annotation.EYES_OPEN,
    },
    "R02": {
        "T0": Annotation.EYES_CLOSED,
    },
    "R03": {
        "T0": Annotation.REST,
        "T1": Annotation.MOVE_LEFT_FIST,
        "T2": Annotation.MOVE_RIGHT_FIST,
    },
    "R04": {
        "T0": Annotation.REST,
        "T1": Annotation.IMAG_LEFT_FIST,
        "T2": Annotation.IMAG_RIGHT_FIST,
    },
    "R05": {
        "T0": Annotation.REST,
        "T1": Annotation.MOVE_BOTH_FIST,
        "T2": Annotation.MOVE_BOTH_FEET,
    },
    "R06": {
        "T0": Annotation.REST,
        "T1": Annotation.IMAG_BOTH_FIST,
        "T2": Annotation.IMAG_BOTH_FEET,
    },
    "R07": {
        "T0": Annotation.REST,
        "T1": Annotation.MOVE_LEFT_FIST,
        "T2": Annotation.MOVE_RIGHT_FIST,
    },
    "R08": {
        "T0": Annotation.REST,
        "T1": Annotation.IMAG_LEFT_FIST,
        "T2": Annotation.IMAG_RIGHT_FIST,
    },
    "R09": {
        "T0": Annotation.REST,
        "T1": Annotation.MOVE_BOTH_FIST,
        "T2": Annotation.MOVE_BOTH_FEET,
    },
    "R10": {
        "T0": Annotation.REST,
        "T1": Annotation.IMAG_BOTH_FIST,
        "T2": Annotation.IMAG_BOTH_FEET,
    },
    "R11": {
        "T0": Annotation.REST,
        "T1": Annotation.MOVE_LEFT_FIST,
        "T2": Annotation.MOVE_RIGHT_FIST,
    },
    "R12": {
        "T0": Annotation.REST,
        "T1": Annotation.IMAG_LEFT_FIST,
        "T2": Annotation.IMAG_RIGHT_FIST,
    },
    "R13": {
        "T0": Annotation.REST,
        "T1": Annotation.MOVE_BOTH_FIST,
        "T2": Annotation.MOVE_BOTH_FEET,
    },
    "R14": {
        "T0": Annotation.REST,
        "T1": Annotation.IMAG_BOTH_FIST,
        "T2": Annotation.IMAG_BOTH_FEET,
    },
}


EMOTIV_EVENT_TYPE_TO_LABEL = {
    "eyesopen_element": Annotation.EYES_OPEN,
    "eyesopen": Annotation.EYES_OPEN,
    "eyesclose_element": Annotation.EYES_CLOSED,
    "eyesclose": Annotation.EYES_CLOSED,
}

EMOTIV_TASK_LABEL = Task.MOVE_EYES

EMOTIV_FILENAME_PATTERN = re.compile(
    r"^(?P<root>Alpha Supression_(?P<headset>[A-Z0-9]+)_.+?)(?P<suffix>_markers)?_S(?P<subject>\d{3})\.(?P<extension>csv|json)$"
)


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


EMOTIV_HEADSET_ALIASES: dict[str, Headset] = {
    "EPOC": Headset.EMOTIV_EPOC_14,
    "EPOCPLUS": Headset.EMOTIV_EPOC_14,
    "EPOCX": Headset.EMOTIV_EPOC_14,
    "INSIGHT": Headset.EMOTIV_INSIGHT_5,
}

HEADSET_TO_CHANNELS: dict[Headset, list[str]] = {
    Headset.EMOTIV_EPOC_14: EPOC14_CHANNELS,
    Headset.EMOTIV_INSIGHT_5: INSIGHT5_CHANNELS,
    Headset.LEMON_61: LEMON_CHANNELS,
    Headset.UNICORN_HYBRID_BLACK_8: NEUROTECHS_CHANNELS,
    Headset.BRAIN_ACTICHAMP_31: RESTING_METHODS_CHANNELS,
    Headset.PHYSIONET_64: PHYSIONET_64_CHANNELS,
}


@dataclass
class EmotivEventInfo:
    label: str
    start_idx: int
    end_idx: int
    num_epochs: int


@dataclass
class EmotivFileBundle:
    data_path: Path
    marker_path: Path
    subject_id: int
    headset: Headset


@dataclass
class EmotivRecordingInfo:
    bundle: EmotivFileBundle
    sample_count: int
    events: list[EmotivEventInfo]


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


@dataclass
class EmotivAlphaDataSplit(DataSplit):
    data_source: DataSource = DataSource.EMOTIVE_ALPHA
    max_sessions: int = 1

    def __post_init__(self) -> None:
        if not self.sessions:
            self.sessions = ["1"]
        normalized_sessions: list[str] = []
        for session in self.sessions:
            session_str = str(session).strip()
            if not session_str:
                session_str = "1"
            normalized_sessions.append(session_str)
        self.sessions = normalized_sessions
        normalized_subjects: list[str] = []
        for subject in self.subjects:
            if not isinstance(subject, str):
                raise ValueError("Emotiv Alpha subjects must be strings like 'S001'.")
            subject_code = subject.strip().upper()
            if not subject_code.startswith("S"):
                subject_code = f"S{subject_code}"
            if not subject_code[1:].isdigit():
                raise ValueError(f"Invalid Emotiv Alpha subject identifier: {subject}.")
            normalized_subjects.append(f"S{int(subject_code[1:]):03d}")
        self.subjects = normalized_subjects
        super().__post_init__()


@dataclass
class EmotivAlphaInsightDataSplit(EmotivAlphaDataSplit):
    max_subjects: int = 14
    headset: Headset = Headset.EMOTIV_INSIGHT_5


@dataclass
class EmotivAlphaEpocDataSplit(EmotivAlphaDataSplit):
    max_subjects: int = 13
    headset: Headset = Headset.EMOTIV_EPOC_14


@dataclass
class LemonRestingStateDataSplit(DataSplit):
    data_source: DataSource = DataSource.LEMON_REST
    headset: Headset = Headset.LEMON_61
    max_subjects: int = 215
    max_sessions: int = 1
    resample_sfreq: float | None = None

    def __post_init__(self) -> None:
        if not self.sessions:
            self.sessions = ["1"]
        normalized_sessions: list[str] = []
        for session in self.sessions:
            session_str = str(session).strip().lower()
            if session_str.isdigit():
                session_str = f"ses-{int(session_str):02d}"
            elif not session_str.startswith("ses-"):
                session_str = f"ses-{session_str}"
            normalized_sessions.append(session_str)
        self.sessions = normalized_sessions
        if self.max_subjects == 0 and self.subjects:
            self.max_subjects = len(self.subjects)
        normalized_subjects: list[str] = []
        for subject in self.subjects:
            if not isinstance(subject, str):
                raise ValueError(
                    "LEMON resting-state subjects must be strings like 'sub-010002'."
                )
            subj = subject.strip().lower()
            if not subj.startswith("sub-"):
                subj = f"sub-{subj}"
            normalized_subjects.append(subj)
        self.subjects = normalized_subjects
        super().__post_init__()
        if self.resample_sfreq is None and self.sampling_rate is not None:
            self.resample_sfreq = float(self.sampling_rate)

    def subject_ids(self) -> list[str]:
        return list(self.subjects)


@dataclass
class NeurotechsEyesDataSplit(DataSplit):
    data_source: DataSource = DataSource.NEUROTECHS
    headset: Headset = Headset.UNICORN_HYBRID_BLACK_8
    max_subjects: int = 100
    max_sessions: int = 2
    eyes_closed_duration_sec: float = 60.0
    eyes_open_duration_sec: float = 60.0
    baseline_offset_sec: float = 0.0
    task_name: str = "STEMSKILLS"

    def __post_init__(self) -> None:
        if not self.sessions:
            self.sessions = ["1", "2"]
        normalized_sessions: list[str] = []
        for session in self.sessions:
            session_str = str(session).strip().lower()
            if session_str.isdigit():
                session_str = f"ses-{int(session_str):02d}"
            elif not session_str.startswith("ses-"):
                session_str = f"ses-{session_str}"
            normalized_sessions.append(session_str)
        self.sessions = normalized_sessions
        normalized_subjects: list[str] = []
        for subject in self.subjects:
            if not isinstance(subject, str):
                raise ValueError(
                    "Neurotechs splits require subject identifiers as strings like 'sub-01c'."
                )
            subj = subject.strip().lower()
            if not subj.startswith("sub-"):
                subj = f"sub-{subj}"
            normalized_subjects.append(subj)
        self.subjects = normalized_subjects
        super().__post_init__()

    def subject_ids(self) -> list[str]:
        return list(self.subjects)


@dataclass
class RestingEEGMethodsDataSplit(DataSplit):
    data_source: DataSource = DataSource.RESTING_METHODS
    headset: Headset = Headset.BRAIN_ACTICHAMP_31
    max_subjects: int = 99
    max_sessions: int = 2
    block_sequence: tuple[Annotation, ...] = (
        Annotation.EYES_OPEN,
        Annotation.EYES_CLOSED,
        Annotation.EYES_OPEN,
        Annotation.EYES_CLOSED,
    )
    block_duration_sec: float = 135.0
    baseline_offset_sec: float = 0.0
    task_name: str = "rest"

    def __post_init__(self) -> None:
        if not self.sessions:
            self.sessions = ["ses-pre", "ses-post"]
        normalized_sessions: list[str] = []
        for session in self.sessions:
            if isinstance(session, str):
                sess = session.lower()
                if not sess.startswith("ses-"):
                    sess = f"ses-{sess}"
                normalized_sessions.append(sess)
            else:
                raise ValueError(
                    "Resting EEG methods sessions must be provided as strings like 'pre' or 'ses-pre'."
                )
        self.sessions = normalized_sessions
        normalized_subjects: list[str] = []
        for subject in self.subjects:
            if not isinstance(subject, str):
                raise ValueError(
                    "Resting EEG methods subjects must be strings like 'sub-01'."
                )
            subj = subject.strip().lower()
            if not subj.startswith("sub-"):
                subj = f"sub-{subj}"
            normalized_subjects.append(subj)
        self.subjects = normalized_subjects
        super().__post_init__()
        if len(self.block_sequence) == 0:
            raise ValueError("block_sequence must contain at least one entry.")
        for block in self.block_sequence:
            if block not in RESTING_BLOCK_LABELS:
                raise ValueError(
                    f"Unknown block name '{block}'. Expected keys: {list(RESTING_BLOCK_LABELS.keys())}."
                )

    def subject_ids(self) -> list[str]:
        return list(self.subjects)


@dataclass
class EEGMMIDataSplit(DataSplit):
    data_source: DataSource = DataSource.EEG_MMI
    headset: Headset = Headset.PHYSIONET_64
    max_subjects: int = 109
    max_sessions: int = 14

    def __post_init__(self) -> None:
        normalized_subjects: list[str] = []
        for subject in self.subjects:
            if not isinstance(subject, str):
                raise ValueError("EEG MMI subjects must be strings like 'S001'.")
            subject_code = subject.strip().upper()
            if not subject_code.startswith("S"):
                subject_code = f"S{subject_code}"
            if not subject_code[1:].isdigit():
                raise ValueError(f"Invalid EEG MMI subject identifier: {subject}.")
            normalized_subjects.append(f"S{int(subject_code[1:]):03d}")
        self.subjects = normalized_subjects

        normalized_sessions: list[str] = []
        for session in self.sessions:
            if not isinstance(session, str):
                raise ValueError("EEG MMI sessions must be strings like 'R01'.")
            session_code = session.strip().upper()
            if not session_code.startswith("R"):
                session_code = f"R{session_code}"
            if not session_code[1:].isdigit():
                raise ValueError(f"Invalid EEG MMI session identifier: {session}.")
            normalized_sessions.append(f"R{int(session_code[1:]):02d}")
        self.sessions = normalized_sessions

        super().__post_init__()
        assert self.headset == Headset.PHYSIONET_64
        assert self.data_source == DataSource.EEG_MMI


def extract_eeg_mmi_session_data(
    base_path: str,
    output_path: str,
    subject: str,
    session: str,
    ignore_cache: bool = False,
    epoch_length_sec: float | None = None,
    sampling_rate: int | None = None,
    default_event_length_sec: float | None = None,
) -> tuple[np.memmap, np.memmap]:
    """Extract the raw EEG and annotations from an edf file return
    a np.memmap of shape (N_E, N_C, T) where:
        N_E: number of epochs.
        N_C: number of channels.
        T: number of time samples.
    containing EEG data and a memmap of shape (N_E) containing the annotations
    """
    subject_code = subject.strip().upper()
    session_code = session.strip().upper()
    if not subject_code.startswith("S"):
        raise ValueError(f"EEG MMI subject identifiers must start with 'S': {subject}.")
    if not session_code.startswith("R"):
        raise ValueError(f"EEG MMI session identifiers must start with 'R': {session}.")
    output_eeg_path = os.path.join(
        base_path, subject_code, f"{subject_code}{session_code}_eeg.npy"
    )
    output_labels_path = os.path.join(
        base_path, subject_code, f"{subject_code}{session_code}_labels.npy"
    )
    cached = (
        os.path.exists(output_eeg_path)
        and os.path.exists(output_labels_path)
        and not ignore_cache
    )
    source_path = os.path.join(
        base_path, subject_code, f"{subject_code}{session_code}.edf"
    )
    data = mne.io.read_raw_edf(source_path)
    data_sfreq = float(data.info["sfreq"])
    if sampling_rate is None:
        sampling_rate = int(round(data_sfreq))
    else:
        if not np.isclose(sampling_rate, data_sfreq):
            raise ValueError(
                f"Sampling rate mismatch for {subject_code}{session_code}: "
                f"expected {sampling_rate}, found {data_sfreq}."
            )
    if epoch_length_sec is None:
        raise ValueError("epoch_length_sec must be provided for EEG MMI extraction.")
    epoch_length_samples = int(round(epoch_length_sec * sampling_rate))
    if not np.isclose(epoch_length_samples, epoch_length_sec * sampling_rate):
        raise ValueError(
            "epoch_length_sec * sampling_rate must produce an integer number of samples."
        )
    if default_event_length_sec is None:
        default_event_length_sec = epoch_length_sec
    default_event_length_samples = int(round(default_event_length_sec * sampling_rate))
    if not np.isclose(
        default_event_length_samples, default_event_length_sec * sampling_rate
    ):
        raise ValueError(
            "default_event_length_sec * sampling_rate must produce an integer number of samples."
        )

    events, event_map = mne.events_from_annotations(data)
    # Reverse the mapping to go from labels to annotations.
    labels_to_annotations = {value: key for key, value in event_map.items()}
    eeg_data = data.get_data()
    assert isinstance(eeg_data, np.ndarray)
    num_channels, num_samples = eeg_data.shape
    if events.shape[0] == 2:
        # Huge hack for one specific anamoly.
        events = events[0:1]
    # For the eyes open, eyes closed tasks, we have a single event.
    # So split it up into multiple pseudo events.
    if events.shape[0] == 1:
        real_total_duration = (eeg_data.sum(axis=0) != 0).sum()
        num_events = real_total_duration // default_event_length_samples
        events = events.repeat(num_events, axis=0)
        events[:, 0] = np.arange(num_events) * default_event_length_samples

    # Get the maximum duration between two consecutive events.
    if events.shape[0] > 1:
        max_event_duration = int(np.max(np.diff(events[:, 0])))
    else:
        max_event_duration = default_event_length_samples
    session_eeg = np.memmap(
        filename=os.path.join(output_path, f"{subject_code}{session_code}_eeg.npy"),
        mode="r" if cached else "w+",
        shape=(len(events), num_channels, max_event_duration),
        dtype=eeg_data.dtype,
    )
    session_labels = np.memmap(
        filename=os.path.join(output_path, f"{subject_code}{session_code}_labels.npy"),
        mode="r" if cached else "w+",
        shape=(len(events), 2),
        dtype=TASK_LABEL_DTYPE,
    )
    if cached:
        return session_eeg, session_labels

    events_list = events.tolist()
    final_event_stop = events_list[-1][0] + max_event_duration
    for i, (current_event, next_event) in enumerate(
        zip(
            events_list,
            events_list[1:] + [[final_event_stop, 0, 0]],
        )
    ):
        # TODO: cbf i think these are sometimes ints and sometimes lists depending on the task.
        event_start = current_event[0]
        event_stop = next_event[0]
        event_duration = event_stop - event_start
        annotation_code = labels_to_annotations[current_event[2]]
        task = EEG_MMI_SESSION_TO_TASK[session].value
        annotation = EEG_MMI_SESSION_ANNOTATION_CODE_MAP[session][
            annotation_code.item()
        ].value
        epoch_slice = _normalize_epochs(eeg_data[:, event_start:event_stop])
        session_eeg[i, :, 0:event_duration] = epoch_slice
        session_labels[i, :] = [task, annotation]

    session_eeg.flush()
    session_labels.flush()
    return session_eeg, session_labels


def extract_eeg_mmi_split(
    base_path: str,
    split: EEGMMIDataSplit,
    output_path: str,
    epoch_length_sec: float | None = None,
    reset_cache: bool = False,
):
    eeg_path = os.path.join(output_path, f"{split.split_name}_{split.code()}_eeg.npy")
    labels_path = os.path.join(
        output_path, f"{split.split_name}_{split.code()}_labels.npy"
    )
    if os.path.exists(eeg_path) and os.path.exists(labels_path) and not reset_cache:
        return np.load(eeg_path, mmap_mode="r", allow_pickle=True), np.load(
            labels_path, mmap_mode="r", allow_pickle=True
        )
    eegs, task_labels = [], []
    # Store information about the shape of each session.
    # [n_trials, n_channels, n_samples]
    shapes = np.zeros((len(split.subjects) * len(split.sessions), 3), dtype=int)
    target_epoch_length_sec = epoch_length_sec or split.epoch_length_sec
    if target_epoch_length_sec is None:
        raise ValueError("EEG MMI splits require an epoch_length_sec.")
    for i, subject in enumerate(split.subjects):
        for j, session in enumerate(split.sessions):
            eeg, label = extract_eeg_mmi_session_data(
                base_path,
                output_path,
                subject,
                session,
                reset_cache,
                epoch_length_sec=target_epoch_length_sec,
                sampling_rate=split.sampling_rate,
                default_event_length_sec=target_epoch_length_sec,
            )
            eegs.append(eeg)
            task_labels.append(label)
            shapes[i * len(split.sessions) + j] = eeg.shape
    n_trials = int(shapes[:, 0].sum())
    n_channels = int(shapes[:, 1].max())
    n_samples = int(shapes[:, 2].max())
    split_eeg = np.zeros(
        shape=(n_trials, n_channels, n_samples),
        dtype=eegs[0].dtype,
    )
    # We decide here that the labels are strings with 11 characters.
    # Labels are: [task, annotation].
    split_labels = np.zeros(
        shape=(n_trials, 2),
        dtype=TASK_LABEL_DTYPE,
    )
    cum_trial = 0
    for shape, eeg, task_label in zip(shapes, eegs, task_labels):
        split_eeg[cum_trial : cum_trial + shape[0], 0 : shape[1], 0 : shape[2]] = eeg
        split_labels[cum_trial : cum_trial + shape[0]] = task_label
        cum_trial += shape[0]

    np.save(eeg_path, split_eeg)
    np.save(labels_path, split_labels)
    return np.load(eeg_path, mmap_mode="r", allow_pickle=True), np.load(
        labels_path, mmap_mode="r", allow_pickle=True
    )


def _discover_emotiv_recordings(
    base_path: str | Path,
    subjects: list[str],
    expected_headset: Headset,
) -> list[EmotivFileBundle]:
    base = Path(base_path)
    subject_filter: set[str] | None = None
    if subjects:
        normalized: set[str] = set()
        for subject in subjects:
            if not isinstance(subject, str):
                raise ValueError("Emotiv Alpha subjects must be strings like 'S001'.")
            subject_code = subject.strip().upper()
            if not subject_code.startswith("S"):
                subject_code = f"S{subject_code}"
            if not subject_code[1:].isdigit():
                raise ValueError(f"Invalid Emotiv Alpha subject identifier: {subject}.")
            normalized.add(f"S{int(subject_code[1:]):03d}")
        subject_filter = normalized
    recordings: list[EmotivFileBundle] = []

    for csv_path in base.rglob("Alpha Supression_*_S???.csv"):
        stem = csv_path.stem
        if stem.endswith("_markers"):
            continue
        match = EMOTIV_FILENAME_PATTERN.match(csv_path.name)
        if not match or match.group("suffix"):
            continue
        alias = match.group("headset")
        headset = EMOTIV_HEADSET_ALIASES.get(alias)
        if headset is None or headset != expected_headset:
            continue
        subject_id = int(match.group("subject"))
        subject_code = f"S{subject_id:03d}"
        if subject_filter is not None and subject_code not in subject_filter:
            continue
        root = match.group("root")
        marker_name = f"{root}_markers_S{subject_id:03d}.csv"
        marker_path = csv_path.with_name(marker_name)
        if not marker_path.exists():
            logger.warning(f"Missing marker file {marker_name} for {csv_path}")
            continue
        recordings.append(
            EmotivFileBundle(
                data_path=csv_path,
                marker_path=marker_path,
                subject_id=subject_id,
                headset=headset,
            )
        )

    recordings.sort(key=lambda item: (item.subject_id, item.data_path.name))
    return recordings


def _build_emotiv_recording_info(
    bundle: EmotivFileBundle,
    epoch_length: int,
) -> EmotivRecordingInfo:
    try:
        timestamps_df = pd.read_csv(
            bundle.data_path, usecols=["Timestamp"], encoding="utf-8-sig"
        )
    except ValueError:
        full_df = pd.read_csv(bundle.data_path, encoding="utf-8-sig")
        if "Timestamp" not in full_df.columns:
            raise ValueError(f"Timestamp column missing in {bundle.data_path}")
        timestamps_df = full_df[["Timestamp"]]
    timestamps = timestamps_df["Timestamp"].to_numpy(dtype=np.float64)
    sample_count = int(timestamps.shape[0])
    if sample_count == 0:
        return EmotivRecordingInfo(bundle=bundle, sample_count=0, events=[])

    markers_df = pd.read_csv(
        bundle.marker_path,
        usecols=["timestamp", "duration", "type"],
    )
    markers_df["timestamp"] = pd.to_numeric(markers_df["timestamp"], errors="coerce")
    markers_df["duration"] = pd.to_numeric(markers_df["duration"], errors="coerce")
    markers_df = markers_df.dropna(subset=["timestamp", "duration", "type"])

    events: list[EmotivEventInfo] = []
    for _, row in markers_df.iterrows():
        event_type = str(row["type"]).strip().lower()
        label = EMOTIV_EVENT_TYPE_TO_LABEL.get(event_type)
        if label is None:
            continue
        label = label.value
        start_time = float(row["timestamp"])
        duration = float(row["duration"])
        if duration <= 0:
            continue
        start_idx = int(np.searchsorted(timestamps, start_time, side="left"))
        if start_idx >= sample_count:
            continue
        end_time = start_time + duration
        end_idx = int(np.searchsorted(timestamps, end_time, side="right"))
        end_idx = min(end_idx, sample_count)
        num_samples = end_idx - start_idx
        if num_samples <= 0:
            continue
        num_epochs = num_samples // epoch_length
        if num_epochs <= 0:
            continue
        effective_end_idx = start_idx + num_epochs * epoch_length
        events.append(
            EmotivEventInfo(
                label=label,
                start_idx=start_idx,
                end_idx=effective_end_idx,
                num_epochs=num_epochs,
            )
        )

    return EmotivRecordingInfo(bundle=bundle, sample_count=sample_count, events=events)


def _write_emotiv_recording_data(
    recording: EmotivRecordingInfo,
    epoch_length: int,
    channels: list[str],
    eeg_store: np.ndarray,
    labels_store: np.ndarray,
    start_epoch_idx: int,
) -> int:
    usecols = ["Timestamp"] + [f"EEG.{channel}" for channel in channels]
    try:
        df = pd.read_csv(
            recording.bundle.data_path, usecols=usecols, encoding="utf-8-sig"
        )
    except ValueError:
        df = pd.read_csv(recording.bundle.data_path, encoding="utf-8-sig")
        if "Timestamp" not in df.columns:
            raise ValueError(
                f"Timestamp column missing in {recording.bundle.data_path}"
            )
        missing_cols = [col for col in usecols if col not in df.columns]
        for col in missing_cols:
            if col == "Timestamp":
                continue
            df[col] = 0.0
        df = df[usecols]
    timestamps = df.pop("Timestamp").to_numpy(dtype=np.float64)
    if timestamps.shape[0] != recording.sample_count:
        logger.warning(
            f"Timestamp count mismatch for {recording.bundle.data_path} "
            f"(expected {recording.sample_count}, got {timestamps.shape[0]})"
        )

    signals = np.zeros((len(channels), timestamps.shape[0]), dtype=np.float32)
    for idx, channel in enumerate(channels):
        column_name = f"EEG.{channel}"
        if column_name not in df:
            continue
        channel_series = pd.to_numeric(df[column_name], errors="coerce").fillna(0.0)
        data = channel_series.to_numpy(dtype=np.float32, copy=False)
        signals[idx, : data.shape[0]] = data

    next_epoch_idx = start_epoch_idx
    for event in recording.events:
        epoch_start = event.start_idx
        for _ in range(event.num_epochs):
            epoch_end = epoch_start + epoch_length
            if epoch_end > signals.shape[1]:
                break
            epoch_slice = _normalize_epochs(signals[:, epoch_start:epoch_end])
            eeg_store[next_epoch_idx, :, :] = epoch_slice
            labels_store[next_epoch_idx, 0] = EMOTIV_TASK_LABEL.value
            labels_store[next_epoch_idx, 1] = event.label
            next_epoch_idx += 1
            epoch_start = epoch_end
    return next_epoch_idx


def extract_emotiv_alpha_suppression_split(
    split: EmotivAlphaDataSplit,
    reset_cache: bool = False,
) -> tuple[np.memmap, np.memmap]:
    epoch_length_samples = int(round(split.epoch_length_sec * split.sampling_rate))
    if not np.isclose(
        epoch_length_samples, split.epoch_length_sec * split.sampling_rate
    ):
        raise ValueError(
            "epoch_length_sec * sampling_rate must produce an integer number of samples."
        )
    os.makedirs(split.output_path, exist_ok=True)
    cache_prefix = f"{split.split_name}_{split.code()}"
    eeg_output_path = os.path.join(split.output_path, f"{cache_prefix}_eeg.npy")
    labels_output_path = os.path.join(split.output_path, f"{cache_prefix}_labels.npy")

    if (
        os.path.exists(eeg_output_path)
        and os.path.exists(labels_output_path)
        and not reset_cache
    ):
        return np.load(eeg_output_path, mmap_mode="r", allow_pickle=True), np.load(
            labels_output_path, mmap_mode="r", allow_pickle=True
        )

    recordings_files = _discover_emotiv_recordings(
        base_path=split.source_base_path,
        subjects=split.subjects,
        expected_headset=split.headset,
    )
    if not recordings_files:
        raise ValueError(
            "No Emotiv recordings matched the requested subjects/headset configuration."
        )

    recordings: list[EmotivRecordingInfo] = []
    total_epochs = 0
    for bundle in recordings_files:
        recording = _build_emotiv_recording_info(bundle, epoch_length_samples)
        epoch_count = sum(event.num_epochs for event in recording.events)
        if epoch_count == 0:
            logger.warning(f"No eyes-open/closed epochs found in {bundle.data_path}")
            continue
        recordings.append(recording)
        total_epochs += epoch_count

    if total_epochs == 0:
        raise ValueError(
            "No eyes-open or eyes-closed epochs were extracted from the Emotiv dataset."
        )

    channels = HEADSET_TO_CHANNELS.get(split.headset)
    if channels is None:
        raise NotImplementedError(
            f"Unsupported headset for extraction: {split.headset}"
        )

    eeg_store = np.zeros(
        (total_epochs, len(channels), epoch_length_samples), dtype=np.float32
    )
    labels_store = np.empty((total_epochs, 2), dtype=TASK_LABEL_DTYPE)

    epoch_cursor = 0
    for recording in recordings:
        epoch_cursor = _write_emotiv_recording_data(
            recording=recording,
            epoch_length=epoch_length_samples,
            channels=channels,
            eeg_store=eeg_store,
            labels_store=labels_store,
            start_epoch_idx=epoch_cursor,
        )

    np.save(eeg_output_path, eeg_store)
    np.save(labels_output_path, labels_store)

    return np.load(eeg_output_path, mmap_mode="r", allow_pickle=True), np.load(
        labels_output_path, mmap_mode="r", allow_pickle=True
    )


def extract_neurotechs_eyes_split(
    split: NeurotechsEyesDataSplit,
    reset_cache: bool = False,
) -> tuple[np.memmap, np.memmap]:
    """
    Extract the eyes-closed (first minute) and eyes-open (second minute)
    baseline segments provided at the start of each Neurotechs recording.

    Each segment is divided into non-overlapping epochs of length
    ``split.epoch_length_sec`` and interleaved as
    [closed_0, open_0, closed_1, open_1, ...].
    """

    base_path = Path(split.source_base_path).expanduser().resolve()
    output_path = Path(split.output_path).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    cache_prefix = f"{split.split_name}_{split.code()}"
    eeg_output_path = output_path / f"{cache_prefix}_eeg.npy"
    labels_output_path = output_path / f"{cache_prefix}_labels.npy"
    channels_output_path = output_path / f"{cache_prefix}_channels.json"

    if eeg_output_path.exists() and labels_output_path.exists() and not reset_cache:
        return np.load(eeg_output_path, mmap_mode="r", allow_pickle=True), np.load(
            labels_output_path, mmap_mode="r", allow_pickle=True
        )

    epoch_samples = int(round(split.epoch_length_sec * split.sampling_rate))
    if not np.isclose(epoch_samples, split.epoch_length_sec * split.sampling_rate):
        raise ValueError(
            "epoch_length_sec * sampling_rate must produce an integer number of samples."
        )
    closed_samples = int(round(split.eyes_closed_duration_sec * split.sampling_rate))
    if not np.isclose(
        closed_samples, split.eyes_closed_duration_sec * split.sampling_rate
    ):
        raise ValueError(
            "eyes_closed_duration_sec * sampling_rate must produce an integer number of samples."
        )
    open_samples = int(round(split.eyes_open_duration_sec * split.sampling_rate))
    if not np.isclose(open_samples, split.eyes_open_duration_sec * split.sampling_rate):
        raise ValueError(
            "eyes_open_duration_sec * sampling_rate must produce an integer number of samples."
        )
    offset_samples = int(round(split.baseline_offset_sec * split.sampling_rate))
    if not np.isclose(offset_samples, split.baseline_offset_sec * split.sampling_rate):
        raise ValueError(
            "baseline_offset_sec * sampling_rate must produce an integer number of samples."
        )

    segments_per_condition = min(closed_samples, open_samples) // epoch_samples
    if segments_per_condition <= 0:
        raise ValueError(
            "Epoch length is longer than the baseline window; no segments can be created."
        )

    channels = HEADSET_TO_CHANNELS.get(split.headset)
    if channels is None:
        raise NotImplementedError(
            f"Unsupported headset for Neurotechs: {split.headset}"
        )

    recordings: list[tuple[str, str, Path]] = []
    for subject_id in split.subject_ids():
        for session in split.sessions:
            session_str = session.lower()
            eeg_dir = base_path / subject_id / session_str / "eeg"
            if not eeg_dir.exists():
                logger.warning(
                    f"Skipping {subject_id} {session_str}: EEG directory not found: {eeg_dir}."
                )
                continue
            pattern = f"{subject_id}_{session_str}_task-{split.task_name}_eeg.set"
            eeg_files = sorted(eeg_dir.glob(pattern))
            if not eeg_files:
                eeg_files = sorted(eeg_dir.glob("*.set"))
            if not eeg_files:
                logger.warning(
                    f"Skipping {subject_id} {session_str}: no .set file found."
                )
                continue
            recordings.append((subject_id, session_str, eeg_files[0]))

    if not recordings:
        raise ValueError(
            "No Neurotechs recordings matched the requested subjects/sessions."
        )

    total_expected = len(recordings) * segments_per_condition * 2
    eeg_buffer = np.zeros(
        (total_expected, len(channels), epoch_samples), dtype=np.float32
    )
    labels_buffer = np.empty((total_expected, 2), dtype=TASK_LABEL_DTYPE)

    cursor = 0
    progress = tqdm(recordings, desc="Processing Neurotechs baseline")
    for subject_id, session_str, eeg_path in progress:
        raw = mne.io.read_raw_eeglab(eeg_path.as_posix(), preload=True, verbose=False)
        sfreq = float(raw.info["sfreq"])
        if not np.isclose(sfreq, split.sampling_rate):
            raise ValueError(
                f"Sampling rate mismatch for {subject_id} {session_str}: "
                f"expected {split.sampling_rate}, found {sfreq}."
            )
        missing_channels = [ch for ch in channels if ch not in raw.ch_names]
        if missing_channels:
            raise ValueError(
                f"Missing channels {missing_channels} for {subject_id} {session_str}."
            )
        picks = [raw.ch_names.index(ch) for ch in channels]
        data = raw.get_data(picks=picks).astype(np.float32, copy=False)

        baseline_start = offset_samples
        baseline_stop = baseline_start + closed_samples + open_samples
        if baseline_stop > data.shape[1]:
            logger.warning(
                f"Skipping {subject_id} {session_str}: baseline window "
                f"({baseline_stop} samples) exceeds recording length ({data.shape[1]})."
            )
            raw.close()
            continue

        closed_data = data[:, baseline_start : baseline_start + closed_samples][
            :, : segments_per_condition * epoch_samples
        ]
        open_data = data[
            :,
            baseline_start + closed_samples : baseline_start
            + closed_samples
            + open_samples,
        ][:, : segments_per_condition * epoch_samples]

        closed_segments = closed_data.reshape(
            len(channels), segments_per_condition, epoch_samples
        ).transpose(1, 0, 2)
        open_segments = open_data.reshape(
            len(channels), segments_per_condition, epoch_samples
        ).transpose(1, 0, 2)

        for seg_idx in range(segments_per_condition):
            if cursor >= eeg_buffer.shape[0]:
                raise RuntimeError("Cursor exceeded Neurotechs buffer allocation.")
            normalized_closed = _normalize_epochs(closed_segments[seg_idx])
            eeg_buffer[cursor] = normalized_closed
            labels_buffer[cursor, 0] = Task.MOVE_EYES.value
            labels_buffer[cursor, 1] = Annotation.EYES_CLOSED.value
            cursor += 1
            if cursor >= eeg_buffer.shape[0]:
                raise RuntimeError("Cursor exceeded Neurotechs buffer allocation.")
            normalized_open = _normalize_epochs(open_segments[seg_idx])
            eeg_buffer[cursor] = normalized_open
            labels_buffer[cursor, 0] = Task.MOVE_EYES.value
            labels_buffer[cursor, 1] = Annotation.EYES_OPEN.value
            cursor += 1
        raw.close()

    if cursor == 0:
        raise ValueError("No eyes-open/eyes-closed segments were extracted.")

    if cursor != total_expected:
        logger.info(
            f"Extracted {cursor} Neurotechs segments (expected {total_expected}); "
            "some recordings may have been skipped."
        )

    eeg_final = eeg_buffer[:cursor]
    labels_final = labels_buffer[:cursor]
    np.save(eeg_output_path, eeg_final)
    np.save(labels_output_path, labels_final)
    with open(channels_output_path, "w", encoding="utf-8") as fp:
        json.dump(channels, fp, indent=2)

    return np.load(eeg_output_path, mmap_mode="r", allow_pickle=True), np.load(
        labels_output_path, mmap_mode="r", allow_pickle=True
    )


RESTING_BLOCK_LABELS = {
    Annotation.EYES_OPEN: (Task.MOVE_EYES, Annotation.EYES_OPEN),
    Annotation.EYES_CLOSED: (Task.MOVE_EYES, Annotation.EYES_CLOSED),
}


def extract_resting_methods_split(
    split: RestingEEGMethodsDataSplit,
    reset_cache: bool = False,
) -> tuple[np.memmap, np.memmap]:
    """
    Extract alternating eyes-open / eyes-closed blocks from the
    resting-eeg-study-methods dataset. Each recording contains four blocks
    (open, closed, open, closed) of length ``split.block_duration_sec``.
    """

    base_path = Path(split.source_base_path).expanduser().resolve()
    output_path = Path(split.output_path).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    cache_prefix = f"{split.split_name}_{split.code()}"
    eeg_output_path = output_path / f"{cache_prefix}_eeg.npy"
    labels_output_path = output_path / f"{cache_prefix}_labels.npy"
    channels_output_path = output_path / f"{cache_prefix}_channels.json"

    if eeg_output_path.exists() and labels_output_path.exists() and not reset_cache:
        return np.load(eeg_output_path, mmap_mode="r", allow_pickle=True), np.load(
            labels_output_path, mmap_mode="r", allow_pickle=True
        )

    epoch_samples = int(round(split.epoch_length_sec * split.sampling_rate))
    if not np.isclose(epoch_samples, split.epoch_length_sec * split.sampling_rate):
        raise ValueError(
            "epoch_length_sec * sampling_rate must produce an integer number of samples."
        )
    block_samples = int(round(split.block_duration_sec * split.sampling_rate))
    if not np.isclose(block_samples, split.block_duration_sec * split.sampling_rate):
        raise ValueError(
            "block_duration_sec * sampling_rate must produce an integer number of samples."
        )
    offset_samples = int(round(split.baseline_offset_sec * split.sampling_rate))
    if not np.isclose(offset_samples, split.baseline_offset_sec * split.sampling_rate):
        raise ValueError(
            "baseline_offset_sec * sampling_rate must produce an integer number of samples."
        )

    segments_per_block = block_samples // epoch_samples
    if segments_per_block <= 0:
        raise ValueError(
            "Epoch length must be shorter than the block duration for resting EEG extraction."
        )

    channels = HEADSET_TO_CHANNELS.get(split.headset)
    if channels is None:
        raise NotImplementedError(
            f"Unsupported headset for resting methods extraction: {split.headset}"
        )

    recordings: list[tuple[str, str, Path]] = []
    for subject_id in split.subject_ids():
        for session in split.sessions:
            eeg_dir = base_path / subject_id / session / "eeg"
            if not eeg_dir.exists():
                logger.warning(
                    f"Skipping {subject_id} {session}: EEG directory not found: {eeg_dir}."
                )
                continue
            pattern = f"{subject_id}_{session}_task-{split.task_name}_eeg.vhdr"
            vhdr_files = sorted(eeg_dir.glob(pattern))
            if not vhdr_files:
                vhdr_files = sorted(eeg_dir.glob("*.vhdr"))
            if not vhdr_files:
                logger.warning(f"Skipping {subject_id} {session}: no .vhdr file found.")
                continue
            recordings.append((subject_id, session, vhdr_files[0]))

    if not recordings:
        raise ValueError(
            "No resting EEG recordings matched the requested subjects/sessions."
        )

    total_expected = len(recordings) * len(split.block_sequence) * segments_per_block
    eeg_buffer = np.zeros(
        (total_expected, len(channels), epoch_samples), dtype=np.float32
    )
    labels_buffer = np.empty((total_expected, 2), dtype=TASK_LABEL_DTYPE)

    cursor = 0
    progress = tqdm(recordings, desc="Processing resting EEG baseline")
    for subject_id, session, vhdr_path in progress:
        raw = mne.io.read_raw_brainvision(
            vhdr_path.as_posix(), preload=True, verbose=False
        )
        sfreq = float(raw.info["sfreq"])
        if not np.isclose(sfreq, split.sampling_rate):
            raise ValueError(
                f"Sampling rate mismatch for {subject_id} {session}: "
                f"expected {split.sampling_rate}, found {sfreq}."
            )
        missing_channels = [ch for ch in channels if ch not in raw.ch_names]
        if missing_channels:
            raise ValueError(
                f"Missing channels {missing_channels} for {subject_id} {session}."
            )
        picks = [raw.ch_names.index(ch) for ch in channels]
        data = raw.get_data(picks=picks).astype(np.float32, copy=False)

        start_idx = offset_samples
        for block_index, block_name in enumerate(split.block_sequence):
            task_label, base_label = RESTING_BLOCK_LABELS[block_name]
            block_start = start_idx + block_index * block_samples
            block_end = block_start + block_samples
            if block_start >= data.shape[1]:
                logger.warning(
                    f"Block {block_name} for {subject_id} {session} starts beyond data length; "
                    "skipping remaining blocks."
                )
                break
            block_end = min(block_end, data.shape[1])
            available_samples = block_end - block_start
            usable_samples = (available_samples // epoch_samples) * epoch_samples
            if usable_samples < epoch_samples:
                logger.warning(
                    f"Block {block_name} for {subject_id} {session} has insufficient samples after trimming; "
                    "skipping."
                )
                continue
            block_data = data[:, block_start : block_start + usable_samples]
            segments = block_data.reshape(
                len(channels),
                usable_samples // epoch_samples,
                epoch_samples,
            ).transpose(1, 0, 2)
            for segment in segments:
                if cursor >= eeg_buffer.shape[0]:
                    raise RuntimeError("Cursor exceeded resting EEG buffer allocation.")
                normalized_segment = _normalize_epochs(segment)
                eeg_buffer[cursor] = normalized_segment
                labels_buffer[cursor, 0] = task_label.value
                labels_buffer[cursor, 1] = base_label.value
                cursor += 1
        raw.close()

    if cursor == 0:
        raise ValueError("No eyes-open/eyes-closed segments were extracted.")

    eeg_final = eeg_buffer[:cursor]
    labels_final = labels_buffer[:cursor]
    np.save(eeg_output_path, eeg_final)
    np.save(labels_output_path, labels_final)
    with open(channels_output_path, "w", encoding="utf-8") as fp:
        json.dump(channels, fp, indent=2)

    return np.load(eeg_output_path, mmap_mode="r", allow_pickle=True), np.load(
        labels_output_path, mmap_mode="r", allow_pickle=True
    )


def extract_lemon_resting_state(
    split: LemonRestingStateDataSplit,
    ignore_cache: bool = False,
) -> tuple[np.memmap, np.memmap]:
    """
    Extracts resting-state EEG segments from the MPI LEMON dataset.

    Each BrainVision recording contains alternating 2 s epochs marked with
    Stimulus/S200 (eyes-open) and Stimulus/S210 (eyes-closed). This function
    creates a memmap with shape (N, C, T) where N is the number of epochs,
    C is the number of channels, and T is the number of time samples per epoch.

    Args:
        source_root: Path to the dataset root containing sub-*/RSEEG folders.
        output_dir: Directory where the memmaps will be cached.
        subjects: Optional sequence of subject identifiers (e.g. ``sub-010002``).
            If None, all subjects under `source_root` are processed.
        epoch_duration_sec: Length of each extracted segment in seconds.
            Defaults to the canonical 2 second segments.
        resample_sfreq: Optionally resample the data before epoching.
            If None, the native sampling rate (~2500 Hz) is used.
        ignore_cache: If True, recompute the memmaps even if cached files exist.

    Returns:
        A tuple of (eeg_memmap, labels_memmap). Labels are stored as string pairs
        ``[task, label]`` using TASK_LABEL_DTYPE.
    """

    source_path = Path(split.source_base_path).expanduser().resolve()
    out_path = Path(split.output_path).expanduser().resolve()
    out_path.mkdir(parents=True, exist_ok=True)

    if not source_path.exists():
        raise FileNotFoundError(f"LEMON source directory not found: {source_path}")

    lemon_subject_dirs = []
    if split.subjects is None:
        for child in sorted(source_path.iterdir()):
            if child.is_dir() and child.name.startswith("sub-"):
                lemon_subject_dirs.append(child)
    else:
        for subject in split.subject_ids():
            subject_dir = source_path / subject
            if not subject_dir.exists():
                raise FileNotFoundError(f"Missing subject directory: {subject_dir}")
            lemon_subject_dirs.append(subject_dir)

    if not lemon_subject_dirs:
        raise ValueError("No subject directories found for LEMON extraction.")

    def find_vhdr(subject_dir: Path) -> Path:
        rseeg_dir = subject_dir / "RSEEG"
        if not rseeg_dir.exists():
            raise FileNotFoundError(f"Missing RSEEG folder for {subject_dir.name}")
        vhdr_files = sorted(
            file for file in rseeg_dir.glob("*.vhdr") if not file.name.startswith("._")
        )
        if not vhdr_files:
            raise FileNotFoundError(f"No .vhdr file found in {rseeg_dir}")
        return vhdr_files[0]

    event_id = {"Stimulus/S200": 1, "Stimulus/S210": 2}
    label_map = {
        "Stimulus/S200": (Task.MOVE_EYES, Annotation.EYES_OPEN),
        "Stimulus/S210": (Task.MOVE_EYES, Annotation.EYES_CLOSED),
    }

    first_native_sfreq: float | None = None
    for candidate_dir in lemon_subject_dirs:
        try:
            preview_vhdr = find_vhdr(candidate_dir)
        except FileNotFoundError:
            continue
        preview_raw = mne.io.read_raw_brainvision(
            preview_vhdr.as_posix(), preload=False, verbose=False
        )
        first_native_sfreq = float(preview_raw.info["sfreq"])
        preview_raw.close()
        break

    if first_native_sfreq is None:
        raise ValueError("No usable LEMON recordings were discovered.")

    target_sfreq_guess = (
        float(split.resample_sfreq)
        if split.resample_sfreq is not None
        else first_native_sfreq
    )
    epoch_suffix = f"{int(split.epoch_length_sec * 1000)}ms"
    sr_suffix = f"{int(target_sfreq_guess)}hz"
    subject_suffix = (
        f"{len(lemon_subject_dirs)}sub" if split.subjects is not None else "allsubjects"
    )
    cache_prefix = f"lemon_{subject_suffix}_{sr_suffix}_{epoch_suffix}"

    eeg_output_path = out_path / f"{cache_prefix}_eeg.npy"
    labels_output_path = out_path / f"{cache_prefix}_labels.npy"
    channels_output_path = out_path / f"{cache_prefix}_channels.json"

    if not ignore_cache and eeg_output_path.exists() and labels_output_path.exists():
        logger.info(f"Using cached LEMON extraction at {eeg_output_path}")
        return np.load(eeg_output_path, mmap_mode="r", allow_pickle=True), np.load(
            labels_output_path, mmap_mode="r", allow_pickle=True
        )

    subject_info: list[dict[str, Any]] = []
    reference_channels: list[str] | None = None
    native_sfreq: float | None = None
    total_epochs = 0

    logger.info("Scanning LEMON recordings for epoch counts.")
    target_epoch_duration_sec = float(split.epoch_length_sec)
    if target_epoch_duration_sec <= 0:
        raise ValueError("epoch_duration_sec must be positive.")

    for subject_dir in lemon_subject_dirs:
        vhdr_path = find_vhdr(subject_dir)

        raw = mne.io.read_raw_brainvision(
            vhdr_path.as_posix(), preload=False, verbose=False
        )
        current_sfreq = float(raw.info["sfreq"])
        if native_sfreq is None:
            native_sfreq = current_sfreq
        elif not np.isclose(native_sfreq, current_sfreq):
            if split.resample_sfreq is None:
                raise ValueError(
                    f"Inconsistent sampling rate detected for {subject_dir.name}."
                )
            else:
                warning_str = (
                    "Subject {sub_id} has native sfreq {native} Hz (baseline {baseline} Hz); "
                    "proceeding because resample_sfreq={resample} is set."
                ).format(
                    sub_id=subject_dir.name,
                    native=current_sfreq,
                    baseline=native_sfreq,
                    resample=split.resample_sfreq,
                )
                logger.warning(warning_str)

        annotations = raw.annotations
        epoch_samples_native = int(round(target_epoch_duration_sec * raw.info["sfreq"]))
        if not np.isclose(
            epoch_samples_native, target_epoch_duration_sec * raw.info["sfreq"]
        ):
            raise ValueError(
                "epoch_duration_sec produces a non-integer number of samples "
                "at the native sampling rate."
            )
        max_sample = raw.n_times
        valid_count = 0
        per_event_counts: dict[str, int] = {desc: 0 for desc in event_id}
        for desc, onset in zip(annotations.description, annotations.onset):
            if desc not in event_id:
                continue
            sample = int(round(onset * raw.info["sfreq"]))
            if sample + epoch_samples_native <= max_sample:
                valid_count += 1
                per_event_counts[desc] += 1

        if valid_count == 0:
            logger.warning(f"No valid epochs found for {subject_dir.name}")
            continue
        logger.info(f"Found {valid_count} valid epochs for {subject_dir.name}")

        missing_events = [
            desc for desc, count in per_event_counts.items() if count == 0
        ]
        if missing_events:
            missing_summary = ", ".join(sorted(missing_events))
            logger.warning(
                f"Skipping {subject_dir.name}; missing required events: {missing_summary}"
            )
            continue

        if reference_channels is None:
            reference_channels = list(raw.ch_names)
        elif list(raw.ch_names) != reference_channels:
            raise ValueError(
                f"Channel mismatch for {subject_dir.name}. "
                "All subjects must have identical channel ordering."
            )

        subject_info.append({
            "subject": subject_dir.name,
            "vhdr_path": vhdr_path,
            "valid_events": valid_count,
        })
        total_epochs += valid_count

    if not subject_info:
        raise ValueError("No usable LEMON recordings were discovered.")

    assert reference_channels is not None
    assert native_sfreq is not None

    target_sfreq = float(split.resample_sfreq) if split.resample_sfreq else native_sfreq
    if target_sfreq <= 0:
        raise ValueError("resample_sfreq must be positive if provided.")
    sr_suffix = f"{int(target_sfreq)}hz"

    epoch_samples = int(round(target_epoch_duration_sec * target_sfreq))
    if not np.isclose(epoch_samples, target_epoch_duration_sec * target_sfreq):
        raise ValueError(
            "epoch_duration_sec * sampling_rate must be close to an integer number of samples."
        )

    # Update cache prefix with actual subject count when available.
    subject_suffix = (
        f"{len(subject_info)}sub" if split.subjects is not None else "allsubjects"
    )
    cache_prefix = f"lemon_{subject_suffix}_{sr_suffix}_{epoch_suffix}"
    eeg_output_path = out_path / f"{cache_prefix}_eeg.npy"
    labels_output_path = out_path / f"{cache_prefix}_labels.npy"
    channels_output_path = out_path / f"{cache_prefix}_channels.json"

    if not ignore_cache and eeg_output_path.exists() and labels_output_path.exists():
        logger.info(f"Using cached LEMON extraction at {eeg_output_path}")
        return np.load(eeg_output_path, mmap_mode="r", allow_pickle=True), np.load(
            labels_output_path, mmap_mode="r", allow_pickle=True
        )

    logger.info(
        f"Extracting {total_epochs} epochs across {len(subject_info)} subjects (resample={target_sfreq} Hz)."
    )

    eeg_store = open_memmap(
        eeg_output_path,
        mode="w+",
        dtype=np.float32,
        shape=(total_epochs, len(reference_channels), epoch_samples),
    )
    labels_store = open_memmap(
        labels_output_path,
        mode="w+",
        dtype=TASK_LABEL_DTYPE,
        shape=(total_epochs, 2),
    )

    code_to_desc = {code: desc for desc, code in event_id.items()}
    cursor = 0

    progress = tqdm(subject_info, desc="Processing LEMON subjects")
    for info in progress:
        vhdr_path = info["vhdr_path"]
        raw = mne.io.read_raw_brainvision(
            vhdr_path.as_posix(), preload=True, verbose=False
        )
        if split.resample_sfreq:
            raw.resample(target_sfreq, npad="auto")

        annotations = raw.annotations
        epoch_samples_resampled = int(
            round(target_epoch_duration_sec * raw.info["sfreq"])
        )
        if not np.isclose(
            epoch_samples_resampled,
            target_epoch_duration_sec * raw.info["sfreq"],
        ):
            raise ValueError(
                "epoch_duration_sec produces a non-integer number of samples "
                "at the resampled rate."
            )
        max_sample = raw.n_times
        valid_onsets: list[tuple[float, str]] = []
        for desc, onset in zip(annotations.description, annotations.onset):
            if desc not in event_id:
                continue
            sample = int(round(onset * raw.info["sfreq"]))
            if sample + epoch_samples_resampled <= max_sample:
                valid_onsets.append((onset, desc))

        if not valid_onsets:
            logger.warning(f"Skipping {vhdr_path}; no valid events after filtering.")
            continue

        present_events = {desc for _, desc in valid_onsets}
        missing_events = sorted(set(event_id) - present_events)
        if missing_events:
            missing_summary = ", ".join(missing_events)
            logger.warning(
                f"Skipping {vhdr_path}; missing required events after filtering: {missing_summary}"
            )
            continue

        # Reconstruct events array (sample index, 0, event code).
        events = np.zeros((len(valid_onsets), 3), dtype=int)
        for idx, (onset, desc) in enumerate(valid_onsets):
            sample = int(round(onset * raw.info["sfreq"]))
            events[idx, 0] = sample
            events[idx, 2] = event_id[desc]

        epochs = mne.Epochs(
            raw,
            events,
            event_id=event_id,
            tmin=0.0,
            tmax=(epoch_samples - 1) / target_sfreq,
            baseline=None,
            proj=False,
            preload=True,
            reject_by_annotation=False,
            verbose=False,
        )

        epoch_data = epochs.get_data().astype(np.float32)
        epoch_data = _normalize_epochs(epoch_data)
        n_epochs = epoch_data.shape[0]
        if not n_epochs:
            logger.warning(f"No epochs extracted for {vhdr_path}")
            continue

        eeg_store[cursor : cursor + n_epochs, :, :] = epoch_data
        event_codes = epochs.events[:, 2]
        for i, code in enumerate(event_codes):
            desc = code_to_desc[int(code)]
            task, label = label_map[desc]
            labels_store[cursor + i, 0] = task.value
            labels_store[cursor + i, 1] = label.value

        cursor += n_epochs

    if cursor != total_epochs:
        raise RuntimeError(
            f"Expected to write {total_epochs} epochs, but wrote {cursor}."
        )

    with open(channels_output_path, "w", encoding="utf-8") as fp:
        json.dump(reference_channels, fp, indent=2)
    eeg_store.flush()
    labels_store.flush()

    return np.load(eeg_output_path, mmap_mode="r", allow_pickle=True), np.load(
        labels_output_path, mmap_mode="r", allow_pickle=True
    )


def ds_split_factory(splits: list[dict[str, Any]]):
    out = []
    for split in splits:
        data_source = DataSource(split["data_source"])
        headset = Headset(split["headset"])
        if data_source == DataSource.EEG_MMI:
            out.append(EEGMMIDataSplit(**split))
        elif data_source == DataSource.EMOTIVE_ALPHA:
            if headset == Headset.EMOTIV_EPOC_14:
                out.append(EmotivAlphaEpocDataSplit(**split))
            elif headset == Headset.EMOTIV_INSIGHT_5:
                out.append(EmotivAlphaInsightDataSplit(**split))
            else:
                raise NotImplementedError(f"Unknown headset: {headset}")
        elif data_source == DataSource.LEMON_REST:
            out.append(LemonRestingStateDataSplit(**split))
        elif data_source == DataSource.NEUROTECHS:
            out.append(NeurotechsEyesDataSplit(**split))
        elif data_source == DataSource.RESTING_METHODS:
            out.append(RestingEEGMethodsDataSplit(**split))
        else:
            raise NotImplementedError(f"Unknown data source: {data_source}")
    return out


def get_multi_mapped_label_datasets(
    splits: list[DataSplit],
    tasks_map: dict[str, int],
    labels_map: dict[str, int],
    data_config: DataConfig,
    reset_cache: bool = False,
):
    ret = {}
    for split in splits:
        logger.info(f"Creating dataset for split: {split.split_name}")
        os.makedirs(split.output_path, exist_ok=True)

        electrode_positions = None
        if split.headset == Headset.PHYSIONET_64:
            electrode_positions = PHYSIONET_64_CHANNEL_POSITIONS
        elif split.headset == Headset.EMOTIV_INSIGHT_5:
            electrode_positions = INSIGHT5_CHANNEL_POSITIONS
        elif split.headset == Headset.EMOTIV_EPOC_14:
            electrode_positions = EPOC14_CHANNEL_POSITIONS
        elif split.headset == Headset.LEMON_61:
            electrode_positions = LEMON_CHANNEL_POSITIONS
        elif split.headset == Headset.UNICORN_HYBRID_BLACK_8:
            electrode_positions = NEUROTECHS_CHANNEL_POSITIONS
        elif split.headset == Headset.BRAIN_ACTICHAMP_31:
            electrode_positions = RESTING_METHODS_CHANNEL_POSITIONS
        else:
            raise NotImplementedError(f"Unknown headset: {split.headset}")
        assert electrode_positions is not None
        logger.info(f"Got electrode positions for headset: {split.headset}.")

        channel_mask = None
        if split.channel_mask_config is not None:
            logger.info(f"Creating channel mask: {split.channel_mask_config}")
            channel_mask = create_mask(
                electrode_positions.numpy(), split.channel_mask_config
            )

        if split.data_source == DataSource.EEG_MMI:
            logger.info("Extracting EEG MMI data.")
            assert isinstance(split, EEGMMIDataSplit)
            split_eeg, split_labels = extract_eeg_mmi_split(
                split.source_base_path,
                split,
                split.output_path,
                epoch_length_sec=split.epoch_length_sec,
                reset_cache=reset_cache,
            )
        elif split.data_source == DataSource.EMOTIVE_ALPHA:
            logger.info("Extracting Emotiv Alpha data.")
            assert isinstance(split, EmotivAlphaDataSplit)
            split_eeg, split_labels = extract_emotiv_alpha_suppression_split(
                split,
                reset_cache=reset_cache,
            )
        elif split.data_source == DataSource.LEMON_REST:
            logger.info("Extracting LEMON resting-state data.")
            assert isinstance(split, LemonRestingStateDataSplit)
            split_eeg, split_labels = extract_lemon_resting_state(
                split=split,
                ignore_cache=reset_cache,
            )
        elif split.data_source == DataSource.NEUROTECHS:
            logger.info("Extracting Neurotechs eyes-open/closed baseline data.")
            assert isinstance(split, NeurotechsEyesDataSplit)
            split_eeg, split_labels = extract_neurotechs_eyes_split(
                split=split,
                reset_cache=reset_cache,
            )
        elif split.data_source == DataSource.RESTING_METHODS:
            logger.info("Extracting resting EEG study methods data.")
            assert isinstance(split, RestingEEGMethodsDataSplit)
            split_eeg, split_labels = extract_resting_methods_split(
                split=split,
                reset_cache=reset_cache,
            )
        else:
            raise NotImplementedError(f"Unknown data source: {split.data_source}")

        dataset = MappedLabelDataset(
            split_eeg,
            split_labels,
            labels_map,
            tasks_map,
            electrode_positions,
            data_config,
            split.sampling_rate,
            channel_mask,
        )

        ds = ret.get(split.split_name, None)
        if ds is None:
            ret[split.split_name] = MultiMappedLabelDataset([dataset])
        else:
            ds.append_dataset(dataset)
    return ret


def get_libri_brain_speech_dataset(
    output_path: str,
    tmin: float = 0.0,
    tmax: float = 0.8,
    oversample_silence_jitter: int = 0,
    stride: int | None = None,
    partition: str | None = None,
    books: list[int] | None = None,
    books_chapters: list[list[int]] | None = None,
    sessions: list[int] | None = None,
    preload_files: bool = True,
    sensor_mask: list[int] | None = None,
):
    # Download sensor locations JSON
    os.makedirs(output_path, exist_ok=True)
    p = os.path.join(output_path, "sensor_xyz.json")
    if not os.path.exists(p):
        with open(p, "wb") as f:
            f.write(
                requests.get(
                    "https://neural-processing-lab.github.io/2025-libribrain-competition/sensor_xyz.json"
                ).content
            )
    with open(p, "r") as fp:
        sensor_positions = np.array(json.load(fp))
    variant = {}
    if partition is None:
        assert books is not None
        assert books_chapters is not None
        assert sessions is not None
        keys = []
        for book, chapters, session in zip(books, books_chapters, sessions):
            keys.extend([
                ("0", str(chapter), f"Sherlock{book}", str(session))
                for chapter in chapters
            ])
        variant = {"include_run_keys": keys}
    else:
        variant = {"partition": partition}

    return LibriBrainSpeechDataset(
        LibriBrainSpeech(
            os.path.join(output_path, "data"),
            **variant,
            tmin=tmin,
            tmax=tmax,
            preload_files=preload_files,
            stride=stride,
            oversample_silence_jitter=oversample_silence_jitter,
            standardize=True,
        ),
        sensor_positions=torch.tensor(sensor_positions),
        sensors_speech_mask=sensor_mask,
        tmin=tmin,
        tmax=tmax,
    )


def libri_speech_brain_collate_fn(
    items: list[tuple[torch.Tensor, int, np.ndarray, int]],
):
    channel_positions, tasks, channel_signals, labels, metadata = zip(*items)
    channel_positions = torch.stack(channel_positions)
    tasks = torch.tensor(tasks)
    channel_signals = torch.stack(channel_signals)
    labels = torch.tensor(labels)
    metadata = torch.stack(metadata)
    return {
        "channel_positions": channel_positions,
        "tasks": tasks,
        "channel_signals": channel_signals,
        "labels": labels,
        "metadata": metadata,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Data processing standalone script.")
    parser.add_argument("--training-config-path", type=str, required=True)
    parser.add_argument("--model-config-path", type=str, required=True)

    args = parser.parse_args()
    cfg = TrainingConfig(
        **load_yaml(args.training_config_path),
        training_config_path=args.training_config_path,
        model_config_path="",
        world_size=1,
        run_project="",
        run_name="",
        run_group="",
        eval_first=False,
        device="",
        checkpoints=False,
    )
    model_config = MontageNetConfig(**load_yaml(args.model_config_path))
    ds_splits = ds_split_factory(cfg.ds_split_configs)
    ds = get_multi_mapped_label_datasets(
        ds_splits,
        model_config.tasks_map,
        model_config.labels_map,
        model_config.data_config,
        reset_cache=True,
    )
