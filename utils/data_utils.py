from dataclasses import dataclass, field
from enum import Enum
import os
from typing import Callable, Any
import json
import requests

import mne
from loguru import logger
import numpy as np
import torch
from torch import Tensor
from tqdm import tqdm
from pnpl.datasets import LibriBrainPhoneme, LibriBrainSpeech

from utils.torch_datasets import (
    LibriBrainSpeechDataset,
    MappedLabelDataset,
    EEGEyeNetDataset,
)
from utils.electrode_utils import physionet_64_montage


class ValidationType(Enum):
    DEFAULT = "default"
    RANDOM = "random"
    SUBJECT = "subject"
    OBJECT = "object"


ELECTRODE_ORDER = np.array(
    [
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
    ]
)


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
    total_rows = sum(
        [
            session_epochs[split_type] * sessions_per_subject * len(subjects)
            for split_type in ds.keys()
        ]
    )
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
                ch_names = np.array(
                    [name for name in data["ch_names"] if name != "stim"]
                )
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
    # Freq 0 is not needed because the signal is normalized.
    return stft[:, 1:, :-1].abs() ** 2


def get_nth_mask(size: int, n: int, offset: int = 1) -> torch.Tensor:
    mask = torch.ones(size)
    mask[offset - 1 :: n] = False
    return mask.unsqueeze(0).unsqueeze(-1)


def eeg_mmi_collate_fn(
    samples: list[tuple[torch.Tensor, int, np.memmap, int]],
) -> dict[str, torch.Tensor]:
    channel_positons, tasks, channel_signals, labels = [], [], [], []
    for sample in samples:
        channel_positons.append(sample[0])
        tasks.append(sample[1])
        channel_signals.append(sample[2])
        labels.append(sample[3])
    sensor_positons_tensor = torch.tensor(channel_positons)
    tasks_tensor = torch.tensor(tasks)
    channel_signals_tensor = torch.tensor(np.array(channel_signals))
    labels_tensor = torch.tensor(labels)
    return {
        "channel_positions": sensor_positons_tensor,
        "tasks": tasks_tensor,
        "channel_signals": channel_signals_tensor,
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


class EEGMMITask(Enum):
    BASELINE_EYES_OPEN = "m_eyes_open"
    BASELINE_EYES_CLOSED = "m_eyes_clos"
    TASK_1 = "m_l__r_fist"
    TASK_2 = "i_l__r_fist"
    TASK_3 = "m_fist_feet"
    TASK_4 = "i_fist_feet"


SESSION_TO_TASK = {
    1: EEGMMITask.BASELINE_EYES_OPEN,
    2: EEGMMITask.BASELINE_EYES_CLOSED,
    3: EEGMMITask.TASK_1,
    4: EEGMMITask.TASK_2,
    5: EEGMMITask.TASK_3,
    6: EEGMMITask.TASK_4,
    7: EEGMMITask.TASK_1,
    8: EEGMMITask.TASK_2,
    9: EEGMMITask.TASK_3,
    10: EEGMMITask.TASK_4,
    11: EEGMMITask.TASK_1,
    12: EEGMMITask.TASK_2,
    13: EEGMMITask.TASK_3,
    14: EEGMMITask.TASK_4,
}


class Annotation(Enum):
    EYES_OPEN = "base_e_open"
    EYES_CLOSED = "base_e_clos"
    REST = "base_b_rest"
    MOVE_LEFT_FIST = "move_l_fist"
    MOVE_RIGHT_FIST = "move_r_fist"
    MOVE_BOTH_FIST = "move_b_fist"
    MOVE_BOTH_FEET = "move_b_feet"
    IMAG_LEFT_FIST = "imag_l_fist"
    IMAG_RIGHT_FIST = "imag_r_fist"
    IMAG_BOTH_FIST = "imag_b_fist"
    IMAG_BOTH_FEET = "imag_b_feet"


ANNOTATION_CODE_MAP: dict[int, dict[str, Annotation]] = {
    1: {
        "T0": Annotation.EYES_OPEN,
    },
    2: {
        "T0": Annotation.EYES_CLOSED,
    },
    3: {
        "T0": Annotation.REST,
        "T1": Annotation.MOVE_LEFT_FIST,
        "T2": Annotation.MOVE_RIGHT_FIST,
    },
    4: {
        "T0": Annotation.REST,
        "T1": Annotation.IMAG_LEFT_FIST,
        "T2": Annotation.IMAG_RIGHT_FIST,
    },
    5: {
        "T0": Annotation.REST,
        "T1": Annotation.MOVE_BOTH_FIST,
        "T2": Annotation.MOVE_BOTH_FEET,
    },
    6: {
        "T0": Annotation.REST,
        "T1": Annotation.IMAG_BOTH_FIST,
        "T2": Annotation.IMAG_BOTH_FEET,
    },
    7: {
        "T0": Annotation.REST,
        "T1": Annotation.MOVE_LEFT_FIST,
        "T2": Annotation.MOVE_RIGHT_FIST,
    },
    8: {
        "T0": Annotation.REST,
        "T1": Annotation.IMAG_LEFT_FIST,
        "T2": Annotation.IMAG_RIGHT_FIST,
    },
    9: {
        "T0": Annotation.REST,
        "T1": Annotation.MOVE_BOTH_FIST,
        "T2": Annotation.MOVE_BOTH_FEET,
    },
    10: {
        "T0": Annotation.REST,
        "T1": Annotation.IMAG_BOTH_FIST,
        "T2": Annotation.IMAG_BOTH_FEET,
    },
    11: {
        "T0": Annotation.REST,
        "T1": Annotation.MOVE_LEFT_FIST,
        "T2": Annotation.MOVE_RIGHT_FIST,
    },
    12: {
        "T0": Annotation.REST,
        "T1": Annotation.IMAG_LEFT_FIST,
        "T2": Annotation.IMAG_RIGHT_FIST,
    },
    13: {
        "T0": Annotation.REST,
        "T1": Annotation.MOVE_BOTH_FIST,
        "T2": Annotation.MOVE_BOTH_FEET,
    },
    14: {
        "T0": Annotation.REST,
        "T1": Annotation.IMAG_BOTH_FIST,
        "T2": Annotation.IMAG_BOTH_FEET,
    },
}


@dataclass
class EEGMMISplit:
    name: str
    subjects: list[int]
    sessions: list[int]
    max_subjects: int = 109
    max_sessions: int = 14

    def code(self) -> str:
        # Hack bc subjects and sessions are 1-indexed.
        subjects_onehot = np.zeros(self.max_subjects + 1, dtype=int)
        subjects_onehot[self.subjects] = 1
        sessions_onehot = np.zeros(self.max_sessions + 1, dtype=int)
        sessions_onehot[self.sessions] = 1
        return f"sub-{''.join(str(s) for s in subjects_onehot.tolist())}_sess-{''.join(str(s) for s in sessions_onehot.tolist())}"


def extract_eeg_mmi_session_data(
    base_path: str,
    output_path: str,
    subject: int,
    session: int,
    ignore_cache: bool = False,
    default_event_length: int = 180,
) -> tuple[np.memmap, np.memmap]:
    """Extract the raw EEG and annotations from an edf file return
    a np.memmap of shape (N_E, N_C, T) where:
        N_E: number of epochs.
        N_C: number of channels.
        T: number of time samples.
    containing EEG data and a memmap of shape (N_E) containing the annotations
    """
    subject_str = "S" + str(subject).zfill(3)
    session_str = "R" + str(session).zfill(2)
    output_eeg_path = os.path.join(
        base_path, subject_str, f"{subject_str}{session_str}_eeg.npy"
    )
    output_labels_path = os.path.join(
        base_path, subject_str, f"{subject_str}{session_str}_labels.npy"
    )
    cached = (
        os.path.exists(output_eeg_path)
        and os.path.exists(output_labels_path)
        and not ignore_cache
    )
    source_path = os.path.join(
        base_path, subject_str, f"{subject_str}{session_str}.edf"
    )
    data = mne.io.read_raw_edf(source_path)
    events, event_map = mne.events_from_annotations(data)
    # Reverse the mapping to go from labels to annotations.
    labels_to_annotations = {value: key for key, value in event_map.items()}
    eeg_data = data.get_data()
    # Normalize the data.
    eeg_data = (eeg_data - eeg_data.mean(axis=1, keepdims=True)) / eeg_data.std(
        axis=1, keepdims=True
    )
    assert isinstance(eeg_data, np.ndarray)
    num_channels, num_samples = eeg_data.shape
    if events.shape[0] == 2:
        # Huge hack for one specific anamoly.
        events = events[0:1]
    # For the eyes open, eyes closed tasks, we have a single event.
    # So split it up into multiple pseudo events.
    if events.shape[0] == 1:
        real_total_duration = (eeg_data.sum(axis=0) != 0).sum()
        num_events = real_total_duration // default_event_length
        events = events.repeat(num_events, axis=0)
        events[:, 0] = np.arange(num_events) * default_event_length

    # Get the maximum duration between two consecutive events.
    max_event_duration = np.max(np.diff(events[:, 0]))
    session_eeg = np.memmap(
        filename=os.path.join(output_path, f"{subject_str}{session_str}_eeg.npy"),
        mode="r" if cached else "w+",
        shape=(len(events), num_channels, max_event_duration),
        dtype=eeg_data.dtype,
    )
    session_labels = np.memmap(
        filename=os.path.join(output_path, f"{subject_str}{session_str}_labels.npy"),
        mode="r" if cached else "w+",
        shape=(len(events), 2),
        dtype="<U11",
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
        task = SESSION_TO_TASK[session].value
        annotation = ANNOTATION_CODE_MAP[session][annotation_code.item()].value
        session_eeg[i, :, 0:event_duration] = eeg_data[:, event_start:event_stop]
        session_labels[i, :] = [task, annotation]

    session_eeg.flush()
    session_labels.flush()
    return session_eeg, session_labels


def extract_eeg_mmi_split(
    base_path: str,
    split: EEGMMISplit,
    output_path: str,
    reset_cache: bool = False,
):
    eeg_path = os.path.join(output_path, f"{split.name}_{split.code()}_eeg.npy")
    labels_path = os.path.join(output_path, f"{split.name}_{split.code()}_labels.npy")
    if os.path.exists(eeg_path) and os.path.exists(labels_path) and not reset_cache:
        return np.load(eeg_path, mmap_mode="r", allow_pickle=True), np.load(
            labels_path, mmap_mode="r", allow_pickle=True
        )
    eegs, task_labels = [], []
    shapes = np.zeros((len(split.subjects) * len(split.sessions), 3), dtype=int)
    for i, subject in enumerate(split.subjects):
        for j, session in enumerate(split.sessions):
            eeg, label = extract_eeg_mmi_session_data(
                base_path, output_path, subject, session, reset_cache
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
    split_labels = np.zeros(
        shape=(n_trials, 2),
        dtype="<U11",
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


def get_eeg_mmi_dataset(
    source_base_path: str,
    output_path: str,
    splits: dict[str, EEGMMISplit],
    labels_map: dict[str, int],
    tasks_map: dict[str, int],
    reset_cache: bool = False,
) -> dict[str, np.memmap]:
    ret = {}
    os.makedirs(output_path, exist_ok=True)
    electrode_positions = torch.tensor(
        list(physionet_64_montage().get_positions()["ch_pos"].values())
    )
    for split_name, split in splits.items():
        split_eeg, split_labels = extract_eeg_mmi_split(
            source_base_path, split, output_path, reset_cache
        )
        dataset = MappedLabelDataset(
            split_eeg, split_labels, labels_map, tasks_map, electrode_positions
        )
        ret[split_name] = dataset
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
            keys.extend(
                [
                    ("0", str(chapter), f"Sherlock{book}", str(session))
                    for chapter in chapters
                ]
            )
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
    )


def libri_speech_brain_collate_fn(
    items: list[tuple[torch.Tensor, int, np.ndarray, int]]
):
    channel_positions, tasks, channel_signals, labels = zip(*items)
    channel_positions = torch.stack(channel_positions)
    tasks = torch.tensor(tasks)
    channel_signals = torch.stack(channel_signals)
    labels = torch.tensor(labels)
    return {
        "channel_positions": channel_positions,
        "tasks": tasks,
        "channel_signals": channel_signals,
        "labels": labels,
    }
