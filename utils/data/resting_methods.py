from dataclasses import dataclass
import json
from pathlib import Path

from loguru import logger
import mne
import numpy as np
from tqdm import tqdm

from .common import (
    Annotation,
    DataSource,
    DataSplit,
    Headset,
    HEADSET_TO_CHANNELS,
    TASK_LABEL_DTYPE,
    Task,
    _normalize_epochs,
)


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


RESTING_BLOCK_LABELS = {
    Annotation.EYES_OPEN: (Task.MOVE_EYES, Annotation.EYES_OPEN),
    Annotation.EYES_CLOSED: (Task.MOVE_EYES, Annotation.EYES_CLOSED),
}


def extract_resting_methods_split(
    split: RestingEEGMethodsDataSplit,
    reset_cache: bool = False,
) -> tuple[np.memmap, np.memmap, np.memmap]:
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
    metadata_output_path = output_path / f"{cache_prefix}_metadata.npy"
    channels_output_path = output_path / f"{cache_prefix}_channels.json"

    if eeg_output_path.exists() and labels_output_path.exists() and metadata_output_path.exists() and not reset_cache:
        return (
            np.load(eeg_output_path, mmap_mode="r", allow_pickle=True),
            np.load(labels_output_path, mmap_mode="r", allow_pickle=True),
            np.load(metadata_output_path, mmap_mode="r", allow_pickle=True),
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
    metadata_buffer = np.zeros((total_expected,), dtype=int)

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
                metadata_buffer[cursor] = normalized_segment.shape[1]
                cursor += 1
        raw.close()

    if cursor == 0:
        raise ValueError("No eyes-open/eyes-closed segments were extracted.")

    eeg_final = eeg_buffer[:cursor]
    labels_final = labels_buffer[:cursor]
    metadata_final = metadata_buffer[:cursor]
    np.save(eeg_output_path, eeg_final)
    np.save(labels_output_path, labels_final)
    np.save(metadata_output_path, metadata_final)
    with open(channels_output_path, "w", encoding="utf-8") as fp:
        json.dump(channels, fp, indent=2)

    return (
        np.load(eeg_output_path, mmap_mode="r", allow_pickle=True),
        np.load(labels_output_path, mmap_mode="r", allow_pickle=True),
        np.load(metadata_output_path, mmap_mode="r", allow_pickle=True),
    )
