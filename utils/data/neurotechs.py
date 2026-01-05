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


def extract_neurotechs_eyes_split(
    split: NeurotechsEyesDataSplit,
    reset_cache: bool = False,
) -> tuple[np.memmap, np.memmap, np.memmap]:
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
    metadata_buffer = np.zeros((total_expected,), dtype=int)

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
            metadata_buffer[cursor] = normalized_closed.shape[1]
            cursor += 1
            if cursor >= eeg_buffer.shape[0]:
                raise RuntimeError("Cursor exceeded Neurotechs buffer allocation.")
            normalized_open = _normalize_epochs(open_segments[seg_idx])
            eeg_buffer[cursor] = normalized_open
            labels_buffer[cursor, 0] = Task.MOVE_EYES.value
            labels_buffer[cursor, 1] = Annotation.EYES_OPEN.value
            metadata_buffer[cursor] = normalized_open.shape[1]
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
