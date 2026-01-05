from dataclasses import dataclass
import os
from pathlib import Path
import re

from loguru import logger
import numpy as np
import pandas as pd

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

EMOTIV_HEADSET_ALIASES: dict[str, Headset] = {
    "EPOC": Headset.EMOTIV_EPOC_14,
    "EPOCPLUS": Headset.EMOTIV_EPOC_14,
    "EPOCX": Headset.EMOTIV_EPOC_14,
    "INSIGHT": Headset.EMOTIV_INSIGHT_5,
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
    metadata_store: np.ndarray,
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
            metadata_store[next_epoch_idx] = epoch_slice.shape[1]
            next_epoch_idx += 1
            epoch_start = epoch_end
    return next_epoch_idx


def extract_emotiv_alpha_suppression_split(
    split: EmotivAlphaDataSplit,
    reset_cache: bool = False,
) -> tuple[np.memmap, np.memmap, np.memmap]:
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
    metadata_output_path = os.path.join(split.output_path, f"{cache_prefix}_metadata.npy")

    if (
        os.path.exists(eeg_output_path)
        and os.path.exists(labels_output_path)
        and os.path.exists(metadata_output_path)
        and not reset_cache
    ):
        return (
            np.load(eeg_output_path, mmap_mode="r", allow_pickle=True),
            np.load(labels_output_path, mmap_mode="r", allow_pickle=True),
            np.load(metadata_output_path, mmap_mode="r", allow_pickle=True),
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
    metadata_store = np.zeros((total_epochs,), dtype=int)

    epoch_cursor = 0
    for recording in recordings:
        epoch_cursor = _write_emotiv_recording_data(
            recording=recording,
            epoch_length=epoch_length_samples,
            channels=channels,
            eeg_store=eeg_store,
            labels_store=labels_store,
            metadata_store=metadata_store,
            start_epoch_idx=epoch_cursor,
        )

    np.save(eeg_output_path, eeg_store)
    np.save(labels_output_path, labels_store)
    np.save(metadata_output_path, metadata_store)

    return (
        np.load(eeg_output_path, mmap_mode="r", allow_pickle=True),
        np.load(labels_output_path, mmap_mode="r", allow_pickle=True),
        np.load(metadata_output_path, mmap_mode="r", allow_pickle=True),
    )
