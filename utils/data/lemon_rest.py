from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from loguru import logger
import mne
import numpy as np
from numpy.lib.format import open_memmap
from tqdm import tqdm

from .common import (
    Annotation,
    DataSource,
    DataSplit,
    Headset,
    TASK_LABEL_DTYPE,
    Task,
    _normalize_epochs,
)


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


def extract_lemon_resting_state(
    split: LemonRestingStateDataSplit,
    ignore_cache: bool = False,
) -> tuple[np.memmap, np.memmap, np.memmap]:
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
        A tuple of (eeg_memmap, labels_memmap, metadata_memmap). Labels are stored
        as string pairs ``[task, label]`` using TASK_LABEL_DTYPE. Metadata stores
        the true epoch length.
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
    metadata_output_path = out_path / f"{cache_prefix}_metadata.npy"
    channels_output_path = out_path / f"{cache_prefix}_channels.json"

    if not ignore_cache and eeg_output_path.exists() and labels_output_path.exists() and metadata_output_path.exists():
        logger.info(f"Using cached LEMON extraction at {eeg_output_path}")
        return (
            np.load(eeg_output_path, mmap_mode="r", allow_pickle=True),
            np.load(labels_output_path, mmap_mode="r", allow_pickle=True),
            np.load(metadata_output_path, mmap_mode="r", allow_pickle=True),
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
    metadata_output_path = out_path / f"{cache_prefix}_metadata.npy"
    channels_output_path = out_path / f"{cache_prefix}_channels.json"

    if not ignore_cache and eeg_output_path.exists() and labels_output_path.exists() and metadata_output_path.exists():
        logger.info(f"Using cached LEMON extraction at {eeg_output_path}")
        return (
            np.load(eeg_output_path, mmap_mode="r", allow_pickle=True),
            np.load(labels_output_path, mmap_mode="r", allow_pickle=True),
            np.load(metadata_output_path, mmap_mode="r", allow_pickle=True),
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
    metadata_store = open_memmap(
        metadata_output_path,
        mode="w+",
        dtype=int,
        shape=(total_epochs,),
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
            metadata_store[cursor + i] = epoch_data.shape[2]

        cursor += n_epochs

    if cursor != total_epochs:
        raise RuntimeError(
            f"Expected to write {total_epochs} epochs, but wrote {cursor}."
        )

    with open(channels_output_path, "w", encoding="utf-8") as fp:
        json.dump(reference_channels, fp, indent=2)
    eeg_store.flush()
    labels_store.flush()
    metadata_store.flush()

    return (
        np.load(eeg_output_path, mmap_mode="r", allow_pickle=True),
        np.load(labels_output_path, mmap_mode="r", allow_pickle=True),
        np.load(metadata_output_path, mmap_mode="r", allow_pickle=True),
    )
