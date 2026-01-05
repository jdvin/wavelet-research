from dataclasses import dataclass
import os

import mne
import numpy as np

from .common import (
    Annotation,
    DataSource,
    DataSplit,
    Headset,
    TASK_LABEL_DTYPE,
    Task,
    _normalize_epochs,
)


EEG_MMI_SESSION_TO_TASK = {
    "R01": Task.MOVE_EYES,
    "R02": Task.MOVE_EYES,
    "R03": Task.MOVE_IMAGE_LEFT_RIGHT_FIST_BOTH_FIST_FEET,
    "R04": Task.MOVE_IMAGE_LEFT_RIGHT_FIST_BOTH_FIST_FEET,
    "R05": Task.MOVE_IMAGE_LEFT_RIGHT_FIST_BOTH_FIST_FEET,
    "R06": Task.MOVE_IMAGE_LEFT_RIGHT_FIST_BOTH_FIST_FEET,
    "R07": Task.MOVE_IMAGE_LEFT_RIGHT_FIST_BOTH_FIST_FEET,
    "R08": Task.MOVE_IMAGE_LEFT_RIGHT_FIST_BOTH_FIST_FEET,
    "R09": Task.MOVE_IMAGE_LEFT_RIGHT_FIST_BOTH_FIST_FEET,
    "R10": Task.MOVE_IMAGE_LEFT_RIGHT_FIST_BOTH_FIST_FEET,
    "R11": Task.MOVE_IMAGE_LEFT_RIGHT_FIST_BOTH_FIST_FEET,
    "R12": Task.MOVE_IMAGE_LEFT_RIGHT_FIST_BOTH_FIST_FEET,
    "R13": Task.MOVE_IMAGE_LEFT_RIGHT_FIST_BOTH_FIST_FEET,
    "R14": Task.MOVE_IMAGE_LEFT_RIGHT_FIST_BOTH_FIST_FEET,
}

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
) -> tuple[np.memmap, np.memmap, np.memmap]:
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
    session_metadata = np.memmap(
        filename=os.path.join(
            output_path, f"{subject_code}{session_code}_metadata.npy"
        ),
        mode="r" if cached else "w+",
        shape=(len(events),),
        dtype=int,
    )
    if cached:
        return session_eeg, session_labels, session_metadata

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
        annotation_code = labels_to_annotations[current_event[2]]
        task = EEG_MMI_SESSION_TO_TASK[session].value
        annotation = EEG_MMI_SESSION_ANNOTATION_CODE_MAP[session][
            annotation_code.item()
        ].value
        epoch_slice = _normalize_epochs(eeg_data[:, event_start:event_stop])
        session_eeg[i, :, 0 : epoch_slice.shape[-1]] = epoch_slice
        session_labels[i, :] = [task, annotation]
        session_metadata[i] = epoch_slice.shape[1]

    session_eeg.flush()
    session_labels.flush()
    session_metadata.flush()
    return session_eeg, session_labels, session_metadata


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
    metadata_path = os.path.join(
        output_path, f"{split.split_name}_{split.code()}_metadata.npy"
    )
    if (
        os.path.exists(eeg_path)
        and os.path.exists(labels_path)
        and os.path.exists(metadata_path)
        and not reset_cache
    ):
        return (
            np.load(eeg_path, mmap_mode="r", allow_pickle=True),
            np.load(labels_path, mmap_mode="r", allow_pickle=True),
            np.load(metadata_path, mmap_mode="r", allow_pickle=True),
        )

    target_epoch_length_sec = epoch_length_sec or split.epoch_length_sec
    if target_epoch_length_sec is None:
        raise ValueError("EEG MMI splits require an epoch_length_sec.")

    # First pass: extract all sessions and collect their shapes.
    # Session memmaps are stored on disk, so we only keep references.
    session_info: list[tuple[np.memmap, np.memmap, np.memmap, tuple[int, ...]]] = []
    for subject in split.subjects:
        for session in split.sessions:
            eeg, label, metadata = extract_eeg_mmi_session_data(
                base_path,
                output_path,
                subject,
                session,
                reset_cache,
                epoch_length_sec=target_epoch_length_sec,
                sampling_rate=split.sampling_rate,
                default_event_length_sec=target_epoch_length_sec,
            )
            session_info.append((eeg, label, metadata, eeg.shape))

    # Calculate total dimensions.
    n_trials = sum(shape[0] for _, _, _, shape in session_info)
    n_channels = max(shape[1] for _, _, _, shape in session_info)
    n_samples = max(shape[2] for _, _, _, shape in session_info)
    dtype = session_info[0][0].dtype

    # Create output memmaps directly on disk.
    split_eeg = np.memmap(
        eeg_path,
        dtype=dtype,
        mode="w+",
        shape=(n_trials, n_channels, n_samples),
    )
    split_labels = np.memmap(
        labels_path,
        dtype=TASK_LABEL_DTYPE,
        mode="w+",
        shape=(n_trials, 2),
    )
    split_metadata = np.memmap(
        metadata_path,
        dtype=int,
        mode="w+",
        shape=(n_trials,),
    )

    # Second pass: copy data from session memmaps to output memmaps.
    cum_trial = 0
    for eeg, label, metadata, shape in session_info:
        n_sess_trials, n_sess_channels, n_sess_samples = shape
        split_eeg[
            cum_trial : cum_trial + n_sess_trials,
            0:n_sess_channels,
            0:n_sess_samples,
        ] = eeg
        split_labels[cum_trial : cum_trial + n_sess_trials] = label
        split_metadata[cum_trial : cum_trial + n_sess_trials] = metadata
        cum_trial += n_sess_trials

    # Flush to ensure data is written to disk.
    split_eeg.flush()
    split_labels.flush()
    split_metadata.flush()

    # Return read-only memmaps.
    return (
        np.memmap(
            eeg_path, dtype=dtype, mode="r", shape=(n_trials, n_channels, n_samples)
        ),
        np.memmap(labels_path, dtype=TASK_LABEL_DTYPE, mode="r", shape=(n_trials, 2)),
        np.memmap(metadata_path, dtype=int, mode="r", shape=(n_trials,)),
    )
