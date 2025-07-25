import os
import numpy as np
from torch.utils.data import Dataset
import torch


class EEGEyeNetDataset(Dataset):
    def __init__(
        self,
        dataset_path: str,
        labels_map: dict[str, int],
    ):
        self.dataset_path = dataset_path
        self.labels_map = labels_map
        self.inputs = np.load(os.path.join(dataset_path, "EEG.npy"), mmap_mode="r")
        self.labels = np.load(os.path.join(dataset_path, "labels.npy"), mmap_mode="r")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index) -> tuple[np.memmap, list[int]]:
        return self.inputs[index], [
            self.labels_map[label.item().split("_")[1]] for label in self.labels[index]
        ]


class MappedLabelDataset(Dataset):
    def __init__(
        self,
        inputs: np.memmap,
        labels: np.memmap,
        labels_map: dict[str, int],
        tasks_map: dict[str, int],
        sensor_positions: torch.Tensor,
        sensor_mask: None | list[int] = None,
    ):
        self.labels_map = labels_map
        self.tasks_map = tasks_map
        self.inputs = inputs
        self.labels = labels
        self.sensor_mask = sensor_mask
        if sensor_mask is not None:
            self.sensor_positions = sensor_positions[sensor_mask]
        else:
            self.sensor_positions = sensor_positions

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, np.ndarray, int]:
        input = self.inputs[index]
        if self.sensor_mask is not None:
            input = input[self.sensor_mask]
        task, label = self.labels[index, :]
        return (
            self.sensor_positions,
            self.tasks_map[task],
            input,
            self.labels_map[label],
        )


class LibriBrainSpeechDataset(Dataset):
    """
    Parameters:
        dataset: LibriBrain dataset.
        limit_samples (int, optional): If provided, limits the length of the dataset to this
                          number of samples.
        speech_silence_only (bool, optional): If True, only includes segments that are either
                          purely speech or purely silence (with additional balancing).
        apply_sensors_speech_mask (bool, optional): If True, applies a fixed sensor mask to the sensor
                          data in each sample.
    """

    def __init__(
        self,
        dataset,
        sensor_positions: torch.Tensor,
        sensors_speech_mask: None | list[int] = None,
    ):
        self.dataset = dataset
        self.sensor_positions = sensor_positions
        # These are the sensors we identified:
        self.sensors_speech_mask = sensors_speech_mask

    def __len__(self):
        return len(self.dataset.samples)

    def __getitem__(self, index):
        if self.sensors_speech_mask is not None:
            sensors = self.dataset[index][0][self.sensors_speech_mask]
        else:
            sensors = self.dataset[index][0][:]

        label_from_the_middle_idx = self.dataset[index][1].shape[0] // 2
        # try:
        #     label_from_the_middle_idx = self.dataset[index][1].shape[0] // 2
        # except AttributeError:
        #     print(self.dataset[index])
        #     breakpoint()
        return [
            self.sensor_positions,
            0,  # Task HACK.
            sensors,
            self.dataset[index][1][label_from_the_middle_idx],
        ]
