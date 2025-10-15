import os
from typing import Sequence

import numpy as np
from pnpl.datasets import LibriBrainCompetitionHoldout, LibriBrainSpeech
from torch.utils.data import Dataset
import torch
from torch.nn import functional as F

from src.montagenet import DataConfig, lcmN


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
        channel_positions: torch.Tensor,
        data_config: DataConfig,
        sr: int,
        channel_mask: None | list[int] | np.ndarray | torch.Tensor = None,
    ):
        self.labels_map = labels_map
        self.tasks_map = tasks_map
        self.inputs = inputs
        self.labels = labels
        self.channel_mask = channel_mask
        if channel_mask is not None:
            self.channel_positions = channel_positions[channel_mask]
        else:
            self.channel_positions = channel_positions
        sr_lcm = lcmN(*data_config.sampling_rates)
        self.sequence_positions = torch.arange(
            0, int(sr_lcm * data_config.sequence_length_seconds), sr_lcm // sr
        )
        assert sr in data_config.sampling_rates

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | int | np.ndarray]:
        input = self.inputs[index]
        if self.channel_mask is not None:
            input = input[self.channel_mask]
        task, label = self.labels[index, :]
        return {
            "channel_signals": input,
            "channel_positions": self.channel_positions,
            "sequence_positions": self.sequence_positions,
            "tasks": self.tasks_map[task],
            "labels": self.labels_map[label],
        }


class MultiMappedLabelDataset(Dataset):
    def __init__(self, datasets: Sequence[MappedLabelDataset]):
        if len(datasets) == 0:
            raise ValueError("At least one dataset is required.")
        self.datasets = list(datasets)
        self.lengths = [len(ds) for ds in self.datasets]
        self.total_length = sum(self.lengths)

    def __len__(self) -> int:
        return self.total_length

    def __getitem__(self, index: int):
        if index < 0:
            index += self.total_length
        if index < 0 or index >= self.total_length:
            raise IndexError("Index out of range.")
        for dataset, size in zip(self.datasets, self.lengths):
            if index < size:
                return dataset[index]
            index -= size
        # Should never reach here.
        raise IndexError("Index out of range.")

    def append_dataset(self, dataset: MappedLabelDataset) -> None:
        self.datasets.append(dataset)
        dataset_length = len(dataset)
        self.lengths.append(dataset_length)
        self.total_length += dataset_length


LIBRI_BRAIN_SR = 250


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
        dataset: LibriBrainSpeech | LibriBrainCompetitionHoldout,
        sensor_positions: torch.Tensor,
        tmax: float,
        tmin: float,
        sensors_speech_mask: None | list[int] = None,
        holdout: bool = False,
    ):
        self.dataset = dataset
        self.sensor_positions = sensor_positions
        # These are the sensors we identified:
        self.sensors_speech_mask = sensors_speech_mask
        self.holdout = holdout
        self.samples_per_item = (tmax - tmin) * LIBRI_BRAIN_SR

    def __len__(self):
        return len(self.dataset.samples)

    def pad_to_length(self, tensor: torch.Tensor, index: int):
        """Incredibly complicated function for the incredibly rare event that, whilst
        iterating over the competition holdout set, we get an example on either edge.
        """
        if tensor.shape[-1] == self.samples_per_item:
            return tensor
        # Left, right padding for each axis.
        pad = torch.zeros(2, dtype=torch.long)
        # Pad on the left if we are on the left half of the dataset, otherwise on the right.
        pad_pos = 0 if index < self.__len__() // 2 else 1
        pad[pad_pos] = self.samples_per_item - tensor.shape[-1]
        return F.pad(
            tensor,
            pad.tolist(),
            mode="constant",
            value=0,
        )

    def __getitem__(self, index):
        if self.sensors_speech_mask is not None:
            sensors = self.dataset[index][0][self.sensors_speech_mask]
        else:
            sensors = self.dataset[index][0][:]

        if not self.holdout:
            label_from_the_middle_idx = self.dataset[index][1].shape[0] // 2

            # Speech density is the proportion of non-zero labels.
            speech_density = (
                self.dataset[index][1].sum() / self.dataset[index][1].shape[0]
            )
            # Speech proximity is the minimum absolute distance between the target label and the closest non-zero label
            # normalised by the maximum possible distance.
            pos_labels = self.dataset[index][1].nonzero()
            if pos_labels.numel() > 0:
                speech_proximity = (
                    (pos_labels - label_from_the_middle_idx).abs().min()
                ) / (self.dataset[index][1].shape[0] // 2)
            else:
                speech_proximity = 0
            label = self.dataset[index][1][label_from_the_middle_idx]
            metadata = torch.tensor([speech_density, speech_proximity])
        else:
            sensors = self.pad_to_length(sensors, index)
            label = torch.tensor(0)
            metadata = torch.tensor([0, 0])

        assert sensors.shape[0] == 306, f"{index}: {sensors.shape}"

        return [
            self.sensor_positions,
            0,  # Task HACK.
            sensors,
            label,
            metadata,
        ]
