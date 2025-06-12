import os
import numpy as np
from torch.utils.data import Dataset


class EEGEyeNetDataset(Dataset):
    def __init__(self, dataset_path: str, labels_map: dict[str, int]):
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
