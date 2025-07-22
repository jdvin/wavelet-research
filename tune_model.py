import os
import argparse
from loguru import logger
import requests
import json

import numpy as np
from sklearn.linear_model import LogisticRegression
from rotary_embedding_torch.rotary_embedding_torch import default
from torch.utils.data import DataLoader
from pnpl.datasets import LibriBrainSpeech, LibriBrainCompetitionHoldout
from tqdm import tqdm
import torch

from utils.torch_datasets import (
    LibriBrainSpeechDataset,
)
from src.montagenet import MontageNetConfig, MontageNet, TaskConfig
from utils.train_utils import load_yaml


def get_datasets(output_path: str, tune_split: str, t_max: float, task: str):
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

    tune_dataset = LibriBrainSpeechDataset(
        LibriBrainSpeech(
            os.path.join(output_path, "data"),
            partition=tune_split,
            tmax=t_max,
            stride=1,
            preload_files=True,
            standardize=True,
        ),
        sensor_positions=torch.tensor(sensor_positions),
    )
    test_dataset = LibriBrainSpeechDataset(
        LibriBrainCompetitionHoldout(
            os.path.join(output_path, "data"),
            tmax=t_max,
            task=task,
        ),
        sensor_positions=torch.tensor(sensor_positions),
    )

    return tune_dataset, test_dataset


def generate_logits(model, dataloader):
    predictions = []
    labels = []
    for i, batch in enumerate(tqdm(dataloader)):
        # For your submission, this is where you would generate your model prediction:
        loss, logits, labels = model(batch)  # Assuming model has a predict method
        predictions.append(logits)
        labels.append(labels)
    return torch.stack(predictions).cpu().numpy(), torch.stack(labels).cpu().numpy()


def main(
    model_checkpoint_path: str,
    model_config_path: str,
    tune_split: str,
    dataset_path: str,
    t_max: float,
    output_path: str,
    batch_size: int,
    num_workers: int,
    device: str,
):
    model_config = MontageNetConfig(
        **load_yaml(model_config_path), tasks=[TaskConfig(key="speech", n_classes=2)]
    )
    model = MontageNet(model_config, rank=0, world_size=1)
    model.load_state_dict(torch.load(model_checkpoint_path, map_location=device))
    tune_dataset, test_dataset = get_datasets(
        dataset_path, tune_split=tune_split, t_max=t_max, task="speech"
    )
    tune_dataloader = DataLoader(
        tune_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    tune_logits, tune_labels = generate_logits(model, tune_dataloader)
    regressor = LogisticRegression(random_state=42).fit(tune_logits, tune_labels)
    logger.info(f"Tuning accuracy: {regressor.score(tune_logits, tune_labels)}")
    test_logits, test_labels = generate_logits(model, test_dataloader)
    test_probs = regressor.predict_proba(test_logits)

    test_dataset.generate_submission_in_csv(...)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tune a model.")
    parser.add_argument("--model-checkpoint-path", type=str, required=True)
    parser.add_argument("--dataset-path", type=str, required=True)
    parser.add_argument("--model-config-path", type=str, required=True)
    parser.add_argument("--tune-split", type=str, required=True)
    parser.add_argument("--output-path", type=str, required=True)
    parser.add_argument("--t-max", type=float, default=0.8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    main(**vars(args))
