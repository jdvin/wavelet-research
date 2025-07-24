import os
import argparse
from loguru import logger
import requests
import json

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from rotary_embedding_torch.rotary_embedding_torch import default
from torch.utils.data import DataLoader
from pnpl.datasets import LibriBrainSpeech, LibriBrainCompetitionHoldout
from tqdm import tqdm
import torch

from utils.data_utils import libri_speech_brain_collate_fn
from utils.torch_datasets import (
    LibriBrainSpeechDataset,
)
from src.montagenet import MontageNetConfig, MontageNet, TaskConfig
from utils.train_utils import get_microbatch, load_yaml


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


def generate_logits(model, dataloader, device):
    all_logits = []
    all_labels = []
    iterator = iter(dataloader)
    while True:
        try:
            batch = get_microbatch(iterator, device, torch.bfloat16)
        except StopIteration:
            break
        # For your submission, this is where you would generate your model prediction:
        loss, logits, labels = model(batch)  # Assuming model has a predict method
        all_logits.append(logits)
        all_labels.append(labels)
    return torch.stack(all_logits).cpu().numpy(), torch.stack(all_labels).cpu().numpy()


def main(
    model_checkpoint_path: str | None,
    model_config_path: str,
    tune_split: str,
    dataset_path: str,
    t_max: float,
    output_path: str,
    batch_size: int,
    num_workers: int,
    device: str,
    dtype: str,
):
    logger.info("Creating model instance.")
    model_config = MontageNetConfig(
        **load_yaml(model_config_path), tasks=[TaskConfig(key="speech", n_classes=2)]
    )
    model = MontageNet(model_config, rank=0, world_size=1).to(
        device, dtype=torch.bfloat16
    )
    if model_checkpoint_path is not None:
        logger.info(f"Loading model from {model_checkpoint_path}")
        model.load_state_dict(
            torch.load(model_checkpoint_path, map_location=device, dtype=torch.bfloat16)
        )
    else:
        logger.warning(
            "No model checkpoint path provided. Running random initialization."
        )
    model.eval()
    logger.info("Loading Datasetss")
    tune_dataset, test_dataset = get_datasets(
        dataset_path, tune_split=tune_split, t_max=t_max, task="speech"
    )
    tune_dataloader = DataLoader(
        tune_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=libri_speech_brain_collate_fn,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=libri_speech_brain_collate_fn,
    )
    logger.info("Generating tune logits.")
    tune_logits, tune_labels = generate_logits(model, tune_dataloader, device)
    logger.info("Fitting regressor.")
    regressor = LogisticRegression(random_state=42).fit(tune_logits, tune_labels)
    logger.info("Predicting probabilities for tune set.")
    predicted_probs = regressor.predict_proba(tune_logits)
    logger.info("Optimising decision boundary.")
    best_f1, best_thresh = 0.0, 0.0
    for thresh in np.arange(0.0, 1.0, 0.1):
        predicted_labels = (predicted_probs > thresh).astype(int)
        score = f1_score(tune_labels, predicted_labels, labels=[0, 1], average="macro")
        logger.info(f"Tuning macro f1 @ thresh {thresh}: {score}")
        if score > best_f1:
            best_f1 = score
            best_thresh = thresh
            logger.info(f"New best macro f1 @ thresh {thresh}!")
    logger.info("=" * 80)
    logger.info(f"Best thresh: {best_thresh}")
    logger.info(f"Predicted F1 @ best thresh: {best_f1}")

    test_logits, test_labels = generate_logits(model, test_dataloader, device)
    test_probs = regressor.predict_proba(test_logits)
    test_dataset.dataset.generate_submission_in_csv(test_probs, output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tune a model.")
    parser.add_argument("--model-checkpoint-path", type=str, default=None)
    parser.add_argument("--dataset-path", type=str, required=True)
    parser.add_argument("--model-config-path", type=str, required=True)
    parser.add_argument("--tune-split", type=str, required=True)
    parser.add_argument("--output-path", type=str, required=True)
    parser.add_argument("--t-max", type=float, default=0.8)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="bf16")
    args = parser.parse_args()
    main(**vars(args))
