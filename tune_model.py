import os
import argparse
from loguru import logger
import requests
import json
import gc
from typing import Iterator, Tuple, Optional
import tempfile

import numpy as np
from sklearn.linear_model import LogisticRegression, SGDClassifier
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


def get_datasets(
    output_path: str, tune_split: str, t_max: float, t_min: float, task: str
):
    p = os.path.join(output_path, "sensor_xyz.json")
    with open(p, "r") as fp:
        sensor_positions = np.array(json.load(fp))

    tune_dataset = LibriBrainSpeechDataset(
        LibriBrainSpeech(
            os.path.join(output_path, "data"),
            partition=tune_split,
            tmax=t_max,
            stride=1,
            preload_files=True,  # Changed to False to avoid preloading
            standardize=True,
        ),
        sensor_positions=torch.tensor(sensor_positions),
        tmax=t_max,
        tmin=t_min,
    )
    test_dataset = LibriBrainSpeechDataset(
        LibriBrainCompetitionHoldout(
            os.path.join(output_path, "data"),
            tmax=t_max,
            task=task,
        ),
        sensor_positions=torch.tensor(sensor_positions),
        holdout=True,
        tmax=t_max,
        tmin=t_min,
    )

    return tune_dataset, test_dataset


@torch.no_grad()
def generate_logits_streaming(
    model, dataloader, device
) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """
    Generator that yields logits and labels batch by batch for constant memory usage.
    """
    iterator = iter(dataloader)
    pbar = tqdm(total=len(dataloader), desc="Processing batches")

    while True:
        try:
            batch = get_microbatch(iterator, device, torch.bfloat16)
            pbar.update()
        except StopIteration:
            break

        # For your submission, this is where you would generate your model prediction:
        # loss, logits, labels = model(batch)
        labels = batch["labels"]
        logits = torch.randn((labels.shape[0], 2))

        # Convert to numpy and yield immediately to avoid accumulation
        batch_logits = logits.detach().cpu().to(dtype=torch.float32).numpy()
        batch_labels = labels.detach().cpu().to(dtype=torch.long).numpy()

        yield batch_logits, batch_labels

    pbar.close()


def fit_incremental_regressor(model, dataloader, device) -> SGDClassifier:
    """
    Fit logistic regression incrementally using streaming data to maintain constant memory.
    """
    regressor = SGDClassifier(
        loss="log_loss",  # Equivalent to logistic regression
        random_state=42,
        learning_rate="constant",
        eta0=0.01,
        max_iter=1,
    )

    logger.info("Fitting regressor incrementally...")
    first_batch = True

    for batch_logits, batch_labels in generate_logits_streaming(
        model, dataloader, device
    ):
        if first_batch:
            # Initialize with first batch
            regressor.partial_fit(batch_logits, batch_labels, classes=np.array([0, 1]))
            first_batch = False
        else:
            # Update with subsequent batches
            regressor.partial_fit(batch_logits, batch_labels)

    return regressor


def predict_streaming(regressor, model, dataloader, device, output_file: str):
    """
    Generate predictions in streaming fashion and save directly to file.
    """
    logger.info("Generating predictions in streaming mode...")

    with tempfile.NamedTemporaryFile(
        mode="w", delete=False, suffix=".csv"
    ) as temp_file:
        temp_filename = temp_file.name

        # Write header if needed
        predictions = []

        for batch_logits, batch_labels in generate_logits_streaming(
            model, dataloader, device
        ):
            batch_probs = regressor.predict_proba(batch_logits)
            predictions.append(batch_probs)
            break

        all_predictions = np.vstack(predictions)

        return all_predictions


def evaluate_streaming(regressor, model, dataloader, device) -> float:
    """
    Evaluate model performance in streaming fashion to maintain constant memory.
    """
    logger.info("Evaluating model performance...")

    all_predicted_labels = []
    all_true_labels = []

    for batch_logits, batch_labels in generate_logits_streaming(
        model, dataloader, device
    ):
        batch_probs = regressor.predict_proba(batch_logits)
        batch_predicted_labels = batch_probs.argmax(axis=1)

        all_predicted_labels.append(batch_predicted_labels)
        all_true_labels.append(batch_labels)

    # Final concatenation for F1 calculation
    predicted_labels = np.concatenate(all_predicted_labels)
    true_labels = np.concatenate(all_true_labels)

    score = f1_score(true_labels, predicted_labels, labels=[0, 1], average="macro")

    return score


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
    logger.info("Loading Datasets")
    tune_dataset, test_dataset = get_datasets(
        dataset_path, tune_split=tune_split, t_max=t_max, t_min=0.0, task="speech"
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
    # Fit regressor incrementally using streaming data
    logger.info("Fitting regressor incrementally using streaming data...")
    regressor = fit_incremental_regressor(model, tune_dataloader, device)

    # # # Evaluate performance on tune set using streaming
    logger.info("Evaluating performance on tune set...")
    score = evaluate_streaming(regressor, model, tune_dataloader, device)
    logger.info(f"Tune F1 Score: {score}")

    # Clean up tune dataset and dataloader
    del tune_dataset, tune_dataloader
    gc.collect()

    # Generate predictions for test set using streaming
    logger.info("Generating test predictions using streaming...")
    test_probs = predict_streaming(
        regressor, model, test_dataloader, device, output_path
    )

    # Generate submission CSV
    test_dataset.dataset.generate_submission_in_csv(test_probs[:, 1], output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tune a model.")
    parser.add_argument("--model-checkpoint-path", type=str, default=None)
    parser.add_argument("--dataset-path", type=str, required=True)
    parser.add_argument("--model-config-path", type=str, required=True)
    parser.add_argument("--tune-split", type=str, required=True)
    parser.add_argument("--output-path", type=str, required=True)
    parser.add_argument("--t-max", type=float, default=0.8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="bf16")
    args = parser.parse_args()
    main(**vars(args))
