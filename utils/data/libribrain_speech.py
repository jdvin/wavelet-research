import json
import os

import numpy as np
import requests
import torch
from pnpl.datasets import LibriBrainSpeech

from utils.torch_datasets import LibriBrainSpeechDataset


def get_libri_brain_speech_dataset(
    output_path: str,
    tmin: float = 0.0,
    tmax: float = 0.8,
    oversample_silence_jitter: int = 0,
    stride: int | None = None,
    partition: str | None = None,
    books: list[int] | None = None,
    books_chapters: list[list[int]] | None = None,
    sessions: list[int] | None = None,
    preload_files: bool = True,
    sensor_mask: list[int] | None = None,
):
    # Download sensor locations JSON
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
    variant = {}
    if partition is None:
        assert books is not None
        assert books_chapters is not None
        assert sessions is not None
        keys = []
        for book, chapters, session in zip(books, books_chapters, sessions):
            keys.extend([
                ("0", str(chapter), f"Sherlock{book}", str(session))
                for chapter in chapters
            ])
        variant = {"include_run_keys": keys}
    else:
        variant = {"partition": partition}

    return LibriBrainSpeechDataset(
        LibriBrainSpeech(
            os.path.join(output_path, "data"),
            **variant,
            tmin=tmin,
            tmax=tmax,
            preload_files=preload_files,
            stride=stride,
            oversample_silence_jitter=oversample_silence_jitter,
            standardize=True,
        ),
        sensor_positions=torch.tensor(sensor_positions),
        sensors_speech_mask=sensor_mask,
        tmin=tmin,
        tmax=tmax,
    )


def libri_speech_brain_collate_fn(
    items: list[tuple[torch.Tensor, int, np.ndarray, int]],
):
    channel_positions, tasks, channel_signals, labels, metadata = zip(*items)
    channel_positions = torch.stack(channel_positions)
    tasks = torch.tensor(tasks)
    channel_signals = torch.stack(channel_signals)
    labels = torch.tensor(labels)
    metadata = torch.stack(metadata)
    return {
        "channel_positions": channel_positions,
        "tasks": tasks,
        "channel_signals": channel_signals,
        "labels": labels,
        "metadata": metadata,
    }
