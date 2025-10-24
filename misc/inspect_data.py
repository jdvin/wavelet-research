"""Plot random EEG epochs from NumPy memmaps.

Usage example:
    python misc/inspect_data.py data/sample_eeg.npy --num-epochs 5 --seed 0
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")  # Use a non-interactive backend for script execution.
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot random epochs from EEG NumPy memmaps."
    )
    parser.add_argument(
        "files",
        nargs="+",
        help="Path(s) to .npy files containing EEG data with shape (n, n_channels, n_samples).",
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=3,
        help="Number of random epochs to plot per file (default: 3).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducible epoch selection.",
    )
    return parser.parse_args()


def load_memmap(path: Path) -> np.memmap:
    # Using np.load with mmap_mode leverages the metadata stored in .npy files.
    memmap = np.load(path, mmap_mode="r")
    if memmap.ndim != 3:
        raise ValueError(
            f"Expected 3D array (n, n_channels, n_samples) in {path}, got shape {memmap.shape}"
        )
    return memmap


def sample_random_epochs(
    memmap: np.memmap,
    path: Path,
    num_plots: int,
    rng: np.random.Generator,
) -> list[tuple[int, np.ndarray]]:
    if num_plots <= 0:
        print(f"Skipping plots for {path} because num-epochs <= 0.")
        return []

    n_epochs, n_channels, _ = memmap.shape
    if n_epochs == 0:
        raise ValueError(f"{path} contains zero epochs.")

    epochs_to_plot = min(num_plots, n_epochs)
    chosen_indices = rng.choice(n_epochs, size=epochs_to_plot, replace=False)
    return [(int(index), np.asarray(memmap[index])) for index in chosen_indices]


def main() -> None:
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    rng = np.random.default_rng(args.seed)

    dataset_epochs: list[tuple[Path, list[tuple[int, np.ndarray]]]] = []
    channel_counts: dict[Path, int] = {}

    for file_path_str in args.files:
        file_path = Path(file_path_str)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        memmap = load_memmap(file_path)
        sampled = sample_random_epochs(memmap, file_path, args.num_epochs, rng)
        if sampled:
            dataset_epochs.append((file_path, sampled))
            channel_counts[file_path] = memmap.shape[1]

    if not dataset_epochs:
        print("No epochs sampled; nothing to plot.")
        return

    max_epochs = max(len(epochs) for _, epochs in dataset_epochs)
    n_rows = len(dataset_epochs)

    fig, axes = plt.subplots(
        n_rows,
        max_epochs,
        figsize=(12.0 * max_epochs, 3.5 * n_rows),
        squeeze=False,
    )

    for row_idx, (file_path, epochs) in enumerate(dataset_epochs):
        for col_idx in range(max_epochs):
            ax = axes[row_idx][col_idx]
            if col_idx >= len(epochs):
                ax.axis("off")
                continue

            epoch_index, epoch_data = epochs[col_idx]
            n_channels = channel_counts[file_path]
            n_samples = epoch_data.shape[-1]
            time_axis = np.linspace(0.0, 5.0, num=n_samples, endpoint=False)

            for channel_idx in range(n_channels):
                ax.plot(time_axis, epoch_data[channel_idx], linewidth=0.9, alpha=0.7)

            ax.set_xlim(0.0, 5.0)
            ax.set_title(f"{file_path.name} - epoch {epoch_index}")
            if col_idx == 0:
                ax.set_ylabel("Amplitude")

            if n_channels <= 10:
                ax.legend(
                    [f"ch {i}" for i in range(n_channels)],
                    loc="upper right",
                    fontsize="small",
                )

        for col_idx in range(max_epochs):
            ax = axes[row_idx][col_idx]
            if row_idx == n_rows - 1 and ax.has_data():
                ax.set_xlabel("Time (s)")
            elif ax.has_data():
                ax.set_xticklabels([])

    fig.tight_layout()
    output_path = script_dir / "combined_epochs.png"
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
