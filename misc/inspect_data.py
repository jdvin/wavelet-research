"""Plot random EEG epochs from NumPy memmaps.

Usage example:
    python misc/inspect_data.py data/sample --num-epochs 5 --seed 0
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")  # Use a non-interactive backend for script execution.
import matplotlib.pyplot as plt

from misc.filter_stream import main as filter_stream


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot random epochs from EEG NumPy memmaps."
    )
    parser.add_argument(
        "stubs",
        nargs="+",
        help=(
            "File stubs that resolve to `<stub>_eeg.npy` and `<stub>_labels.npy` pairs "
            "with EEG data of shape (n, n_channels, n_samples)."
        ),
    )
    parser.add_argument("--filter-stream", action="store_true", help="Filter EEG data.")
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=3,
        help="Number of random epochs to plot per label (default: 3).",
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


def load_labels(path: Path) -> np.ndarray:
    labels = np.load(path)
    if labels.ndim != 2 or labels.shape[1] < 2:
        raise ValueError(
            f"Expected labels with shape (n, 2) in {path}, got shape {labels.shape}"
        )
    return labels


def resolve_stub_paths(stub: Path) -> tuple[Path, Path]:
    eeg_path = stub.parent / f"{stub.name}_eeg.npy"
    label_path = stub.parent / f"{stub.name}_labels.npy"
    return eeg_path, label_path


def sample_balanced_epochs(
    memmap: np.memmap,
    labels: np.ndarray,
    path: Path,
    num_epochs_per_label: int,
    rng: np.random.Generator,
) -> dict[int, list[tuple[int, np.ndarray]]]:
    if num_epochs_per_label <= 0:
        print(f"Skipping plots for {path} because num-epochs <= 0.")
        return {}

    if memmap.shape[0] != labels.shape[0]:
        raise ValueError(
            f"EEG data ({memmap.shape[0]}) and labels ({labels.shape[0]}) have different lengths for {path}."
        )

    label_column = labels[:, 1]
    unique_labels = np.unique(label_column)
    if unique_labels.size == 0:
        raise ValueError(f"No labels found for {path}.")

    label_indices = {
        label: np.flatnonzero(label_column == label) for label in unique_labels
    }
    per_label = min([
        num_epochs_per_label,
        *[len(indices) for indices in label_indices.values()],
    ])
    if per_label == 0:
        raise ValueError(
            f"Insufficient samples to draw from every class for {path}. "
            "At least one label has zero samples."
        )
    if per_label < num_epochs_per_label:
        print(
            f"Requested {num_epochs_per_label} epochs per label for {path}, "
            f"but the smallest class only has {per_label} samples. Using {per_label} per label."
        )

    sampled_epochs: dict[int, list[tuple[int, np.ndarray]]] = {}
    for label, indices in label_indices.items():
        chosen = rng.choice(indices, size=per_label, replace=False)
        sampled_epochs[label] = [
            (int(index), np.asarray(memmap[index])) for index in chosen
        ]
    return sampled_epochs


def main() -> None:
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    rng = np.random.default_rng(args.seed)

    dataset_epochs: list[tuple[Path, dict[int, list[tuple[int, np.ndarray]]]]] = []
    channel_counts: dict[Path, int] = {}
    observed_labels: set[int] = set()

    for stub_str in args.stubs:
        if args.filter_stream:
            filter_stream(
                stub_str,
                stub_str + "_epoched",
                125,
                16,
                5,
                0,
            )
            stub_str = stub_str + "_epoched"

        stub_path = Path(stub_str)
        eeg_path, label_path = resolve_stub_paths(stub_path)
        if not eeg_path.exists():
            raise FileNotFoundError(f"EEG file not found: {eeg_path}")
        if not label_path.exists():
            raise FileNotFoundError(f"Label file not found: {label_path}")

        memmap = load_memmap(eeg_path)
        labels = load_labels(label_path)
        sampled = sample_balanced_epochs(memmap, labels, eeg_path, args.num_epochs, rng)
        if sampled:
            dataset_epochs.append((eeg_path, sampled))
            channel_counts[eeg_path] = memmap.shape[1]
            observed_labels.update(sampled.keys())

    if not dataset_epochs:
        print("No epochs sampled; nothing to plot.")
        return

    if not observed_labels:
        print("No labels observed; nothing to plot.")
        return

    label_list = sorted(observed_labels)
    n_rows = len(label_list)
    n_cols = args.num_epochs
    if n_cols <= 0:
        print("num-epochs must be positive to plot.")
        return

    for file_path, samples_by_label in dataset_epochs:
        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=(4.5 * n_cols, 3.5 * n_rows),
            squeeze=False,
        )
        n_channels = channel_counts[file_path]
        for row_idx, label_value in enumerate(label_list):
            label_samples = samples_by_label.get(label_value, [])
            for col_idx in range(n_cols):
                ax = axes[row_idx][col_idx]
                title_text = "N/A"
                if col_idx >= len(label_samples):
                    ax.text(
                        0.5,
                        0.5,
                        "N/A",
                        ha="center",
                        va="center",
                        fontsize="large",
                        transform=ax.transAxes,
                    )
                    ax.set_xticks([])
                    ax.set_yticks([])
                else:
                    epoch_index, epoch_data = label_samples[col_idx]
                    n_samples = epoch_data.shape[-1]
                    time_axis = np.linspace(0.0, 5.0, num=n_samples, endpoint=False)
                    for channel_idx in range(n_channels):
                        ax.plot(
                            time_axis,
                            epoch_data[channel_idx],
                            linewidth=0.9,
                            alpha=0.7,
                        )
                    ax.set_xlim(0.0, 5.0)
                    title_text = f"Epoch {epoch_index}"
                    if n_channels <= 10:
                        ax.legend(
                            [f"ch {i}" for i in range(n_channels)],
                            loc="upper right",
                            fontsize="x-small",
                        )

                if row_idx == 0:
                    ax.set_title(title_text, fontsize="small")
                if col_idx == 0:
                    ax.set_ylabel(f"Label {label_value}")
                if row_idx == n_rows - 1:
                    ax.set_xlabel("Time (s)")
                else:
                    ax.set_xticklabels([])

        fig.suptitle(file_path.name, fontsize="large")
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
        output_path = script_dir / f"combined_epochs_{file_path.stem}.png"
        fig.savefig(output_path, dpi=150)
        plt.close(fig)
        print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
