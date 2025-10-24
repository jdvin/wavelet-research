from copy import deepcopy
from collections import OrderedDict
import tempfile
import os
from dataclasses import dataclass
from enum import Enum
from typing import Any
import mne
import requests, io, numpy as np
from stl import mesh
import plotly.graph_objects as go
import plotly.io as pio
import torch

PHYSIONET_64_CHANNELS = [
    "FC5",
    "FC3",
    "FC1",
    "FCz",
    "FC2",
    "FC4",
    "FC6",
    "C5",
    "C3",
    "C1",
    "Cz",
    "C2",
    "C4",
    "C6",
    "CP5",
    "CP3",
    "CP1",
    "CPz",
    "CP2",
    "CP4",
    "CP6",
    "Fp1",
    "Fpz",
    "Fp2",
    "AF7",
    "AF3",
    "AFz",
    "AF4",
    "AF8",
    "F7",
    "F5",
    "F3",
    "F1",
    "Fz",
    "F2",
    "F4",
    "F6",
    "F8",
    "FT7",
    "FC5",
    "FC3",
    "FC1",
    "FCz",
    "FC2",
    "FC4",
    "FC6",
    "FT8",
    "T7",
    "T8",
    "T9",
    "T10",
    "TP7",
    "TP8",
    "P7",
    "P5",
    "P3",
    "P1",
    "Pz",
    "P2",
    "P4",
    "P6",
    "P8",
    "PO7",
    "PO3",
    "POz",
    "PO4",
    "PO8",
    "O1",
    "Oz",
    "O2",
    "Iz",
]

# --- Emotiv channel sets (actives only) ---
EPOC14_CHANNELS = [
    "AF3",
    "F7",
    "F3",
    "FC5",
    "T7",
    "P7",
    "O1",
    "O2",
    "P8",
    "T8",
    "FC6",
    "F4",
    "F8",
    "AF4",
]
INSIGHT5_CHANNELS = ["AF3", "AF4", "T7", "T8", "Pz"]

NEUROTECHS_CHANNELS = ["Fz", "C3", "Cz", "C4", "Pz", "PO7", "Oz", "PO8"]

RESTING_METHODS_CHANNELS = [
    "Fp1",
    "F3",
    "F7",
    "FT9",
    "FC5",
    "FC1",
    "C3",
    "T7",
    "TP9",
    "CP5",
    "CP1",
    "Pz",
    "P3",
    "P7",
    "O1",
    "Oz",
    "O2",
    "P4",
    "P8",
    "TP10",
    "CP6",
    "CP2",
    "Cz",
    "C4",
    "T8",
    "FT10",
    "FC6",
    "FC2",
    "F4",
    "F8",
    "Fp2",
]

LEMON_CHANNELS = [
    "Fp1",
    "Fp2",
    "F7",
    "F3",
    "Fz",
    "F4",
    "F8",
    "FC5",
    "FC1",
    "FC2",
    "FC6",
    "T7",
    "C3",
    "Cz",
    "C4",
    "T8",
    "CP5",
    "CP1",
    "CP2",
    "CP6",
    "AFz",
    "P7",
    "P3",
    "Pz",
    "P4",
    "P8",
    "PO9",
    "O1",
    "Oz",
    "O2",
    "PO10",
    "AF7",
    "AF3",
    "AF4",
    "AF8",
    "F5",
    "F1",
    "F2",
    "F6",
    "FT7",
    "FC3",
    "FC4",
    "FT8",
    "C5",
    "C1",
    "C2",
    "C6",
    "TP7",
    "CP3",
    "CPz",
    "CP4",
    "TP8",
    "P5",
    "P1",
    "P2",
    "P6",
    "PO7",
    "PO3",
    "POz",
    "PO4",
    "PO8",
]

# Choose a standard montage that includes AF3/AF4/T7/T8/Pz, etc.
# "standard_1020" works for these; you can also try "standard_1005" for denser sets.
STANDARD_1020 = mne.channels.make_standard_montage("standard_1020").get_positions()[
    "ch_pos"
]

INSIGHT5_CHANNEL_POSITIONS = torch.tensor(
    np.vstack([STANDARD_1020[ch] for ch in INSIGHT5_CHANNELS]), dtype=torch.float32
)
EPOC14_CHANNEL_POSITIONS = torch.tensor(
    np.vstack([STANDARD_1020[ch] for ch in EPOC14_CHANNELS]), dtype=torch.float32
)
NEUROTECHS_CHANNEL_POSITIONS = torch.tensor(
    np.vstack([STANDARD_1020[ch] for ch in NEUROTECHS_CHANNELS]), dtype=torch.float32
)
RESTING_METHODS_CHANNEL_POSITIONS = torch.tensor(
    np.vstack([STANDARD_1020[ch] for ch in RESTING_METHODS_CHANNELS]),
    dtype=torch.float32,
)
LEMON_CHANNEL_POSITIONS = torch.tensor(
    np.vstack(
        [np.asarray(STANDARD_1020[ch], dtype=np.float32) for ch in LEMON_CHANNELS]
    ),
    dtype=torch.float32,
)


def physionet_64_montage():
    dense = mne.channels.make_standard_montage("standard_1005")
    pos = dense.get_positions()  # dict of xyz + fiducials

    ch_pos = OrderedDict((ch, pos["ch_pos"][ch]) for ch in PHYSIONET_64_CHANNELS)
    mont_physio = mne.channels.make_dig_montage(
        ch_pos=ch_pos,
        nasion=pos["nasion"],
        lpa=pos["lpa"],
        rpa=pos["rpa"],
        coord_frame="head",
    )

    return mont_physio


PHYSIONET_64_CHANNEL_POSITIONS = torch.tensor(
    np.vstack(list(physionet_64_montage().get_positions()["ch_pos"].values())),
    dtype=torch.float32,
)


# ---------- helper -----------------------------------------------------------
def fetch_stl(url: str) -> mesh.Mesh:
    raw = requests.get(url, timeout=30)
    raw.raise_for_status()

    # Save to temporary file and load
    with tempfile.NamedTemporaryFile(suffix=".stl", delete=False) as tmp:
        tmp.write(raw.content)
        tmp_path = tmp.name

    try:
        stl_mesh = mesh.Mesh.from_file(tmp_path)
        return stl_mesh
    finally:
        os.unlink(tmp_path)


def mesh_to_plotly(mesh_obj, color, opacity, name):
    """Convert STL mesh to plotly mesh3d format"""
    # Extract vertices and faces
    vertices = mesh_obj.vectors.reshape(-1, 3)

    # Create faces (triangles) - each set of 3 vertices forms a triangle
    n_triangles = len(mesh_obj.vectors)
    faces = np.arange(n_triangles * 3).reshape(-1, 3)

    return go.Mesh3d(
        x=vertices[:, 0],
        y=vertices[:, 1],
        z=vertices[:, 2],
        i=faces[:, 0],
        j=faces[:, 1],
        k=faces[:, 2],
        color=color,
        opacity=opacity,
        name=name,
        showscale=False,
        lighting=dict(ambient=0.4, diffuse=0.8, specular=0.1),
        lightposition=dict(x=100, y=200, z=0),
    )


class Region(Enum):
    FRONTAL = "frontal"
    TEMPORAL = "temporal"
    PARIETAL = "parietal"
    OCCIPITAL = "occipital"


@dataclass
class Montage:
    name: str
    electrode_names: list[str]
    pos_array: np.ndarray


class ChannelMaskType(Enum):
    REGION = "region"
    NTH = "nth"


def get_nth_mask(size: int, n: int, offset: int = 1) -> np.ndarray:
    mask = np.ones(size)
    mask[offset - 1 :: n] = 0
    return mask


def get_region_electrodes(
    electrode_positions: np.ndarray, regions: list[Region]
) -> np.ndarray:
    """Get a boolean mask for the specified regions"""
    # ---- 2. convert to head-centric spherical coordinates ------------------
    # x = Right(+), y = Anterior(+), z = Superior(+)
    x, y, z = electrode_positions.T
    r = np.linalg.norm(electrode_positions, axis=1)  # radius (≈ constant)
    az = np.degrees(np.arctan2(y, x)) % 360  # 0°=R, 90°=A, 180°=L, 270°=P
    el = np.degrees(np.arcsin(z / r))  # +90° = vertex (Cz)

    # ---- 3. crude lobar boundaries in spherical coords --------------------
    # You can tweak these thresholds for your net size / ROI granularity.
    mask = np.zeros(electrode_positions.shape[0], dtype=bool)
    frontal = (az >= 45) & (az <= 135)
    occipital = (az >= 225) & (az <= 315)
    temporal = (((az > 135) & (az < 225)) | ((az > 315) | (az < 45))) & (el < 15)
    parietal = (~frontal & ~occipital & ~temporal) & (el > 15)
    if Region.FRONTAL in regions:
        mask = mask | frontal
    if Region.TEMPORAL in regions:
        mask = mask | temporal
    if Region.PARIETAL in regions:
        mask = mask | parietal
    if Region.OCCIPITAL in regions:
        mask = mask | occipital
    return mask


class ChannelMaskConfig:
    def __init__(self, mask_type: str, args: dict[str, Any]):
        self.mask_type = ChannelMaskType(mask_type)
        self.args = args


def create_mask(pos_array: np.ndarray, mask_config: ChannelMaskConfig) -> np.ndarray:
    if mask_config.mask_type == ChannelMaskType.REGION:
        mask = ~get_region_electrodes(pos_array, mask_config.args["regions"])
    elif mask_config.mask_type == ChannelMaskType.NTH:
        mask = get_nth_mask(
            pos_array.shape[0], mask_config.args["n"], mask_config.args["offset"]
        )
    else:
        raise NotImplementedError(f"Unknown mask type: {mask_config.mask_type}")

    return mask[np.newaxis, ..., np.newaxis]


def mask_regions(montage: Montage, regions: list[Region]) -> Montage:
    """Mask the montage to only include the specified regions"""
    mask = ~get_region_electrodes(montage.pos_array, regions)
    return Montage(
        str(regions),
        [name for m, name in zip(mask, montage.electrode_names) if m],
        montage.pos_array[mask],
    )


def get_montage(montage_name: str) -> Montage:
    """Load a montage from the montage library"""
    montage = mne.channels.make_standard_montage(montage_name)
    electrode_pos = montage.get_positions()["ch_pos"]
    return Montage(
        montage_name, list(electrode_pos.keys()), np.array(list(electrode_pos.values()))
    )


def plot_montages(
    fig,
    montages: list[Montage],
    colors: list[str],
    ref_montage: Montage,
    skull_bounds=None,
):
    """Plot multiple electrode montages on the figure with different colors

    Args:
        fig: Plotly figure object to add traces to
        montage_names: List of montage names (e.g., ['GSN-HydroCel-129', 'standard_1020'])
        colors: List of colors for each montage (defaults to cycle through common colors)
        skull_bounds: Skull mesh bounds for scaling (optional)

    Returns:
        List of electrode position arrays for bounds calculation
    """

    all_electrode_positions = []

    for i, montage in enumerate(montages):
        color = colors[i]

        pos_array = montage.pos_array
        electrode_names = montage.electrode_names

        # Scale electrode positions to approximate skull size if skull_bounds provided
        if skull_bounds is not None:
            skull_scale = np.max(skull_bounds) - np.min(skull_bounds)
            montage_scale = np.max(pos_array) - np.min(ref_montage.pos_array)
            scale_factor = skull_scale / montage_scale * 0.75

            # Center and scale the electrode positions
            pos_centered = pos_array - np.mean(ref_montage.pos_array, axis=0)
            pos_scaled = pos_centered * scale_factor
            pos_final = pos_scaled
        else:
            pos_final = pos_array

        # Store electrode positions for bounds calculation
        all_electrode_positions.append(pos_final)

        fig.add_trace(
            go.Scatter3d(
                x=pos_final[:, 0],
                y=pos_final[:, 1],
                z=pos_final[:, 2],
                mode="markers",
                marker=dict(size=4, color=color, opacity=0.8),
                text=electrode_names,
                name=f"Electrodes ({montage.name})",
                hovertemplate=f"<b>%{{text}}</b><br>Montage: {montage.name}<br>x: %{{x:.2f}}<br>y: %{{y:.2f}}<br>z: %{{z:.2f}}<extra></extra>",
            )
        )

        fig.add_trace(
            go.Scatter3d(
                x=pos_final[:, 0],
                y=pos_final[:, 1],
                z=pos_final[:, 2] + 5,  # Offset labels above markers
                mode="text",
                text=electrode_names,
                textposition="middle center",
                textfont=dict(size=8, color=color),
                name=f"Labels ({montage.name})",
                showlegend=False,
                hoverinfo="skip",
            )
        )

    return all_electrode_positions


brain_url = (
    "https://raw.githubusercontent.com/garretthinton/BrainSPECT/" "master/brain.stl"
)
skull_url = (
    "https://raw.githubusercontent.com/XiaoxiaoLiu/3D-printing/"
    "master/final_skull_visible_female_full.stl"
)


def main():
    # Set plotly to open in browser
    pio.renderers.default = "browser"

    # Load meshes
    print("Loading meshes...")
    brain_mesh = fetch_stl(brain_url)
    skull_mesh = fetch_stl(skull_url)

    # Fix skull orientation - rotate 180 degrees around X-axis
    skull_mesh.rotate([0, 1, 0], np.radians(180))
    brain_mesh.rotate([0, 0, 1], np.radians(180))

    # Scale up skull to better fit montage
    skull_center = np.mean(
        np.concatenate([skull_mesh.v0, skull_mesh.v1, skull_mesh.v2]), axis=0
    )
    skull_offset = np.array([0, 25, -65])
    skull_mesh.translate(-skull_center + skull_offset)  # Center at origin
    skull_mesh.vectors *= 1.4  # Scale up by 40%

    # Also align and scale brain to match skull better
    brain_center = np.mean(
        np.concatenate([brain_mesh.v0, brain_mesh.v1, brain_mesh.v2]), axis=0
    )
    brain_offset = np.array([0, 0, -25])
    brain_mesh.translate(-brain_center + brain_offset)  # Center at origin
    brain_mesh.vectors *= 1.2  # Scale up slightly
    # brain_mesh.rotate([1, 0, 0], np.radians(180))  # Same orientation as skull

    print("Creating 3D visualization...")

    # Create plotly figure
    fig = go.Figure()

    # Add skull mesh
    skull_trace = mesh_to_plotly(skull_mesh, "lightgrey", 0.3, "Skull")
    fig.add_trace(skull_trace)

    # Add brain mesh
    brain_trace = mesh_to_plotly(brain_mesh, "tomato", 0.6, "Brain")
    fig.add_trace(brain_trace)

    # Get skull bounds for montage scaling
    skull_bounds = np.concatenate([skull_mesh.v0, skull_mesh.v1, skull_mesh.v2])

    # Plot multiple montages with different colors
    # montage_names = ["GSN-HydroCel-129"]  # Default single montage
    # Example of multiple montages:
    physionet_dict = physionet_64_montage().get_positions()["ch_pos"]
    physionet_montage = Montage(
        "physionet-64",
        list(physionet_dict.keys()),
        np.array(list(physionet_dict.values())),
    )
    insight_dict = {ch: STANDARD_1020[ch] for ch in INSIGHT5_CHANNELS}
    insight_montage = Montage(
        "insight-5", list(insight_dict.keys()), np.array(list(insight_dict.values()))
    )
    epoch_dict = {ch: STANDARD_1020[ch] for ch in EPOC14_CHANNELS}
    epoch_montage = Montage(
        "epoch-14", list(epoch_dict.keys()), np.array(list(epoch_dict.values()))
    )
    lemon_dict = {ch: STANDARD_1020[ch] for ch in LEMON_CHANNELS}
    lemon_montage = Montage(
        "lemon-61", list(lemon_dict.keys()), np.array(list(lemon_dict.values()))
    )

    # occipital = mask_regions(full_montage, [Region.OCCIPITAL])
    # parietal = mask_regions(full_montage, [Region.PARIETAL])
    # temporal = mask_regions(full_montage, [Region.TEMPORAL])
    # frontal = mask_regions(full_montage, [Region.FRONTAL])
    colors = [
        "blue",
        # "red",
    ]  # "green"]
    # montages = [occipital, parietal, temporal, frontal]
    montages = [
        physionet_montage,
        # lemon_montage,
    ]

    electrode_positions = plot_montages(
        fig, montages, colors, physionet_montage, skull_bounds=skull_bounds
    )

    # Configure layout for better performance and appearance - include electrode positions in bounds
    all_pts_list = [
        brain_mesh.v0,
        brain_mesh.v1,
        brain_mesh.v2,
        skull_mesh.v0,
        skull_mesh.v1,
        skull_mesh.v2,
    ]

    # Add all electrode positions to bounds calculation
    for pos_array in electrode_positions:
        all_pts_list.append(pos_array)

    all_pts = np.concatenate(all_pts_list)

    # Set equal aspect ratio and clean layout
    x_range = [all_pts[:, 0].min(), all_pts[:, 0].max()]
    y_range = [all_pts[:, 1].min(), all_pts[:, 1].max()]
    z_range = [all_pts[:, 2].min(), all_pts[:, 2].max()]
    fig.update_layout(
        title=dict(
            text="Brain and Skull with EEG Electrode Montages<br><sup>Real-time 3D interaction enabled</sup>",
            x=0.5,
            font=dict(size=16),
        ),
        scene=dict(
            xaxis=dict(range=x_range, showgrid=False, showticklabels=False, title=""),
            yaxis=dict(range=y_range, showgrid=False, showticklabels=False, title=""),
            zaxis=dict(range=z_range, showgrid=False, showticklabels=False, title=""),
            bgcolor="white",
            aspectmode="cube",
        ),
        width=900,
        height=700,
        margin=dict(l=0, r=0, t=50, b=0),
        showlegend=True,
        legend=dict(x=0, y=1),
    )

    print("3D visualization ready with real-time rotation!")
    print(
        "Controls: Click and drag to rotate, scroll to zoom, double-click to reset view"
    )
    fig.show()


if __name__ == "__main__":
    main()
