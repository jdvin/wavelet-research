from copy import deepcopy
import tempfile
import os
from dataclasses import dataclass
from enum import Enum
import mne
import requests, io, numpy as np
from stl import mesh
import plotly.graph_objects as go
import plotly.io as pio


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


def get_region_mask(
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


def mask_regions(montage: Montage, regions: list[Region]) -> Montage:
    """Mask the montage to only include the specified regions"""
    mask = get_region_mask(montage.pos_array, regions)
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
    full_montage = get_montage("GSN-HydroCel-129")
    occipital = mask_regions(full_montage, [Region.OCCIPITAL])
    parietal = mask_regions(full_montage, [Region.PARIETAL])
    temporal = mask_regions(full_montage, [Region.TEMPORAL])
    frontal = mask_regions(full_montage, [Region.FRONTAL])
    colors = ["blue", "red", "green", "orange"]
    montages = [occipital, parietal, temporal, frontal]

    electrode_positions = plot_montages(
        fig, montages, colors, full_montage, skull_bounds=skull_bounds
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
