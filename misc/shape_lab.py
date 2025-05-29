import mne
import matplotlib.pyplot as plt

import requests, io, numpy as np
from stl import mesh
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt


# ---------- helper -----------------------------------------------------------
def fetch_stl(url: str) -> mesh.Mesh:
    raw = requests.get(url, timeout=30)
    raw.raise_for_status()
    return mesh.Mesh(np.frombuffer(raw.content, dtype=np.byte))


brain_url = (
    "https://raw.githubusercontent.com/garretthinton/BrainSPECT/" "master/brain.stl"
)
skull_url = (
    "https://raw.githubusercontent.com/XiaoxiaoLiu/3D-printing/"
    "master/final_skull_visible_female_full.stl"
)

brain_mesh = fetch_stl(brain_url)
skull_mesh = fetch_stl(skull_url)

# ---------- quick visual check ----------------------------------------------
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection="3d")

# skull first (opaque), then brain (semi-transparent)
ax.add_collection3d(
    mplot3d.art3d.Poly3DCollection(
        skull_mesh.vectors, facecolor="lightgrey", edgecolor="none", alpha=0.8
    )
)
ax.add_collection3d(
    mplot3d.art3d.Poly3DCollection(
        brain_mesh.vectors, facecolor="tomato", edgecolor="none", alpha=0.35
    )
)

# autoscale so both meshes fit
all_pts = np.concatenate(
    [
        brain_mesh.v0,
        brain_mesh.v1,
        brain_mesh.v2,
        skull_mesh.v0,
        skull_mesh.v1,
        skull_mesh.v2,
    ]
)
ax.auto_scale_xyz(all_pts[:, 0], all_pts[:, 1], all_pts[:, 2])
ax.set_axis_off()
plt.tight_layout()
plt.show()


def main():
    montage = mne.channels.make_standard_montage("GSN-HydroCel-129")
    breakpoint()
    fig = montage.plot(kind="3d", show=False)
    ax = fig.gca()
    ax.view_init(elev=15, azim=70)
    plt.show()


if __name__ == "__main__":
    main()
