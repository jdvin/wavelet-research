# pip install mne pandas
import mne
import pandas as pd
from pathlib import Path

# --- Emotiv channel sets (actives only) ---
EPOC14 = ["AF3","F7","F3","FC5","T7","P7","O1","O2","P8","T8","FC6","F4","F8","AF4"]
INSIGHT5 = ["AF3","AF4","T7","T8","Pz"]

# Choose a standard montage that includes AF3/AF4/T7/T8/Pz, etc.
# "standard_1020" works for these; you can also try "standard_1005" for denser sets.
montage = mne.channels.make_standard_montage("standard_1020")
pos = montage.get_positions()["ch_pos"]  # dict: label -> (x,y,z) in meters (head frame)

def to_df(labels, name):
    rows = []
    for ch in labels:
        if ch not in pos:
            raise KeyError(f"{ch} not found in montage. Try 'standard_1005' or check label spelling.")
        x, y, z = pos[ch]
        rows.append({"label": ch, "x": x, "y": y, "z": z})
    df = pd.DataFrame(rows)
    df = df[["label","x","y","z"]]
    df.to_csv(f"{name}_xyz_standard1020.csv", index=False)
    return df

epoc_df = to_df(EPOC14, "emotiv_epoc14")
insight_df = to_df(INSIGHT5, "emotiv_insight5")

print("Wrote:")
print(" - emotiv_epoc14_xyz_standard1020.csv")
print(" - emotiv_insight5_xyz_standard1020.csv")
