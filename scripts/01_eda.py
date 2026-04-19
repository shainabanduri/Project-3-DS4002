"""
=============================================================
01_eda.py  –  Exploratory Data Analysis
=============================================================
DS 4002  |  Group: Model Citizens
Members : Shaina Banduri, Neil Parikh, Nishana Dahal

PURPOSE
-------
Answers the three EDA questions from the MI2 analysis plan:
  Q1. Is the dataset (training set) class-balanced?
  Q2. How are images distributed across the three folds?
  Q3. Do cancer and normal images differ in average brightness?

USAGE
-----
Run from the repo root (the "Cleaned Up" folder):
    python scripts/01_eda.py

OUTPUT
------
Saves three PNG plots to output/:
  q1_class_distribution.png
  q2_fold_distribution.png
  q3_brightness_by_class.png

REQUIREMENTS
------------
    pip install pandas matplotlib Pillow numpy
=============================================================
"""

import pathlib
import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

# ------------------------------------------------------------------
# Path setup  (relative to this script's location)
# ------------------------------------------------------------------
REPO_ROOT  = pathlib.Path(__file__).resolve().parent.parent   # "Cleaned Up/"
DATA_DIR   = REPO_ROOT / "data" / "C-NMC 2019 (PKG)" / "C-NMC_training_data"
OUTPUT_DIR = REPO_ROOT / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

FOLDS   = ["fold_0", "fold_1", "fold_2"]
CLASSES = {"all": "cancer", "hem": "normal"}   # subfolder name → display label

if not DATA_DIR.exists():
    raise FileNotFoundError(
        f"Training data not found at:\n  {DATA_DIR}\n"
        "Make sure the dataset is at data/C-NMC 2019 (PKG)/C-NMC_training_data/"
    )

# ------------------------------------------------------------------
# Build a metadata dataframe from the training folds (no pixels yet)
# ------------------------------------------------------------------
records = []
for fold in FOLDS:
    for cls_folder, label in CLASSES.items():
        cls_path = DATA_DIR / fold / cls_folder
        for img_file in sorted(cls_path.glob("*.bmp")):
            records.append({
                "fold":  fold,
                "label": label,
                "image": img_file.name,
                "path":  str(img_file),
            })

df = pd.DataFrame(records)
print(f"Total training images: {len(df):,}")
print(df.groupby(["fold", "label"]).size().unstack(fill_value=0).to_string())
print()

# ------------------------------------------------------------------
# Q1 – Class distribution (cancer vs. normal)
# ------------------------------------------------------------------
class_counts = df["label"].value_counts()

fig, ax = plt.subplots(figsize=(5, 4))
bars = ax.bar(
    class_counts.index, class_counts.values,
    color=["#E53935", "#1E88E5"], edgecolor="black", linewidth=0.7,
)
ax.set_title("Class Distribution (Cancer vs Normal)", fontsize=13, fontweight="bold")
ax.set_ylabel("Number of Images")
ax.set_xlabel("Class")
for bar, v in zip(bars, class_counts.values):
    ax.text(
        bar.get_x() + bar.get_width() / 2, v + 50,
        f"{v:,}", ha="center", va="bottom", fontsize=10,
    )
plt.tight_layout()
fig.savefig(OUTPUT_DIR / "q1_class_distribution.png", dpi=150)
plt.close()
print("Saved: q1_class_distribution.png")

# ------------------------------------------------------------------
# Q2 – Image distribution across folds (stacked bar)
# ------------------------------------------------------------------
fold_counts = df.groupby(["fold", "label"]).size().unstack(fill_value=0)

fig, ax = plt.subplots(figsize=(6, 4))
fold_counts.plot(
    kind="bar", stacked=True, ax=ax,
    color=["#E53935", "#1E88E5"], edgecolor="black", linewidth=0.5,
)
ax.set_title("Image Distribution Across Folds", fontsize=13, fontweight="bold")
ax.set_ylabel("Number of Images")
ax.set_xlabel("Fold")
ax.tick_params(axis="x", rotation=0)
ax.legend(title="Label", loc="upper right")
plt.tight_layout()
fig.savefig(OUTPUT_DIR / "q2_fold_distribution.png", dpi=150)
plt.close()
print("Saved: q2_fold_distribution.png")

# ------------------------------------------------------------------
# Q3 – Average image brightness by class
#       Sample 150 images per class for speed (total 300)
# ------------------------------------------------------------------
cancer_sample = df[df["label"] == "cancer"].sample(150, random_state=42)
normal_sample = df[df["label"] == "normal"].sample(150, random_state=42)
sample_df     = pd.concat([cancer_sample, normal_sample])

brightness_rows = []
for _, row in sample_df.iterrows():
    # Convert to greyscale and compute mean pixel intensity
    arr = np.array(Image.open(row["path"]).convert("L"))
    brightness_rows.append({"label": row["label"], "brightness": arr.mean()})

brightness_df = pd.DataFrame(brightness_rows)

fig, ax = plt.subplots(figsize=(6, 4))
colors = {"cancer": "#E53935", "normal": "#1E88E5"}
for label, grp in brightness_df.groupby("label"):
    ax.hist(grp["brightness"], bins=30, alpha=0.55, label=label, color=colors[label])
ax.set_title("Average Image Brightness by Class", fontsize=13, fontweight="bold")
ax.set_xlabel("Mean Pixel Brightness (greyscale, 0–255)")
ax.set_ylabel("Frequency")
ax.legend()
plt.tight_layout()
fig.savefig(OUTPUT_DIR / "q3_brightness_by_class.png", dpi=150)
plt.close()
print("Saved: q3_brightness_by_class.png")

print(f"\nEDA complete — all plots saved to {OUTPUT_DIR}/")
