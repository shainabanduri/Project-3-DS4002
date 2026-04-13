import os
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import random
import numpy as np

# =========================
# PATH SETUP (UPDATED)
# =========================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(
    BASE_DIR,
    "data",
    "datasetproject3",
    "C-NMC 2019 (PKG)",
    "C-NMC_training_data"
)

OUTPUT_DIR = os.path.join(BASE_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================
# LOAD DATA
# =========================
data = []

for fold in os.listdir(DATA_DIR):
    fold_path = os.path.join(DATA_DIR, fold)

    if os.path.isdir(fold_path):
        for label in ["all", "hem"]:
            label_path = os.path.join(fold_path, label)

            if os.path.exists(label_path):
                for img in os.listdir(label_path):
                    data.append({
                        "fold": fold,
                        "label": "cancer" if label == "all" else "normal",
                        "image": img
                    })

df = pd.DataFrame(data)

print("\nDataset Preview:")
print(df.head())

# =========================
# Q1: CLASS DISTRIBUTION
# =========================
class_counts = df["label"].value_counts()

plt.figure()
class_counts.plot(kind="bar")
plt.title("Class Distribution (Cancer vs Normal)")
plt.ylabel("Number of Images")
plt.xlabel("Class")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "q1_class_distribution.png"))
plt.close()

# =========================
# Q2: IMAGES PER FOLD
# =========================
fold_counts = df.groupby(["fold", "label"]).size().unstack()

plt.figure()
fold_counts.plot(kind="bar", stacked=True)
plt.title("Image Distribution Across Folds")
plt.ylabel("Number of Images")
plt.xlabel("Fold")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "q2_fold_distribution.png"))
plt.close()

# =========================
# Q3: AVERAGE IMAGE BRIGHTNESS
# =========================
import numpy as np

brightness = []

# sample subset for speed
sample_df = df.sample(300, random_state=42)

for _, row in sample_df.iterrows():
    img_path = os.path.join(
        DATA_DIR,
        row["fold"],
        "all" if row["label"] == "cancer" else "hem",
        row["image"]
    )

    img = Image.open(img_path).convert("L")  # grayscale
    img_array = np.array(img)

    avg_brightness = img_array.mean()

    brightness.append({
        "label": row["label"],
        "brightness": avg_brightness
    })

brightness_df = pd.DataFrame(brightness)

plt.figure()

for label in ["cancer", "normal"]:
    subset = brightness_df[brightness_df["label"] == label]
    plt.hist(subset["brightness"], bins=30, alpha=0.5, label=label)

plt.title("Average Image Brightness by Class")
plt.xlabel("Brightness")
plt.ylabel("Frequency")
plt.legend()

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "q3_brightness.png"))
plt.close()