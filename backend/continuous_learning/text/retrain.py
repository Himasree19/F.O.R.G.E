# ==========================================================
# retrain.py
# Text Continuous Learning - Dataset Preparation
# ==========================================================

import os
import pandas as pd

# ----------------------------------------------------------
# Base Paths
# ----------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Original Training Dataset
ORIGINAL_DATASET = r"C:\Users\LENOVO\Desktop\final forgery\text_forgery\dataset_2"

REAL_DATASET = os.path.join(ORIGINAL_DATASET, "Real.csv")
FAKE_DATASET = os.path.join(ORIGINAL_DATASET, "Fake.csv")

# Continuous Learning Feedback
FEEDBACK_DIR = os.path.join(BASE_DIR, "feedback")

REAL_REPLAY = os.path.join(FEEDBACK_DIR, "real.csv")
FAKE_REPLAY = os.path.join(FEEDBACK_DIR, "fake.csv")

# Output
OUTPUT_DATASET = os.path.join(BASE_DIR, "merged_training_dataset.csv")


# ----------------------------------------------------------
# Load Original Dataset
# ----------------------------------------------------------

def load_original_dataset():

    # -------- Real --------
    real_df = pd.read_csv(REAL_DATASET)

    real_df = real_df.rename(columns={"sentence": "text"})
    real_df = real_df[["text"]]
    real_df["label"] = "Real"

    # -------- Fake --------
    fake_df = pd.read_csv(FAKE_DATASET)

    fake_df = fake_df[["text"]]
    fake_df["label"] = "Fake"

    return real_df, fake_df


# ----------------------------------------------------------
# Load Replay Dataset
# ----------------------------------------------------------

def load_replay_dataset():

    replay_real = pd.read_csv(REAL_REPLAY)
    replay_fake = pd.read_csv(FAKE_REPLAY)

    replay_real = replay_real.rename(columns={"sentence": "text"})
    replay_fake = replay_fake.rename(columns={"sentence": "text"})

    replay_real = replay_real[["text"]]
    replay_fake = replay_fake[["text"]]

    replay_real["label"] = "Real"
    replay_fake["label"] = "Fake"

    return replay_real, replay_fake


# ----------------------------------------------------------
# Merge Dataset
# ----------------------------------------------------------

def prepare_training_dataset():

    original_real, original_fake = load_original_dataset()

    replay_real, replay_fake = load_replay_dataset()

    merged = pd.concat(
        [
            original_real,
            original_fake,
            replay_real,
            replay_fake
        ],
        ignore_index=True
    )

    merged = merged.sample(frac=1, random_state=42).reset_index(drop=True)

    merged.to_csv(OUTPUT_DATASET, index=False)

    print("\nMerged Dataset Created Successfully")
    print("Saved To :", OUTPUT_DATASET)
    print("Total Samples :", len(merged))

    return merged


# ----------------------------------------------------------
# Test
# ----------------------------------------------------------

if __name__ == "__main__":

    prepare_training_dataset()