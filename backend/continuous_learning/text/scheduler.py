# ==========================================================
# scheduler.py
# Text Continuous Learning Scheduler
# ==========================================================

import os
import pandas as pd

# ----------------------------------------------------------
# Paths
# ----------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

FEEDBACK_DIR = os.path.join(BASE_DIR, "feedback")

REAL_CSV = os.path.join(FEEDBACK_DIR, "real.csv")
FAKE_CSV = os.path.join(FEEDBACK_DIR, "fake.csv")

# ----------------------------------------------------------
# Retraining Threshold
# ----------------------------------------------------------

RETRAIN_THRESHOLD = 500


# ----------------------------------------------------------
# Count Verified Samples
# ----------------------------------------------------------

def get_dataset_statistics():
    """
    Returns the number of verified samples
    available for continuous learning.
    """

    real_count = 0
    fake_count = 0

    if os.path.exists(REAL_CSV):
        real_count = len(pd.read_csv(REAL_CSV))

    if os.path.exists(FAKE_CSV):
        fake_count = len(pd.read_csv(FAKE_CSV))

    total = real_count + fake_count

    return {
        "real": real_count,
        "fake": fake_count,
        "total": total
    }


# ----------------------------------------------------------
# Check Retraining Requirement
# ----------------------------------------------------------

def should_retrain():

    stats = get_dataset_statistics()

    return stats["total"] >= RETRAIN_THRESHOLD


# ----------------------------------------------------------
# Test
# ----------------------------------------------------------

if __name__ == "__main__":

    stats = get_dataset_statistics()

    print("\n========== Replay Dataset ==========")
    print(f"Real Samples : {stats['real']}")
    print(f"Fake Samples : {stats['fake']}")
    print(f"Total Samples: {stats['total']}")
    print("====================================\n")

    if should_retrain():
        print("Retraining Required.")
    else:
        print("Retraining Not Required.")