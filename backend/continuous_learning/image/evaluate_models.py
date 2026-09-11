import os
import joblib
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
# =====================================================
# PATHS
# =====================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

TRAINING_DATASET = os.path.join(
    BASE_DIR,
    "training",
    "training_dataset.csv"
)

CURRENT_MODEL = os.path.join(
    BASE_DIR,
    "..",
    "..",
    "models",
    "final2_randomf_fusion_v3.pkl"
)

NEW_MODEL = os.path.join(
    BASE_DIR,
    "models",
    "RF_v2.pkl"
)
# =====================================================
# LOAD DATASET
# =====================================================

def load_dataset():

    df = pd.read_csv(TRAINING_DATASET)

    X = df.drop(
        columns=["verified_label"]
    )

    y = df["verified_label"]

    return X, y
# =====================================================
# LOAD MODELS
# =====================================================

def load_models():

    current_model = joblib.load(CURRENT_MODEL)

    new_model = joblib.load(NEW_MODEL)

    return current_model, new_model
if __name__ == "__main__":

    X, y = load_dataset()

    current_model, new_model = load_models()

    print("Dataset Loaded :", len(X))

    print("Models Loaded Successfully")
    