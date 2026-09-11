import os
import pandas as pd
# =====================================================
# PATHS
# =====================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

REPLAY_DATASET = os.path.join(
    BASE_DIR,
    "replay",
    "replay_dataset.csv"
)

TRAINING_FOLDER = os.path.join(
    BASE_DIR,
    "training"
)

TRAINING_DATASET = os.path.join(
    TRAINING_FOLDER,
    "training_dataset.csv"
)
# =====================================================
# FEATURE COLUMNS
# =====================================================

FEATURE_COLUMNS = [

    "cnn_score",

    "lighting",
    "shadow",
    "boundary",
    "texture",
    "symmetry",

    "high_frequency_noise",
    "gan_fingerprint",
    "checkerboard_artifact",
    "prnu_score",
    "copy_move_score",
    "splicing_score",
    "jpeg_artifact_score",
    "ela_score",
    "resampling_score",
    "rgb_inconsistency",
    "hsv_anomaly",
    "ycbcr_chroma",
    "saturation_anomaly",
    "cfa_artifact",

    "eye_score",
    "iris_score",
    "landmark_score",
    "skin_texture_score",
    "face_blending_score",

    "object_relationship_score",
    "hand_anatomy_score",
    "eye_alignment_score",
    "object_count_score",
    "shadow_consistency_score",
    "geometry_score",
    "depth_of_field_score",
    "anatomy_score",
    "perspective_score",
    "text_error_score",

    "anomaly_mean",
    "anomaly_max",
    "suspicious_patches",

    "quality_score",
    "noise_score",
    "sharpness_score",
    "compression_score",
    "authenticity_score"
]
# =====================================================
# BUILD TRAINING DATASET
# =====================================================
def build_training_dataset():
    create_training_folder()

    df = pd.read_csv(REPLAY_DATASET)

    # Keep only approved samples
    approved_df = df[df["status"] == "Approved"].copy()

    if approved_df.empty:
        print("[dataset_builder] No Approved samples found to build training dataset.")
        return

    # Keep only features + label
    training_df = approved_df[FEATURE_COLUMNS + ["verified_label"]].copy()

    # Fill any missing NaN values with 0.0 so scikit-learn can fit cleanly
    training_df[FEATURE_COLUMNS] = training_df[FEATURE_COLUMNS].fillna(0.0)
    training_df["verified_label"] = training_df["verified_label"].fillna("Real")

    training_df.to_csv(
        TRAINING_DATASET,
        index=False
    )

    print("=" * 50)
    print("Training Dataset Created Successfully")
    print(f"Approved Samples : {len(training_df)}")
    print(f"Saved To : {TRAINING_DATASET}")
    print("=" * 50)
# =====================================================
# CREATE TRAINING FOLDER
# =====================================================

def create_training_folder():

    os.makedirs(
        TRAINING_FOLDER,
        exist_ok=True
    )

    print("Training folder ready.")

if __name__ == "__main__":

    build_training_dataset()