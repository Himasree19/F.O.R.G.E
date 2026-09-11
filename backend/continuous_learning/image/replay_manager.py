import os
import csv
import pandas as pd
from datetime import datetime

# =====================================================
# PATHS & CONSTANTS
# =====================================================
MODEL_VERSION = "Image_RF_v2.0"
batch_id = datetime.now().strftime("BATCH_%Y%m%d")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

REPLAY_FOLDER = os.path.join(BASE_DIR, "replay")
CSV_PATH = os.path.join(REPLAY_FOLDER, "replay_dataset.csv")

# =====================================================
# CSV COLUMNS
# =====================================================
CSV_COLUMNS = [
    "image_id", "image_path", "cnn_score", "lighting", "shadow", 
    "boundary", "texture", "symmetry", "verified_label", 
    "high_frequency_noise", "gan_fingerprint", "checkerboard_artifact", 
    "prnu_score", "copy_move_score", "splicing_score", "jpeg_artifact_score", 
    "ela_score", "resampling_score", "rgb_inconsistency", "hsv_anomaly", 
    "ycbcr_chroma", "saturation_anomaly", "cfa_artifact", "eye_score", 
    "iris_score", "landmark_score", "skin_texture_score", "face_blending_score", 
    "object_relationship_score", "hand_anatomy_score", "eye_alignment_score", 
    "object_count_score", "shadow_consistency_score", "geometry_score", 
    "depth_of_field_score", "anatomy_score", "perspective_score", "text_error_score", 
    "anomaly_mean", "anomaly_max", "suspicious_patches", "quality_score", 
    "noise_score", "sharpness_score", "compression_score", "authenticity_score", 
    "predicted_label", "prediction_confidence", "user_feedback", "status", 
    "timestamp", "model_version", "batch_id"
]

FEATURE_COLUMNS = [col for col in CSV_COLUMNS if col not in [
    "image_id", "image_path", "verified_label", "predicted_label",
    "prediction_confidence", "user_feedback", "status", "timestamp",
    "model_version", "batch_id"
]]

KEY_ALIAS_MAP = {
    "cnn": "cnn_score", "ela": "ela_score", "prnu": "prnu_score",
    "splicing": "splicing_score", "copy_move": "copy_move_score",
    "jpeg_artifact": "jpeg_artifact_score", "resampling": "resampling_score",
    "high_freq_noise": "high_frequency_noise", "gan": "gan_fingerprint",
    "checkerboard": "checkerboard_artifact", "hand_anatomy": "hand_anatomy_score",
    "skin_texture": "skin_texture_score", "face_blending": "face_blending_score",
    "eye": "eye_score", "iris": "iris_score"
}

def create_replay_csv():
    os.makedirs(REPLAY_FOLDER, exist_ok=True)
    if not os.path.exists(CSV_PATH):
        with open(CSV_PATH, "w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(CSV_COLUMNS)

def generate_image_id():
    create_replay_csv()
    with open(CSV_PATH, "r", encoding="utf-8") as file:
        rows = list(csv.reader(file))
    next_id = len(rows)
    return f"FORGE_IMG_{next_id:06d}"

def extract_features_recursively(data):
    extracted = {}
    def _search(obj):
        if isinstance(obj, dict):
            for k, v in obj.items():
                col_key = str(k).lower().strip()
                resolved_key = KEY_ALIAS_MAP.get(col_key, col_key)

                for feat in FEATURE_COLUMNS:
                    if resolved_key == feat or resolved_key == feat.replace("_score", ""):
                        if isinstance(v, (int, float)):
                            extracted[feat] = float(v)
                        elif isinstance(v, str):
                            try:
                                extracted[feat] = float(v)
                            except ValueError:
                                pass
                        elif isinstance(v, dict):
                            val = v.get("score") if v.get("score") is not None else v.get("value")
                            if val is not None:
                                try:
                                    extracted[feat] = float(val)
                                except (ValueError, TypeError):
                                    pass
                _search(v)
        elif isinstance(obj, list):
            for item in obj:
                _search(item)

    _search(data)
    return extracted

def _clean_filename(path_str):
    return str(path_str).replace('\\', '/').split('/')[-1]

def append_replay_record(result):
    create_replay_csv()
    
    current_path = result.get("uploaded_file", result.get("image_path", ""))
    target_filename = _clean_filename(current_path)

    df = pd.read_csv(CSV_PATH) if os.path.exists(CSV_PATH) and os.path.getsize(CSV_PATH) > 0 else pd.DataFrame(columns=CSV_COLUMNS)

    existing_idx = None
    if not df.empty and "image_path" in df.columns:
        for idx, row in df.iterrows():
            if _clean_filename(row["image_path"]) == target_filename:
                existing_idx = idx
                break

    features = extract_features_recursively(result)

    if existing_idx is not None:
        image_id = df.loc[existing_idx, "image_id"]
        for feat in FEATURE_COLUMNS:
            if feat in features:
                df.loc[existing_idx, feat] = features[feat]
        
        if result.get("prediction"):
            df.loc[existing_idx, "predicted_label"] = result.get("prediction")
        if result.get("confidence"):
            df.loc[existing_idx, "prediction_confidence"] = float(result.get("confidence"))

        df.to_csv(CSV_PATH, index=False)
        print(f"✓ Replay Record Updated for {image_id} (No duplicate ID created)")
        return image_id
    else:
        image_id = generate_image_id()
        row = {column: 0.0 for column in CSV_COLUMNS}
        row["image_id"] = image_id
        row["image_path"] = current_path

        for feature_name in FEATURE_COLUMNS:
            if feature_name in features:
                row[feature_name] = features[feature_name]

        row["predicted_label"] = result.get("prediction", "UNKNOWN")
        try:
            row["prediction_confidence"] = float(result.get("confidence", 0.0))
        except (ValueError, TypeError):
            row["prediction_confidence"] = 0.0

        row["verified_label"] = ""
        row["user_feedback"] = "No Response"
        row["status"] = "Pending"
        row["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        row["model_version"] = MODEL_VERSION
        row["batch_id"] = batch_id

        with open(CSV_PATH, "a", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=CSV_COLUMNS)
            writer.writerow(row)

        print(f"✓ New Replay Record Logged: {image_id}")
        return image_id