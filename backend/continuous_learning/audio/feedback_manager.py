# ==========================================================
# feedback_manager.py
# Audio Continuous Learning - Feedback Manager
# ==========================================================

import os
import csv
import shutil
from datetime import datetime

# ----------------------------------------------------------
# Paths & Directories
# ----------------------------------------------------------

# Directory where feedback_manager.py lives
# Path: ...\F.O.R.G.E-main\backend\continuous_learning\audio
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

FEEDBACK_DIR = os.path.join(BASE_DIR, "feedback")
FEEDBACK_CSV = os.path.join(FEEDBACK_DIR, "feedback.csv")

# Target directories: ...\backend\continuous_learning\audio\all_audio\correct and \incorrect
ALL_AUDIO_DIR = os.path.join(BASE_DIR, "all_audio")
CORRECT_AUDIO_DIR = os.path.join(ALL_AUDIO_DIR, "correct")
INCORRECT_AUDIO_DIR = os.path.join(ALL_AUDIO_DIR, "incorrect")

# Explicitly point to uploaded source directory 3 levels up:
# Path: ...\F.O.R.G.E-main\uploads\audio
UPLOAD_SOURCE_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "..", "..", "uploads", "audio"))

# ----------------------------------------------------------
# Initialize Directories & CSV Header
# ----------------------------------------------------------

os.makedirs(FEEDBACK_DIR, exist_ok=True)
os.makedirs(CORRECT_AUDIO_DIR, exist_ok=True)
os.makedirs(INCORRECT_AUDIO_DIR, exist_ok=True)

CSV_HEADER = [
    "audio_id",
    "audio_name",
    "predicted_label",
    "verified_label",
    "status",
    "confidence",
    "used_for_training",
    "model_version",
    "timestamp"
]

def initialize_feedback_file():
    """Ensures the feedback CSV exists with headers."""
    if not os.path.exists(FEEDBACK_CSV):
        with open(FEEDBACK_CSV, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(CSV_HEADER)

initialize_feedback_file()

# ----------------------------------------------------------
# Helper Functions
# ----------------------------------------------------------

def _normalize_label(label: str) -> str:
    """Maps frontend labels ('HUMAN'/'REAL' or 'AI'/'FAKE') into standard strings."""
    val = str(label or "").strip().lower()
    if val in ["human", "real", "authentic"]:
        return "Real"
    if val in ["ai", "fake", "synthetic", "generated"]:
        return "Fake"
    return label.capitalize() if label else "Real"


def _resolve_local_audio_path(audio_path: str) -> str:
    """
    Resolves relative web URLs like '/uploads/audio/sample.wav'
    into the absolute path where uploads are saved on disk.
    """
    if not audio_path:
        return ""

    # Return if it is already a valid absolute path on disk
    if os.path.exists(audio_path):
        return os.path.abspath(audio_path)

    # Extract ONLY the pure filename (e.g. 'bc72eefedd20470db0043cfb213522d3_new_audio_(4505).wav')
    filename = os.path.basename(audio_path)

    # 1. Direct check in target uploads source folder: ...\F.O.R.G.E-main\uploads\audio\filename
    target_path = os.path.join(UPLOAD_SOURCE_DIR, filename)
    if os.path.exists(target_path):
        return target_path

    # 2. Candidate fallbacks in case directory structure varies
    candidate_paths = [
        os.path.abspath(os.path.join(BASE_DIR, "..", "..", "uploads", "audio", filename)),
        os.path.abspath(os.path.join(BASE_DIR, "..", "uploads", "audio", filename)),
        os.path.abspath(os.path.join(BASE_DIR, "uploads", "audio", filename)),
    ]

    for cand in candidate_paths:
        if os.path.exists(cand):
            return cand

    return audio_path

# ----------------------------------------------------------
# Save Feedback & Copy Audio
# ----------------------------------------------------------

def save_feedback(
    audio_path,
    audio_name,
    predicted_label,
    verified_label,
    confidence,
    model_version="v1"
):
    """
    Saves audio verification feedback into feedback.csv and 
    copies audio files into all_audio/correct/ or all_audio/incorrect/.
    """
    resolved_path = _resolve_local_audio_path(audio_path)
    file_exists = os.path.exists(resolved_path)

    print(f"\n--- AUDIO FEEDBACK PROCESS ---")
    print("Received path:", audio_path)
    print("Resolved path:", resolved_path)
    print("File exists:", file_exists)

    # Normalize labels (e.g. HUMAN -> Real, AI -> Fake)
    norm_predicted = _normalize_label(predicted_label)
    norm_verified = _normalize_label(verified_label)

    initialize_feedback_file()

    # Generate Audio ID based on row count
    with open(FEEDBACK_CSV, "r", newline="", encoding="utf-8") as f:
        reader = list(csv.reader(f))
        total_rows = len(reader)

    audio_id = f"A{total_rows:03d}"

    # Determine status and target directory in all_audio
    if norm_predicted.lower() == norm_verified.lower():
        status = "Correct"
        destination_dir = CORRECT_AUDIO_DIR
    else:
        status = "Incorrect"
        destination_dir = INCORRECT_AUDIO_DIR

    # File extension resolution
    extension = os.path.splitext(resolved_path)[1] or ".wav"
    saved_audio_name = f"{audio_id}{extension}"

    destination_path = os.path.join(destination_dir, saved_audio_name)

    # Copy file to all_audio/correct or all_audio/incorrect
    if file_exists:
        try:
            shutil.copy2(resolved_path, destination_path)
            print(f"✅ SUCCESS: Audio file copied to {destination_path}")
        except Exception as err:
            print(f"❌ ERROR: Failed to copy audio file: {err}")
    else:
        print(f"⚠️ Warning: Binary file missing at {resolved_path}. CSV record created.")

    # Current timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Format confidence float
    try:
        conf_val = f"{float(confidence):.2f}"
    except (ValueError, TypeError):
        conf_val = "0.00"

    # Append record to feedback.csv
    row = [
        audio_id,
        saved_audio_name,
        norm_predicted,
        norm_verified,
        status,
        conf_val,
        "No",
        model_version,
        timestamp
    ]

    with open(FEEDBACK_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(row)

    print("-----------------------------------\n")

    return {
        "success": True,
        "audio_id": audio_id,
        "audio_name": saved_audio_name,
        "status": status,
        "destination_path": destination_path
    }


if __name__ == "__main__":
    print("Audio Feedback Manager ready.")