# ==========================================================
# feedback_manager.py
# Text Continuous Learning - Feedback Manager
# ==========================================================

import os
import csv
import uuid
from datetime import datetime

# ----------------------------------------------------------
# Paths (Absolute Path Resolution)
# ----------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

FEEDBACK_DIR = os.path.join(BASE_DIR, "feedback")
os.makedirs(FEEDBACK_DIR, exist_ok=True)

FEEDBACK_CSV = os.path.join(FEEDBACK_DIR, "feedback.csv")
REAL_CSV = os.path.join(FEEDBACK_DIR, "real.csv")
FAKE_CSV = os.path.join(FEEDBACK_DIR, "fake.csv")

# ----------------------------------------------------------
# CSV Header
# ----------------------------------------------------------

CSV_HEADER = [
    "sentence_id",
    "sentence",
    "predicted_label",
    "confidence",
    "verified_label",
    "status",
    "timestamp"
]

# ----------------------------------------------------------
# Create CSV Files
# ----------------------------------------------------------

def initialize_feedback_files():
    """
    Creates feedback.csv, real.csv and fake.csv
    if they do not already exist.
    """
    for file_path in [FEEDBACK_CSV, REAL_CSV, FAKE_CSV]:
        if not os.path.exists(file_path):
            with open(file_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
                writer.writerow(CSV_HEADER)

# ----------------------------------------------------------
# Label Normalization Helper
# ----------------------------------------------------------

def _normalize_label(label: str) -> str:
    """Normalizes variations like 'HUMAN'/'Real' or 'AI'/'Fake'."""
    val = str(label or "").strip().lower()
    if val in ["human", "real", "authentic"]:
        return "Real"
    if val in ["ai", "fake", "synthetic", "generated"]:
        return "Fake"
    return label.capitalize() if label else "Real"

# ----------------------------------------------------------
# Save Feedback
# ----------------------------------------------------------

def save_feedback(
    sentence,
    predicted_label,
    confidence,
    verified_label,
    sentence_id=None
):
    """
    Saves verified feedback.

    Every entry goes into:
        feedback.csv

    If verified_label == Real / HUMAN:
        also save into real.csv

    If verified_label == Fake / AI:
        also save into fake.csv
    """
    initialize_feedback_files()

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Clean sentence text: collapse newlines/extra spaces so CSV format stays clean
    raw_sentence = str(sentence or "").strip()
    clean_sentence = " ".join(raw_sentence.split()) if raw_sentence else "Direct Text Entry"

    norm_predicted = _normalize_label(predicted_label)
    norm_verified = _normalize_label(verified_label)

    # Format confidence as float or percentage string
    try:
        conf_float = float(confidence)
        conf_str = f"{conf_float:.2f}"
    except (ValueError, TypeError):
        conf_str = "0.00"

    # Generate a unique sentence_id if one wasn't provided
    if not sentence_id:
        sentence_id = f"txt_{uuid.uuid4().hex[:8]}"

    # Determine status based on agreement between predicted and verified
    status = (
        "Verified"
        if norm_predicted.lower() == norm_verified.lower()
        else "Corrected"
    )

    # Exact row layout matching CSV_HEADER
    row = [
        sentence_id,
        clean_sentence,
        norm_predicted,
        conf_str,
        norm_verified,
        status,
        timestamp
    ]

    # Helper function to write row with safe quoting
    def _append_row(file_path):
        with open(file_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
            writer.writerow(row)

    # 1. Always save into overall feedback.csv
    _append_row(FEEDBACK_CSV)

    # 2. Save into replay buffer according to verified label
    if norm_verified.lower() == "real":
        _append_row(REAL_CSV)
    elif norm_verified.lower() == "fake":
        _append_row(FAKE_CSV)

    return {
        "success": True,
        "sentence_id": sentence_id,
        "status": status,
        "message": "Feedback saved successfully."
    }


if __name__ == "__main__":
    initialize_feedback_files()
    print("Feedback CSV files initialized successfully.")