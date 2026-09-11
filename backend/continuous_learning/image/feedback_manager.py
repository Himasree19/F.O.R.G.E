# ==========================================================
# feedback_manager.py
# Image Continuous Learning - Feedback Manager
# ==========================================================

import os
import csv
import shutil
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ----------------------------------------------------------
# Paths & Directories
# ----------------------------------------------------------

FEEDBACK_DIR = os.path.join(BASE_DIR, "feedback")
FEEDBACK_CSV = os.path.join(FEEDBACK_DIR, "feedback.csv")

APPROVED_CSV = os.path.join(FEEDBACK_DIR, "approved.csv")
PENDING_CSV = os.path.join(FEEDBACK_DIR, "pending.csv")

USER_DATASET_DIR = os.path.join(BASE_DIR, "user_dataset")
APPROVED_IMG_DIR = os.path.join(USER_DATASET_DIR, "approved")
PENDING_IMG_DIR = os.path.join(USER_DATASET_DIR, "pending")

for folder in [FEEDBACK_DIR, USER_DATASET_DIR, APPROVED_IMG_DIR, PENDING_IMG_DIR]:
    os.makedirs(folder, exist_ok=True)

CSV_HEADER = [
    "image_id",
    "saved_image_name",
    "predicted_label",
    "verified_label",
    "status",
    "confidence",
    "source_path",
    "timestamp"
]

def initialize_csv_files():
    """Ensures all CSV tracking files exist with headers."""
    for csv_file in [FEEDBACK_CSV, APPROVED_CSV, PENDING_CSV]:
        if not os.path.exists(csv_file):
            with open(csv_file, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(CSV_HEADER)

initialize_csv_files()

def _normalize_label(label: str) -> str:
    val = str(label or "").strip().lower()
    if val in ["human", "real", "authentic"]:
        return "Real"
    if val in ["ai", "fake", "synthetic", "generated"]:
        return "Fake"
    return label.capitalize() if label else "Real"

def _resolve_image_path(source_path: str) -> str:
    """Resolves web paths like /uploads/image.jpg to local file paths."""

    if not source_path:
        return ""

    if os.path.exists(source_path):
        print("✓ Direct path exists:", os.path.abspath(source_path))
        return os.path.abspath(source_path)

    clean_name = source_path.lstrip("/").replace("uploads/", "")
    project_root = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))

    candidate_paths = [
        os.path.join(project_root, "uploads", clean_name),
        os.path.join(project_root, "backend", "uploads", clean_name),
    ]

    print("\n========== PATH DEBUG ==========")
    print("Source Path :", source_path)
    print("Clean Name  :", clean_name)
    print("Project Root:", project_root)

    for cand in candidate_paths:
        print("Checking :", cand)
        print("Exists   :", os.path.exists(cand))
        if os.path.exists(cand):
            print("✓ Using :", cand)
            print("===============================\n")
            return cand

    print("❌ No valid file found.")
    print("===============================\n")
    return source_path

def process_image_feedback(data: dict):
    """
    Saves image feedback automatically to CSV and copies the image binary 
    into user_dataset/approved/ or user_dataset/pending/.
    """
    initialize_csv_files()

    image_id = data.get("image_id") or f"IMG_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    raw_source = data.get("source_path") or data.get("image_path") or ""
    resolved_path = _resolve_image_path(raw_source)
    print("\nIMAGE FEEDBACK")

    print("Raw Source :", raw_source)

    print("Resolved :", resolved_path)

    print("Exists :", os.path.exists(resolved_path))

    print()

    norm_predicted = _normalize_label(data.get("predicted_label", "AI"))
    norm_verified = _normalize_label(data.get("verified_label", "HUMAN"))
    
    try:
        confidence = f"{float(data.get('confidence', 0.0)):.2f}"
    except (ValueError, TypeError):
        confidence = "0.00"

    # Determine status & target image folder
    user_feedback = str(data.get("user_feedback", "")).lower()
    
    if user_feedback == "agree" or norm_predicted.lower() == norm_verified.lower():
        status = "Approved"
        target_dir = APPROVED_IMG_DIR
        target_csv = APPROVED_CSV
    else:
        status = "Pending"
        target_dir = PENDING_IMG_DIR
        target_csv = PENDING_CSV

    # Format destination image filename
    ext = os.path.splitext(resolved_path)[1] or ".jpg"
    saved_img_name = f"{image_id}{ext}"
    destination_path = os.path.join(target_dir, saved_img_name)

    if os.path.exists(resolved_path):

        shutil.copy2(
            resolved_path,
            destination_path
        )

        print("Image copied successfully.")

    else:

        print("IMAGE NOT FOUND")

    print(resolved_path)

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    row = [
        image_id,
        saved_img_name,
        norm_predicted,
        norm_verified,
        status,
        confidence,
        resolved_path,
        timestamp
    ]

    # Save to main feedback.csv
 
    print("\n========== CSV DEBUG ==========")
    print("Feedback CSV :", FEEDBACK_CSV)

    with open(FEEDBACK_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(row)

    print("Main CSV Updated")
    print("Feedback CSV Path:", os.path.abspath(FEEDBACK_CSV))

    # Save to category CSV (approved.csv or pending.csv)
    print("Target CSV :", target_csv)

    with open(target_csv, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(row)

    print("Category CSV Updated")
    print("Category CSV Path:", os.path.abspath(target_csv))
    print("Image Exists:", os.path.exists(destination_path))

    return {
        "success": True,
        "image_id": image_id,
        "status": status,
        "destination_path": destination_path,
        "message": f"Image saved to user_dataset/{status.lower()} and CSV updated."
    }
