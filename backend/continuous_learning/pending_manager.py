# continuous_learning/pending_manager.py
import os
import csv
import shutil
import uuid
from datetime import datetime

# ==========================================================
# PATH CONFIGURATION
# ==========================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODALITIES = ["image", "audio", "text"]

CSV_HEADERS = [
    "id", 
    "modality", 
    "identifier",  # filepath for image/audio, or raw text content
    "prediction", 
    "confidence", 
    "timestamp", 
    "status"
]

def _get_modality_dir(modality: str) -> str:
    return os.path.join(BASE_DIR, modality, "feedback")

def _get_pending_csv(modality: str) -> str:
    path = os.path.join(_get_modality_dir(modality), "pending_feedback.csv")
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(CSV_HEADERS)
    return path

# ==========================================================
# PENDING MANAGEMENT FUNCTIONS
# ==========================================================

def save_pending_item(modality: str, identifier: str, prediction: str, confidence: float, source_filepath: str = None) -> str:
    """Stores an item into pending_feedback.csv and copies media files to pending_uploads/."""
    item_id = str(uuid.uuid4())
    modality_dir = _get_modality_dir(modality)
    stored_identifier = identifier

    # Copy binary files for image/audio to pending storage
    if modality in ["image", "audio"] and source_filepath and os.path.exists(source_filepath):
        uploads_dir = os.path.join(modality_dir, "pending_uploads")
        os.makedirs(uploads_dir, exist_ok=True)
        ext = os.path.splitext(source_filepath)[1]
        dest_filename = f"{item_id}{ext}"
        dest_path = os.path.join(uploads_dir, dest_filename)
        shutil.copy2(source_filepath, dest_path)
        stored_identifier = dest_path

    csv_file = _get_pending_csv(modality)
    row = [item_id, modality, stored_identifier, prediction, confidence, datetime.utcnow().isoformat(), "pending"]

    with open(csv_file, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(row)

    return item_id

def get_all_pending_items() -> list:
    """Aggregates all pending verifications from Image, Audio, and Text CSVs."""
    all_items = []
    for mod in MODALITIES:
        csv_file = _get_pending_csv(mod)
        if not os.path.exists(csv_file):
            continue
        with open(csv_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("status") == "pending":
                    all_items.append(row)
    return all_items

def get_pending_count() -> int:
    """Returns the total pending count across all 3 modalities."""
    return len(get_all_pending_items())

def resolve_pending_item(item_id: str, final_label: str = None) -> dict:
    """Removes the record from pending_feedback.csv, cleans up staged files, and returns record details."""
    target_record = None

    for mod in MODALITIES:
        csv_file = _get_pending_csv(mod)
        if not os.path.exists(csv_file):
            continue

        rows = []
        found = False
        with open(csv_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("id") == item_id:
                    target_record = row
                    found = True
                else:
                    rows.append(row)

        if found:
            # Overwrite CSV without the resolved row
            with open(csv_file, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=CSV_HEADERS)
                writer.writeheader()
                writer.writerows(rows)
            break

    return target_record

if __name__ == "__main__":
    print(f"Pending Manager initialized. Current pending items count: {get_pending_count()}")# continuous_learning/pending_manager.py
import os
import csv
import shutil
import uuid
from datetime import datetime

# ==========================================================
# PATH CONFIGURATION
# ==========================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODALITIES = ["image", "audio", "text"]

CSV_HEADERS = [
    "id", 
    "modality", 
    "identifier",  # filepath for image/audio, or raw text content
    "prediction", 
    "confidence", 
    "timestamp", 
    "status"
]

def _get_modality_dir(modality: str) -> str:
    return os.path.join(BASE_DIR, modality, "feedback")

def _get_pending_csv(modality: str) -> str:
    path = os.path.join(_get_modality_dir(modality), "pending_feedback.csv")
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(CSV_HEADERS)
    return path

# ==========================================================
# PENDING MANAGEMENT FUNCTIONS
# ==========================================================

def save_pending_item(modality: str, identifier: str, prediction: str, confidence: float, source_filepath: str = None) -> str:
    """Stores an item into pending_feedback.csv and copies media files to pending_uploads/."""
    item_id = str(uuid.uuid4())
    modality_dir = _get_modality_dir(modality)
    stored_identifier = identifier

    # Copy binary files for image/audio to pending storage
    if modality in ["image", "audio"] and source_filepath and os.path.exists(source_filepath):
        uploads_dir = os.path.join(modality_dir, "pending_uploads")
        os.makedirs(uploads_dir, exist_ok=True)
        ext = os.path.splitext(source_filepath)[1]
        dest_filename = f"{item_id}{ext}"
        dest_path = os.path.join(uploads_dir, dest_filename)
        shutil.copy2(source_filepath, dest_path)
        stored_identifier = dest_path

    csv_file = _get_pending_csv(modality)
    row = [item_id, modality, stored_identifier, prediction, confidence, datetime.utcnow().isoformat(), "pending"]

    with open(csv_file, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(row)

    return item_id

def get_all_pending_items() -> list:
    """Aggregates all pending verifications from Image, Audio, and Text CSVs."""
    all_items = []
    for mod in MODALITIES:
        csv_file = _get_pending_csv(mod)
        if not os.path.exists(csv_file):
            continue
        with open(csv_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("status") == "pending":
                    all_items.append(row)
    return all_items

def get_pending_count() -> int:
    """Returns the total pending count across all 3 modalities."""
    return len(get_all_pending_items())

def resolve_pending_item(item_id: str, final_label: str = None) -> dict:
    """Removes the record from pending_feedback.csv, cleans up staged files, and returns record details."""
    target_record = None

    for mod in MODALITIES:
        csv_file = _get_pending_csv(mod)
        if not os.path.exists(csv_file):
            continue

        rows = []
        found = False
        with open(csv_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("id") == item_id:
                    target_record = row
                    found = True
                else:
                    rows.append(row)

        if found:
            # Overwrite CSV without the resolved row
            with open(csv_file, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=CSV_HEADERS)
                writer.writeheader()
                writer.writerows(rows)
            break

    return target_record

if __name__ == "__main__":
    print(f"Pending Manager initialized. Current pending items count: {get_pending_count()}")