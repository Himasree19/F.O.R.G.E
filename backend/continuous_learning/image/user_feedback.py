import os
import pandas as pd
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "replay", "replay_dataset.csv")

def process_feedback(image_id: str, verified_label: str, user_feedback: str = "Agree"):
    """
    Locates the record in replay_dataset.csv matching image_id and updates:
    - verified_label ('Real' or 'Fake')
    - user_feedback ('Agree' or 'Disagree')
    - status ('Approved')
    - timestamp
    """
    if not os.path.exists(CSV_PATH):
        return {
            "success": False,
            "message": "replay_dataset.csv not found."
        }

    df = pd.read_csv(CSV_PATH)

    # Check if image ID exists in replay record
    if image_id not in df["image_id"].values:
        return {
            "success": False,
            "message": f"Image ID '{image_id}' not found in replay dataset."
        }

    idx = df[df["image_id"] == image_id].index[0]

    # Standardize label for Random Forest training (Real / Fake)
    val = str(verified_label).strip().lower()
    if val in ["human", "real", "authentic"]:
        norm_label = "Real"
    elif val in ["ai", "fake", "synthetic"]:
        norm_label = "Fake"
    else:
        norm_label = verified_label.capitalize()

    # Update row attributes
    df.loc[idx, "verified_label"] = norm_label
    df.loc[idx, "user_feedback"] = user_feedback
    df.loc[idx, "status"] = "Approved"  # Must be 'Approved' for dataset_builder.py
    df.loc[idx, "timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Save changes back to CSV
    df.to_csv(CSV_PATH, index=False)

    print(f"✓ Replay Record Updated for {image_id} -> Label: {norm_label} | Status: Approved")

    return {
        "success": True,
        "message": "Replay dataset updated successfully.",
        "image_id": image_id,
        "verified_label": norm_label,
        "status": "Approved"
    }