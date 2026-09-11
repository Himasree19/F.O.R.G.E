import os
import shutil
from datetime import datetime

# Root of the image continuous learning module
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

VERIFIED_CASES_DIR = os.path.join(BASE_DIR, "verified_cases")


def save_verified_case(image_path: str, label: str):
    """
    Save a verified image for future continuous learning.

    Parameters
    ----------
    image_path : str
        Path to the verified image.

    label : str
        'real' or 'fake'
    """

    label = label.lower()

    if label not in ["real", "fake"]:
        raise ValueError("Label must be 'real' or 'fake'.")

    destination_folder = os.path.join(VERIFIED_CASES_DIR, label)

    os.makedirs(destination_folder, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    filename = os.path.basename(image_path)

    new_filename = f"{timestamp}_{filename}"

    destination = os.path.join(destination_folder, new_filename)

    shutil.copy2(image_path, destination)

    print(f"[✓] Verified image stored at:\n{destination}")

    return destination