import os
import csv
import shutil

# ==========================================
# FEEDBACK PATHS
# ==========================================

FEEDBACK_DIR = "feedback"

FAKE_DIR = os.path.join(
    FEEDBACK_DIR,
    "corrected_fake"
)

REAL_DIR = os.path.join(
    FEEDBACK_DIR,
    "corrected_real"
)

CSV_FILE = os.path.join(
    FEEDBACK_DIR,
    "feedback.csv"
)

# ==========================================
# CREATE DIRECTORIES
# ==========================================

os.makedirs(
    FAKE_DIR,
    exist_ok=True
)

os.makedirs(
    REAL_DIR,
    exist_ok=True
)

# ==========================================
# INIT CSV
# ==========================================

if not os.path.exists(CSV_FILE):

    with open(
        CSV_FILE,
        "w",
        newline=""
    ) as file:

        writer = csv.writer(file)

        writer.writerow([

            "filename",

            "predicted",

            "corrected"
        ])

# ==========================================
# SAVE FEEDBACK
# ==========================================

def save_feedback(

    image_path,

    predicted,

    corrected
):

    try:

        filename = os.path.basename(
            image_path
        )

        # ==============================
        # MOVE IMAGE
        # ==============================

        if corrected.upper() == "AI":

            destination = os.path.join(

                FAKE_DIR,

                filename
            )

        else:

            destination = os.path.join(

                REAL_DIR,

                filename
            )

        shutil.copy(

            image_path,

            destination
        )

        # ==============================
        # LOG FEEDBACK
        # ==============================

        with open(

            CSV_FILE,

            "a",

            newline=""
        ) as file:

            writer = csv.writer(file)

            writer.writerow([

                filename,

                predicted,

                corrected
            ])

        return {

            "status":
                "success",

            "message":
                "Feedback stored"
        }

    except Exception as e:

        return {

            "status":
                "error",

            "message":
                str(e)
        }