import os
# =====================================================
# PATHS
# =====================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

APPROVED_REAL = os.path.join(
    BASE_DIR,
    "user_dataset",
    "approved",
    "real"
)

APPROVED_FAKE = os.path.join(
    BASE_DIR,
    "user_dataset",
    "approved",
    "fake"
)

RETRAIN_THRESHOLD = 50
# =====================================================
# COUNT APPROVED IMAGES
# =====================================================

def count_approved_images():

    real_count = len(os.listdir(APPROVED_REAL))

    fake_count = len(os.listdir(APPROVED_FAKE))

    total = real_count + fake_count

    return real_count, fake_count, total
# =====================================================
# RETRAINING STATUS
# =====================================================

def retraining_status():

    real_count, fake_count, total = count_approved_images()

    if total >= RETRAIN_THRESHOLD:

        return {

            "status": "Ready for Retraining",

            "approved_real": real_count,

            "approved_fake": fake_count,

            "total": total,

            "remaining": 0

        }

    return {

        "status": "Collect More Samples",

        "approved_real": real_count,

        "approved_fake": fake_count,

        "total": total,

        "remaining": RETRAIN_THRESHOLD - total

    }
if __name__ == "__main__":

    print(retraining_status())