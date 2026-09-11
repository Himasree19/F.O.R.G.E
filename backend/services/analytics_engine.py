import os
import json

ANALYTICS_FILE = "analytics.json"

# ==========================================
# INITIALIZE
# ==========================================

def initialize_analytics():

    default_data = {
        "total_uploads": 0,

        "text_uploads": 0,
        "image_uploads": 0,
        "audio_uploads": 0,

        "fake_detected": 0,
        "real_detected": 0,

        "text_fake": 0,
        "text_real": 0,

        "image_fake": 0,
        "image_real": 0,

        "audio_fake": 0,
        "audio_real": 0,

        "avg_risk_score": 0,
        "total_risk_score": 0
    }

    if not os.path.exists(ANALYTICS_FILE):

        with open(ANALYTICS_FILE, "w") as f:
            json.dump(default_data, f, indent=4)

    else:

        with open(ANALYTICS_FILE, "r") as f:
            data = json.load(f)

        updated = False

        for key, value in default_data.items():

            if key not in data:
                data[key] = value
                updated = True

        if updated:

            with open(ANALYTICS_FILE, "w") as f:
                json.dump(data, f, indent=4)


# ==========================================
# UPDATE
# ==========================================

def update_analytics(file_type, prediction, risk_score):

    initialize_analytics()

    with open(ANALYTICS_FILE, "r") as f:
        data = json.load(f)

    file_type = str(file_type).lower()
    prediction = str(prediction).upper()

    try:
        risk_score = float(risk_score)
    except Exception:
        risk_score = 0

    data["total_uploads"] += 1

    if file_type == "text":
        data["text_uploads"] += 1

    elif file_type == "image":
        data["image_uploads"] += 1

    elif file_type == "audio":
        data["audio_uploads"] += 1

    is_fake = prediction in [
        "AI",
        "FAKE",
        "AI GENERATED",
        "SYNTHETIC"
    ]

    if is_fake:

        data["fake_detected"] += 1

        if file_type == "text":
            data["text_fake"] += 1

        elif file_type == "image":
            data["image_fake"] += 1

        elif file_type == "audio":
            data["audio_fake"] += 1

    else:

        data["real_detected"] += 1

        if file_type == "text":
            data["text_real"] += 1

        elif file_type == "image":
            data["image_real"] += 1

        elif file_type == "audio":
            data["audio_real"] += 1

    data["total_risk_score"] += risk_score

    if data["total_uploads"] > 0:

        data["avg_risk_score"] = round(
            data["total_risk_score"] / data["total_uploads"],
            2
        )

    with open(ANALYTICS_FILE, "w") as f:
        json.dump(data, f, indent=4)


# ==========================================
# READ
# ==========================================

def get_analytics():

    initialize_analytics()

    with open(ANALYTICS_FILE, "r") as f:
        return json.load(f)