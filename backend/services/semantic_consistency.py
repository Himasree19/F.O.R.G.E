from PIL import Image


def analyze_semantics(image_path):

    try:

        img = Image.open(image_path)

        width, height = img.size

        ratio = width / height

        if ratio > 3 or ratio < 0.3:

            score = 75

            risk = "HIGH"

        elif ratio > 2 or ratio < 0.5:

            score = 55

            risk = "MEDIUM"

        else:

            score = 20

            risk = "LOW"

        return {

            "score": score,

            "risk": risk
        }

    except Exception:

        return {

            "score": 50,

            "risk": "UNKNOWN"
        }