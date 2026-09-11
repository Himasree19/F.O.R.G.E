import cv2
import numpy as np


def analyze_artifacts(image_path):

    try:

        img = cv2.imread(image_path)

        gray = cv2.cvtColor(
            img,
            cv2.COLOR_BGR2GRAY
        )

        variance = cv2.Laplacian(
            gray,
            cv2.CV_64F
        ).var()

        if variance < 80:

            score = 85

            risk = "HIGH"

        elif variance < 150:

            score = 60

            risk = "MEDIUM"

        else:

            score = 20

            risk = "LOW"

        return {

            "score": score,

            "risk": risk,

            "artifact_strength": round(
                float(variance),
                2
            )
        }

    except Exception:

        return {

            "score": 50,

            "risk": "UNKNOWN"
        }