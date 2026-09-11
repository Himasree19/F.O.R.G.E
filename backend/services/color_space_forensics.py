import cv2
import numpy as np


def analyze_color_space(image_path):

    try:

        img = cv2.imread(image_path)

        b, g, r = cv2.split(img)

        score_value = np.std(r) + np.std(g) + np.std(b)

        if score_value < 60:

            score = 80

            risk = "HIGH"

        elif score_value < 100:

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