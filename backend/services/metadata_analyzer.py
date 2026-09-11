from PIL import Image


def analyze_metadata(image_path):

    try:

        img = Image.open(image_path)

        exif = img.getexif()

        metadata_count = len(exif)

        if metadata_count == 0:

            score = 85

            risk = "HIGH"

        elif metadata_count < 5:

            score = 60

            risk = "MEDIUM"

        else:

            score = 20

            risk = "LOW"

        return {

            "score": score,

            "risk": risk,

            "metadata_entries": metadata_count
        }

    except Exception:

        return {

            "score": 50,

            "risk": "UNKNOWN"
        }