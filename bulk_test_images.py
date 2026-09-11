import os
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from backend.services.fake_image_detector import analyze_image

DATASET_DIR = "test_images"

REAL_DIR = os.path.join(DATASET_DIR, "real")
FAKE_DIR = os.path.join(DATASET_DIR, "fake")

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp")

results = []


def test_folder(folder_path, actual_label):
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(IMAGE_EXTENSIONS):
            image_path = os.path.join(folder_path, filename)

            print("Testing:", image_path)

            result = analyze_image(image_path)

            print("Result:", result)

            predicted_label = result.get("prediction", "ERROR")

            results.append({
                "filename": filename,
                "path": image_path,
                "actual": actual_label,
                "prediction": predicted_label,
                "confidence": result.get("confidence", 0),
                "ai_probability": result.get("raw_ai_probability", 0),
                "human_probability": result.get("raw_human_probability", 0),
                "risk_level": result.get("risk_level", "N/A"),
                "risk_score": result.get("risk_score", 0),
                "error": result.get("error", "")
            })


test_folder(REAL_DIR, "HUMAN")
test_folder(FAKE_DIR, "AI")

df = pd.DataFrame(results)

df.to_csv("image_bulk_test_results.csv", index=False)

valid_df = df[df["prediction"] != "ERROR"]

if len(valid_df) == 0:
    print("\n❌ No valid predictions found.")
    print("This means the model is not loading or analyze_image() is returning errors.")
    print("\nErrors:")
    print(df[["filename", "actual", "prediction", "error"]])
    exit()

accuracy = accuracy_score(valid_df["actual"], valid_df["prediction"])

cm = confusion_matrix(
    valid_df["actual"],
    valid_df["prediction"],
    labels=["HUMAN", "AI"]
)

report = classification_report(
    valid_df["actual"],
    valid_df["prediction"],
    labels=["HUMAN", "AI"]
)

print("\n===================================")
print("IMAGE MODEL BULK TEST RESULTS")
print("===================================")
print(f"Total Images Tested: {len(df)}")
print(f"Valid Predictions : {len(valid_df)}")
print(f"Accuracy          : {accuracy * 100:.2f}%")

print("\nConfusion Matrix")
print("Labels: ['HUMAN', 'AI']")
print(cm)

print("\nClassification Report")
print(report)

print("\nResults saved as:")
print("image_bulk_test_results.csv")