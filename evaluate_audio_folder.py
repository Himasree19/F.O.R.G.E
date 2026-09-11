from __future__ import annotations

import argparse
import csv
import json
import mimetypes
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import requests
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


SUPPORTED_EXTENSIONS = {
    ".wav",
    ".mp3",
    ".flac",
    ".m4a",
    ".ogg",
    ".aac",
}


def normalize_prediction(value: object) -> int:
    """
    Return:
        0 -> REAL
        1 -> FAKE
    """
    text = str(value or "").strip().upper()

    fake_labels = {
        "FAKE",
        "AI",
        "AI-GENERATED",
        "AI GENERATED",
        "SYNTHETIC",
        "DEEPFAKE",
        "SPOOF",
    }

    real_labels = {
        "REAL",
        "HUMAN",
        "AUTHENTIC",
        "GENUINE",
        "BONAFIDE",
        "BONA FIDE",
    }

    if text in fake_labels:
        return 1

    if text in real_labels:
        return 0

    if any(word in text for word in ["FAKE", "SYNTHETIC", "AI", "SPOOF"]):
        return 1

    if any(word in text for word in ["REAL", "HUMAN", "AUTHENTIC", "GENUINE"]):
        return 0

    raise ValueError(f"Unrecognized prediction label: {value!r}")


def extract_fake_probability(result: dict) -> float:
    candidates = [
        result.get("raw_probability_fake"),
        result.get("fake_probability"),
        result.get("probability_fake"),
        result.get("ai_probability"),
        result.get("risk_score"),
        result.get("confidence")
        if normalize_prediction(result.get("prediction")) == 1
        else None,
    ]

    probabilities = result.get("probabilities")

    if isinstance(probabilities, dict):
        candidates.insert(0, probabilities.get("fake"))
        candidates.insert(0, probabilities.get("ai"))

    for value in candidates:
        if value is None:
            continue

        number = float(value)

        if 0 <= number <= 1:
            number *= 100

        return max(0.0, min(100.0, number))

    return 0.0


def collect_audio_files(dataset_dir: Path):
    items: list[tuple[Path, int]] = []

    for class_name, label in (("real", 0), ("fake", 1)):
        class_dir = dataset_dir / class_name

        if not class_dir.exists():
            raise FileNotFoundError(
                f"Required folder not found: {class_dir}"
            )

        for path in sorted(class_dir.rglob("*")):
            if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
                items.append((path, label))

    return items


def analyze_audio(api_url: str, audio_path: Path) -> dict:
    mime_type = (
        mimetypes.guess_type(audio_path.name)[0]
        or "application/octet-stream"
    )

    with audio_path.open("rb") as file_handle:
        response = requests.post(
            api_url,
            files={
                "file": (
                    audio_path.name,
                    file_handle,
                    mime_type,
                )
            },
            timeout=300,
        )

    try:
        data = response.json()
    except ValueError as error:
        raise RuntimeError(
            f"Backend returned non-JSON response for {audio_path.name}: "
            f"{response.text[:300]}"
        ) from error

    if not response.ok:
        raise RuntimeError(
            f"Backend error for {audio_path.name}: "
            f"{data.get('error') or data}"
        )

    if data.get("error"):
        raise RuntimeError(
            f"Analysis error for {audio_path.name}: {data['error']}"
        )

    return data


def save_confusion_matrix(
    matrix: np.ndarray,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 6.2))

    image = ax.imshow(matrix)

    ax.set_title(
        "FORGE Audio Model Confusion Matrix",
        fontsize=17,
        pad=18,
    )

    ax.set_xlabel(
        "Predicted Label",
        fontsize=13,
        labelpad=12,
    )

    ax.set_ylabel(
        "Actual Label",
        fontsize=13,
        labelpad=12,
    )

    labels = ["REAL", "FAKE"]

    ax.set_xticks([0, 1], labels=labels)
    ax.set_yticks([0, 1], labels=labels)

    threshold = matrix.max() / 2 if matrix.size else 0

    for row in range(2):
        for column in range(2):
            value = int(matrix[row, column])

            ax.text(
                column,
                row,
                str(value),
                ha="center",
                va="center",
                fontsize=24,
                fontweight="bold",
                color="white" if value > threshold else "black",
            )

    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the FORGE audio model on labelled "
            "real and fake audio folders."
        )
    )

    parser.add_argument(
        "--dataset",
        required=True,
        help="Folder containing real/ and fake/ directories.",
    )

    parser.add_argument(
        "--api",
        default="http://127.0.0.1:8000/analyze",
        help="FORGE audio analysis endpoint.",
    )

    parser.add_argument(
        "--output",
        default="audio_evaluation_results",
        help="Directory for metrics, CSV and confusion matrix.",
    )

    args = parser.parse_args()

    dataset_dir = Path(args.dataset).expanduser().resolve()
    output_dir = Path(args.output).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    samples = collect_audio_files(dataset_dir)

    if not samples:
        print("No supported audio files were found.", file=sys.stderr)
        return 1

    y_true: list[int] = []
    y_pred: list[int] = []
    rows: list[dict] = []
    failures: list[dict] = []

    print(f"Found {len(samples)} labelled audio samples.")

    for index, (audio_path, actual_label) in enumerate(samples, start=1):
        print(
            f"[{index}/{len(samples)}] "
            f"Analysing {audio_path.name}"
        )

        try:
            result = analyze_audio(
                args.api,
                audio_path,
            )

            predicted_label = normalize_prediction(
                result.get("prediction")
            )

            fake_probability = extract_fake_probability(result)

            y_true.append(actual_label)
            y_pred.append(predicted_label)

            rows.append(
                {
                    "filename": audio_path.name,
                    "actual_label": "FAKE" if actual_label else "REAL",
                    "predicted_label": (
                        "FAKE" if predicted_label else "REAL"
                    ),
                    "fake_probability": round(fake_probability, 4),
                    "confidence": result.get("confidence"),
                    "correct": actual_label == predicted_label,
                }
            )

        except Exception as error:
            failures.append(
                {
                    "filename": audio_path.name,
                    "error": str(error),
                }
            )

            print(f"  ERROR: {error}")

    if not y_true:
        print(
            "No files were evaluated successfully.",
            file=sys.stderr,
        )
        return 1

    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    )

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(
        y_true,
        y_pred,
        pos_label=1,
        zero_division=0,
    )
    recall = recall_score(
        y_true,
        y_pred,
        pos_label=1,
        zero_division=0,
    )
    f1 = f1_score(
        y_true,
        y_pred,
        pos_label=1,
        zero_division=0,
    )

    report = classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
        target_names=["REAL", "FAKE"],
        zero_division=0,
        output_dict=True,
    )

    metrics = {
        "evaluated_samples": len(y_true),
        "failed_samples": len(failures),
        "positive_class": "FAKE",
        "accuracy": accuracy,
        "precision_fake": precision,
        "recall_fake": recall,
        "f1_fake": f1,
        "confusion_matrix": {
            "labels": ["REAL", "FAKE"],
            "values": matrix.tolist(),
            "true_negative": int(matrix[0, 0]),
            "false_positive": int(matrix[0, 1]),
            "false_negative": int(matrix[1, 0]),
            "true_positive": int(matrix[1, 1]),
        },
        "classification_report": report,
    }

    with (output_dir / "audio_predictions.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file_handle:
        writer = csv.DictWriter(
            file_handle,
            fieldnames=[
                "filename",
                "actual_label",
                "predicted_label",
                "fake_probability",
                "confidence",
                "correct",
            ],
        )

        writer.writeheader()
        writer.writerows(rows)

    with (output_dir / "audio_metrics.json").open(
        "w",
        encoding="utf-8",
    ) as file_handle:
        json.dump(
            metrics,
            file_handle,
            indent=2,
        )

    with (output_dir / "audio_classification_report.txt").open(
        "w",
        encoding="utf-8",
    ) as file_handle:
        file_handle.write(
            classification_report(
                y_true,
                y_pred,
                labels=[0, 1],
                target_names=["REAL", "FAKE"],
                zero_division=0,
            )
        )

    if failures:
        with (output_dir / "audio_failures.json").open(
            "w",
            encoding="utf-8",
        ) as file_handle:
            json.dump(
                failures,
                file_handle,
                indent=2,
            )

    save_confusion_matrix(
        matrix,
        output_dir / "audio_confusion_matrix.png",
    )

    print("\n" + "=" * 60)
    print("FORGE AUDIO EVALUATION")
    print("=" * 60)
    print(f"Evaluated samples : {len(y_true)}")
    print(f"Failed samples    : {len(failures)}")
    print(f"Accuracy          : {accuracy * 100:.2f}%")
    print(f"Precision (FAKE)  : {precision * 100:.2f}%")
    print(f"Recall (FAKE)     : {recall * 100:.2f}%")
    print(f"F1-score (FAKE)   : {f1 * 100:.2f}%")
    print("\nConfusion matrix [REAL, FAKE]:")
    print(matrix)
    print(f"\nSaved results to: {output_dir}")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())