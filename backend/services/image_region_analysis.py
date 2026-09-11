from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np


FACE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades
    + "haarcascade_frontalface_default.xml"
)

PATCH_SIZE = 48
PATCH_STRIDE = 32
MAX_PATCHES = 500


# =========================================================
# GENERAL HELPERS
# =========================================================

def clamp(
    value: float,
    minimum: float = 0.0,
    maximum: float = 1.0,
) -> float:
    return max(
        minimum,
        min(
            maximum,
            float(value),
        ),
    )


def safe_crop(
    image: np.ndarray,
    box: Tuple[int, int, int, int],
) -> np.ndarray:
    x1, y1, x2, y2 = box

    image_height, image_width = image.shape[:2]

    x1 = max(
        0,
        min(
            image_width - 1,
            int(x1),
        ),
    )

    y1 = max(
        0,
        min(
            image_height - 1,
            int(y1),
        ),
    )

    x2 = max(
        x1 + 1,
        min(
            image_width,
            int(x2),
        ),
    )

    y2 = max(
        y1 + 1,
        min(
            image_height,
            int(y2),
        ),
    )

    return image[
        y1:y2,
        x1:x2,
    ]


def normalized_box(
    box: Tuple[int, int, int, int],
    image_width: int,
    image_height: int,
) -> Dict[str, float]:
    x1, y1, x2, y2 = box

    return {
        "x": round(
            (x1 / image_width) * 100,
            5,
        ),
        "y": round(
            (y1 / image_height) * 100,
            5,
        ),
        "width": round(
            ((x2 - x1) / image_width) * 100,
            5,
        ),
        "height": round(
            ((y2 - y1) / image_height) * 100,
            5,
        ),
    }


def risk_level(
    score: float,
) -> str:
    percent = score * 100

    if percent >= 75:
        return "HIGH"

    if percent >= 45:
        return "MEDIUM"

    return "LOW"


# =========================================================
# LOCAL FORENSIC METRICS
# =========================================================

def calculate_texture_score(
    region: np.ndarray,
) -> float:
    if region.size == 0:
        return 0.0

    gray = cv2.cvtColor(
        region,
        cv2.COLOR_BGR2GRAY,
    )

    laplacian_variance = float(
        cv2.Laplacian(
            gray,
            cv2.CV_64F,
        ).var()
    )

    normalized = clamp(
        laplacian_variance / 900.0
    )

    return clamp(
        abs(
            normalized - 0.42
        ) / 0.58
    )


def calculate_edge_score(
    region: np.ndarray,
) -> float:
    if region.size == 0:
        return 0.0

    gray = cv2.cvtColor(
        region,
        cv2.COLOR_BGR2GRAY,
    )

    edges = cv2.Canny(
        gray,
        70,
        170,
    )

    edge_density = float(
        np.mean(
            edges > 0
        )
    )

    return clamp(
        abs(
            edge_density - 0.10
        ) / 0.18
    )


def calculate_noise_score(
    region: np.ndarray,
) -> float:
    if region.size == 0:
        return 0.0

    gray = cv2.cvtColor(
        region,
        cv2.COLOR_BGR2GRAY,
    )

    blurred = cv2.GaussianBlur(
        gray,
        (5, 5),
        0,
    )

    residual = cv2.absdiff(
        gray,
        blurred,
    )

    residual_std = float(
        np.std(
            residual
        )
    )

    normalized = clamp(
        residual_std / 35.0
    )

    return clamp(
        abs(
            normalized - 0.30
        ) / 0.70
    )


def calculate_lighting_score(
    region: np.ndarray,
) -> float:
    if region.size == 0:
        return 0.0

    lab = cv2.cvtColor(
        region,
        cv2.COLOR_BGR2LAB,
    )

    luminance = lab[:, :, 0].astype(
        np.float32
    )

    width = luminance.shape[1]

    if width < 4:
        return 0.0

    middle = width // 2

    left_mean = float(
        np.mean(
            luminance[:, :middle]
        )
    )

    right_mean = float(
        np.mean(
            luminance[:, middle:]
        )
    )

    return clamp(
        abs(
            left_mean - right_mean
        ) / 70.0
    )


def calculate_frequency_score(
    region: np.ndarray,
) -> float:
    if region.size == 0:
        return 0.0

    gray = cv2.cvtColor(
        region,
        cv2.COLOR_BGR2GRAY,
    ).astype(
        np.float32
    )

    spectrum = np.fft.fftshift(
        np.fft.fft2(
            gray
        )
    )

    magnitude = np.log1p(
        np.abs(
            spectrum
        )
    )

    height, width = magnitude.shape

    center_y = height // 2
    center_x = width // 2

    radius_y = max(
        1,
        height // 8,
    )

    radius_x = max(
        1,
        width // 8,
    )

    total_energy = float(
        np.sum(
            magnitude
        )
    )

    if total_energy <= 0:
        return 0.0

    low_energy = float(
        np.sum(
            magnitude[
                max(
                    0,
                    center_y - radius_y,
                ):
                min(
                    height,
                    center_y + radius_y,
                ),
                max(
                    0,
                    center_x - radius_x,
                ):
                min(
                    width,
                    center_x + radius_x,
                ),
            ]
        )
    )

    high_frequency_ratio = (
        1.0
        - (
            low_energy
            / total_energy
        )
    )

    return clamp(
        abs(
            high_frequency_ratio - 0.72
        ) / 0.28
    )


def calculate_compression_score(
    region: np.ndarray,
) -> float:
    if region.size == 0:
        return 0.0

    gray = cv2.cvtColor(
        region,
        cv2.COLOR_BGR2GRAY,
    ).astype(
        np.float32
    )

    height, width = gray.shape

    vertical_values = []
    horizontal_values = []

    for x in range(
        8,
        width,
        8,
    ):
        vertical_values.append(
            float(
                np.mean(
                    np.abs(
                        gray[:, x]
                        - gray[:, x - 1]
                    )
                )
            )
        )

    for y in range(
        8,
        height,
        8,
    ):
        horizontal_values.append(
            float(
                np.mean(
                    np.abs(
                        gray[y, :]
                        - gray[y - 1, :]
                    )
                )
            )
        )

    block_score = float(
        np.mean(
            vertical_values
            + horizontal_values
        )
    ) if (
        vertical_values
        or horizontal_values
    ) else 0.0

    return clamp(
        block_score / 35.0
    )


def calculate_color_score(
    region: np.ndarray,
) -> float:
    if region.size == 0:
        return 0.0

    hsv = cv2.cvtColor(
        region,
        cv2.COLOR_BGR2HSV,
    )

    saturation = hsv[:, :, 1].astype(
        np.float32
    )

    saturation_mean = float(
        np.mean(
            saturation
        )
    )

    saturation_std = float(
        np.std(
            saturation
        )
    )

    overly_uniform = clamp(
        (
            22.0 - saturation_std
        ) / 22.0
    )

    overly_saturated = clamp(
        (
            saturation_mean - 170.0
        ) / 85.0
    )

    return clamp(
        (
            overly_uniform * 0.65
            + overly_saturated * 0.35
        )
    )


def calculate_attention_score(
    heatmap: Optional[np.ndarray],
    box: Tuple[int, int, int, int],
) -> float:
    if heatmap is None:
        return 0.0

    region = safe_crop(
        heatmap,
        box,
    )

    if region.size == 0:
        return 0.0

    if region.ndim == 3:
        region = cv2.cvtColor(
            region,
            cv2.COLOR_BGR2GRAY,
        )

    return clamp(
        float(
            np.mean(
                region
            )
        ) / 255.0
    )


# =========================================================
# METRIC AGGREGATION
# =========================================================

def calculate_local_metrics(
    region: np.ndarray,
    heatmap: Optional[np.ndarray],
    box: Tuple[int, int, int, int],
) -> Dict[str, float]:
    return {
        "attention": calculate_attention_score(
            heatmap,
            box,
        ),
        "texture": calculate_texture_score(
            region
        ),
        "edge": calculate_edge_score(
            region
        ),
        "noise": calculate_noise_score(
            region
        ),
        "lighting": calculate_lighting_score(
            region
        ),
        "frequency": calculate_frequency_score(
            region
        ),
        "compression": calculate_compression_score(
            region
        ),
        "color": calculate_color_score(
            region
        ),
    }


def calculate_combined_score(
    metrics: Dict[str, float],
) -> float:
    return clamp(
        metrics["attention"] * 0.23
        + metrics["texture"] * 0.16
        + metrics["edge"] * 0.13
        + metrics["noise"] * 0.13
        + metrics["frequency"] * 0.12
        + metrics["lighting"] * 0.08
        + metrics["compression"] * 0.08
        + metrics["color"] * 0.07
    )


def generate_reasons(
    metrics: Dict[str, float],
) -> List[str]:
    reasons: List[str] = []

    if metrics["attention"] >= 0.70:
        reasons.append(
            "Strong model attention is concentrated in this area."
        )

    if metrics["texture"] >= 0.65:
        reasons.append(
            "Local texture differs from expected natural variation."
        )

    if metrics["edge"] >= 0.65:
        reasons.append(
            "Edge transitions appear irregular or artificially blended."
        )

    if metrics["noise"] >= 0.65:
        reasons.append(
            "Noise residuals are inconsistent with natural camera acquisition."
        )

    if metrics["frequency"] >= 0.65:
        reasons.append(
            "Frequency-domain energy distribution appears abnormal."
        )

    if metrics["lighting"] >= 0.65:
        reasons.append(
            "Local illumination is inconsistent across the selected area."
        )

    if metrics["compression"] >= 0.65:
        reasons.append(
            "Compression block behaviour differs from surrounding image regions."
        )

    if metrics["color"] >= 0.65:
        reasons.append(
            "Local colour distribution appears unusually uniform or saturated."
        )

    if not reasons:
        reasons.append(
            "No major forensic irregularity was detected in this area."
        )

    return reasons[:4]


def format_metrics(
    metrics: Dict[str, float],
) -> Dict[str, float]:
    return {
        key: round(
            value * 100,
            2,
        )
        for key, value in metrics.items()
    }


# =========================================================
# PATCH-LEVEL HOVER GRID
# =========================================================

def create_patch_record(
    patch_id: str,
    image: np.ndarray,
    heatmap: Optional[np.ndarray],
    box: Tuple[int, int, int, int],
    row: int,
    column: int,
) -> Dict[str, Any]:
    region = safe_crop(
        image,
        box,
    )

    metrics = calculate_local_metrics(
        region,
        heatmap,
        box,
    )

    score = calculate_combined_score(
        metrics
    )

    height, width = image.shape[:2]

    x1, y1, x2, y2 = box

    return {
        "id": patch_id,
        "row": row,
        "column": column,
        "box": normalized_box(
            box,
            width,
            height,
        ),
        "pixel_box": {
            "x1": int(x1),
            "y1": int(y1),
            "x2": int(x2),
            "y2": int(y2),
        },
        "center": {
            "x": round(
                (
                    (
                        x1 + x2
                    ) / 2
                    / width
                ) * 100,
                5,
            ),
            "y": round(
                (
                    (
                        y1 + y2
                    ) / 2
                    / height
                ) * 100,
                5,
            ),
        },
        "forgery_score": round(
            score * 100,
            2,
        ),
        "risk_level": risk_level(
            score
        ),
        "metrics": format_metrics(
            metrics
        ),
        "reasons": generate_reasons(
            metrics
        ),
    }


def generate_hover_grid(
    image: np.ndarray,
    heatmap: Optional[np.ndarray],
) -> Dict[str, Any]:
    image_height, image_width = image.shape[:2]

    patches: List[Dict[str, Any]] = []

    row_index = 0

    for y in range(
        0,
        image_height,
        PATCH_STRIDE,
    ):
        column_index = 0

        for x in range(
            0,
            image_width,
            PATCH_STRIDE,
        ):
            if len(
                patches
            ) >= MAX_PATCHES:
                break

            x2 = min(
                image_width,
                x + PATCH_SIZE,
            )

            y2 = min(
                image_height,
                y + PATCH_SIZE,
            )

            if (
                x2 - x < 16
                or y2 - y < 16
            ):
                column_index += 1
                continue

            patch_id = (
                f"patch_"
                f"{row_index}_"
                f"{column_index}"
            )

            patches.append(
                create_patch_record(
                    patch_id=patch_id,
                    image=image,
                    heatmap=heatmap,
                    box=(
                        x,
                        y,
                        x2,
                        y2,
                    ),
                    row=row_index,
                    column=column_index,
                )
            )

            column_index += 1

        if len(
            patches
        ) >= MAX_PATCHES:
            break

        row_index += 1

    ranked_patches = sorted(
        patches,
        key=lambda item: (
            item["forgery_score"]
        ),
        reverse=True,
    )

    return {
        "patch_size": PATCH_SIZE,
        "patch_stride": PATCH_STRIDE,
        "image_width": image_width,
        "image_height": image_height,
        "patch_count": len(
            patches
        ),
        "patches": patches,
        "ranked_patches": ranked_patches[:40],
    }


# =========================================================
# SEMANTIC FACE REGIONS
# =========================================================

def build_face_regions(
    face: Tuple[int, int, int, int],
) -> List[
    Tuple[
        str,
        str,
        Tuple[int, int, int, int],
    ]
]:
    x, y, width, height = face

    def create_box(
        left: float,
        top: float,
        right: float,
        bottom: float,
    ) -> Tuple[int, int, int, int]:
        return (
            int(
                x + left * width
            ),
            int(
                y + top * height
            ),
            int(
                x + right * width
            ),
            int(
                y + bottom * height
            ),
        )

    return [
        (
            "forehead",
            "Forehead",
            create_box(
                0.18,
                0.05,
                0.82,
                0.29,
            ),
        ),
        (
            "left_eye",
            "Left Eye",
            create_box(
                0.12,
                0.27,
                0.47,
                0.48,
            ),
        ),
        (
            "right_eye",
            "Right Eye",
            create_box(
                0.53,
                0.27,
                0.88,
                0.48,
            ),
        ),
        (
            "nose",
            "Nose",
            create_box(
                0.35,
                0.38,
                0.65,
                0.70,
            ),
        ),
        (
            "left_cheek",
            "Left Cheek",
            create_box(
                0.07,
                0.46,
                0.40,
                0.74,
            ),
        ),
        (
            "right_cheek",
            "Right Cheek",
            create_box(
                0.60,
                0.46,
                0.93,
                0.74,
            ),
        ),
        (
            "mouth",
            "Mouth",
            create_box(
                0.28,
                0.66,
                0.72,
                0.87,
            ),
        ),
        (
            "jaw",
            "Jaw and Chin",
            create_box(
                0.18,
                0.74,
                0.82,
                1.00,
            ),
        ),
    ]


def analyze_semantic_region(
    region_id: str,
    name: str,
    image: np.ndarray,
    heatmap: Optional[np.ndarray],
    box: Tuple[int, int, int, int],
) -> Dict[str, Any]:
    region = safe_crop(
        image,
        box,
    )

    metrics = calculate_local_metrics(
        region,
        heatmap,
        box,
    )

    score = calculate_combined_score(
        metrics
    )

    image_height, image_width = image.shape[:2]

    return {
        "id": region_id,
        "name": name,
        "box": normalized_box(
            box,
            image_width,
            image_height,
        ),
        "forgery_score": round(
            score * 100,
            2,
        ),
        "risk_level": risk_level(
            score
        ),
        "metrics": format_metrics(
            metrics
        ),
        "reasons": generate_reasons(
            metrics
        ),
    }


# =========================================================
# MAIN ANALYSIS
# =========================================================

def analyse_image_regions(
    image_path: str,
    heatmap_path: Optional[str] = None,
) -> Dict[str, Any]:
    image = cv2.imread(
        image_path
    )

    if image is None:
        return {
            "face_detected": False,
            "regions": [],
            "ranked_regions": [],
            "hover_grid": {
                "patches": [],
                "ranked_patches": [],
            },
            "error": (
                "Unable to read image "
                "for investigation analysis."
            ),
        }

    heatmap = None

    if heatmap_path:
        heatmap = cv2.imread(
            heatmap_path,
            cv2.IMREAD_GRAYSCALE,
        )

        if heatmap is not None:
            heatmap = cv2.resize(
                heatmap,
                (
                    image.shape[1],
                    image.shape[0],
                ),
            )

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY,
    )

    detected_faces = (
        FACE_CASCADE.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(80, 80),
        )
    )

    image_height, image_width = image.shape[:2]

    semantic_regions: List[
        Dict[str, Any]
    ] = []

    if len(
        detected_faces
    ) > 0:
        largest_face = max(
            detected_faces,
            key=lambda item: (
                item[2] * item[3]
            ),
        )

        face = tuple(
            map(
                int,
                largest_face,
            )
        )

        region_definitions = build_face_regions(
            face
        )

        face_detected = True

    else:
        region_definitions = [
            (
                "upper_left",
                "Upper Left",
                (
                    0,
                    0,
                    image_width // 2,
                    image_height // 2,
                ),
            ),
            (
                "upper_right",
                "Upper Right",
                (
                    image_width // 2,
                    0,
                    image_width,
                    image_height // 2,
                ),
            ),
            (
                "lower_left",
                "Lower Left",
                (
                    0,
                    image_height // 2,
                    image_width // 2,
                    image_height,
                ),
            ),
            (
                "lower_right",
                "Lower Right",
                (
                    image_width // 2,
                    image_height // 2,
                    image_width,
                    image_height,
                ),
            ),
        ]

        face_detected = False

    for (
        region_id,
        display_name,
        box,
    ) in region_definitions:
        semantic_regions.append(
            analyze_semantic_region(
                region_id=region_id,
                name=display_name,
                image=image,
                heatmap=heatmap,
                box=box,
            )
        )

    ranked_regions = sorted(
        semantic_regions,
        key=lambda item: (
            item["forgery_score"]
        ),
        reverse=True,
    )

    hover_grid = generate_hover_grid(
        image,
        heatmap,
    )

    return {
        "face_detected": face_detected,
        "regions": semantic_regions,
        "ranked_regions": ranked_regions,
        "hover_grid": hover_grid,
        "image_dimensions": {
            "width": image_width,
            "height": image_height,
        },
        "analysis_version": (
            "FORGE-IMAGE-INVESTIGATION-2.0"
        ),
    }