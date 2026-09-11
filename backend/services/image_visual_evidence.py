from __future__ import annotations

import os
import uuid
from typing import Dict, Tuple

import cv2
import numpy as np


PROJECT_ROOT = os.getcwd()

BASE_OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "backend",
    "uploads",
)

HEATMAP_DIR = os.path.join(
    BASE_OUTPUT_DIR,
    "heatmaps",
)

OVERLAY_DIR = os.path.join(
    BASE_OUTPUT_DIR,
    "overlays",
)

EDGE_DIR = os.path.join(
    BASE_OUTPUT_DIR,
    "edge_maps",
)

FREQUENCY_DIR = os.path.join(
    BASE_OUTPUT_DIR,
    "frequency_maps",
)

NATURALNESS_DIR = os.path.join(
    BASE_OUTPUT_DIR,
    "naturalness_maps",
)


for directory in (
    HEATMAP_DIR,
    OVERLAY_DIR,
    EDGE_DIR,
    FREQUENCY_DIR,
    NATURALNESS_DIR,
):
    os.makedirs(
        directory,
        exist_ok=True,
    )


# =========================================================
# FILE HELPERS
# =========================================================

def save_generated_image(
    directory: str,
    suffix: str,
    image: np.ndarray,
) -> Tuple[str, str]:
    filename = (
        f"{uuid.uuid4().hex}_{suffix}.jpg"
    )

    output_path = os.path.join(
        directory,
        filename,
    )

    saved = cv2.imwrite(
        output_path,
        image,
    )

    if not saved:
        raise RuntimeError(
            f"Unable to save generated image: {output_path}"
        )

    relative_path = os.path.relpath(
        output_path,
        BASE_OUTPUT_DIR,
    ).replace(
        os.sep,
        "/",
    )

    public_url = (
        f"/backend-uploads/{relative_path}"
    )

    return output_path, public_url


def read_image(
    image_path: str,
) -> np.ndarray:
    image = cv2.imread(
        image_path
    )

    if image is None:
        raise ValueError(
            f"Unable to read image: {image_path}"
        )

    return image


# =========================================================
# FORENSIC FEATURE MAPS
# =========================================================

def calculate_texture_anomaly(
    gray: np.ndarray,
) -> np.ndarray:
    local_blur = cv2.GaussianBlur(
        gray,
        (21, 21),
        0,
    )

    residual = cv2.absdiff(
        gray,
        local_blur,
    )

    residual = cv2.GaussianBlur(
        residual,
        (9, 9),
        0,
    )

    return cv2.normalize(
        residual,
        None,
        0,
        255,
        cv2.NORM_MINMAX,
    ).astype(
        np.uint8
    )


def calculate_edge_anomaly(
    gray: np.ndarray,
) -> np.ndarray:
    laplacian = cv2.Laplacian(
        gray,
        cv2.CV_32F,
    )

    laplacian = cv2.convertScaleAbs(
        laplacian
    )

    laplacian = cv2.GaussianBlur(
        laplacian,
        (7, 7),
        0,
    )

    return cv2.normalize(
        laplacian,
        None,
        0,
        255,
        cv2.NORM_MINMAX,
    ).astype(
        np.uint8
    )


def calculate_noise_anomaly(
    gray: np.ndarray,
) -> np.ndarray:
    denoised = cv2.GaussianBlur(
        gray,
        (5, 5),
        0,
    )

    noise = cv2.absdiff(
        gray,
        denoised,
    )

    noise = cv2.GaussianBlur(
        noise,
        (9, 9),
        0,
    )

    return cv2.normalize(
        noise,
        None,
        0,
        255,
        cv2.NORM_MINMAX,
    ).astype(
        np.uint8
    )


def calculate_frequency_anomaly(
    gray: np.ndarray,
) -> np.ndarray:
    gray_float = gray.astype(
        np.float32
    )

    spectrum = np.fft.fftshift(
        np.fft.fft2(
            gray_float
        )
    )

    magnitude = np.log1p(
        np.abs(
            spectrum
        )
    )

    magnitude = cv2.normalize(
        magnitude,
        None,
        0,
        255,
        cv2.NORM_MINMAX,
    ).astype(
        np.uint8
    )

    magnitude = cv2.resize(
        magnitude,
        (
            gray.shape[1],
            gray.shape[0],
        ),
    )

    magnitude = cv2.GaussianBlur(
        magnitude,
        (11, 11),
        0,
    )

    return magnitude


# =========================================================
# COMBINED SUSPICION MAP
# =========================================================

def build_suspicion_map(
    image: np.ndarray,
) -> np.ndarray:
    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY,
    )

    texture = calculate_texture_anomaly(
        gray
    )

    edge = calculate_edge_anomaly(
        gray
    )

    noise = calculate_noise_anomaly(
        gray
    )

    frequency = calculate_frequency_anomaly(
        gray
    )

    combined = (
        texture.astype(
            np.float32
        ) * 0.34
        + edge.astype(
            np.float32
        ) * 0.26
        + noise.astype(
            np.float32
        ) * 0.22
        + frequency.astype(
            np.float32
        ) * 0.18
    )

    combined = cv2.normalize(
        combined,
        None,
        0,
        255,
        cv2.NORM_MINMAX,
    ).astype(
        np.uint8
    )

    combined = cv2.GaussianBlur(
        combined,
        (17, 17),
        0,
    )

    # Increase contrast so suspicious zones become visually clear.
    combined = cv2.equalizeHist(
        combined
    )

    # Suppress weak background signals.
    combined = cv2.threshold(
        combined,
        72,
        255,
        cv2.THRESH_TOZERO,
    )[1]

    return combined


# =========================================================
# HUMAN / NATURALNESS MAP
# =========================================================

def build_naturalness_map(
    suspicion_map: np.ndarray,
) -> np.ndarray:
    naturalness = (
        255
        - suspicion_map
    )

    naturalness = cv2.GaussianBlur(
        naturalness,
        (15, 15),
        0,
    )

    return naturalness


# =========================================================
# COLOUR MAPPING
# =========================================================

def build_forensic_colour_map(
    suspicion_map: np.ndarray,
) -> np.ndarray:
    """
    Colour meaning:

    Dark blue   = strongly natural
    Cyan        = likely natural
    Green       = mostly natural
    Yellow      = uncertain / mixed
    Orange      = suspicious
    Red         = strong AI / manipulation indicator
    """

    normalized = (
        suspicion_map.astype(
            np.float32
        )
        / 255.0
    )

    output = np.zeros(
        (
            suspicion_map.shape[0],
            suspicion_map.shape[1],
            3,
        ),
        dtype=np.uint8,
    )

    # OpenCV uses BGR.

    # 0.00–0.20: dark blue
    mask = normalized < 0.20
    output[mask] = (
        145,
        45,
        0,
    )

    # 0.20–0.40: cyan / blue-green
    mask = (
        (normalized >= 0.20)
        & (normalized < 0.40)
    )
    output[mask] = (
        220,
        180,
        0,
    )

    # 0.40–0.55: green
    mask = (
        (normalized >= 0.40)
        & (normalized < 0.55)
    )
    output[mask] = (
        60,
        190,
        50,
    )

    # 0.55–0.70: yellow
    mask = (
        (normalized >= 0.55)
        & (normalized < 0.70)
    )
    output[mask] = (
        0,
        220,
        255,
    )

    # 0.70–0.85: orange
    mask = (
        (normalized >= 0.70)
        & (normalized < 0.85)
    )
    output[mask] = (
        0,
        120,
        255,
    )

    # 0.85–1.00: red
    mask = normalized >= 0.85
    output[mask] = (
        0,
        0,
        255,
    )

    return output


def build_naturalness_colour_map(
    naturalness_map: np.ndarray,
) -> np.ndarray:
    return cv2.applyColorMap(
        naturalness_map,
        cv2.COLORMAP_WINTER,
    )


# =========================================================
# LEGEND
# =========================================================

def add_heatmap_legend(
    image: np.ndarray,
) -> np.ndarray:
    output = image.copy()

    height, width = output.shape[:2]

    legend_width = min(
        360,
        max(
            240,
            width // 3,
        ),
    )

    legend_height = 72

    x1 = 18
    y1 = height - legend_height - 18
    x2 = x1 + legend_width
    y2 = y1 + legend_height

    overlay = output.copy()

    cv2.rectangle(
        overlay,
        (
            x1,
            y1,
        ),
        (
            x2,
            y2,
        ),
        (
            8,
            12,
            25,
        ),
        -1,
    )

    output = cv2.addWeighted(
        overlay,
        0.82,
        output,
        0.18,
        0,
    )

    cv2.putText(
        output,
        "FORENSIC HEATMAP",
        (
            x1 + 12,
            y1 + 20,
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (
            255,
            255,
            255,
        ),
        1,
        cv2.LINE_AA,
    )

    labels = [
        (
            "NATURAL",
            (
                160,
                80,
                0,
            ),
        ),
        (
            "LOW",
            (
                190,
                170,
                0,
            ),
        ),
        (
            "MIXED",
            (
                0,
                210,
                230,
            ),
        ),
        (
            "SUSPICIOUS",
            (
                0,
                120,
                255,
            ),
        ),
        (
            "HIGH AI",
            (
                0,
                0,
                255,
            ),
        ),
    ]

    segment_width = (
        legend_width - 24
    ) // len(
        labels
    )

    for index, (
        label,
        colour,
    ) in enumerate(
        labels
    ):
        start_x = (
            x1
            + 12
            + index * segment_width
        )

        cv2.rectangle(
            output,
            (
                start_x,
                y1 + 30,
            ),
            (
                start_x
                + segment_width
                - 4,
                y1 + 44,
            ),
            colour,
            -1,
        )

        cv2.putText(
            output,
            label,
            (
                start_x,
                y1 + 61,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.27,
            (
                230,
                235,
                245,
            ),
            1,
            cv2.LINE_AA,
        )

    return output


# =========================================================
# HIGH-RISK REGIONS
# =========================================================

def draw_suspicious_regions(
    image: np.ndarray,
    suspicion_map: np.ndarray,
) -> np.ndarray:
    output = image.copy()

    high_risk_mask = cv2.threshold(
        suspicion_map,
        185,
        255,
        cv2.THRESH_BINARY,
    )[1]

    high_risk_mask = cv2.morphologyEx(
        high_risk_mask,
        cv2.MORPH_CLOSE,
        np.ones(
            (
                15,
                15,
            ),
            dtype=np.uint8,
        ),
    )

    contours, _ = cv2.findContours(
        high_risk_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    image_area = (
        image.shape[0]
        * image.shape[1]
    )

    region_index = 1

    for contour in contours:
        area = cv2.contourArea(
            contour
        )

        if area < (
            image_area * 0.003
        ):
            continue

        x, y, width, height = (
            cv2.boundingRect(
                contour
            )
        )

        region_values = suspicion_map[
            y:y + height,
            x:x + width,
        ]

        region_score = float(
            np.mean(
                region_values
            )
        ) / 255.0 * 100

        cv2.rectangle(
            output,
            (
                x,
                y,
            ),
            (
                x + width,
                y + height,
            ),
            (
                0,
                0,
                255,
            ),
            2,
        )

        label = (
            f"AI RISK {region_index}: "
            f"{region_score:.0f}%"
        )

        text_size, _ = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            1,
        )

        cv2.rectangle(
            output,
            (
                x,
                max(
                    0,
                    y - 22,
                ),
            ),
            (
                x
                + text_size[0]
                + 10,
                y,
            ),
            (
                0,
                0,
                190,
            ),
            -1,
        )

        cv2.putText(
            output,
            label,
            (
                x + 5,
                y - 6,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (
                255,
                255,
                255,
            ),
            1,
            cv2.LINE_AA,
        )

        region_index += 1

    return output


# =========================================================
# MAIN GENERATOR
# =========================================================

def generate_image_visual_evidence(
    image_path: str,
) -> Dict[str, str]:
    image = read_image(
        image_path
    )

    suspicion_map = build_suspicion_map(
        image
    )

    naturalness_map = build_naturalness_map(
        suspicion_map
    )

    forensic_colour_map = (
        build_forensic_colour_map(
            suspicion_map
        )
    )

    naturalness_colour_map = (
        build_naturalness_colour_map(
            naturalness_map
        )
    )

    # More readable combined overlay.
    combined_overlay = cv2.addWeighted(
        image,
        0.58,
        forensic_colour_map,
        0.42,
        0,
    )

    combined_overlay = (
        draw_suspicious_regions(
            combined_overlay,
            suspicion_map,
        )
    )

    combined_overlay = (
        add_heatmap_legend(
            combined_overlay
        )
    )

    # Heatmap-only image with legend.
    labelled_heatmap = (
        add_heatmap_legend(
            forensic_colour_map
        )
    )

    naturalness_overlay = cv2.addWeighted(
        image,
        0.60,
        naturalness_colour_map,
        0.40,
        0,
    )

    cv2.putText(
        naturalness_overlay,
        "GREEN / BLUE = MORE NATURAL",
        (
            18,
            30,
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (
            255,
            255,
            255,
        ),
        2,
        cv2.LINE_AA,
    )

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY,
    )

    edge_map = cv2.Canny(
        gray,
        65,
        165,
    )

    edge_map = cv2.cvtColor(
        edge_map,
        cv2.COLOR_GRAY2BGR,
    )

    frequency_map = calculate_frequency_anomaly(
        gray
    )

    frequency_map = cv2.applyColorMap(
        frequency_map,
        cv2.COLORMAP_TURBO,
    )

    heatmap_path, heatmap_url = (
        save_generated_image(
            HEATMAP_DIR,
            "ai_suspicion_heatmap",
            labelled_heatmap,
        )
    )

    overlay_path, overlay_url = (
        save_generated_image(
            OVERLAY_DIR,
            "forensic_overlay",
            combined_overlay,
        )
    )

    naturalness_path, naturalness_url = (
        save_generated_image(
            NATURALNESS_DIR,
            "naturalness_overlay",
            naturalness_overlay,
        )
    )

    edge_path, edge_url = (
        save_generated_image(
            EDGE_DIR,
            "edge_map",
            edge_map,
        )
    )

    frequency_path, frequency_url = (
        save_generated_image(
            FREQUENCY_DIR,
            "frequency_map",
            frequency_map,
        )
    )

    return {
        "heatmap_path": heatmap_path,
        "heatmap_url": heatmap_url,

        "overlay_path": overlay_path,
        "overlay_url": overlay_url,

        "naturalness_path": naturalness_path,
        "naturalness_url": naturalness_url,

        "edge_map_path": edge_path,
        "edge_map_url": edge_url,

        "frequency_map_path": frequency_path,
        "frequency_map_url": frequency_url,

        "legend": {
            "blue": "Strongly natural region",
            "cyan": "Likely natural region",
            "green": "Mostly natural region",
            "yellow": "Mixed or uncertain region",
            "orange": "Suspicious synthetic indicator",
            "red": "Strong AI or manipulation indicator",
        },

        "interpretation_notice": (
            "Red and orange regions contain stronger forensic "
            "irregularities. Green, cyan and blue regions show "
            "more natural image characteristics."
        ),
    }