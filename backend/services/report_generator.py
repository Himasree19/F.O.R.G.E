from __future__ import annotations

import html
import os
import tempfile
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm, mm
from reportlab.platypus import (
    Image,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


# =========================================================
# CONFIGURATION
# =========================================================

PROJECT_ROOT = os.getcwd()

PAGE_WIDTH, PAGE_HEIGHT = A4

REPORT_TITLE = (
    "FORGE - Multimodal Digital Forensic Examination Report"
)

REPORT_ENGINE_VERSION = "FORGE-REPORT-3.0"

DARK_NAVY = colors.HexColor("#08111F")
NAVY = colors.HexColor("#10233D")
CYAN = colors.HexColor("#00A9C7")
LIGHT_CYAN = colors.HexColor("#DFF8FC")
BLUE = colors.HexColor("#2C6EA3")

RED = colors.HexColor("#D83A56")
LIGHT_RED = colors.HexColor("#FCE7EC")

AMBER = colors.HexColor("#D99000")
LIGHT_AMBER = colors.HexColor("#FFF3D8")

GREEN = colors.HexColor("#178B55")
LIGHT_GREEN = colors.HexColor("#E5F6EE")

LIGHT_GREY = colors.HexColor("#F2F5F8")
MID_GREY = colors.HexColor("#D6DEE7")
TEXT_GREY = colors.HexColor("#475569")
WHITE = colors.white
BLACK = colors.black


# =========================================================
# GENERIC HELPERS
# =========================================================

def safe_number(
    value: Any,
    default: float = 0.0,
) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def clamp(
    value: float,
    minimum: float = 0.0,
    maximum: float = 100.0,
) -> float:
    return max(
        minimum,
        min(
            maximum,
            value,
        ),
    )


def escape_text(
    value: Any,
) -> str:
    if value is None:
        return ""

    return html.escape(
        str(value)
    )


def format_key(
    key: Any,
) -> str:
    return (
        str(key)
        .replace("_", " ")
        .strip()
        .title()
    )


def format_percent(
    value: Any,
) -> str:
    return (
        f"{safe_number(value):.2f}%"
    )


def format_file_size(
    size_bytes: Any,
) -> str:
    size = safe_number(
        size_bytes,
        0,
    )

    if size <= 0:
        return "Not available"

    units = [
        "B",
        "KB",
        "MB",
        "GB",
    ]

    index = 0

    while (
        size >= 1024
        and index < len(units) - 1
    ):
        size /= 1024
        index += 1

    return (
        f"{size:.2f} {units[index]}"
    )


def get_modality(
    result: Dict[str, Any],
) -> str:
    modality = (
        result.get("modality")
        or result.get("file_type")
        or ""
    )

    modality = str(
        modality
    ).lower()

    if modality in {
        "text",
        "image",
        "audio",
    }:
        return modality

    if (
        result.get("full_document")
        or result.get("highlighted_document")
    ):
        return "text"

    if (
        result.get("visual_evidence")
        or result.get("heatmap")
        or result.get("region_analysis")
    ):
        return "image"

    if (
        result.get("waveform")
        or result.get("spectrogram")
        or result.get("audio_heatmap")
    ):
        return "audio"

    return "unknown"


def normalize_probability(
    value: Any,
) -> float:
    number = safe_number(
        value,
        0,
    )

    if 0 <= number <= 1:
        number *= 100

    return clamp(
        number
    )


def get_probabilities(
    result: Dict[str, Any],
) -> Tuple[float, float]:
    probabilities = (
        result.get("probabilities")
        or {}
    )

    ai_probability = (
        probabilities.get("ai")
        if isinstance(
            probabilities,
            dict,
        )
        else None
    )

    human_probability = (
        probabilities.get("human")
        if isinstance(
            probabilities,
            dict,
        )
        else None
    )

    if ai_probability is None:
        ai_probability = (
            result.get(
                "raw_ai_probability"
            )
        )

    if ai_probability is None:
        ai_probability = (
            result.get(
                "raw_probability_fake"
            )
        )

    if ai_probability is None:
        ai_probability = (
            result.get(
                "risk_score"
            )
        )

    ai_probability = (
        normalize_probability(
            ai_probability
        )
    )

    if human_probability is None:
        human_probability = (
            result.get(
                "raw_human_probability"
            )
        )

    if human_probability is None:
        human_probability = (
            result.get(
                "raw_probability_real"
            )
        )

    if human_probability is None:
        human_probability = (
            100 - ai_probability
        )

    human_probability = (
        normalize_probability(
            human_probability
        )
    )

    return (
        ai_probability,
        human_probability,
    )


def risk_colors(
    risk: Any,
) -> Tuple[
    colors.Color,
    colors.Color,
]:
    normalized = str(
        risk or ""
    ).upper()

    if "HIGH" in normalized:
        return RED, LIGHT_RED

    if "MEDIUM" in normalized:
        return AMBER, LIGHT_AMBER

    return GREEN, LIGHT_GREEN


# =========================================================
# PATH RESOLUTION
# =========================================================

def resolve_local_path(
    value: Any,
) -> Optional[str]:
    if not value:
        return None

    raw_path = str(
        value
    ).strip()

    if raw_path.startswith(
        "http://"
    ) or raw_path.startswith(
        "https://"
    ):
        return None

    if os.path.isfile(
        raw_path
    ):
        return raw_path

    if raw_path.startswith(
        "/uploads/"
    ):
        candidate = os.path.join(
            PROJECT_ROOT,
            raw_path.lstrip("/"),
        )

        if os.path.isfile(
            candidate
        ):
            return candidate

    if raw_path.startswith(
        "/backend-uploads/"
    ):
        relative = raw_path.replace(
            "/backend-uploads/",
            "",
            1,
        )

        candidate = os.path.join(
            PROJECT_ROOT,
            "backend",
            "uploads",
            relative,
        )

        if os.path.isfile(
            candidate
        ):
            return candidate

    if raw_path.startswith(
        "/audio-visuals/"
    ):
        relative = raw_path.replace(
            "/audio-visuals/",
            "",
            1,
        )

        candidate = os.path.join(
            PROJECT_ROOT,
            "backend",
            "audio_visuals",
            relative,
        )

        if os.path.isfile(
            candidate
        ):
            return candidate

    if raw_path.startswith(
        "/reports/"
    ):
        candidate = os.path.join(
            PROJECT_ROOT,
            raw_path.lstrip("/"),
        )

        if os.path.isfile(
            candidate
        ):
            return candidate

    candidate = os.path.join(
        PROJECT_ROOT,
        raw_path.lstrip("/"),
    )

    if os.path.isfile(
        candidate
    ):
        return candidate

    return None


# =========================================================
# STYLES
# =========================================================

def build_styles() -> Dict[str, ParagraphStyle]:
    styles = getSampleStyleSheet()

    return {
        "cover_title": ParagraphStyle(
            "ForgeCoverTitle",
            parent=styles["Title"],
            fontName="Helvetica-Bold",
            fontSize=25,
            leading=31,
            textColor=WHITE,
            alignment=TA_CENTER,
            spaceAfter=16,
        ),

        "cover_subtitle": ParagraphStyle(
            "ForgeCoverSubtitle",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=12,
            leading=18,
            textColor=LIGHT_CYAN,
            alignment=TA_CENTER,
        ),

        "section": ParagraphStyle(
            "ForgeSection",
            parent=styles["Heading1"],
            fontName="Helvetica-Bold",
            fontSize=17,
            leading=21,
            textColor=NAVY,
            spaceBefore=8,
            spaceAfter=12,
        ),

        "subsection": ParagraphStyle(
            "ForgeSubsection",
            parent=styles["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=13,
            leading=17,
            textColor=BLUE,
            spaceBefore=7,
            spaceAfter=8,
        ),

        "body": ParagraphStyle(
            "ForgeBody",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=9.4,
            leading=14,
            textColor=TEXT_GREY,
            alignment=TA_LEFT,
            spaceAfter=7,
        ),

        "small": ParagraphStyle(
            "ForgeSmall",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=7.7,
            leading=11,
            textColor=TEXT_GREY,
        ),

        "table_header": ParagraphStyle(
            "ForgeTableHeader",
            parent=styles["BodyText"],
            fontName="Helvetica-Bold",
            fontSize=8,
            leading=10,
            textColor=WHITE,
            alignment=TA_LEFT,
        ),

        "table_cell": ParagraphStyle(
            "ForgeTableCell",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=7.8,
            leading=10.5,
            textColor=TEXT_GREY,
        ),

        "verdict": ParagraphStyle(
            "ForgeVerdict",
            parent=styles["Heading1"],
            fontName="Helvetica-Bold",
            fontSize=21,
            leading=26,
            alignment=TA_CENTER,
        ),

        "disclaimer": ParagraphStyle(
            "ForgeDisclaimer",
            parent=styles["BodyText"],
            fontName="Helvetica-Oblique",
            fontSize=7.5,
            leading=10,
            textColor=colors.HexColor(
                "#64748B"
            ),
        ),
    }


# =========================================================
# PAGE DECORATION
# =========================================================

def draw_header_footer(
    canvas,
    document,
) -> None:
    canvas.saveState()

    page_number = (
        canvas.getPageNumber()
    )

    canvas.setFillColor(
        DARK_NAVY
    )

    canvas.rect(
        0,
        PAGE_HEIGHT - 17 * mm,
        PAGE_WIDTH,
        17 * mm,
        stroke=0,
        fill=1,
    )

    canvas.setFillColor(
        WHITE
    )

    canvas.setFont(
        "Helvetica-Bold",
        10,
    )

    canvas.drawString(
        18 * mm,
        PAGE_HEIGHT - 10.5 * mm,
        "FORGE",
    )

    canvas.setFont(
        "Helvetica",
        7.5,
    )

    canvas.setFillColor(
        LIGHT_CYAN
    )

    canvas.drawRightString(
        PAGE_WIDTH - 18 * mm,
        PAGE_HEIGHT - 10.5 * mm,
        "Multimodal Digital Forensic Examination",
    )

    canvas.setStrokeColor(
        MID_GREY
    )

    canvas.line(
        18 * mm,
        15 * mm,
        PAGE_WIDTH - 18 * mm,
        15 * mm,
    )

    canvas.setFont(
        "Helvetica",
        7,
    )

    canvas.setFillColor(
        TEXT_GREY
    )

    canvas.drawString(
        18 * mm,
        10 * mm,
        REPORT_ENGINE_VERSION,
    )

    canvas.drawRightString(
        PAGE_WIDTH - 18 * mm,
        10 * mm,
        f"Page {page_number}",
    )

    canvas.restoreState()


def draw_cover_background(
    canvas,
    document,
) -> None:
    canvas.saveState()

    canvas.setFillColor(
        DARK_NAVY
    )

    canvas.rect(
        0,
        0,
        PAGE_WIDTH,
        PAGE_HEIGHT,
        stroke=0,
        fill=1,
    )

    canvas.setFillColor(
        colors.HexColor(
            "#0C3049"
        )
    )

    canvas.circle(
        PAGE_WIDTH * 0.15,
        PAGE_HEIGHT * 0.82,
        75 * mm,
        stroke=0,
        fill=1,
    )

    canvas.setFillColor(
        colors.HexColor(
            "#123B5C"
        )
    )

    canvas.circle(
        PAGE_WIDTH * 0.93,
        PAGE_HEIGHT * 0.15,
        85 * mm,
        stroke=0,
        fill=1,
    )

    canvas.restoreState()


def first_page(
    canvas,
    document,
) -> None:
    draw_cover_background(
        canvas,
        document,
    )


def later_pages(
    canvas,
    document,
) -> None:
    draw_header_footer(
        canvas,
        document,
    )


# =========================================================
# TABLE HELPERS
# =========================================================

def paragraph_cell(
    value: Any,
    style: ParagraphStyle,
) -> Paragraph:
    return Paragraph(
        escape_text(
            value
        ),
        style,
    )


def make_key_value_table(
    rows: Iterable[
        Tuple[Any, Any]
    ],
    styles: Dict[
        str,
        ParagraphStyle,
    ],
    widths: Optional[
        List[float]
    ] = None,
) -> Table:
    data = []

    for key, value in rows:
        data.append(
            [
                Paragraph(
                    f"<b>{escape_text(key)}</b>",
                    styles["table_cell"],
                ),

                paragraph_cell(
                    value,
                    styles["table_cell"],
                ),
            ]
        )

    table = Table(
        data,
        colWidths=(
            widths
            or [
                48 * mm,
                115 * mm,
            ]
        ),
        hAlign="LEFT",
    )

    table.setStyle(
        TableStyle(
            [
                (
                    "BACKGROUND",
                    (0, 0),
                    (0, -1),
                    LIGHT_GREY,
                ),

                (
                    "GRID",
                    (0, 0),
                    (-1, -1),
                    0.45,
                    MID_GREY,
                ),

                (
                    "VALIGN",
                    (0, 0),
                    (-1, -1),
                    "TOP",
                ),

                (
                    "LEFTPADDING",
                    (0, 0),
                    (-1, -1),
                    7,
                ),

                (
                    "RIGHTPADDING",
                    (0, 0),
                    (-1, -1),
                    7,
                ),

                (
                    "TOPPADDING",
                    (0, 0),
                    (-1, -1),
                    6,
                ),

                (
                    "BOTTOMPADDING",
                    (0, 0),
                    (-1, -1),
                    6,
                ),
            ]
        )
    )

    return table


def make_data_table(
    headers: List[str],
    rows: List[List[Any]],
    styles: Dict[
        str,
        ParagraphStyle,
    ],
    col_widths: Optional[
        List[float]
    ] = None,
) -> Table:
    data = [
        [
            paragraph_cell(
                header,
                styles["table_header"],
            )
            for header in headers
        ]
    ]

    for row in rows:
        data.append(
            [
                paragraph_cell(
                    value,
                    styles["table_cell"],
                )
                for value in row
            ]
        )

    table = Table(
        data,
        colWidths=col_widths,
        repeatRows=1,
        hAlign="LEFT",
    )

    table.setStyle(
        TableStyle(
            [
                (
                    "BACKGROUND",
                    (0, 0),
                    (-1, 0),
                    NAVY,
                ),

                (
                    "TEXTCOLOR",
                    (0, 0),
                    (-1, 0),
                    WHITE,
                ),

                (
                    "GRID",
                    (0, 0),
                    (-1, -1),
                    0.4,
                    MID_GREY,
                ),

                (
                    "ROWBACKGROUNDS",
                    (0, 1),
                    (-1, -1),
                    [
                        WHITE,
                        LIGHT_GREY,
                    ],
                ),

                (
                    "VALIGN",
                    (0, 0),
                    (-1, -1),
                    "TOP",
                ),

                (
                    "LEFTPADDING",
                    (0, 0),
                    (-1, -1),
                    5,
                ),

                (
                    "RIGHTPADDING",
                    (0, 0),
                    (-1, -1),
                    5,
                ),

                (
                    "TOPPADDING",
                    (0, 0),
                    (-1, -1),
                    5,
                ),

                (
                    "BOTTOMPADDING",
                    (0, 0),
                    (-1, -1),
                    5,
                ),
            ]
        )
    )

    return table


# =========================================================
# CHART HELPERS
# =========================================================

def create_probability_chart(
    ai_probability: float,
    human_probability: float,
    output_directory: str,
) -> str:
    output_path = os.path.join(
        output_directory,
        "probability_chart.png",
    )

    figure, axis = plt.subplots(
        figsize=(7.2, 2.7)
    )

    labels = [
        "AI / Fake",
        "Human / Real",
    ]

    values = [
        ai_probability,
        human_probability,
    ]

    bars = axis.barh(
        labels,
        values,
    )

    axis.set_xlim(
        0,
        100,
    )

    axis.set_xlabel(
        "Probability (%)"
    )

    axis.set_title(
        "Model Probability Distribution"
    )

    axis.grid(
        axis="x",
        alpha=0.25,
    )

    for bar, value in zip(
        bars,
        values,
    ):
        axis.text(
            min(
                value + 1,
                96,
            ),
            bar.get_y()
            + bar.get_height() / 2,
            f"{value:.2f}%",
            va="center",
            fontsize=9,
        )

    figure.tight_layout()

    figure.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )

    return output_path


def create_parameter_chart(
    parameter_contribution: Dict[
        str,
        Any,
    ],
    output_directory: str,
) -> Optional[str]:
    if not parameter_contribution:
        return None

    labels = []
    scores = []

    for key, value in (
        parameter_contribution.items()
    ):
        labels.append(
            format_key(
                key
            )
        )

        if isinstance(
            value,
            dict,
        ):
            score = value.get(
                "score",
                0,
            )
        else:
            score = value

        scores.append(
            clamp(
                safe_number(
                    score
                )
            )
        )

    if not labels:
        return None

    output_path = os.path.join(
        output_directory,
        "parameter_chart.png",
    )

    figure_height = max(
        3,
        len(labels) * 0.38,
    )

    figure, axis = plt.subplots(
        figsize=(
            7.2,
            figure_height,
        )
    )

    positions = list(
        range(
            len(labels)
        )
    )

    bars = axis.barh(
        positions,
        scores,
    )

    axis.set_yticks(
        positions,
        labels,
        fontsize=8,
    )

    axis.invert_yaxis()

    axis.set_xlim(
        0,
        100,
    )

    axis.set_xlabel(
        "Contribution score (%)"
    )

    axis.set_title(
        "Explainable Parameter Contribution"
    )

    axis.grid(
        axis="x",
        alpha=0.25,
    )

    for bar, value in zip(
        bars,
        scores,
    ):
        axis.text(
            min(
                value + 1,
                96,
            ),
            bar.get_y()
            + bar.get_height() / 2,
            f"{value:.1f}%",
            va="center",
            fontsize=7.5,
        )

    figure.tight_layout()

    figure.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )

    return output_path


def create_region_chart(
    region_analysis: Dict[
        str,
        Any,
    ],
    output_directory: str,
) -> Optional[str]:
    ranked = (
        region_analysis.get(
            "ranked_regions"
        )
        or region_analysis.get(
            "regions"
        )
        or []
    )

    if not ranked:
        return None

    ranked = sorted(
        ranked,
        key=lambda item: safe_number(
            item.get(
                "forgery_score",
                0,
            )
        ),
        reverse=True,
    )[:10]

    labels = [
        item.get(
            "name",
            "Region",
        )
        for item in ranked
    ]

    scores = [
        clamp(
            safe_number(
                item.get(
                    "forgery_score",
                    0,
                )
            )
        )
        for item in ranked
    ]

    output_path = os.path.join(
        output_directory,
        "regional_chart.png",
    )

    figure, axis = plt.subplots(
        figsize=(
            7.2,
            max(
                3,
                len(labels) * 0.42,
            ),
        )
    )

    positions = list(
        range(
            len(labels)
        )
    )

    bars = axis.barh(
        positions,
        scores,
    )

    axis.set_yticks(
        positions,
        labels,
        fontsize=8,
    )

    axis.invert_yaxis()

    axis.set_xlim(
        0,
        100,
    )

    axis.set_xlabel(
        "Regional suspicion score (%)"
    )

    axis.set_title(
        "Ranked Suspicious Image Regions"
    )

    axis.grid(
        axis="x",
        alpha=0.25,
    )

    for bar, score in zip(
        bars,
        scores,
    ):
        axis.text(
            min(
                score + 1,
                96,
            ),
            bar.get_y()
            + bar.get_height() / 2,
            f"{score:.1f}%",
            va="center",
            fontsize=7.5,
        )

    figure.tight_layout()

    figure.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )

    return output_path


def create_sentence_chart(
    sentences: List[
        Dict[str, Any]
    ],
    output_directory: str,
) -> Optional[str]:
    if not sentences:
        return None

    limited = sentences[:30]

    scores = []

    for item in limited:
        score = (
            item.get("score")
            or item.get("confidence")
            or item.get("ai_score")
            or 0
        )

        scores.append(
            clamp(
                safe_number(
                    score
                )
            )
        )

    if not scores:
        return None

    output_path = os.path.join(
        output_directory,
        "sentence_chart.png",
    )

    figure, axis = plt.subplots(
        figsize=(
            7.2,
            3.1,
        )
    )

    axis.plot(
        range(
            1,
            len(scores) + 1,
        ),
        scores,
        marker="o",
        linewidth=1.5,
        markersize=3.5,
    )

    axis.axhline(
        75,
        linestyle="--",
        linewidth=1,
        alpha=0.7,
    )

    axis.axhline(
        45,
        linestyle="--",
        linewidth=1,
        alpha=0.7,
    )

    axis.set_ylim(
        0,
        100,
    )

    axis.set_xlabel(
        "Sentence number"
    )

    axis.set_ylabel(
        "AI suspicion score (%)"
    )

    axis.set_title(
        "Sentence Suspicion Timeline"
    )

    axis.grid(
        alpha=0.25,
    )

    figure.tight_layout()

    figure.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )

    return output_path


def create_audio_timeline_chart(
    segments: List[
        Dict[str, Any]
    ],
    output_directory: str,
) -> Optional[str]:
    if not segments:
        return None

    scores = []

    for segment in segments:
        score = (
            segment.get("score")
            or segment.get("confidence")
            or segment.get("risk_score")
            or 0
        )

        scores.append(
            clamp(
                safe_number(
                    score
                )
            )
        )

    if not scores:
        return None

    output_path = os.path.join(
        output_directory,
        "audio_timeline_chart.png",
    )

    figure, axis = plt.subplots(
        figsize=(
            7.2,
            3,
        )
    )

    axis.bar(
        range(
            1,
            len(scores) + 1,
        ),
        scores,
    )

    axis.set_ylim(
        0,
        100,
    )

    axis.set_xlabel(
        "Suspicious interval"
    )

    axis.set_ylabel(
        "Suspicion score (%)"
    )

    axis.set_title(
        "Suspicious Audio Interval Timeline"
    )

    axis.grid(
        axis="y",
        alpha=0.25,
    )

    figure.tight_layout()

    figure.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )

    return output_path



# =========================================================
# ADVANCED FORENSIC CHARTS AND LEGENDS
# =========================================================

def safe_list(value: Any) -> List[Any]:
    return value if isinstance(value, list) else []


def safe_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def extract_curve_points(result: Dict[str, Any], key: str) -> List[Dict[str, float]]:
    curves = safe_dict(result.get("audio_curves"))
    if not curves:
        curves = safe_dict(safe_dict(result.get("advanced_audio_analysis")).get("curves"))

    raw_points = safe_list(curves.get(key))
    points: List[Dict[str, float]] = []

    for item in raw_points:
        if not isinstance(item, dict):
            continue
        time_value = safe_number(item.get("time"), float("nan"))
        curve_value = safe_number(item.get("value"), float("nan"))
        if not (time_value == time_value and curve_value == curve_value):
            continue
        points.append({"time": time_value, "value": curve_value})

    return points


def create_audio_curve_chart(
    points: List[Dict[str, float]],
    title: str,
    y_label: str,
    output_directory: str,
    filename: str,
) -> Optional[str]:
    if len(points) < 2:
        return None

    times = [safe_number(item.get("time")) for item in points]
    values = [safe_number(item.get("value")) for item in points]

    output_path = os.path.join(output_directory, filename)
    figure, axis = plt.subplots(figsize=(7.2, 3.0))

    axis.plot(times, values, linewidth=1.45)
    axis.fill_between(times, values, alpha=0.12)
    axis.set_xlabel("Time (seconds)")
    axis.set_ylabel(y_label)
    axis.set_title(title)
    axis.grid(alpha=0.24)

    if values:
        minimum = min(values)
        maximum = max(values)
        if abs(maximum - minimum) < 1e-10:
            padding = max(abs(maximum) * 0.1, 0.01)
            axis.set_ylim(minimum - padding, maximum + padding)

    figure.tight_layout()
    figure.savefig(output_path, dpi=190, bbox_inches="tight")
    plt.close(figure)
    return output_path


def create_voice_dna_chart(
    voice_dna: Dict[str, Any],
    output_directory: str,
) -> Optional[str]:
    if not voice_dna:
        return None

    labels: List[str] = []
    scores: List[float] = []

    for key, value in voice_dna.items():
        if not isinstance(value, dict):
            continue
        labels.append(format_key(key))
        scores.append(clamp(safe_number(value.get("score"))))

    if not labels:
        return None

    output_path = os.path.join(output_directory, "voice_dna_chart.png")
    figure, axis = plt.subplots(figsize=(7.2, max(3.2, len(labels) * 0.42)))
    positions = list(range(len(labels)))
    bars = axis.barh(positions, scores)
    axis.set_yticks(positions, labels, fontsize=8)
    axis.invert_yaxis()
    axis.set_xlim(0, 100)
    axis.set_xlabel("Score (%)")
    axis.set_title("Voice DNA and Synthetic Signature Profile")
    axis.grid(axis="x", alpha=0.24)

    for bar, score in zip(bars, scores):
        axis.text(min(score + 1, 96), bar.get_y() + bar.get_height() / 2,
                  f"{score:.1f}%", va="center", fontsize=7.5)

    figure.tight_layout()
    figure.savefig(output_path, dpi=190, bbox_inches="tight")
    plt.close(figure)
    return output_path


def build_heatmap_legend(
    styles: Dict[str, ParagraphStyle],
    title: str = "Forensic Heatmap Interpretation",
) -> List[Any]:
    rows = [
        ["Blue", "0-20%", "Low anomaly", "The region is comparatively consistent with surrounding evidence."],
        ["Cyan / Green", "20-40%", "Mild deviation", "A small statistical deviation is present; usually insufficient on its own."],
        ["Yellow", "40-60%", "Review zone", "Moderate anomaly requiring comparison with model and metadata evidence."],
        ["Orange", "60-80%", "High anomaly", "Strong inconsistency that may indicate editing, synthesis or compression artefacts."],
        ["Red", "80-100%", "Critical anomaly", "The strongest model attention or anomaly intensity in the examined evidence."],
    ]

    return [
        Paragraph(title, styles["subsection"]),
        make_data_table(
            ["Colour", "Indicative range", "Meaning", "How to interpret"],
            rows,
            styles,
            col_widths=[22 * mm, 27 * mm, 31 * mm, 89 * mm],
        ),
        Spacer(1, 3 * mm),
        Paragraph(
            "Heatmap colours represent relative model attention or anomaly intensity, not a definitive pixel-level manipulation mask. "
            "Interpret them together with the final model probability, regional metrics, metadata and source context.",
            styles["disclaimer"],
        ),
    ]


def build_audio_heatmap_legend(styles: Dict[str, ParagraphStyle]) -> List[Any]:
    rows = [
        ["Blue / Dark", "Low acoustic anomaly", "Stable or comparatively natural acoustic behaviour."],
        ["Green", "Low-to-moderate anomaly", "Minor deviation in energy, pitch or spectral behaviour."],
        ["Yellow", "Review zone", "Potential synthetic or edited pattern requiring timeline inspection."],
        ["Orange", "High anomaly", "Strong discontinuity, spectral inconsistency or vocoder-like behaviour."],
        ["Red", "Critical anomaly", "Highest synthetic-speech suspicion or acoustic inconsistency."],
    ]
    return [
        Paragraph("Audio Heatmap Interpretation", styles["subsection"]),
        make_data_table(
            ["Colour", "Meaning", "Forensic interpretation"],
            rows,
            styles,
            col_widths=[28 * mm, 46 * mm, 95 * mm],
        ),
        Spacer(1, 3 * mm),
        Paragraph(
            "Audio heatmap intensity is an explainability aid. It does not independently prove voice cloning or editing. "
            "Use the suspicious timeline, pitch, energy, spectral flux, spectral flatness and Voice DNA findings together.",
            styles["disclaimer"],
        ),
    ]

# =========================================================
# IMAGE HELPERS
# =========================================================

def scaled_report_image(
    image_path: str,
    max_width: float,
    max_height: float,
) -> Optional[Image]:
    if not (
        image_path
        and os.path.isfile(
            image_path
        )
    ):
        return None

    try:
        image = Image(
            image_path
        )

        scale = min(
            max_width / image.imageWidth,
            max_height / image.imageHeight,
        )

        image.drawWidth = (
            image.imageWidth
            * scale
        )

        image.drawHeight = (
            image.imageHeight
            * scale
        )

        return image

    except Exception:
        return None


def image_panel(
    title: str,
    path: Optional[str],
    styles: Dict[
        str,
        ParagraphStyle,
    ],
    max_width: float = 78 * mm,
    max_height: float = 62 * mm,
) -> List[Any]:
    elements: List[Any] = [
        Paragraph(
            escape_text(
                title
            ),
            styles["subsection"],
        )
    ]

    image = (
        scaled_report_image(
            path,
            max_width,
            max_height,
        )
        if path
        else None
    )

    if image:
        elements.append(
            image
        )
    else:
        elements.append(
            Paragraph(
                "Visual evidence was not available for embedding.",
                styles["small"],
            )
        )

    return elements


# =========================================================
# REPORT SECTIONS
# =========================================================

def build_cover(
    result: Dict[
        str,
        Any,
    ],
    styles: Dict[
        str,
        ParagraphStyle,
    ],
) -> List[Any]:
    modality = get_modality(
        result
    ).upper()

    prediction = result.get(
        "prediction",
        "UNKNOWN",
    )

    confidence = safe_number(
        result.get(
            "confidence",
            0,
        )
    )

    risk = result.get(
        "risk_level",
        "N/A",
    )

    case_id = result.get(
        "case_id",
        "Not assigned",
    )

    evidence = (
        result.get(
            "evidence"
        )
        or {}
    )

    generated_at = (
        evidence.get(
            "analysis_timestamp_utc"
        )
        or datetime.now(
            timezone.utc
        ).isoformat()
    )

    risk_color, risk_background = (
        risk_colors(
            risk
        )
    )

    verdict_style = ParagraphStyle(
        "DynamicVerdict",
        parent=styles["verdict"],
        textColor=risk_color,
    )

    cover_table = Table(
        [
            [
                Paragraph(
                    "<b>CASE ID</b>",
                    styles["table_header"],
                ),
                Paragraph(
                    escape_text(
                        case_id
                    ),
                    styles["table_cell"],
                ),
            ],

            [
                Paragraph(
                    "<b>MODALITY</b>",
                    styles["table_header"],
                ),
                Paragraph(
                    escape_text(
                        modality
                    ),
                    styles["table_cell"],
                ),
            ],

            [
                Paragraph(
                    "<b>ANALYSIS TIME</b>",
                    styles["table_header"],
                ),
                Paragraph(
                    escape_text(
                        generated_at
                    ),
                    styles["table_cell"],
                ),
            ],

            [
                Paragraph(
                    "<b>ENGINE VERSION</b>",
                    styles["table_header"],
                ),
                Paragraph(
                    escape_text(
                        result.get(
                            "analysis_version",
                            REPORT_ENGINE_VERSION,
                        )
                    ),
                    styles["table_cell"],
                ),
            ],
        ],
        colWidths=[
            48 * mm,
            102 * mm,
        ],
        hAlign="CENTER",
    )

    cover_table.setStyle(
        TableStyle(
            [
                (
                    "BACKGROUND",
                    (0, 0),
                    (0, -1),
                    colors.HexColor(
                        "#164A68"
                    ),
                ),

                (
                    "BACKGROUND",
                    (1, 0),
                    (1, -1),
                    colors.HexColor(
                        "#E8F6FA"
                    ),
                ),

                (
                    "GRID",
                    (0, 0),
                    (-1, -1),
                    0.45,
                    colors.HexColor(
                        "#4B7C98"
                    ),
                ),

                (
                    "VALIGN",
                    (0, 0),
                    (-1, -1),
                    "MIDDLE",
                ),

                (
                    "LEFTPADDING",
                    (0, 0),
                    (-1, -1),
                    8,
                ),

                (
                    "RIGHTPADDING",
                    (0, 0),
                    (-1, -1),
                    8,
                ),

                (
                    "TOPPADDING",
                    (0, 0),
                    (-1, -1),
                    8,
                ),

                (
                    "BOTTOMPADDING",
                    (0, 0),
                    (-1, -1),
                    8,
                ),
            ]
        )
    )

    verdict_table = Table(
        [
            [
                Paragraph(
                    escape_text(
                        prediction
                    ),
                    verdict_style,
                )
            ],

            [
                Paragraph(
                    (
                        f"<b>Confidence:</b> "
                        f"{confidence:.2f}%"
                        f"&nbsp;&nbsp;&nbsp;&nbsp;"
                        f"<b>Risk:</b> "
                        f"{escape_text(risk)}"
                    ),
                    styles["body"],
                )
            ],
        ],
        colWidths=[
            145 * mm
        ],
        hAlign="CENTER",
    )

    verdict_table.setStyle(
        TableStyle(
            [
                (
                    "BACKGROUND",
                    (0, 0),
                    (-1, -1),
                    risk_background,
                ),

                (
                    "BOX",
                    (0, 0),
                    (-1, -1),
                    1,
                    risk_color,
                ),

                (
                    "ALIGN",
                    (0, 0),
                    (-1, -1),
                    "CENTER",
                ),

                (
                    "TOPPADDING",
                    (0, 0),
                    (-1, -1),
                    12,
                ),

                (
                    "BOTTOMPADDING",
                    (0, 0),
                    (-1, -1),
                    12,
                ),
            ]
        )
    )

    return [
        Spacer(
            1,
            32 * mm,
        ),

        Paragraph(
            "FORGE",
            styles["cover_title"],
        ),

        Paragraph(
            REPORT_TITLE,
            styles["cover_title"],
        ),

        Spacer(
            1,
            7 * mm,
        ),

        Paragraph(
            (
                "Explainable artificial intelligence "
                "for text, image and audio forensic examination"
            ),
            styles["cover_subtitle"],
        ),

        Spacer(
            1,
            18 * mm,
        ),

        cover_table,

        Spacer(
            1,
            14 * mm,
        ),

        verdict_table,

        Spacer(
            1,
            15 * mm,
        ),

        Paragraph(
            (
                "This report records model outputs, "
                "forensic indicators and explainability evidence. "
                "The findings should be interpreted with contextual, "
                "technical and investigative evidence."
            ),
            ParagraphStyle(
                "CoverDisclaimer",
                parent=styles["cover_subtitle"],
                fontSize=8,
                leading=12,
                textColor=colors.HexColor(
                    "#A9D7E4"
                ),
            ),
        ),

        PageBreak(),
    ]


def build_executive_summary(
    result: Dict[
        str,
        Any,
    ],
    styles: Dict[
        str,
        ParagraphStyle,
    ],
    temporary_directory: str,
) -> List[Any]:
    prediction = result.get(
        "prediction",
        "UNKNOWN",
    )

    confidence = safe_number(
        result.get(
            "confidence",
            0,
        )
    )

    risk_level = result.get(
        "risk_level",
        "N/A",
    )

    risk_score = safe_number(
        result.get(
            "risk_score",
            0,
        )
    )

    modality = get_modality(
        result
    )

    ai_probability, human_probability = (
        get_probabilities(
            result
        )
    )

    recommendation = (
        result.get(
            "recommendation"
        )
        or (
            "Review the explainability evidence and "
            "modality-specific findings before reaching "
            "a final investigative conclusion."
        )
    )

    probability_chart = (
        create_probability_chart(
            ai_probability,
            human_probability,
            temporary_directory,
        )
    )

    summary_text = (
        f"The FORGE {modality} forensic engine classified "
        f"the submitted evidence as "
        f"<b>{escape_text(prediction)}</b> with "
        f"<b>{confidence:.2f}% confidence</b>. "
        f"The examination produced a risk level of "
        f"<b>{escape_text(risk_level)}</b> and a risk score of "
        f"<b>{risk_score:.2f}%</b>. "
        f"The result is supported by explainable feature groups, "
        f"visual or temporal evidence, and modality-specific "
        f"forensic indicators."
    )

    decision_table = make_key_value_table(
        [
            (
                "Prediction",
                prediction,
            ),

            (
                "Confidence",
                format_percent(
                    confidence
                ),
            ),

            (
                "Risk level",
                risk_level,
            ),

            (
                "Risk score",
                format_percent(
                    risk_score
                ),
            ),

            (
                "AI / Fake probability",
                format_percent(
                    ai_probability
                ),
            ),

            (
                "Human / Real probability",
                format_percent(
                    human_probability
                ),
            ),

            (
                "Decision strength",
                result.get(
                    "decision_strength",
                    "Computed",
                ),
            ),
        ],
        styles,
    )

    elements: List[Any] = [
        Paragraph(
            "1. Executive Summary",
            styles["section"],
        ),

        Paragraph(
            summary_text,
            styles["body"],
        ),

        Spacer(
            1,
            3 * mm,
        ),

        decision_table,

        Spacer(
            1,
            5 * mm,
        ),
    ]

    chart = scaled_report_image(
        probability_chart,
        160 * mm,
        62 * mm,
    )

    if chart:
        elements.append(
            chart
        )

    elements.extend(
        [
            Spacer(
                1,
                4 * mm,
            ),

            Paragraph(
                "Investigator Interpretation",
                styles["subsection"],
            ),

            Paragraph(
                escape_text(
                    recommendation
                ),
                styles["body"],
            ),
        ]
    )

    return elements


def build_evidence_section(
    result: Dict[
        str,
        Any,
    ],
    styles: Dict[
        str,
        ParagraphStyle,
    ],
) -> List[Any]:
    evidence = (
        result.get(
            "evidence"
        )
        or {}
    )

    rows = [
        (
            "Case ID",
            result.get(
                "case_id",
                "Not assigned",
            ),
        ),

        (
            "Original filename",
            evidence.get(
                "original_filename",
                "Not available",
            ),
        ),

        (
            "Modality",
            get_modality(
                result
            ).upper(),
        ),

        (
            "MIME type",
            evidence.get(
                "mime_type",
                "Not available",
            ),
        ),

        (
            "File size",
            format_file_size(
                evidence.get(
                    "size_bytes"
                )
            ),
        ),

        (
            "Analysis timestamp",
            evidence.get(
                "analysis_timestamp_utc",
                datetime.now(
                    timezone.utc
                ).isoformat(),
            ),
        ),

        (
            "Analysis version",
            result.get(
                "analysis_version",
                REPORT_ENGINE_VERSION,
            ),
        ),

        (
            "SHA-256",
            evidence.get(
                "sha256",
                "Not available",
            ),
        ),
    ]

    return [
        Paragraph(
            "2. Evidence and Integrity Information",
            styles["section"],
        ),

        make_key_value_table(
            rows,
            styles,
        ),

        Spacer(
            1,
            5 * mm,
        ),

        Paragraph(
            (
                "The SHA-256 value, when available, identifies "
                "the exact file processed during this examination. "
                "Any modification to the file will result in a different hash."
            ),
            styles["disclaimer"],
        ),
    ]


def build_parameter_section(
    result: Dict[
        str,
        Any,
    ],
    styles: Dict[
        str,
        ParagraphStyle,
    ],
    temporary_directory: str,
) -> List[Any]:
    parameter_contribution = (
        result.get(
            "parameter_contribution"
        )
        or {}
    )

    if not parameter_contribution:
        return []

    chart_path = (
        create_parameter_chart(
            parameter_contribution,
            temporary_directory,
        )
    )

    rows = []

    for key, value in (
        parameter_contribution.items()
    ):
        if isinstance(
            value,
            dict,
        ):
            score = value.get(
                "score",
                0,
            )

            risk = value.get(
                "risk",
                "N/A",
            )

            reason = value.get(
                "reason",
                "No explanation was generated.",
            )
        else:
            score = value
            risk = "N/A"
            reason = (
                "No detailed explanation "
                "was generated."
            )

        rows.append(
            [
                format_key(
                    key
                ),

                format_percent(
                    score
                ),

                risk,

                reason,
            ]
        )

    elements: List[Any] = [
        Paragraph(
            "3. Explainable Parameter Analysis",
            styles["section"],
        )
    ]

    chart = scaled_report_image(
        chart_path,
        160 * mm,
        80 * mm,
    )

    if chart:
        elements.extend(
            [
                chart,
                Spacer(
                    1,
                    4 * mm,
                ),
            ]
        )

    elements.append(
        make_data_table(
            [
                "Parameter group",
                "Score",
                "Risk",
                "Forensic explanation",
            ],
            rows,
            styles,
            col_widths=[
                36 * mm,
                21 * mm,
                20 * mm,
                88 * mm,
            ],
        )
    )

    fusion = (
        result.get(
            "fusion_breakdown"
        )
        or {}
    )

    if fusion:
        elements.extend(
            [
                Spacer(
                    1,
                    5 * mm,
                ),

                Paragraph(
                    "Model Fusion Breakdown",
                    styles["subsection"],
                ),

                make_key_value_table(
                    [
                        (
                            format_key(
                                key
                            ),
                            value,
                        )
                        for key, value
                        in fusion.items()
                    ],
                    styles,
                ),
            ]
        )

    return elements


# =========================================================
# TEXT REPORT
# =========================================================

def extract_text_sentences(
    result: Dict[
        str,
        Any,
    ],
) -> List[Dict[str, Any]]:
    document = (
        result.get(
            "full_document"
        )
        or result.get(
            "highlighted_document"
        )
        or result.get(
            "highlighted_sentences"
        )
        or []
    )

    if not isinstance(
        document,
        list,
    ):
        return []

    normalized = []

    for index, item in enumerate(
        document,
        start=1,
    ):
        if isinstance(
            item,
            dict,
        ):
            sentence = (
                item.get(
                    "sentence"
                )
                or item.get(
                    "text"
                )
                or item.get(
                    "content"
                )
                or ""
            )

            score = (
                item.get(
                    "score"
                )
                or item.get(
                    "confidence"
                )
                or item.get(
                    "ai_score"
                )
                or 0
            )

            risk = (
                item.get(
                    "risk"
                )
                or item.get(
                    "level"
                )
                or (
                    "HIGH"
                    if safe_number(
                        score
                    ) >= 75
                    else "MEDIUM"
                    if safe_number(
                        score
                    ) >= 45
                    else "LOW"
                )
            )

            reason = (
                item.get(
                    "reason"
                )
                or item.get(
                    "explanation"
                )
                or "No sentence-level reason was generated."
            )
        else:
            sentence = str(
                item
            )

            score = 0
            risk = "LOW"
            reason = (
                "No sentence-level reason "
                "was generated."
            )

        normalized.append(
            {
                "index": index,
                "sentence": sentence,
                "score": safe_number(
                    score
                ),
                "risk": risk,
                "reason": reason,
            }
        )

    return normalized


def build_text_section(
    result: Dict[
        str,
        Any,
    ],
    styles: Dict[
        str,
        ParagraphStyle,
    ],
    temporary_directory: str,
) -> List[Any]:
    sentences = extract_text_sentences(
        result
    )

    elements: List[Any] = [
        PageBreak(),

        Paragraph(
            "4. Text Forensic Examination",
            styles["section"],
        ),

        Paragraph(
            (
                "The text examination combines stylometric, lexical, "
                "phrase-pattern and semantic evidence. Sentence-level "
                "analysis identifies portions of the document exhibiting "
                "stronger or weaker indicators of synthetic generation."
            ),
            styles["body"],
        ),
    ]

    if not sentences:
        elements.append(
            Paragraph(
                "No sentence-level evidence was returned by the text pipeline.",
                styles["body"],
            )
        )

        return elements

    chart_path = (
        create_sentence_chart(
            sentences,
            temporary_directory,
        )
    )

    chart = scaled_report_image(
        chart_path,
        160 * mm,
        65 * mm,
    )

    if chart:
        elements.extend(
            [
                chart,
                Spacer(
                    1,
                    4 * mm,
                ),
            ]
        )

    ranked = sorted(
        sentences,
        key=lambda item: item[
            "score"
        ],
        reverse=True,
    )

    elements.extend(
        [
            Paragraph(
                "Highest-Risk Sentences",
                styles["subsection"],
            ),

            make_data_table(
                [
                    "#",
                    "Sentence",
                    "AI score",
                    "Risk",
                    "Reason",
                ],
                [
                    [
                        item["index"],
                        item["sentence"],
                        format_percent(
                            item["score"]
                        ),
                        item["risk"],
                        item["reason"],
                    ]
                    for item in ranked[:15]
                ],
                styles,
                col_widths=[
                    10 * mm,
                    75 * mm,
                    20 * mm,
                    18 * mm,
                    47 * mm,
                ],
            ),

            Spacer(
                1,
                6 * mm,
            ),

            Paragraph(
                "Complete Sentence Examination",
                styles["subsection"],
            ),

            make_data_table(
                [
                    "#",
                    "Sentence",
                    "Score",
                    "Risk",
                ],
                [
                    [
                        item["index"],
                        item["sentence"],
                        format_percent(
                            item["score"]
                        ),
                        item["risk"],
                    ]
                    for item in sentences
                ],
                styles,
                col_widths=[
                    10 * mm,
                    115 * mm,
                    23 * mm,
                    21 * mm,
                ],
            ),
        ]
    )

    return elements


# =========================================================
# IMAGE REPORT
# =========================================================

def build_image_visual_grid(
    result: Dict[
        str,
        Any,
    ],
    styles: Dict[
        str,
        ParagraphStyle,
    ],
) -> Optional[Table]:
    visual_evidence = (
        result.get(
            "visual_evidence"
        )
        or {}
    )

    original_path = resolve_local_path(
        result.get(
            "uploaded_file"
        )
    )

    overlay_path = resolve_local_path(
        visual_evidence.get(
            "overlay"
        )
        or result.get(
            "heatmap"
        )
    )

    heatmap_path = resolve_local_path(
        visual_evidence.get(
            "heatmap"
        )
    )

    edge_path = resolve_local_path(
        visual_evidence.get(
            "edge_map"
        )
    )

    frequency_path = resolve_local_path(
        visual_evidence.get(
            "frequency_map"
        )
    )

    panels = [
        (
            "Original evidence",
            original_path,
        ),

        (
            "Forensic overlay",
            overlay_path,
        ),

        (
            "Heatmap",
            heatmap_path,
        ),

        (
            "Edge response",
            edge_path,
        ),

        (
            "Frequency map",
            frequency_path,
        ),
    ]

    cells = []

    for title, path in panels:
        image = (
            scaled_report_image(
                path,
                77 * mm,
                58 * mm,
            )
            if path
            else None
        )

        cell = [
            Paragraph(
                escape_text(
                    title
                ),
                styles["subsection"],
            )
        ]

        if image:
            cell.append(
                image
            )
        else:
            cell.append(
                Paragraph(
                    "Not available",
                    styles["small"],
                )
            )

        cells.append(
            cell
        )

    rows = []

    for index in range(
        0,
        len(cells),
        2,
    ):
        row = [
            cells[index]
        ]

        if index + 1 < len(
            cells
        ):
            row.append(
                cells[index + 1]
            )
        else:
            row.append(
                []
            )

        rows.append(
            row
        )

    table = Table(
        rows,
        colWidths=[
            84 * mm,
            84 * mm,
        ],
        hAlign="LEFT",
    )

    table.setStyle(
        TableStyle(
            [
                (
                    "VALIGN",
                    (0, 0),
                    (-1, -1),
                    "TOP",
                ),

                (
                    "BOX",
                    (0, 0),
                    (-1, -1),
                    0.4,
                    MID_GREY,
                ),

                (
                    "INNERGRID",
                    (0, 0),
                    (-1, -1),
                    0.35,
                    MID_GREY,
                ),

                (
                    "LEFTPADDING",
                    (0, 0),
                    (-1, -1),
                    6,
                ),

                (
                    "RIGHTPADDING",
                    (0, 0),
                    (-1, -1),
                    6,
                ),

                (
                    "TOPPADDING",
                    (0, 0),
                    (-1, -1),
                    6,
                ),

                (
                    "BOTTOMPADDING",
                    (0, 0),
                    (-1, -1),
                    6,
                ),
            ]
        )
    )

    return table


def build_image_section(
    result: Dict[str, Any],
    styles: Dict[str, ParagraphStyle],
    temporary_directory: str,
) -> List[Any]:
    region_analysis = safe_dict(result.get("region_analysis"))
    regions = safe_list(region_analysis.get("ranked_regions") or region_analysis.get("regions"))

    elements: List[Any] = [
        PageBreak(),
        Paragraph("4. Image Forensic Examination", styles["section"]),
        Paragraph(
            "The image examination combines the base CNN and Random Forest fusion result with handcrafted forensic features, "
            "regional analysis and visual evidence. The heatmap is a relative anomaly and attention representation rather than "
            "a definitive pixel-level mask.",
            styles["body"],
        ),
    ]

    visual_grid = build_image_visual_grid(result, styles)
    if visual_grid:
        elements.extend([visual_grid, Spacer(1, 5 * mm)])

    elements.extend(build_heatmap_legend(styles))

    if regions:
        chart_path = create_region_chart(region_analysis, temporary_directory)
        chart = scaled_report_image(chart_path, 160 * mm, 75 * mm)
        if chart:
            elements.extend([
                Spacer(1, 5 * mm),
                Paragraph("Regional Suspicion Ranking", styles["subsection"]),
                chart,
                Spacer(1, 4 * mm),
            ])

        regional_rows: List[List[Any]] = []
        for region in regions:
            if not isinstance(region, dict):
                continue
            metrics = safe_dict(region.get("metrics"))
            metric_summary = ", ".join(
                f"{format_key(key)}: {safe_number(value):.1f}%"
                for key, value in metrics.items()
            )
            reasons = "; ".join(str(item) for item in safe_list(region.get("reasons")))
            regional_rows.append([
                region.get("name", "Region"),
                format_percent(region.get("forgery_score", region.get("score", 0))),
                region.get("risk_level", region.get("risk", "N/A")),
                metric_summary or "No regional metrics returned.",
                reasons or region.get("reason", "No regional explanation returned."),
            ])

        if regional_rows:
            elements.extend([
                make_data_table(
                    ["Region", "Score", "Risk", "Regional metrics", "Findings"],
                    regional_rows,
                    styles,
                    col_widths=[25 * mm, 18 * mm, 17 * mm, 51 * mm, 58 * mm],
                ),
                Spacer(1, 5 * mm),
            ])

        top_region = max(
            (item for item in regions if isinstance(item, dict)),
            key=lambda item: safe_number(item.get("forgery_score", item.get("score", 0))),
            default=None,
        )
        if top_region:
            elements.extend([
                Paragraph("Investigator-Focused Interpretation", styles["subsection"]),
                Paragraph(
                    f"The highest-ranked region was <b>{escape_text(top_region.get('name', 'Region'))}</b> with an indicative "
                    f"suspicion score of <b>{safe_number(top_region.get('forgery_score', top_region.get('score', 0))):.2f}%</b>. "
                    "This region should be compared with the original image, overlay, edge response and frequency map before a conclusion is reached.",
                    styles["body"],
                ),
            ])

        elements.append(Paragraph(
            f"Face-specific analysis: {'Yes' if region_analysis.get('face_detected') else 'No'}. "
            f"Regional engine: {escape_text(region_analysis.get('analysis_version', 'N/A'))}.",
            styles["disclaimer"],
        ))
    else:
        elements.append(Paragraph("No regional image evidence was returned.", styles["body"]))

    return elements


# =========================================================
# AUDIO REPORT
# =========================================================

def build_audio_visual_grid(
    result: Dict[str, Any],
    styles: Dict[str, ParagraphStyle],
) -> Optional[Table]:
    visual_evidence = safe_dict(result.get("audio_visual_evidence"))
    panels = [
        ("Waveform", resolve_local_path(result.get("waveform") or visual_evidence.get("waveform"))),
        ("Spectrogram", resolve_local_path(result.get("spectrogram") or visual_evidence.get("spectrogram"))),
        ("Audio heatmap", resolve_local_path(result.get("audio_heatmap") or visual_evidence.get("audio_heatmap"))),
        ("LFCC heatmap", resolve_local_path(result.get("lfcc_heatmap") or visual_evidence.get("lfcc_heatmap"))),
        ("Pitch plot", resolve_local_path(result.get("pitch_plot") or visual_evidence.get("pitch_plot"))),
        ("Energy plot", resolve_local_path(result.get("energy_plot") or visual_evidence.get("energy_plot"))),
    ]

    cells: List[List[Any]] = []
    for title, path in panels:
        image = scaled_report_image(path, 77 * mm, 58 * mm) if path else None
        cell: List[Any] = [Paragraph(escape_text(title), styles["subsection"])]
        cell.append(image if image else Paragraph("Not available", styles["small"]))
        cells.append(cell)

    rows: List[List[Any]] = []
    for index in range(0, len(cells), 2):
        rows.append([cells[index], cells[index + 1] if index + 1 < len(cells) else []])

    table = Table(rows, colWidths=[84 * mm, 84 * mm], hAlign="LEFT")
    table.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("GRID", (0, 0), (-1, -1), 0.35, MID_GREY),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    return table


def build_audio_section(
    result: Dict[str, Any],
    styles: Dict[str, ParagraphStyle],
    temporary_directory: str,
) -> List[Any]:
    suspicious_segments = safe_list(
        result.get("suspicious_intervals") or result.get("suspicious_segments")
    )
    audio_summary = safe_dict(
        result.get("audio_summary") or safe_dict(result.get("advanced_audio_analysis")).get("summary")
    )
    voice_dna = safe_dict(
        result.get("voice_dna") or safe_dict(result.get("advanced_audio_analysis")).get("voice_dna")
    )
    pause_intervals = safe_list(
        result.get("pause_intervals") or safe_dict(result.get("advanced_audio_analysis")).get("pause_intervals")
    )
    breathing_events = safe_list(
        result.get("breathing_events") or safe_dict(result.get("advanced_audio_analysis")).get("breathing_events")
    )

    elements: List[Any] = [
        PageBreak(),
        Paragraph("4. Audio Forensic Examination", styles["section"]),
        Paragraph(
            "The audio examination combines LFCC representations, CNN-BiLSTM inference, acoustic feature analysis, "
            "segment-level explainability and visual evidence. Suspicious intervals identify portions with stronger "
            "synthetic-speech or manipulation indicators.",
            styles["body"],
        ),
    ]

    visual_grid = build_audio_visual_grid(result, styles)
    if visual_grid:
        elements.extend([visual_grid, Spacer(1, 5 * mm)])

    elements.extend(build_audio_heatmap_legend(styles))

    probability_rows = [
        ("Real probability", format_percent(normalize_probability(result.get("raw_probability_real", 0)))),
        ("Fake probability", format_percent(normalize_probability(result.get("raw_probability_fake", 0)))),
        ("Prediction", result.get("prediction", "N/A")),
        ("Confidence", format_percent(result.get("confidence", 0))),
        ("Risk level", result.get("risk_level", "N/A")),
        ("Risk score", format_percent(result.get("risk_score", 0))),
    ]
    elements.extend([
        Spacer(1, 5 * mm),
        Paragraph("Audio Model Output", styles["subsection"]),
        make_key_value_table(probability_rows, styles),
        Spacer(1, 5 * mm),
    ])

    if audio_summary:
        summary_rows = [
            ("Duration", f"{safe_number(audio_summary.get('duration_seconds')):.3f} seconds"),
            ("Mean pitch", f"{safe_number(audio_summary.get('pitch_mean_hz')):.2f} Hz"),
            ("Pitch variation", f"{safe_number(audio_summary.get('pitch_std_hz')):.2f} Hz"),
            ("Mean RMS energy", f"{safe_number(audio_summary.get('energy_mean')):.6f}"),
            ("Energy variation", f"{safe_number(audio_summary.get('energy_variation')):.6f}"),
            ("Mean spectral flux", f"{safe_number(audio_summary.get('spectral_flux_mean')):.6f}"),
            ("Mean spectral flatness", f"{safe_number(audio_summary.get('spectral_flatness_mean')):.6f}"),
            ("Detected pauses", audio_summary.get("pause_count", len(pause_intervals))),
            ("Pause ratio", format_percent(safe_number(audio_summary.get("pause_ratio")) * 100)),
            ("Breathing candidates", audio_summary.get("breathing_event_count", len(breathing_events))),
            ("Estimated breaths/min", f"{safe_number(audio_summary.get('estimated_breaths_per_minute')):.2f}"),
        ]
        elements.extend([
            Paragraph("Advanced Acoustic Summary", styles["subsection"]),
            make_key_value_table(summary_rows, styles),
            Spacer(1, 5 * mm),
        ])

    curve_specs = [
        ("pitch", "Pitch Contour", "Frequency (Hz)", "audio_pitch_curve.png"),
        ("energy", "Energy Timeline", "RMS energy", "audio_energy_curve.png"),
        ("spectral_flux", "Spectral Flux", "Flux", "audio_spectral_flux_curve.png"),
        ("spectral_flatness", "Spectral Flatness", "Flatness ratio", "audio_spectral_flatness_curve.png"),
    ]
    generated_curves: List[Tuple[str, str]] = []
    for key, title, y_label, filename in curve_specs:
        chart_path = create_audio_curve_chart(
            extract_curve_points(result, key), title, y_label, temporary_directory, filename
        )
        if chart_path:
            generated_curves.append((title, chart_path))

    if generated_curves:
        elements.append(Paragraph("Time-Series Acoustic Evidence", styles["subsection"]))
        for title, chart_path in generated_curves:
            chart = scaled_report_image(chart_path, 160 * mm, 62 * mm)
            if chart:
                elements.extend([Paragraph(title, styles["small"]), chart, Spacer(1, 4 * mm)])

    if voice_dna:
        voice_chart_path = create_voice_dna_chart(voice_dna, temporary_directory)
        voice_chart = scaled_report_image(voice_chart_path, 160 * mm, 82 * mm)
        elements.append(Paragraph("Voice DNA and Synthetic Signature", styles["subsection"]))
        if voice_chart:
            elements.extend([voice_chart, Spacer(1, 4 * mm)])

        voice_rows: List[List[Any]] = []
        for key, value in voice_dna.items():
            if not isinstance(value, dict):
                continue
            voice_rows.append([
                format_key(key),
                format_percent(value.get("score", 0)),
                value.get("risk", "N/A"),
                value.get("observed", "N/A"),
                value.get("reason", "No explanation returned."),
            ])
        if voice_rows:
            elements.extend([
                make_data_table(
                    ["Indicator", "Score", "Risk", "Observed", "Interpretation"],
                    voice_rows,
                    styles,
                    col_widths=[33 * mm, 19 * mm, 18 * mm, 24 * mm, 75 * mm],
                ),
                Spacer(1, 5 * mm),
            ])

    if suspicious_segments:
        timeline_chart_path = create_audio_timeline_chart(suspicious_segments, temporary_directory)
        timeline_chart = scaled_report_image(timeline_chart_path, 160 * mm, 65 * mm)
        elements.append(Paragraph("Suspicious Audio Timeline", styles["subsection"]))
        if timeline_chart:
            elements.extend([timeline_chart, Spacer(1, 4 * mm)])

        segment_rows: List[List[Any]] = []
        for index, segment in enumerate(suspicious_segments, start=1):
            if not isinstance(segment, dict):
                continue
            reasons = safe_list(segment.get("reasons"))
            evidence_text = segment.get("reason") or (reasons[0] if reasons else "No explanation generated.")
            segment_rows.append([
                index,
                segment.get("start", "N/A"),
                segment.get("end", "N/A"),
                format_percent(segment.get("score", segment.get("confidence", segment.get("risk_score", 0)))),
                segment.get("risk", segment.get("risk_level", "N/A")),
                evidence_text,
            ])
        if segment_rows:
            elements.append(make_data_table(
                ["#", "Start", "End", "Score", "Risk", "Evidence"],
                segment_rows,
                styles,
                col_widths=[9 * mm, 19 * mm, 19 * mm, 20 * mm, 18 * mm, 84 * mm],
            ))
    else:
        elements.append(Paragraph(
            "No suspicious timestamp intervals were returned by the audio pipeline.", styles["body"]
        ))

    if pause_intervals or breathing_events:
        elements.extend([Spacer(1, 5 * mm), Paragraph("Pause and Breathing Evidence", styles["subsection"])])
        event_rows: List[List[Any]] = []
        for item in pause_intervals[:50]:
            if isinstance(item, dict):
                event_rows.append([
                    "Pause",
                    item.get("start", item.get("start_seconds", "N/A")),
                    item.get("end", item.get("end_seconds", "N/A")),
                    f"Duration: {safe_number(item.get('duration')):.3f}s",
                ])
        for item in breathing_events[:50]:
            if isinstance(item, dict):
                event_rows.append([
                    "Breathing candidate",
                    item.get("time", item.get("time_seconds", "N/A")),
                    "-",
                    f"Confidence: {safe_number(item.get('confidence')):.2f}%",
                ])
        if event_rows:
            elements.append(make_data_table(
                ["Event type", "Start / Time", "End", "Details"],
                event_rows,
                styles,
                col_widths=[38 * mm, 30 * mm, 30 * mm, 71 * mm],
            ))

    return elements


# =========================================================
# CONCLUSION
# =========================================================

def build_conclusion(
    result: Dict[
        str,
        Any,
    ],
    styles: Dict[
        str,
        ParagraphStyle,
    ],
) -> List[Any]:
    prediction = result.get(
        "prediction",
        "UNKNOWN",
    )

    confidence = safe_number(
        result.get(
            "confidence",
            0,
        )
    )

    risk = result.get(
        "risk_level",
        "N/A",
    )

    recommendation = (
        result.get(
            "recommendation"
        )
        or (
            "The finding should be interpreted with the "
            "explainability evidence and any external "
            "investigative information."
        )
    )

    conclusion = (
        f"The examined evidence was classified as "
        f"<b>{escape_text(prediction)}</b> with "
        f"<b>{confidence:.2f}% confidence</b> and "
        f"a <b>{escape_text(risk)}</b> risk designation. "
        f"The output reflects the statistical behaviour learned by "
        f"the deployed FORGE models and the forensic indicators "
        f"available for this specific item."
    )

    return [
        PageBreak(),

        Paragraph(
            "5. Forensic Conclusion",
            styles["section"],
        ),

        Paragraph(
            conclusion,
            styles["body"],
        ),

        Paragraph(
            "Recommended Interpretation",
            styles["subsection"],
        ),

        Paragraph(
            escape_text(
                recommendation
            ),
            styles["body"],
        ),

        Spacer(
            1,
            6 * mm,
        ),

        Table(
            [
                [
                    Paragraph(
                        "<b>Important limitation</b>",
                        styles["table_cell"],
                    )
                ],

                [
                    Paragraph(
                        (
                            "FORGE provides machine-learning-based forensic "
                            "indicators. Its output is not, by itself, proof of "
                            "authorship, manipulation, identity or legal liability. "
                            "High-stakes conclusions should include manual review, "
                            "source verification, contextual investigation and, "
                            "where applicable, examination using validated forensic "
                            "laboratory procedures."
                        ),
                        styles["table_cell"],
                    )
                ],
            ],
            colWidths=[
                165 * mm
            ],
            style=TableStyle(
                [
                    (
                        "BACKGROUND",
                        (0, 0),
                        (-1, 0),
                        LIGHT_AMBER,
                    ),

                    (
                        "BACKGROUND",
                        (0, 1),
                        (-1, -1),
                        colors.HexColor(
                            "#FFF9EB"
                        ),
                    ),

                    (
                        "BOX",
                        (0, 0),
                        (-1, -1),
                        0.8,
                        AMBER,
                    ),

                    (
                        "LEFTPADDING",
                        (0, 0),
                        (-1, -1),
                        9,
                    ),

                    (
                        "RIGHTPADDING",
                        (0, 0),
                        (-1, -1),
                        9,
                    ),

                    (
                        "TOPPADDING",
                        (0, 0),
                        (-1, -1),
                        8,
                    ),

                    (
                        "BOTTOMPADDING",
                        (0, 0),
                        (-1, -1),
                        8,
                    ),
                ]
            ),
        ),

        Spacer(
            1,
            8 * mm,
        ),

        Paragraph(
            (
                f"Report generated by {REPORT_ENGINE_VERSION} on "
                f"{datetime.now(timezone.utc).isoformat()}."
            ),
            styles["disclaimer"],
        ),
    ]


# =========================================================
# MAIN REPORT GENERATOR
# =========================================================

def generate_pdf_report(
    result: Dict[str, Any],
    output_path: str,
) -> None:
    """
    Generate a complete FORGE forensic PDF report.

    Existing call remains valid:

        generate_pdf_report(result, output_path)

    The function supports:
        - text forensic output
        - image forensic output
        - audio forensic output
        - shared explainability information
        - charts and visual evidence
    """

    if not isinstance(
        result,
        dict,
    ):
        raise TypeError(
            "result must be a dictionary"
        )

    output_directory = os.path.dirname(
        output_path
    )

    if output_directory:
        os.makedirs(
            output_directory,
            exist_ok=True,
        )

    styles = build_styles()

    document = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        rightMargin=18 * mm,
        leftMargin=18 * mm,
        topMargin=24 * mm,
        bottomMargin=20 * mm,
        title=REPORT_TITLE,
        author="FORGE",
        subject=(
            "Multimodal digital forensic examination report"
        ),
    )

    modality = get_modality(
        result
    )

    with tempfile.TemporaryDirectory(
        prefix="forge_report_"
    ) as temporary_directory:

        story: List[Any] = []

        story.extend(
            build_cover(
                result,
                styles,
            )
        )

        story.extend(
            build_executive_summary(
                result,
                styles,
                temporary_directory,
            )
        )

        story.extend(
            [
                Spacer(
                    1,
                    6 * mm,
                )
            ]
        )

        story.extend(
            build_evidence_section(
                result,
                styles,
            )
        )

        story.extend(
            [
                Spacer(
                    1,
                    7 * mm,
                )
            ]
        )

        story.extend(
            build_parameter_section(
                result,
                styles,
                temporary_directory,
            )
        )

        if modality == "text":
            story.extend(
                build_text_section(
                    result,
                    styles,
                    temporary_directory,
                )
            )

        elif modality == "image":
            story.extend(
                build_image_section(
                    result,
                    styles,
                    temporary_directory,
                )
            )

        elif modality == "audio":
            story.extend(
                build_audio_section(
                    result,
                    styles,
                    temporary_directory,
                )
            )

        else:
            story.extend(
                [
                    PageBreak(),

                    Paragraph(
                        "4. Modality-Specific Evidence",
                        styles["section"],
                    ),

                    Paragraph(
                        (
                            "The modality could not be determined from "
                            "the provided result object. Shared model and "
                            "explainability findings are included above."
                        ),
                        styles["body"],
                    ),
                ]
            )

        story.extend(
            build_conclusion(
                result,
                styles,
            )
        )

        document.build(
            story,
            onFirstPage=first_page,
            onLaterPages=later_pages,
        )