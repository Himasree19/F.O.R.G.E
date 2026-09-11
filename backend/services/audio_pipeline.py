from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np

from backend.services.audio_detector import analyze_audio_model
from backend.services.audio_segment_analyzer import analyze_audio_segments
from backend.services.audio_advanced_analysis import analyze_advanced_audio
from backend.services.audio_visualizer import generate_audio_visuals
from backend.services.audio_xai import build_audio_xai


# =========================================================
# GENERIC HELPERS
# =========================================================

def normalize_probability(
    value: Any,
) -> float:
    """
    Convert either a 0-1 probability or a 0-100 percentage
    into a probability between 0 and 1.
    """

    try:
        probability = float(
            np.asarray(value).reshape(-1)[0]
        )

    except (
        TypeError,
        ValueError,
        IndexError,
    ):
        return 0.0

    if not np.isfinite(probability):
        return 0.0

    if probability > 1.0:
        probability /= 100.0

    return max(
        0.0,
        min(
            1.0,
            probability,
        ),
    )


def safe_dictionary(
    value: Any,
) -> Dict[str, Any]:
    if isinstance(value, dict):
        return value

    return {}


def safe_list(
    value: Any,
) -> List[Any]:
    if isinstance(value, list):
        return value

    return []


def serialize_array(
    value: Any,
) -> List[float]:
    """
    Convert NumPy arrays into JSON-safe Python lists.
    """

    try:
        array = np.asarray(
            value,
            dtype=np.float32,
        ).reshape(-1)

        output = []

        for item in array:
            if np.isfinite(item):
                output.append(
                    round(
                        float(item),
                        8,
                    )
                )

        return output

    except Exception:
        return []


def risk_level_from_score(
    score: float,
) -> str:
    if score >= 75:
        return "HIGH"

    if score >= 45:
        return "MEDIUM"

    return "LOW"


# =========================================================
# SAFE MODULE EXECUTION
# =========================================================

def run_audio_xai(
    raw_parameters: Any,
    confidence: float,
    fake_probability: float,
) -> Tuple[
    Dict[str, Any],
    List[Dict[str, Any]],
]:
    try:
        result = build_audio_xai(
            raw_parameters,
            confidence,
            fake_probability,
        )

        if (
            isinstance(result, tuple)
            and len(result) == 2
        ):
            parameter_contribution = (
                safe_dictionary(
                    result[0]
                )
            )

            suspicious_segments = (
                safe_list(
                    result[1]
                )
            )

            return (
                parameter_contribution,
                suspicious_segments,
            )

        if isinstance(result, dict):
            parameter_contribution = (
                safe_dictionary(
                    result.get(
                        "parameter_contribution",
                        result,
                    )
                )
            )

            suspicious_segments = (
                safe_list(
                    result.get(
                        "suspicious_segments",
                        [],
                    )
                )
            )

            return (
                parameter_contribution,
                suspicious_segments,
            )

        return {}, []

    except Exception as error:
        print(
            "⚠️ Audio XAI warning:",
            error,
        )

        return {}, []


def run_visual_generation(
    file_path: str,
) -> Dict[str, Any]:
    try:
        visuals = generate_audio_visuals(
            file_path
        )

        if not isinstance(
            visuals,
            dict,
        ):
            raise ValueError(
                "Audio visualizer returned an invalid response."
            )

        return visuals

    except Exception as error:
        print(
            "⚠️ Audio visualization warning:",
            error,
        )

        return {
            "waveform": None,
            "spectrogram": None,
            "audio_heatmap": None,
            "lfcc_heatmap": None,
            "pitch_plot": None,
            "energy_plot": None,
        }


def run_segment_analysis(
    file_path: str,
    fake_probability: float,
) -> Dict[str, Any]:
    try:
        result = analyze_audio_segments(
            file_path=file_path,
            global_fake_probability=(
                fake_probability
            ),
        )

        if not isinstance(
            result,
            dict,
        ):
            raise ValueError(
                "Audio segment analyzer returned an invalid response."
            )

        return result

    except Exception as error:
        print(
            "⚠️ Audio segment analysis warning:",
            error,
        )

        return {
            "sample_rate": 16000,
            "audio_duration": 0,
            "segment_duration": 0,
            "segment_hop_duration": 0,
            "segment_count": 0,
            "segments": [],
            "ranked_segments": [],
            "suspicious_intervals": [],
            "analysis_version": (
                "FORGE-AUDIO-SEGMENT-XAI-1.0"
            ),
            "error": str(error),
        }


def run_advanced_analysis(
    file_path: str,
) -> Dict[str, Any]:
    try:
        result = analyze_advanced_audio(
            file_path
        )

        if not isinstance(
            result,
            dict,
        ):
            raise ValueError(
                "Advanced audio analyzer returned an invalid response."
            )

        return result

    except Exception as error:
        print(
            "⚠️ Advanced audio analysis warning:",
            error,
        )

        return {
            "summary": {},
            "voice_dna": {},
            "curves": {
                "pitch": [],
                "energy": [],
                "spectral_flux": [],
                "spectral_flatness": [],
            },
            "pause_intervals": [],
            "breathing_events": [],
            "analysis_version": (
                "FORGE-AUDIO-ADVANCED-XAI-1.1"
            ),
            "error": str(error),
        }


# =========================================================
# INTERVAL NORMALIZATION
# =========================================================

def normalize_interval(
    interval: Dict[str, Any],
    index: int,
) -> Dict[str, Any]:
    score = interval.get(
        "risk_score",
        interval.get(
            "score",
            0,
        ),
    )

    try:
        score = float(score)

    except (
        TypeError,
        ValueError,
    ):
        score = 0.0

    if 0 <= score <= 1:
        score *= 100

    score = max(
        0.0,
        min(
            100.0,
            score,
        ),
    )

    risk = (
        interval.get(
            "risk_level"
        )
        or interval.get(
            "risk"
        )
        or risk_level_from_score(
            score
        )
    )

    reasons = interval.get(
        "reasons"
    )

    if not isinstance(
        reasons,
        list,
    ):
        reasons = []

    reason = (
        interval.get(
            "reason"
        )
        or (
            reasons[0]
            if reasons
            else (
                "Synthetic acoustic indicators "
                "were detected in this interval."
            )
        )
    )

    return {
        "id": (
            interval.get(
                "id"
            )
            or f"audio_interval_{index:03d}"
        ),

        "start_seconds": round(
            float(
                interval.get(
                    "start_seconds",
                    0,
                )
                or 0
            ),
            3,
        ),

        "end_seconds": round(
            float(
                interval.get(
                    "end_seconds",
                    0,
                )
                or 0
            ),
            3,
        ),

        "start": interval.get(
            "start",
            "00:00.00",
        ),

        "end": interval.get(
            "end",
            "00:00.00",
        ),

        "score": round(
            score,
            2,
        ),

        "risk_score": round(
            score,
            2,
        ),

        "risk": str(
            risk
        ).upper(),

        "risk_level": str(
            risk
        ).upper(),

        "reason": reason,

        "reasons": reasons[:5],
    }


def build_suspicious_intervals(
    timeline_analysis: Dict[str, Any],
    legacy_segments: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    intervals = safe_list(
        timeline_analysis.get(
            "suspicious_intervals"
        )
    )

    if not intervals:
        intervals = safe_list(
            legacy_segments
        )

    output = []

    for index, interval in enumerate(
        intervals,
        start=1,
    ):
        if not isinstance(
            interval,
            dict,
        ):
            continue

        output.append(
            normalize_interval(
                interval,
                index,
            )
        )

    return output


# =========================================================
# VOICE DNA
# =========================================================

def enrich_voice_dna(
    voice_dna: Dict[str, Any],
    fake_probability: float,
) -> Dict[str, Any]:
    output: Dict[str, Any] = {}

    for key, value in voice_dna.items():
        if not isinstance(
            value,
            dict,
        ):
            continue

        enriched = dict(value)

        try:
            score = float(
                enriched.get(
                    "score",
                    0,
                )
            )

        except (
            TypeError,
            ValueError,
        ):
            score = 0.0

        score = max(
            0.0,
            min(
                100.0,
                score,
            ),
        )

        enriched["score"] = round(
            score,
            2,
        )

        if not enriched.get(
            "risk"
        ):
            enriched["risk"] = (
                risk_level_from_score(
                    score
                )
            )

        output[key] = enriched

    synthetic_score = (
        fake_probability * 100
    )

    if (
        "model_synthetic_probability"
        not in output
    ):
        output[
            "model_synthetic_probability"
        ] = {
            "score": round(
                synthetic_score,
                2,
            ),
            "risk": risk_level_from_score(
                synthetic_score
            ),
            "reason": (
                "Synthetic speech probability produced "
                "by the CNN-BiLSTM fusion model."
            ),
        }

    return output


# =========================================================
# AUDIO CURVES
# =========================================================

def extract_audio_curves(
    advanced_analysis: Dict[str, Any],
) -> Dict[str, List[Dict[str, float]]]:
    """
    Safely extract chart-ready curves from advanced analysis.
    """

    raw_curves = advanced_analysis.get(
        "curves",
        {},
    )

    if not isinstance(
        raw_curves,
        dict,
    ):
        raw_curves = {}

    curve_names = [
        "pitch",
        "energy",
        "spectral_flux",
        "spectral_flatness",
    ]

    audio_curves: Dict[
        str,
        List[Dict[str, float]],
    ] = {}

    for curve_name in curve_names:
        raw_curve = raw_curves.get(
            curve_name,
            [],
        )

        if not isinstance(
            raw_curve,
            list,
        ):
            audio_curves[
                curve_name
            ] = []

            continue

        valid_points = []

        for point in raw_curve:
            if not isinstance(
                point,
                dict,
            ):
                continue

            try:
                time_value = float(
                    point.get(
                        "time",
                        0,
                    )
                )

                curve_value = float(
                    point.get(
                        "value",
                        0,
                    )
                )

            except (
                TypeError,
                ValueError,
            ):
                continue

            if (
                not np.isfinite(
                    time_value
                )
                or not np.isfinite(
                    curve_value
                )
            ):
                continue

            valid_points.append(
                {
                    "time": round(
                        time_value,
                        4,
                    ),

                    "value": round(
                        curve_value,
                        8,
                    ),
                }
            )

        audio_curves[
            curve_name
        ] = valid_points

    return audio_curves


# =========================================================
# INVESTIGATION SUMMARY
# =========================================================

def build_investigation_summary(
    *,
    model_result: Dict[str, Any],
    timeline_analysis: Dict[str, Any],
    advanced_analysis: Dict[str, Any],
    suspicious_intervals: List[Dict[str, Any]],
) -> Dict[str, Any]:
    ranked_segments = safe_list(
        timeline_analysis.get(
            "ranked_segments"
        )
    )

    most_suspicious_segment = (
        ranked_segments[0]
        if ranked_segments
        and isinstance(
            ranked_segments[0],
            dict,
        )
        else None
    )

    summary = safe_dictionary(
        advanced_analysis.get(
            "summary"
        )
    )

    voice_dna = safe_dictionary(
        advanced_analysis.get(
            "voice_dna"
        )
    )

    naturalness = safe_dictionary(
        voice_dna.get(
            "naturalness"
        )
    )

    synthetic_signature = safe_dictionary(
        voice_dna.get(
            "synthetic_signature"
        )
    )

    return {
        "overall_prediction": (
            model_result.get(
                "prediction",
                "UNKNOWN",
            )
        ),

        "overall_confidence": round(
            float(
                model_result.get(
                    "confidence",
                    0,
                )
                or 0
            ),
            2,
        ),

        "overall_risk_score": round(
            float(
                model_result.get(
                    "risk_score",
                    0,
                )
                or 0
            ),
            2,
        ),

        "audio_duration": (
            timeline_analysis.get(
                "audio_duration",
                summary.get(
                    "duration_seconds",
                    0,
                ),
            )
        ),

        "segment_count": (
            timeline_analysis.get(
                "segment_count",
                0,
            )
        ),

        "suspicious_interval_count": len(
            suspicious_intervals
        ),

        "pause_count": summary.get(
            "pause_count",
            0,
        ),

        "breathing_event_count": summary.get(
            "breathing_event_count",
            0,
        ),

        "pitch_mean_hz": summary.get(
            "pitch_mean_hz",
            0,
        ),

        "pitch_variation_hz": summary.get(
            "pitch_std_hz",
            0,
        ),

        "naturalness_score": naturalness.get(
            "score",
            0,
        ),

        "synthetic_signature_score": (
            synthetic_signature.get(
                "score",
                0,
            )
        ),

        "most_suspicious_segment": (
            most_suspicious_segment
        ),
    }


# =========================================================
# RECOMMENDATION
# =========================================================

def build_recommendation(
    prediction: str,
    suspicious_intervals: List[Dict[str, Any]],
    advanced_analysis: Dict[str, Any],
) -> str:
    high_risk_count = sum(
        1
        for interval in suspicious_intervals
        if str(
            interval.get(
                "risk_level",
                "",
            )
        ).upper() == "HIGH"
    )

    summary = safe_dictionary(
        advanced_analysis.get(
            "summary"
        )
    )

    breathing_count = int(
        summary.get(
            "breathing_event_count",
            0,
        )
        or 0
    )

    if str(
        prediction
    ).upper() == "FAKE":
        recommendation = (
            "The CNN-BiLSTM fusion model detected synthetic "
            "voice indicators. Review the high-risk timeline "
            "segments, LFCC heatmap, pitch contour, energy "
            "variation, phase continuity and Voice DNA profile."
        )

        if high_risk_count:
            recommendation += (
                f" {high_risk_count} high-risk acoustic "
                "interval(s) require focused examination."
            )

        if breathing_count == 0:
            recommendation += (
                " No reliable breathing event was detected, "
                "which may support additional manual review."
            )

        return recommendation

    if suspicious_intervals:
        return (
            "The recording appears predominantly natural, "
            "but localized acoustic irregularities were detected. "
            "Review the marked timeline intervals and acoustic "
            "curves before using the result in a high-stakes case."
        )

    return (
        "The recording appears predominantly natural and no "
        "major synthetic acoustic interval was identified. "
        "Manual verification is still recommended for legal, "
        "journalistic or evidentiary use."
    )


# =========================================================
# MAIN PIPELINE
# =========================================================

def process_audio(
    file_path: str,
) -> Dict[str, Any]:
    try:
        # =================================================
        # 1. BASE CNN-BILSTM MODEL
        # =================================================

        model_result = analyze_audio_model(
            file_path
        )

        if not isinstance(
            model_result,
            dict,
        ):
            return {
                "error": (
                    "Audio model returned an invalid response."
                ),
                "modality": "audio",
                "file_type": "audio",
            }

        if model_result.get(
            "error"
        ):
            return model_result

        prediction = str(
            model_result.get(
                "prediction",
                "UNKNOWN",
            )
        ).upper()

        confidence = float(
            model_result.get(
                "confidence",
                0,
            )
            or 0
        )

        fake_probability = normalize_probability(
            model_result.get(
                "raw_probability_fake",
                model_result.get(
                    "risk_score",
                    0,
                ),
            )
        )

        real_probability = normalize_probability(
            model_result.get(
                "raw_probability_real",
                1.0 - fake_probability,
            )
        )

        raw_parameters = model_result.get(
            "raw_parameters",
            [],
        )

        scaled_parameters = model_result.get(
            "scaled_parameters",
            [],
        )

        parameter_names = safe_list(
            model_result.get(
                "parameter_names"
            )
        )

        # =================================================
        # 2. GLOBAL AUDIO XAI
        # =================================================

        (
            parameter_contribution,
            legacy_suspicious_segments,
        ) = run_audio_xai(
            raw_parameters=raw_parameters,
            confidence=confidence,
            fake_probability=fake_probability,
        )

        # =================================================
        # 3. SEGMENT TIMELINE
        # =================================================

        timeline_analysis = run_segment_analysis(
            file_path=file_path,
            fake_probability=fake_probability,
        )

        # =================================================
        # 4. ADVANCED AUDIO ANALYSIS
        # =================================================

        advanced_analysis = run_advanced_analysis(
            file_path
        )

        # =================================================
        # 5. VISUAL EVIDENCE
        # =================================================

        visuals = run_visual_generation(
            file_path
        )

        # =================================================
        # 6. TIMELINE DATA
        # =================================================

        audio_segments = safe_list(
            timeline_analysis.get(
                "segments"
            )
        )

        ranked_audio_segments = safe_list(
            timeline_analysis.get(
                "ranked_segments"
            )
        )

        suspicious_intervals = (
            build_suspicious_intervals(
                timeline_analysis,
                legacy_suspicious_segments,
            )
        )

        # =================================================
        # 7. ADVANCED DATA
        # =================================================

        audio_summary = safe_dictionary(
            advanced_analysis.get(
                "summary"
            )
        )

        raw_voice_dna = safe_dictionary(
            advanced_analysis.get(
                "voice_dna"
            )
        )

        voice_dna = enrich_voice_dna(
            raw_voice_dna,
            fake_probability,
        )

        audio_curves = extract_audio_curves(
            advanced_analysis
        )

        pause_intervals = safe_list(
            advanced_analysis.get(
                "pause_intervals"
            )
        )

        breathing_events = safe_list(
            advanced_analysis.get(
                "breathing_events"
            )
        )

        print(
            "\n"
            + "=" * 60
        )

        print(
            "FORGE AUDIO CURVES PASSED TO FRONTEND"
        )

        print(
            "Pitch:",
            len(
                audio_curves[
                    "pitch"
                ]
            ),
        )

        print(
            "Energy:",
            len(
                audio_curves[
                    "energy"
                ]
            ),
        )

        print(
            "Spectral flux:",
            len(
                audio_curves[
                    "spectral_flux"
                ]
            ),
        )

        print(
            "Spectral flatness:",
            len(
                audio_curves[
                    "spectral_flatness"
                ]
            ),
        )

        if advanced_analysis.get(
            "error"
        ):
            print(
                "Advanced analysis error:",
                advanced_analysis.get(
                    "error"
                ),
            )

        print(
            "=" * 60
            + "\n"
        )

        # =================================================
        # 8. SUMMARY AND RECOMMENDATION
        # =================================================

        investigation_summary = (
            build_investigation_summary(
                model_result=model_result,
                timeline_analysis=timeline_analysis,
                advanced_analysis=advanced_analysis,
                suspicious_intervals=(
                    suspicious_intervals
                ),
            )
        )

        recommendation = build_recommendation(
            prediction=prediction,
            suspicious_intervals=(
                suspicious_intervals
            ),
            advanced_analysis=(
                advanced_analysis
            ),
        )

        # =================================================
        # 9. FINAL RESPONSE
        # =================================================

        return {
            "modality": "audio",
            "file_type": "audio",

            "prediction": prediction,

            "confidence": round(
                confidence,
                2,
            ),

            "risk_level": (
                model_result.get(
                    "risk_level"
                )
                or risk_level_from_score(
                    fake_probability
                    * 100
                )
            ),

            "risk_score": round(
                fake_probability
                * 100,
                2,
            ),

            "recommendation": recommendation,

            "probabilities": {
                "ai": round(
                    fake_probability
                    * 100,
                    2,
                ),

                "fake": round(
                    fake_probability
                    * 100,
                    2,
                ),

                "human": round(
                    real_probability
                    * 100,
                    2,
                ),

                "real": round(
                    real_probability
                    * 100,
                    2,
                ),
            },

            # Model information
            "raw_model_output": (
                model_result.get(
                    "raw_model_output"
                )
            ),

            "raw_probability_real": round(
                real_probability,
                8,
            ),

            "raw_probability_fake": round(
                fake_probability,
                8,
            ),

            "positive_class": (
                model_result.get(
                    "positive_class"
                )
            ),

            "model_input_order": (
                model_result.get(
                    "model_input_order"
                )
            ),

            "model_diagnostics": (
                model_result.get(
                    "diagnostics",
                    {},
                )
            ),

            # Acoustic features
            "parameter_names": parameter_names,

            "raw_parameters": serialize_array(
                raw_parameters
            ),

            "scaled_parameters": serialize_array(
                scaled_parameters
            ),

            "parameter_contribution": (
                parameter_contribution
            ),

            # Timeline analysis
            "audio_timeline": timeline_analysis,

            "audio_segments": audio_segments,

            "ranked_audio_segments": (
                ranked_audio_segments
            ),

            "suspicious_segments": (
                suspicious_intervals
            ),

            "suspicious_intervals": (
                suspicious_intervals
            ),

            # Advanced analysis
            "advanced_audio_analysis": (
                advanced_analysis
            ),

            "audio_summary": audio_summary,

            "voice_dna": voice_dna,

            "audio_curves": audio_curves,

            "pause_intervals": (
                pause_intervals
            ),

            "breathing_events": (
                breathing_events
            ),

            "audio_investigation_summary": (
                investigation_summary
            ),

            # Visual evidence
            "waveform": visuals.get(
                "waveform"
            ),

            "spectrogram": visuals.get(
                "spectrogram"
            ),

            "audio_heatmap": visuals.get(
                "audio_heatmap"
            ),

            "lfcc_heatmap": visuals.get(
                "lfcc_heatmap"
            ),

            "pitch_plot": visuals.get(
                "pitch_plot"
            ),

            "energy_plot": visuals.get(
                "energy_plot"
            ),

            "audio_visual_evidence": {
                "waveform": visuals.get(
                    "waveform"
                ),

                "spectrogram": visuals.get(
                    "spectrogram"
                ),

                "audio_heatmap": visuals.get(
                    "audio_heatmap"
                ),

                "lfcc_heatmap": visuals.get(
                    "lfcc_heatmap"
                ),

                "pitch_plot": visuals.get(
                    "pitch_plot"
                ),

                "energy_plot": visuals.get(
                    "energy_plot"
                ),
            },

            # Versions
            "audio_analysis_version": (
                "FORGE-AUDIO-INVESTIGATION-3.1"
            ),

            "model_analysis_version": (
                "FORGE-CNN-BILSTM-FUSION"
            ),

            "segment_analysis_version": (
                timeline_analysis.get(
                    "analysis_version"
                )
            ),

            "advanced_analysis_version": (
                advanced_analysis.get(
                    "analysis_version"
                )
            ),

            "visual_analysis_version": (
                visuals.get(
                    "analysis_version",
                    "FORGE-AUDIO-VISUAL-XAI",
                )
            ),
        }

    except Exception as error:
        print(
            "❌ Audio pipeline failed:",
            error,
        )

        return {
            "error": (
                "Audio pipeline failed: "
                f"{str(error)}"
            ),
            "modality": "audio",
            "file_type": "audio",
        }