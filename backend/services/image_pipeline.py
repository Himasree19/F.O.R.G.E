from __future__ import annotations
import os
from typing import Any, Dict

from backend.continuous_learning.image.replay_manager import (
    FEATURE_COLUMNS,
    append_replay_record,
)
from backend.services.fake_image_detector import analyze_image
from backend.services.image_region_analysis import analyse_image_regions
from backend.services.image_visual_evidence import (
    generate_image_visual_evidence,
)

# =====================================================
# FEATURE EXTRACTION FUNCTION
# =====================================================

def extract_all_forensic_features(image_path: str) -> dict:
    features = {}

    # 1. Base detector features if analyze_image returned them
    try:
        base_features = analyze_image(image_path)
        if isinstance(base_features, dict):
            features.update(base_features.get("forensic_features", {}))
            features.update(base_features)
    except Exception:
        pass

    # Ensure all required FEATURE_COLUMNS exist in the returned dict
    for col in FEATURE_COLUMNS:
        if col not in features or features[col] is None:
            features[col] = 0.0

    return features


def _safe_result(result: Any) -> Dict[str, Any]:
    """Ensures the base image detector returned a valid dictionary."""
    if not isinstance(result, dict):
        return {"error": "The image detector returned an invalid response."}
    return result


def _build_visual_evidence_response(
    visual_evidence: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Removes internal filesystem paths from the API response and
    exposes only frontend-accessible URLs and interpretation metadata.
    """
    return {
        "heatmap": visual_evidence.get("heatmap_url"),
        "overlay": visual_evidence.get("overlay_url"),
        "naturalness_map": visual_evidence.get("naturalness_url"),
        "edge_map": visual_evidence.get("edge_map_url"),
        "frequency_map": visual_evidence.get("frequency_map_url"),
        "legend": visual_evidence.get(
            "legend",
            {
                "blue": "Strongly natural region",
                "cyan": "Likely natural region",
                "green": "Mostly natural region",
                "yellow": "Mixed or uncertain region",
                "orange": "Suspicious synthetic indicator",
                "red": "Strong AI or manipulation indicator",
            },
        ),
        "interpretation_notice": visual_evidence.get(
            "interpretation_notice",
            (
                "Red and orange regions contain stronger "
                "forensic irregularities. Green, cyan and blue "
                "regions show more natural image characteristics."
            ),
        ),
    }


# =====================================================
# MAIN PIPELINE FUNCTION
# =====================================================

def process_image(image_path: str) -> Dict[str, Any]:
    """
    Complete FORGE image-processing pipeline.
    """
    try:
        # 1. Extract complete forensic feature vector
        forensic_features = extract_all_forensic_features(image_path)

        # 2. Base Image Detector
        base_result = analyze_image(image_path)
        result = _safe_result(base_result)

        if result.get("error"):
            return result

        # 3. Visual Forensic Evidence
        visual_evidence = generate_image_visual_evidence(image_path)
        if not isinstance(visual_evidence, dict):
            return {"error": "Visual evidence generator returned an invalid response."}

        heatmap_path = visual_evidence.get("heatmap_path")

        # 4. Region + Patch Investigation
        region_analysis = analyse_image_regions(
            image_path=image_path,
            heatmap_path=heatmap_path,
        )

        if not isinstance(region_analysis, dict):
            region_analysis = {
                "face_detected": False,
                "regions": [],
                "ranked_regions": [],
                "hover_grid": {"patches": [], "ranked_patches": []},
                "error": "Regional investigation engine returned an invalid response.",
            }

        hover_analysis = region_analysis.get("hover_grid") or {
            "patches": [],
            "ranked_patches": [],
        }

        # 5. Frontend Visual Response
        frontend_visuals = _build_visual_evidence_response(visual_evidence)

        # 6. Model Probability Normalisation
        raw_ai_probability = result.get("raw_ai_probability") or result.get("raw_probability_fake") or result.get("risk_score", 0)
        raw_human_probability = result.get("raw_human_probability") or result.get("raw_probability_real")

        try:
            ai_probability = float(raw_ai_probability or 0)
            if 0 <= ai_probability <= 1:
                ai_probability *= 100
            ai_probability = max(0.0, min(100.0, ai_probability))
        except (TypeError, ValueError):
            ai_probability = 0.0

        if raw_human_probability is None:
            human_probability = 100.0 - ai_probability
        else:
            try:
                human_probability = float(raw_human_probability)
                if 0 <= human_probability <= 1:
                    human_probability *= 100
                human_probability = max(0.0, min(100.0, human_probability))
            except (TypeError, ValueError):
                human_probability = 100.0 - ai_probability

        # 7. Merge Complete Response
        result.update(
            {
                "heatmap": frontend_visuals.get("overlay"),
                "visual_evidence": frontend_visuals,
                "region_analysis": region_analysis,
                "hover_analysis": hover_analysis,
                "file_type": "image",
                "modality": "image",
                "probabilities": {
                    "ai": round(ai_probability, 2),
                    "human": round(human_probability, 2),
                },
                "image_analysis_version": "FORGE-IMAGE-INVESTIGATION-2.1",
                "visual_evidence_version": "FORGE-VISUAL-XAI-2.1",
                "regional_analysis_version": region_analysis.get(
                    "analysis_version", "FORGE-IMAGE-REGION-XAI"
                ),
                "uploaded_file": image_path,
                "forensic_features": forensic_features,
            }
        )

        # 8. Investigation Summary
        ranked_regions = region_analysis.get("ranked_regions") or []
        ranked_patches = hover_analysis.get("ranked_patches") or []

        result["image_investigation_summary"] = {
            "face_detected": region_analysis.get("face_detected", False),
            "semantic_region_count": len(region_analysis.get("regions", [])),
            "hover_patch_count": len(hover_analysis.get("patches", [])),
            "most_suspicious_region": ranked_regions[0] if ranked_regions else None,
            "most_suspicious_patch": ranked_patches[0] if ranked_patches else None,
            "interpretation": frontend_visuals.get("interpretation_notice"),
        }

        # 9. Log Record into Replay Dataset
        result["prediction"] = result.get("prediction", "AI" if ai_probability >= 50 else "HUMAN")
        result["confidence"] = max(result["probabilities"]["ai"], result["probabilities"]["human"])

        image_id = append_replay_record(result)
        result["image_id"] = image_id
        result["prediction"] = result.get("prediction", "AI" if ai_probability >= 50 else "HUMAN")
        result["confidence"] = max(result["probabilities"]["ai"], result["probabilities"]["human"])

        # Log record into replay dataset (updates if exists, creates 1 ID if new)
        image_id = append_replay_record(result)
        result["image_id"] = image_id

        return result

    except Exception as error:
        return {
            "error": f"Image pipeline failed: {str(error)}",
            "modality": "image",
            "file_type": "image",
        }