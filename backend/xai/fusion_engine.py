# ==========================================
# IMAGE FORENSIC FUSION ENGINE
# ==========================================

def calculate_fusion_score(

    cnn_confidence,

    metadata_score,

    artifact_score,

    semantic_score,

    color_score
):

    # ======================================
    # WEIGHTS
    # ======================================

    cnn_weight = 0.40

    metadata_weight = 0.15

    artifact_weight = 0.20

    semantic_weight = 0.15

    color_weight = 0.10

    # ======================================
    # FUSION
    # ======================================

    fusion_score = (

        cnn_confidence * cnn_weight +

        metadata_score * metadata_weight +

        artifact_score * artifact_weight +

        semantic_score * semantic_weight +

        color_score * color_weight
    )

    return round(
        fusion_score,
        2
    )


# ==========================================
# FINAL DECISION
# ==========================================

def fusion_decision(score):

    if score >= 70:

        return "AI"

    return "HUMAN"


# ==========================================
# FULL FUSION
# ==========================================

def run_fusion(

    cnn_result,

    metadata_result,

    artifact_result,

    semantic_result,

    color_result
):

    fusion_score = calculate_fusion_score(

        cnn_result["confidence"],

        metadata_result["score"],

        artifact_result["score"],

        semantic_result["score"],

        color_result["score"]
    )

    prediction = fusion_decision(
        fusion_score
    )

    return {

        "prediction": prediction,

        "confidence": fusion_score,

        "fusion_breakdown": {

            "cnn":
                cnn_result["confidence"],

            "metadata":
                metadata_result["score"],

            "artifacts":
                artifact_result["score"],

            "semantic":
                semantic_result["score"],

            "color":
                color_result["score"]
        }
    }