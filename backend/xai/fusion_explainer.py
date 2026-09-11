# ==========================================
# FUSION EXPLAINER
# ==========================================

def explain_fusion(fusion_breakdown):

    dominant_factor = max(
        fusion_breakdown,
        key=fusion_breakdown.get
    )

    explanations = {

        "cnn":
        "CNN model detected patterns associated with AI-generated imagery.",

        "metadata":
        "Image metadata contains anomalies or missing authenticity indicators.",

        "artifacts":
        "Visual artifacts suggest synthetic image generation or manipulation.",

        "semantic":
        "Semantic consistency checks found unusual visual relationships.",

        "color":
        "Color-space analysis detected abnormal spectral distributions."
    }

    return {

        "dominant_factor":
            dominant_factor,

        "explanation":
            explanations.get(
                dominant_factor,
                "No explanation available."
            )
    }