# ==========================================
# RISK ENGINE
# ==========================================

def get_risk(score):

    if score >= 85:
        return "HIGH"

    elif score >= 60:
        return "MEDIUM"

    return "LOW"


# ==========================================
# VISUAL ARTIFACTS
# ==========================================

def visual_artifact_reason(score):

    return {

        "score": round(score, 2),

        "risk": get_risk(score),

        "reason":
            "Synthetic texture inconsistencies detected"
            if score >= 60
            else
            "Natural visual texture patterns observed"
    }


# ==========================================
# FREQUENCY ANALYSIS
# ==========================================

def frequency_reason(score):

    return {

        "score": round(score, 2),

        "risk": get_risk(score),

        "reason":
            "Abnormal GAN frequency signatures detected"
            if score >= 60
            else
            "Natural spectral frequency distribution observed"
    }


# ==========================================
# METADATA ANALYSIS
# ==========================================

def metadata_reason(score):

    return {

        "score": round(score, 2),

        "risk": get_risk(score),

        "reason":
            "Metadata inconsistencies detected"
            if score >= 60
            else
            "Metadata structure appears authentic"
    }


# ==========================================
# SEMANTIC CONSISTENCY
# ==========================================

def semantic_reason(score):

    return {

        "score": round(score, 2),

        "risk": get_risk(score),

        "reason":
            "Semantic inconsistencies detected"
            if score >= 60
            else
            "Visual semantics appear natural"
    }