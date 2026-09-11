# ==========================================
# PARAMETER REASONING ENGINE
# ==========================================

def get_risk_level(score):

    if score >= 75:
        return "VERY HIGH"

    elif score >= 55:
        return "HIGH"

    elif score >= 35:
        return "MEDIUM"

    else:
        return "LOW"


# ==========================================
# STYLOMETRIC
# ==========================================

def stylometric_reason(score):

    risk = get_risk_level(score)

    if score >= 75:

        reason = (
            "Highly uniform sentence structure "
            "and low linguistic variation detected"
        )

    elif score >= 55:

        reason = (
            "Consistent writing patterns "
            "typical of generated text detected"
        )

    elif score >= 35:

        reason = (
            "Moderate stylometric irregularities found"
        )

    else:

        reason = (
            "Natural human writing variability observed"
        )

    return {
        "score": round(score, 2),
        "risk": risk,
        "reason": reason
    }


# ==========================================
# TF-IDF
# ==========================================

def tfidf_reason(score):

    risk = get_risk_level(score)

    if score >= 75:

        reason = (
            "High repetitive keyword frequency "
            "detected across the text"
        )

    elif score >= 55:

        reason = (
            "Repeated lexical patterns "
            "suggest possible AI generation"
        )

    elif score >= 35:

        reason = (
            "Moderate token repetition detected"
        )

    else:

        reason = (
            "Natural vocabulary distribution observed"
        )

    return {
        "score": round(score, 2),
        "risk": risk,
        "reason": reason
    }


# ==========================================
# NGRAM
# ==========================================

def ngram_reason(score):

    risk = get_risk_level(score)

    if score >= 75:

        reason = (
            "Highly repeated phrase-transition "
            "patterns detected"
        )

    elif score >= 55:

        reason = (
            "Structured phrase sequencing "
            "suggests generated content"
        )

    elif score >= 35:

        reason = (
            "Moderate n-gram repetition observed"
        )

    else:

        reason = (
            "Natural phrase diversity detected"
        )

    return {
        "score": round(score, 2),
        "risk": risk,
        "reason": reason
    }


# ==========================================
# SEMANTIC
# ==========================================

def semantic_reason(score):

    risk = get_risk_level(score)

    if score >= 75:

        reason = (
            "High semantic consistency and "
            "low contextual diversity detected"
        )

    elif score >= 55:

        reason = (
            "Semantically coherent patterns "
            "similar to AI-generated text"
        )

    elif score >= 35:

        reason = (
            "Moderate semantic regularity observed"
        )

    else:

        reason = (
            "Natural contextual variation detected"
        )

    return {
        "score": round(score, 2),
        "risk": risk,
        "reason": reason
    }