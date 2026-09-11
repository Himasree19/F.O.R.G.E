# ==========================================
# STYLOMETRIC
# ==========================================

def stylometric_reason(score):

    if score >= 80:

        return {
            "score": score,
            "risk": "HIGH",
            "reason":
            "Low linguistic variation and repetitive writing patterns detected"
        }

    elif score >= 50:

        return {
            "score": score,
            "risk": "MEDIUM",
            "reason":
            "Moderate sentence uniformity observed"
        }

    return {
        "score": score,
        "risk": "LOW",
        "reason":
        "Natural human writing variability observed"
    }

# ==========================================
# TFIDF
# ==========================================

def tfidf_reason(score):

    if score >= 80:

        return {
            "score": score,
            "risk": "HIGH",
            "reason":
            "Strong AI-correlated keyword repetition detected"
        }

    elif score >= 50:

        return {
            "score": score,
            "risk": "MEDIUM",
            "reason":
            "Partial repetitive vocabulary usage observed"
        }

    return {
        "score": score,
        "risk": "LOW",
        "reason":
        "Natural vocabulary distribution observed"
    }

# ==========================================
# NGRAM
# ==========================================

def ngram_reason(score):

    if score >= 80:

        return {
            "score": score,
            "risk": "HIGH",
            "reason":
            "Repeated phrase structures associated with AI text detected"
        }

    elif score >= 50:

        return {
            "score": score,
            "risk": "MEDIUM",
            "reason":
            "Moderate phrase repetition observed"
        }

    return {
        "score": score,
        "risk": "LOW",
        "reason":
        "Natural phrase diversity detected"
    }

# ==========================================
# SEMANTIC
# ==========================================

def semantic_reason(score):

    if score >= 80:

        return {
            "score": score,
            "risk": "VERY HIGH",
            "reason":
            "High semantic consistency and low contextual diversity detected"
        }

    elif score >= 50:

        return {
            "score": score,
            "risk": "HIGH",
            "reason":
            "Semantically coherent patterns similar to AI-generated text"
        }

    return {
        "score": score,
        "risk": "LOW",
        "reason":
        "Natural contextual variation detected"
    }