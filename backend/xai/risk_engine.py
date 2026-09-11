# ==========================================
# DEEPFAKECONNECT RISK ENGINE
# ==========================================

def calculate_risk(confidence):

    if confidence >= 85:

        return {

            "risk_level": "CRITICAL",

            "risk_score": confidence,

            "recommendation":
                "High probability of AI manipulation. Manual verification strongly recommended."
        }

    elif confidence >= 70:

        return {

            "risk_level": "HIGH",

            "risk_score": confidence,

            "recommendation":
                "Likely AI-generated or manipulated content."
        }

    elif confidence >= 40:

        return {

            "risk_level": "MEDIUM",

            "risk_score": confidence,

            "recommendation":
                "Suspicious content detected. Additional review advised."
        }

    return {

        "risk_level": "LOW",

        "risk_score": confidence,

        "recommendation":
            "Content appears authentic."
    }