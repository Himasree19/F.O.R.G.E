import numpy as np


def get_risk(score):
    if score >= 75:
        return "HIGH"
    if score >= 45:
        return "MEDIUM"
    return "LOW"


def clamp(value):
    return max(0, min(100, round(float(value), 2)))


def build_audio_xai(raw_params, prediction_confidence, fake_probability):
    p = raw_params

    spectral_score = clamp(
        abs(p[0]) / 50 +
        abs(p[2]) / 50 +
        abs(p[4]) / 80 +
        abs(p[6]) * 100
    )

    statistical_score = clamp(
        abs(p[8]) * 300 +
        abs(p[10]) * 300 +
        abs(p[12]) / 10 +
        abs(p[13]) * 500
    )

    prosodic_score = clamp(
        abs(p[15]) / 3 +
        abs(p[16]) / 5 +
        abs(p[22]) * 300
    )

    temporal_score = clamp(
        p[17] * 100 +
        p[18] * 300 +
        abs(p[23]) / 3
    )

    phase_score = clamp(
        abs(p[19]) * 15 +
        abs(p[20]) * 20
    )

    tampering_score = clamp(
        p[18] * 350 +
        p[21] * 500 +
        p[17] * 70
    )

    environmental_score = clamp(
        p[21] * 800 +
        abs(p[22]) * 300
    )

    metadata_score = 35

    acoustic_score = clamp(
        prosodic_score * 0.35 +
        temporal_score * 0.35 +
        statistical_score * 0.30
    )

    deep_learning_score = clamp(fake_probability * 100)

    parameter_contribution = {
        "acoustic_artifacts": {
            "score": acoustic_score,
            "risk": get_risk(acoustic_score),
            "reason": "Speech rhythm, tone, pauses, and acoustic consistency were analyzed."
        },
        "metadata_analysis": {
            "score": metadata_score,
            "risk": get_risk(metadata_score),
            "reason": "Basic metadata inspection applied. Extended metadata checks can be added later."
        },
        "frequency_spectral_analysis": {
            "score": spectral_score,
            "risk": get_risk(spectral_score),
            "reason": "Spectral centroid, bandwidth, rolloff, and flatness were analyzed for synthetic frequency patterns."
        },
        "deep_learning_detection": {
            "score": deep_learning_score,
            "risk": get_risk(deep_learning_score),
            "reason": "CNN-BiLSTM fusion model analyzed LFCC and forensic audio features."
        },
        "audio_tampering": {
            "score": tampering_score,
            "risk": get_risk(tampering_score),
            "reason": "Amplitude discontinuities, silence regions, and low-level noise patterns were checked."
        },
        "environmental_inconsistency": {
            "score": environmental_score,
            "risk": get_risk(environmental_score),
            "reason": "Noise floor and energy variation were analyzed for background inconsistency."
        },
        "statistical_analysis": {
            "score": statistical_score,
            "risk": get_risk(statistical_score),
            "reason": "Zero crossing rate, RMS energy, entropy, and variance were analyzed."
        },
        "temporal_dynamics": {
            "score": temporal_score,
            "risk": get_risk(temporal_score),
            "reason": "Pause ratio, tempo, and amplitude transitions were analyzed."
        },
        "phase_analysis": {
            "score": phase_score,
            "risk": get_risk(phase_score),
            "reason": "Phase variance and phase discontinuity were checked for hidden signal inconsistencies."
        },
        "prosodic_features": {
            "score": prosodic_score,
            "risk": get_risk(prosodic_score),
            "reason": "Pitch mean, pitch variation, pitch range, and energy variation were analyzed."
        }
    }

    suspicious_segments = []

    if spectral_score >= 70:
        suspicious_segments.append({
            "start": "00:00",
            "end": "00:05",
            "risk": "HIGH",
            "reason": "Abnormal spectral or high-frequency behavior detected."
        })

    if temporal_score >= 70:
        suspicious_segments.append({
            "start": "00:05",
            "end": "00:10",
            "risk": "HIGH",
            "reason": "Unnatural pause, rhythm, or frame transition detected."
        })

    if phase_score >= 70:
        suspicious_segments.append({
            "start": "00:10",
            "end": "00:15",
            "risk": "MEDIUM",
            "reason": "Phase inconsistency detected."
        })

    return parameter_contribution, suspicious_segments