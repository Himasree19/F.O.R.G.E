from __future__ import annotations

from typing import Any, Dict, List

import librosa
import numpy as np


SAMPLE_RATE = 16000

SEGMENT_DURATION = 1.5
SEGMENT_HOP_DURATION = 0.75

MIN_SEGMENT_DURATION = 0.35


def clamp(
    value: float,
    minimum: float = 0.0,
    maximum: float = 1.0,
) -> float:
    return max(
        minimum,
        min(
            maximum,
            float(value),
        ),
    )


def safe_mean(
    values: np.ndarray,
) -> float:
    if values is None or values.size == 0:
        return 0.0

    return float(
        np.mean(values)
    )


def safe_std(
    values: np.ndarray,
) -> float:
    if values is None or values.size == 0:
        return 0.0

    return float(
        np.std(values)
    )


def safe_percentile(
    values: np.ndarray,
    percentile: float,
) -> float:
    if values is None or values.size == 0:
        return 0.0

    return float(
        np.percentile(
            values,
            percentile,
        )
    )


def format_timestamp(
    seconds: float,
) -> str:
    seconds = max(
        0.0,
        float(seconds),
    )

    minutes = int(
        seconds // 60
    )

    remaining_seconds = (
        seconds % 60
    )

    return (
        f"{minutes:02d}:"
        f"{remaining_seconds:05.2f}"
    )


def calculate_entropy(
    signal: np.ndarray,
) -> float:
    if signal.size == 0:
        return 0.0

    histogram, _ = np.histogram(
        signal,
        bins=50,
        density=True,
    )

    histogram = histogram + 1e-8

    return float(
        -np.sum(
            histogram
            * np.log2(histogram)
        )
    )


def extract_pitch_features(
    segment: np.ndarray,
) -> Dict[str, float]:
    try:
        pitch = librosa.yin(
            segment,
            fmin=50,
            fmax=500,
            sr=SAMPLE_RATE,
        )

        pitch = pitch[
            np.isfinite(pitch)
        ]

        if pitch.size == 0:
            return {
                "pitch_mean": 0.0,
                "pitch_std": 0.0,
                "pitch_range": 0.0,
                "pitch_stability": 0.0,
            }

        pitch_mean = safe_mean(
            pitch
        )

        pitch_std = safe_std(
            pitch
        )

        pitch_range = float(
            np.max(pitch)
            - np.min(pitch)
        )

        pitch_stability = clamp(
            1.0
            - (
                pitch_std
                / max(
                    pitch_mean,
                    1.0,
                )
            )
        )

        return {
            "pitch_mean": pitch_mean,
            "pitch_std": pitch_std,
            "pitch_range": pitch_range,
            "pitch_stability": pitch_stability,
        }

    except Exception:
        return {
            "pitch_mean": 0.0,
            "pitch_std": 0.0,
            "pitch_range": 0.0,
            "pitch_stability": 0.0,
        }


def extract_phase_features(
    segment: np.ndarray,
) -> Dict[str, float]:
    try:
        stft = librosa.stft(
            segment
        )

        phase = np.angle(
            stft
        )

        phase_difference = np.diff(
            phase,
            axis=1,
        )

        return {
            "phase_variance": float(
                np.var(phase)
            ),

            "phase_discontinuity": safe_mean(
                np.abs(
                    phase_difference
                )
            ),
        }

    except Exception:
        return {
            "phase_variance": 0.0,
            "phase_discontinuity": 0.0,
        }


def extract_segment_features(
    segment: np.ndarray,
) -> Dict[str, float]:
    spectral_centroid = (
        librosa.feature.spectral_centroid(
            y=segment,
            sr=SAMPLE_RATE,
        )[0]
    )

    spectral_bandwidth = (
        librosa.feature.spectral_bandwidth(
            y=segment,
            sr=SAMPLE_RATE,
        )[0]
    )

    spectral_rolloff = (
        librosa.feature.spectral_rolloff(
            y=segment,
            sr=SAMPLE_RATE,
        )[0]
    )

    spectral_flatness = (
        librosa.feature.spectral_flatness(
            y=segment
        )[0]
    )

    zero_crossing_rate = (
        librosa.feature.zero_crossing_rate(
            segment
        )[0]
    )

    rms_energy = (
        librosa.feature.rms(
            y=segment
        )[0]
    )

    pitch_features = (
        extract_pitch_features(
            segment
        )
    )

    phase_features = (
        extract_phase_features(
            segment
        )
    )

    amplitude_difference = np.diff(
        segment
    )

    pause_ratio = float(
        np.mean(
            np.abs(segment)
            < 0.02
        )
    )

    noise_floor = safe_percentile(
        np.abs(segment),
        10,
    )

    peak_amplitude = float(
        np.max(
            np.abs(segment)
        )
    ) if segment.size else 0.0

    crest_factor = (
        peak_amplitude
        / max(
            safe_mean(rms_energy),
            1e-8,
        )
    )

    features = {
        "spectral_centroid": safe_mean(
            spectral_centroid
        ),

        "spectral_centroid_variation": safe_std(
            spectral_centroid
        ),

        "spectral_bandwidth": safe_mean(
            spectral_bandwidth
        ),

        "spectral_rolloff": safe_mean(
            spectral_rolloff
        ),

        "spectral_flatness": safe_mean(
            spectral_flatness
        ),

        "zero_crossing_rate": safe_mean(
            zero_crossing_rate
        ),

        "rms_energy": safe_mean(
            rms_energy
        ),

        "energy_variation": safe_std(
            rms_energy
        ),

        "pause_ratio": pause_ratio,

        "noise_floor": noise_floor,

        "amplitude_discontinuity": safe_mean(
            np.abs(
                amplitude_difference
            )
        ),

        "entropy": calculate_entropy(
            segment
        ),

        "crest_factor": crest_factor,

        **pitch_features,

        **phase_features,
    }

    return features


def calculate_feature_risks(
    features: Dict[str, float],
) -> Dict[str, float]:
    pitch_mean = features[
        "pitch_mean"
    ]

    pitch_std = features[
        "pitch_std"
    ]

    pitch_ratio = (
        pitch_std
        / max(
            pitch_mean,
            1.0,
        )
    )

    pitch_risk = clamp(
        abs(
            pitch_ratio - 0.12
        ) / 0.22
    )

    energy_risk = clamp(
        abs(
            features[
                "energy_variation"
            ] - 0.018
        ) / 0.045
    )

    flatness_risk = clamp(
        abs(
            features[
                "spectral_flatness"
            ] - 0.12
        ) / 0.28
    )

    zcr_risk = clamp(
        abs(
            features[
                "zero_crossing_rate"
            ] - 0.08
        ) / 0.20
    )

    pause_risk = clamp(
        abs(
            features[
                "pause_ratio"
            ] - 0.15
        ) / 0.45
    )

    phase_risk = clamp(
        features[
            "phase_discontinuity"
        ] / 2.2
    )

    noise_risk = clamp(
        abs(
            features[
                "noise_floor"
            ] - 0.005
        ) / 0.035
    )

    frequency_risk = clamp(
        abs(
            features[
                "spectral_centroid"
            ] - 1900
        ) / 2600
    )

    bandwidth_risk = clamp(
        abs(
            features[
                "spectral_bandwidth"
            ] - 1700
        ) / 2400
    )

    amplitude_risk = clamp(
        features[
            "amplitude_discontinuity"
        ] / 0.09
    )

    entropy_risk = clamp(
        abs(
            features[
                "entropy"
            ] - 18
        ) / 22
    )

    stability_risk = clamp(
        (
            features[
                "pitch_stability"
            ] - 0.92
        ) / 0.08
    )

    return {
        "pitch": pitch_risk,
        "energy": energy_risk,
        "spectral_flatness": flatness_risk,
        "zero_crossing": zcr_risk,
        "pause_pattern": pause_risk,
        "phase": phase_risk,
        "noise": noise_risk,
        "frequency": frequency_risk,
        "bandwidth": bandwidth_risk,
        "amplitude": amplitude_risk,
        "entropy": entropy_risk,
        "voice_stability": stability_risk,
    }


def calculate_segment_risk(
    risks: Dict[str, float],
    global_fake_probability: float,
) -> float:
    acoustic_score = (
        risks["pitch"] * 0.13
        + risks["energy"] * 0.09
        + risks["spectral_flatness"] * 0.10
        + risks["zero_crossing"] * 0.07
        + risks["pause_pattern"] * 0.09
        + risks["phase"] * 0.11
        + risks["noise"] * 0.08
        + risks["frequency"] * 0.09
        + risks["bandwidth"] * 0.06
        + risks["amplitude"] * 0.07
        + risks["entropy"] * 0.05
        + risks["voice_stability"] * 0.06
    )

    global_probability = clamp(
        global_fake_probability
    )

    combined_score = (
        acoustic_score * 0.72
        + global_probability * 0.28
    )

    return clamp(
        combined_score
    )


def get_risk_level(
    score: float,
) -> str:
    percent = score * 100

    if percent >= 75:
        return "HIGH"

    if percent >= 45:
        return "MEDIUM"

    return "LOW"


def get_prediction(
    score: float,
) -> str:
    if score >= 0.55:
        return "FAKE"

    return "REAL"


def generate_segment_reasons(
    risks: Dict[str, float],
) -> List[str]:
    reason_map = {
        "pitch": (
            "Pitch variation differs from expected natural speech behaviour."
        ),

        "energy": (
            "Energy movement is unusually uniform or unstable."
        ),

        "spectral_flatness": (
            "Spectral flatness suggests artificial harmonic or noise structure."
        ),

        "zero_crossing": (
            "Zero-crossing behaviour differs from typical recorded speech."
        ),

        "pause_pattern": (
            "Pause and silence behaviour appears mechanically distributed."
        ),

        "phase": (
            "Phase continuity contains abrupt or synthetic-looking transitions."
        ),

        "noise": (
            "Background noise is inconsistent with natural recording conditions."
        ),

        "frequency": (
            "Spectral centroid indicates unusual high-frequency distribution."
        ),

        "bandwidth": (
            "Spectral bandwidth differs from expected human speech variation."
        ),

        "amplitude": (
            "Amplitude transitions contain abrupt discontinuities."
        ),

        "entropy": (
            "Signal complexity differs from normal speech variability."
        ),

        "voice_stability": (
            "Voice pitch is unusually stable, which may indicate synthesis."
        ),
    }

    ranked = sorted(
        risks.items(),
        key=lambda item: item[1],
        reverse=True,
    )

    reasons = []

    for key, value in ranked:
        if value >= 0.58:
            reasons.append(
                reason_map[key]
            )

    if not reasons:
        reasons.append(
            "No dominant synthetic acoustic irregularity was identified."
        )

    return reasons[:4]


def build_segment_record(
    *,
    segment: np.ndarray,
    segment_index: int,
    start_time: float,
    end_time: float,
    global_fake_probability: float,
) -> Dict[str, Any]:
    features = extract_segment_features(
        segment
    )

    risks = calculate_feature_risks(
        features
    )

    segment_risk = calculate_segment_risk(
        risks,
        global_fake_probability,
    )

    risk_level = get_risk_level(
        segment_risk
    )

    prediction = get_prediction(
        segment_risk
    )

    confidence = (
        segment_risk
        if prediction == "FAKE"
        else 1.0 - segment_risk
    )

    return {
        "id": (
            f"audio_segment_"
            f"{segment_index:04d}"
        ),

        "index": segment_index,

        "start_seconds": round(
            start_time,
            3,
        ),

        "end_seconds": round(
            end_time,
            3,
        ),

        "start": format_timestamp(
            start_time
        ),

        "end": format_timestamp(
            end_time
        ),

        "duration_seconds": round(
            end_time - start_time,
            3,
        ),

        "prediction": prediction,

        "confidence": round(
            confidence * 100,
            2,
        ),

        "risk_score": round(
            segment_risk * 100,
            2,
        ),

        "risk_level": risk_level,

        "features": {
            key: round(
                float(value),
                6,
            )
            for key, value
            in features.items()
        },

        "metrics": {
            key: round(
                value * 100,
                2,
            )
            for key, value
            in risks.items()
        },

        "reasons": generate_segment_reasons(
            risks
        ),
    }


def merge_suspicious_segments(
    segments: List[
        Dict[str, Any]
    ],
) -> List[Dict[str, Any]]:
    suspicious = [
        segment
        for segment in segments
        if segment[
            "risk_level"
        ] in {
            "MEDIUM",
            "HIGH",
        }
    ]

    if not suspicious:
        return []

    merged = []

    current = {
        "start_seconds": suspicious[0][
            "start_seconds"
        ],

        "end_seconds": suspicious[0][
            "end_seconds"
        ],

        "scores": [
            suspicious[0][
                "risk_score"
            ]
        ],

        "reasons": list(
            suspicious[0][
                "reasons"
            ]
        ),
    }

    for segment in suspicious[1:]:
        if (
            segment[
                "start_seconds"
            ]
            <= current[
                "end_seconds"
            ] + SEGMENT_HOP_DURATION
        ):
            current[
                "end_seconds"
            ] = max(
                current[
                    "end_seconds"
                ],
                segment[
                    "end_seconds"
                ],
            )

            current[
                "scores"
            ].append(
                segment[
                    "risk_score"
                ]
            )

            current[
                "reasons"
            ].extend(
                segment[
                    "reasons"
                ]
            )

        else:
            merged.append(
                current
            )

            current = {
                "start_seconds": segment[
                    "start_seconds"
                ],

                "end_seconds": segment[
                    "end_seconds"
                ],

                "scores": [
                    segment[
                        "risk_score"
                    ]
                ],

                "reasons": list(
                    segment[
                        "reasons"
                    ]
                ),
            }

    merged.append(
        current
    )

    output = []

    for index, interval in enumerate(
        merged,
        start=1,
    ):
        average_score = float(
            np.mean(
                interval["scores"]
            )
        )

        risk = (
            "HIGH"
            if average_score >= 75
            else "MEDIUM"
        )

        unique_reasons = list(
            dict.fromkeys(
                interval["reasons"]
            )
        )

        output.append(
            {
                "id": (
                    f"suspicious_interval_"
                    f"{index:03d}"
                ),

                "start_seconds": interval[
                    "start_seconds"
                ],

                "end_seconds": interval[
                    "end_seconds"
                ],

                "start": format_timestamp(
                    interval[
                        "start_seconds"
                    ]
                ),

                "end": format_timestamp(
                    interval[
                        "end_seconds"
                    ]
                ),

                "score": round(
                    average_score,
                    2,
                ),

                "risk_score": round(
                    average_score,
                    2,
                ),

                "risk": risk,

                "risk_level": risk,

                "reason": (
                    unique_reasons[0]
                    if unique_reasons
                    else (
                        "Synthetic acoustic "
                        "indicators detected."
                    )
                ),

                "reasons": (
                    unique_reasons[:4]
                ),
            }
        )

    return output


def analyze_audio_segments(
    file_path: str,
    global_fake_probability: float,
) -> Dict[str, Any]:
    audio, _ = librosa.load(
        file_path,
        sr=SAMPLE_RATE,
        mono=True,
    )

    if audio.size == 0:
        raise ValueError(
            "Audio file contains no samples."
        )

    segment_samples = int(
        SEGMENT_DURATION
        * SAMPLE_RATE
    )

    hop_samples = int(
        SEGMENT_HOP_DURATION
        * SAMPLE_RATE
    )

    segments = []

    segment_index = 0

    for start_sample in range(
        0,
        len(audio),
        hop_samples,
    ):
        end_sample = min(
            len(audio),
            start_sample
            + segment_samples,
        )

        segment = audio[
            start_sample:end_sample
        ]

        duration = (
            len(segment)
            / SAMPLE_RATE
        )

        if (
            duration
            < MIN_SEGMENT_DURATION
        ):
            continue

        start_time = (
            start_sample
            / SAMPLE_RATE
        )

        end_time = (
            end_sample
            / SAMPLE_RATE
        )

        segments.append(
            build_segment_record(
                segment=segment,
                segment_index=segment_index,
                start_time=start_time,
                end_time=end_time,
                global_fake_probability=(
                    global_fake_probability
                ),
            )
        )

        segment_index += 1

    ranked_segments = sorted(
        segments,
        key=lambda item: item[
            "risk_score"
        ],
        reverse=True,
    )

    suspicious_intervals = (
        merge_suspicious_segments(
            segments
        )
    )

    return {
        "sample_rate": SAMPLE_RATE,

        "audio_duration": round(
            len(audio)
            / SAMPLE_RATE,
            3,
        ),

        "segment_duration": (
            SEGMENT_DURATION
        ),

        "segment_hop_duration": (
            SEGMENT_HOP_DURATION
        ),

        "segment_count": len(
            segments
        ),

        "segments": segments,

        "ranked_segments": (
            ranked_segments[:30]
        ),

        "suspicious_intervals": (
            suspicious_intervals
        ),

        "analysis_version": (
            "FORGE-AUDIO-SEGMENT-XAI-1.0"
        ),
    }