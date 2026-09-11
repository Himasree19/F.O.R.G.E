from __future__ import annotations

from typing import Any, Dict, List, Tuple

import librosa
import numpy as np
from scipy.signal import find_peaks


SAMPLE_RATE = 16000
FRAME_LENGTH = 2048
HOP_LENGTH = 512

MIN_PITCH = 50
MAX_PITCH = 500


# =========================================================
# GENERIC HELPERS
# =========================================================

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
    values = np.asarray(values)

    if values.size == 0:
        return 0.0

    finite = values[
        np.isfinite(values)
    ]

    if finite.size == 0:
        return 0.0

    return float(
        np.mean(finite)
    )


def safe_std(
    values: np.ndarray,
) -> float:
    values = np.asarray(values)

    if values.size == 0:
        return 0.0

    finite = values[
        np.isfinite(values)
    ]

    if finite.size == 0:
        return 0.0

    return float(
        np.std(finite)
    )


def safe_percentile(
    values: np.ndarray,
    percentile: float,
) -> float:
    values = np.asarray(values)

    if values.size == 0:
        return 0.0

    finite = values[
        np.isfinite(values)
    ]

    if finite.size == 0:
        return 0.0

    return float(
        np.percentile(
            finite,
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

    remaining = (
        seconds % 60
    )

    return (
        f"{minutes:02d}:"
        f"{remaining:05.2f}"
    )


def normalize_series(
    values: np.ndarray,
) -> np.ndarray:
    values = np.asarray(
        values,
        dtype=np.float32,
    )

    if values.size == 0:
        return values

    minimum = float(
        np.min(values)
    )

    maximum = float(
        np.max(values)
    )

    difference = (
        maximum - minimum
    )

    if difference <= 1e-8:
        return np.zeros_like(
            values
        )

    return (
        values - minimum
    ) / difference


def risk_level(
    score: float,
) -> str:
    if score >= 75:
        return "HIGH"

    if score >= 45:
        return "MEDIUM"

    return "LOW"


# =========================================================
# AUDIO LOADING
# =========================================================

def load_audio(
    file_path: str,
) -> np.ndarray:
    audio, _ = librosa.load(
        file_path,
        sr=SAMPLE_RATE,
        mono=True,
    )

    audio = np.asarray(
        audio,
        dtype=np.float32,
    )

    if audio.size == 0:
        raise ValueError(
            "Audio file contains no samples."
        )

    if not np.all(
        np.isfinite(audio)
    ):
        raise ValueError(
            "Audio contains NaN or infinite values."
        )

    return audio


# =========================================================
# TIME-SERIES EXTRACTION
# =========================================================

def extract_pitch_curve(
    audio: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    pitch = librosa.yin(
        audio,
        fmin=MIN_PITCH,
        fmax=MAX_PITCH,
        sr=SAMPLE_RATE,
        frame_length=FRAME_LENGTH,
        hop_length=HOP_LENGTH,
    )

    pitch = np.asarray(
        pitch,
        dtype=np.float32,
    )

    times = librosa.frames_to_time(
        np.arange(
            len(pitch)
        ),
        sr=SAMPLE_RATE,
        hop_length=HOP_LENGTH,
    )

    invalid = (
        ~np.isfinite(pitch)
        | (pitch < MIN_PITCH)
        | (pitch > MAX_PITCH)
    )

    pitch[invalid] = 0.0

    return times, pitch


def extract_energy_curve(
    audio: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    rms = librosa.feature.rms(
        y=audio,
        frame_length=FRAME_LENGTH,
        hop_length=HOP_LENGTH,
    )[0]

    rms = np.asarray(
        rms,
        dtype=np.float32,
    )

    times = librosa.frames_to_time(
        np.arange(
            len(rms)
        ),
        sr=SAMPLE_RATE,
        hop_length=HOP_LENGTH,
    )

    return times, rms


def extract_spectral_flux(
    audio: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    stft = np.abs(
        librosa.stft(
            audio,
            n_fft=FRAME_LENGTH,
            hop_length=HOP_LENGTH,
        )
    )

    normalized = stft / (
        np.sum(
            stft,
            axis=0,
            keepdims=True,
        )
        + 1e-8
    )

    difference = np.diff(
        normalized,
        axis=1,
    )

    flux = np.sqrt(
        np.sum(
            np.maximum(
                difference,
                0,
            ) ** 2,
            axis=0,
        )
    )

    flux = np.pad(
        flux,
        (1, 0),
        mode="constant",
    )

    times = librosa.frames_to_time(
        np.arange(
            len(flux)
        ),
        sr=SAMPLE_RATE,
        hop_length=HOP_LENGTH,
    )

    return times, flux


# =========================================================
# VOICE STABILITY
# =========================================================

def calculate_jitter(
    pitch: np.ndarray,
) -> float:
    voiced_pitch = pitch[
        pitch > 0
    ]

    if voiced_pitch.size < 3:
        return 0.0

    periods = (
        1.0 / voiced_pitch
    )

    period_difference = np.abs(
        np.diff(periods)
    )

    jitter = (
        safe_mean(
            period_difference
        )
        / max(
            safe_mean(periods),
            1e-8,
        )
    )

    return float(jitter)


def calculate_shimmer(
    rms: np.ndarray,
) -> float:
    voiced_energy = rms[
        rms > 1e-5
    ]

    if voiced_energy.size < 3:
        return 0.0

    difference = np.abs(
        np.diff(
            voiced_energy
        )
    )

    shimmer = (
        safe_mean(difference)
        / max(
            safe_mean(
                voiced_energy
            ),
            1e-8,
        )
    )

    return float(shimmer)


def calculate_pitch_stability(
    pitch: np.ndarray,
) -> float:
    voiced_pitch = pitch[
        pitch > 0
    ]

    if voiced_pitch.size < 3:
        return 0.0

    coefficient_of_variation = (
        safe_std(
            voiced_pitch
        )
        / max(
            safe_mean(
                voiced_pitch
            ),
            1.0,
        )
    )

    return clamp(
        1.0
        - coefficient_of_variation
    )


# =========================================================
# PAUSE AND BREATHING ANALYSIS
# =========================================================

def detect_pause_intervals(
    rms: np.ndarray,
    times: np.ndarray,
) -> List[Dict[str, Any]]:
    if rms.size == 0:
        return []

    threshold = max(
        safe_percentile(
            rms,
            20,
        ),
        0.003,
    )

    silent = (
        rms <= threshold
    )

    intervals: List[
        Dict[str, Any]
    ] = []

    start_index = None

    for index, is_silent in enumerate(
        silent
    ):
        if is_silent and start_index is None:
            start_index = index

        if (
            not is_silent
            and start_index is not None
        ):
            end_index = (
                index - 1
            )

            start_time = float(
                times[start_index]
            )

            end_time = float(
                times[end_index]
            )

            duration = (
                end_time
                - start_time
            )

            if duration >= 0.18:
                intervals.append(
                    {
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
                        "duration": round(
                            duration,
                            3,
                        ),
                    }
                )

            start_index = None

    return intervals


def detect_breath_candidates(
    audio: np.ndarray,
    rms: np.ndarray,
    spectral_flatness: np.ndarray,
    times: np.ndarray,
) -> List[Dict[str, Any]]:
    if (
        rms.size == 0
        or spectral_flatness.size == 0
    ):
        return []

    length = min(
        len(rms),
        len(spectral_flatness),
        len(times),
    )

    rms = rms[:length]
    spectral_flatness = (
        spectral_flatness[:length]
    )
    times = times[:length]

    low_energy_threshold = (
        safe_percentile(
            rms,
            40,
        )
    )

    high_flatness_threshold = (
        safe_percentile(
            spectral_flatness,
            65,
        )
    )

    candidate_score = (
        (
            rms
            <= low_energy_threshold
        ).astype(
            np.float32
        )
        * 0.45
        + (
            spectral_flatness
            >= high_flatness_threshold
        ).astype(
            np.float32
        )
        * 0.55
    )

    peaks, properties = find_peaks(
        candidate_score,
        height=0.8,
        distance=max(
            1,
            int(
                0.3
                * SAMPLE_RATE
                / HOP_LENGTH
            ),
        ),
    )

    output = []

    for index, peak in enumerate(
        peaks,
        start=1,
    ):
        timestamp = float(
            times[peak]
        )

        output.append(
            {
                "id": (
                    f"breath_"
                    f"{index:03d}"
                ),
                "time_seconds": round(
                    timestamp,
                    3,
                ),
                "time": format_timestamp(
                    timestamp
                ),
                "confidence": round(
                    float(
                        properties[
                            "peak_heights"
                        ][index - 1]
                    )
                    * 100,
                    2,
                ),
            }
        )

    return output


# =========================================================
# RISK CALCULATION
# =========================================================

def calculate_voice_dna(
    *,
    pitch: np.ndarray,
    rms: np.ndarray,
    spectral_flux: np.ndarray,
    spectral_flatness: np.ndarray,
    pause_intervals: List[
        Dict[str, Any]
    ],
    breathing_events: List[
        Dict[str, Any]
    ],
    duration: float,
) -> Dict[str, Dict[str, Any]]:
    voiced_pitch = pitch[
        pitch > 0
    ]

    jitter = calculate_jitter(
        pitch
    )

    shimmer = calculate_shimmer(
        rms
    )

    pitch_stability = (
        calculate_pitch_stability(
            pitch
        )
    )

    energy_variation = safe_std(
        rms
    )

    flux_variation = safe_std(
        spectral_flux
    )

    flatness_mean = safe_mean(
        spectral_flatness
    )

    total_pause_duration = sum(
        interval["duration"]
        for interval
        in pause_intervals
    )

    pause_ratio = (
        total_pause_duration
        / max(
            duration,
            1e-8,
        )
    )

    breathing_rate = (
        len(
            breathing_events
        )
        / max(
            duration / 60,
            1e-8,
        )
    )

    pitch_uniformity_risk = clamp(
        (
            pitch_stability
            - 0.88
        )
        / 0.12
    )

    jitter_risk = clamp(
        abs(
            jitter - 0.012
        )
        / 0.035
    )

    shimmer_risk = clamp(
        abs(
            shimmer - 0.06
        )
        / 0.20
    )

    energy_risk = clamp(
        abs(
            energy_variation - 0.025
        )
        / 0.07
    )

    flux_risk = clamp(
        abs(
            flux_variation - 0.04
        )
        / 0.15
    )

    flatness_risk = clamp(
        abs(
            flatness_mean - 0.12
        )
        / 0.32
    )

    pause_risk = clamp(
        abs(
            pause_ratio - 0.15
        )
        / 0.40
    )

    breathing_risk = clamp(
        abs(
            breathing_rate - 12
        )
        / 18
    )

    naturalness = clamp(
        1.0
        - (
            pitch_uniformity_risk * 0.18
            + jitter_risk * 0.12
            + shimmer_risk * 0.12
            + energy_risk * 0.14
            + flux_risk * 0.12
            + flatness_risk * 0.12
            + pause_risk * 0.10
            + breathing_risk * 0.10
        )
    )

    synthetic_signature = (
        1.0 - naturalness
    )

    temporal_consistency_risk = clamp(
        pitch_uniformity_risk * 0.35
        + energy_risk * 0.25
        + flux_risk * 0.25
        + pause_risk * 0.15
    )

    prosody_risk = clamp(
        pitch_uniformity_risk * 0.45
        + jitter_risk * 0.20
        + energy_risk * 0.20
        + pause_risk * 0.15
    )

    breathing_naturalness = (
        1.0 - breathing_risk
    )

    return {
        "naturalness": {
            "score": round(
                naturalness * 100,
                2,
            ),
            "risk": risk_level(
                synthetic_signature * 100
            ),
            "reason": (
                "Measures how closely the recording follows "
                "natural human voice variation."
            ),
        },

        "synthetic_signature": {
            "score": round(
                synthetic_signature * 100,
                2,
            ),
            "risk": risk_level(
                synthetic_signature * 100
            ),
            "reason": (
                "Combines pitch, energy, spectral, pause and "
                "breathing irregularities associated with synthesis."
            ),
        },

        "pitch_stability": {
            "score": round(
                pitch_stability * 100,
                2,
            ),
            "risk": risk_level(
                pitch_uniformity_risk * 100
            ),
            "reason": (
                "Extremely stable pitch may indicate generated "
                "or vocoder-produced speech."
            ),
        },

        "prosody_risk": {
            "score": round(
                prosody_risk * 100,
                2,
            ),
            "risk": risk_level(
                prosody_risk * 100
            ),
            "reason": (
                "Evaluates natural changes in rhythm, emphasis, "
                "intonation and vocal energy."
            ),
        },

        "temporal_consistency": {
            "score": round(
                temporal_consistency_risk
                * 100,
                2,
            ),
            "risk": risk_level(
                temporal_consistency_risk
                * 100
            ),
            "reason": (
                "Measures abnormal consistency or discontinuity "
                "across successive speech frames."
            ),
        },

        "breathing_naturalness": {
            "score": round(
                breathing_naturalness
                * 100,
                2,
            ),
            "risk": risk_level(
                breathing_risk * 100
            ),
            "reason": (
                "Estimates whether breathing-like events appear "
                "at plausible human speech intervals."
            ),
        },

        "jitter": {
            "score": round(
                jitter_risk * 100,
                2,
            ),
            "risk": risk_level(
                jitter_risk * 100
            ),
            "observed": round(
                jitter,
                6,
            ),
            "reason": (
                "Jitter measures cycle-to-cycle pitch-period variation."
            ),
        },

        "shimmer": {
            "score": round(
                shimmer_risk * 100,
                2,
            ),
            "risk": risk_level(
                shimmer_risk * 100
            ),
            "observed": round(
                shimmer,
                6,
            ),
            "reason": (
                "Shimmer measures cycle-to-cycle amplitude variation."
            ),
        },
    }


# =========================================================
# TIMELINE SERIALIZATION
# =========================================================

def serialize_curve(
    times: np.ndarray,
    values: np.ndarray,
    maximum_points: int = 500,
) -> List[Dict[str, float]]:
    times = np.asarray(
        times,
        dtype=np.float64,
    ).reshape(
        -1
    )

    values = np.asarray(
        values,
        dtype=np.float64,
    ).reshape(
        -1
    )

    length = min(
        len(times),
        len(values),
    )

    if length == 0:
        return []

    times = times[
        :length
    ]

    values = values[
        :length
    ]

    valid_mask = (
        np.isfinite(times)
        & np.isfinite(values)
    )

    times = times[
        valid_mask
    ]

    values = values[
        valid_mask
    ]

    if len(times) == 0:
        return []

    step = max(
        1,
        int(
            np.ceil(
                len(times)
                / maximum_points
            )
        ),
    )

    output = []

    for index in range(
        0,
        len(times),
        step,
    ):
        output.append(
            {
                "time": round(
                    float(
                        times[index]
                    ),
                    4,
                ),

                "value": round(
                    float(
                        values[index]
                    ),
                    8,
                ),
            }
        )

    return output

# =========================================================
# MAIN ANALYSIS
# =========================================================

def analyze_advanced_audio(
    file_path: str,
) -> Dict[str, Any]:
    audio = load_audio(
        file_path
    )

    duration = float(
        len(audio) / SAMPLE_RATE
    )

    # -----------------------------------------------------
    # Pitch
    # -----------------------------------------------------

    try:
        pitch_times, pitch = extract_pitch_curve(
            audio
        )
    except Exception as error:
        print(
            "Advanced pitch extraction warning:",
            error,
        )

        pitch_times = np.array(
            [],
            dtype=np.float32,
        )

        pitch = np.array(
            [],
            dtype=np.float32,
        )

    # -----------------------------------------------------
    # Energy
    # -----------------------------------------------------

    try:
        energy_times, rms = extract_energy_curve(
            audio
        )
    except Exception as error:
        print(
            "Advanced energy extraction warning:",
            error,
        )

        energy_times = np.array(
            [],
            dtype=np.float32,
        )

        rms = np.array(
            [],
            dtype=np.float32,
        )

    # -----------------------------------------------------
    # Spectral flux
    # -----------------------------------------------------

    try:
        flux_times, spectral_flux = extract_spectral_flux(
            audio
        )
    except Exception as error:
        print(
            "Advanced spectral-flux warning:",
            error,
        )

        flux_times = np.array(
            [],
            dtype=np.float32,
        )

        spectral_flux = np.array(
            [],
            dtype=np.float32,
        )

    # -----------------------------------------------------
    # Spectral flatness
    # -----------------------------------------------------

    try:
        spectral_flatness = (
            librosa.feature.spectral_flatness(
                y=audio,
                n_fft=FRAME_LENGTH,
                hop_length=HOP_LENGTH,
            )[0]
        ).astype(
            np.float32
        )

        flatness_times = librosa.frames_to_time(
            np.arange(
                len(spectral_flatness)
            ),
            sr=SAMPLE_RATE,
            hop_length=HOP_LENGTH,
        ).astype(
            np.float32
        )

    except Exception as error:
        print(
            "Advanced spectral-flatness warning:",
            error,
        )

        flatness_times = np.array(
            [],
            dtype=np.float32,
        )

        spectral_flatness = np.array(
            [],
            dtype=np.float32,
        )

    # -----------------------------------------------------
    # Pauses
    # -----------------------------------------------------

    try:
        pause_intervals = detect_pause_intervals(
            rms,
            energy_times,
        )

    except Exception as error:
        print(
            "Pause analysis warning:",
            error,
        )

        pause_intervals = []

    # -----------------------------------------------------
    # Breathing
    # -----------------------------------------------------

    try:
        breathing_events = detect_breath_candidates(
            audio,
            rms,
            spectral_flatness,
            energy_times,
        )

    except Exception as error:
        print(
            "Breathing analysis warning:",
            error,
        )

        breathing_events = []

    # -----------------------------------------------------
    # Voice DNA
    # -----------------------------------------------------

    try:
        voice_dna = calculate_voice_dna(
            pitch=pitch,
            rms=rms,
            spectral_flux=spectral_flux,
            spectral_flatness=spectral_flatness,
            pause_intervals=pause_intervals,
            breathing_events=breathing_events,
            duration=duration,
        )

    except Exception as error:
        print(
            "Voice DNA analysis warning:",
            error,
        )

        voice_dna = {}

    voiced_pitch = pitch[
        np.isfinite(pitch)
        & (pitch > 0)
    ]

    summary = {
        "duration_seconds": round(
            duration,
            3,
        ),

        "pitch_mean_hz": round(
            safe_mean(
                voiced_pitch
            ),
            3,
        ),

        "pitch_std_hz": round(
            safe_std(
                voiced_pitch
            ),
            3,
        ),

        "pitch_min_hz": round(
            float(
                np.min(
                    voiced_pitch
                )
            )
            if voiced_pitch.size
            else 0.0,
            3,
        ),

        "pitch_max_hz": round(
            float(
                np.max(
                    voiced_pitch
                )
            )
            if voiced_pitch.size
            else 0.0,
            3,
        ),

        "energy_mean": round(
            safe_mean(
                rms
            ),
            6,
        ),

        "energy_variation": round(
            safe_std(
                rms
            ),
            6,
        ),

        "spectral_flux_mean": round(
            safe_mean(
                spectral_flux
            ),
            6,
        ),

        "spectral_flatness_mean": round(
            safe_mean(
                spectral_flatness
            ),
            6,
        ),

        "pause_count": len(
            pause_intervals
        ),

        "pause_ratio": round(
            sum(
                float(
                    item.get(
                        "duration",
                        0,
                    )
                )
                for item in pause_intervals
            )
            / max(
                duration,
                1e-8,
            ),
            6,
        ),

        "breathing_event_count": len(
            breathing_events
        ),

        "estimated_breaths_per_minute": round(
            len(
                breathing_events
            )
            / max(
                duration / 60.0,
                1e-8,
            ),
            3,
        ),
    }

    curves = {
        "pitch": serialize_curve(
            pitch_times,
            pitch,
        ),

        "energy": serialize_curve(
            energy_times,
            rms,
        ),

        "spectral_flux": serialize_curve(
            flux_times,
            spectral_flux,
        ),

        "spectral_flatness": serialize_curve(
            flatness_times,
            spectral_flatness,
        ),
    }

    print(
        "\n"
        + "=" * 65
    )

    print(
        "FORGE ADVANCED AUDIO CURVES"
    )

    print(
        "Pitch points:",
        len(
            curves["pitch"]
        ),
    )

    print(
        "Energy points:",
        len(
            curves["energy"]
        ),
    )

    print(
        "Spectral flux points:",
        len(
            curves["spectral_flux"]
        ),
    )

    print(
        "Spectral flatness points:",
        len(
            curves["spectral_flatness"]
        ),
    )

    print(
        "=" * 65
        + "\n"
    )

    return {
        "summary": summary,

        "voice_dna": voice_dna,

        "curves": curves,

        "pause_intervals": pause_intervals,

        "breathing_events": breathing_events,

        "analysis_version": (
            "FORGE-AUDIO-ADVANCED-XAI-1.1"
        ),
    }