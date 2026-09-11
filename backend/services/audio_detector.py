from __future__ import annotations

import os
from typing import Any, Dict, Tuple

import librosa
import numpy as np

from spafe.features.lfcc import lfcc
from tensorflow.keras.models import load_model


# =========================================================
# PATHS
# =========================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "audio_fusion_cnn_bilstm_model_fixed.keras",
)

SCALER_MEAN_PATH = os.path.join(
    BASE_DIR,
    "models",
    "scaler_mean.npy",
)

SCALER_SCALE_PATH = os.path.join(
    BASE_DIR,
    "models",
    "scaler_scale.npy",
)


# =========================================================
# INFERENCE CONFIGURATION
# =========================================================

SAMPLE_RATE = 16000
MAX_FRAMES = 100
NUM_CEPS = 20
NUM_FILTERS = 70

THRESHOLD = 0.5

# IMPORTANT:
#
# Set POSITIVE_CLASS according to the model training labels.
#
# If training labels were:
#     REAL = 0
#     FAKE = 1
# use:
#     POSITIVE_CLASS = "FAKE"
#
# If training labels were:
#     FAKE = 0
#     REAL = 1
# use:
#     POSITIVE_CLASS = "REAL"
#
POSITIVE_CLASS = "FAKE"

# Set this to True only when LFCC features were standardized
# during model training using per-sample mean and standard deviation.
NORMALIZE_LFCC = False

DEBUG_AUDIO_MODEL = True


PARAMETER_NAMES = [
    "spectral_centroid_mean",
    "spectral_centroid_std",
    "spectral_bandwidth_mean",
    "spectral_bandwidth_std",
    "spectral_rolloff_mean",
    "spectral_rolloff_std",
    "spectral_flatness_mean",
    "spectral_flatness_std",
    "zero_crossing_rate_mean",
    "zero_crossing_rate_std",
    "rms_energy_mean",
    "rms_energy_std",
    "entropy",
    "variance",
    "pitch_mean",
    "pitch_std",
    "pitch_range",
    "pause_ratio",
    "amplitude_discontinuity",
    "phase_variance",
    "phase_discontinuity",
    "noise_floor",
    "energy_variation",
    "tempo",
    "chroma_mean",
    "chroma_std",
    "audio_duration",
]

import keras
from keras.src.saving import serialization_lib

# Store Keras's original internal deserializer
_original_deserialize = serialization_lib.deserialize_keras_object

# List of keys from older Keras versions that crash Keras 3 deserialization
UNSUPPORTED_KEYS = {
    "quantization_config",
    "renorm",
    "renorm_clipping",
    "renorm_momentum",
}

def _clean_config(config):
    """Recursively strip unsupported legacy keys from layer config dictionaries."""
    if isinstance(config, dict):
        for key in list(config.keys()):
            if key in UNSUPPORTED_KEYS:
                config.pop(key, None)
            else:
                _clean_config(config[key])
    elif isinstance(config, list):
        for item in config:
            _clean_config(item)

def _patched_deserialize(config, *args, **kwargs):
    _clean_config(config)
    return _original_deserialize(config, *args, **kwargs)

# Monkey-patch the global Keras 3 deserializer
serialization_lib.deserialize_keras_object = _patched_deserialize

# =========================================================
# LOAD MODEL
# =========================================================

try:
    audio_model = load_model(
        MODEL_PATH,
        compile=False
    )
    print("✅ Audio model loaded successfully!")
except Exception as error:
    print("❌ Audio model loading failed:", error)
    audio_model = None
# =========================================================
# LOAD SCALER
# =========================================================

try:
    SCALER_MEAN = np.load(
        SCALER_MEAN_PATH
    ).astype(
        np.float32
    )

    SCALER_SCALE = np.load(
        SCALER_SCALE_PATH
    ).astype(
        np.float32
    )

    if SCALER_MEAN.shape != (
        len(PARAMETER_NAMES),
    ):
        raise ValueError(
            "Invalid scaler mean shape. "
            f"Expected {(len(PARAMETER_NAMES),)}, "
            f"received {SCALER_MEAN.shape}."
        )

    if SCALER_SCALE.shape != (
        len(PARAMETER_NAMES),
    ):
        raise ValueError(
            "Invalid scaler scale shape. "
            f"Expected {(len(PARAMETER_NAMES),)}, "
            f"received {SCALER_SCALE.shape}."
        )

    SCALER_SCALE = np.where(
        SCALER_SCALE == 0,
        1.0,
        SCALER_SCALE,
    ).astype(
        np.float32
    )

    print(
        "✅ Audio parameter scaler loaded"
    )

    print(
        "Scaler mean shape:",
        SCALER_MEAN.shape,
    )

    print(
        "Scaler scale shape:",
        SCALER_SCALE.shape,
    )

except Exception as error:
    print(
        "⚠️ Audio scaler loading failed:",
        error,
    )

    SCALER_MEAN = np.zeros(
        len(PARAMETER_NAMES),
        dtype=np.float32,
    )

    SCALER_SCALE = np.ones(
        len(PARAMETER_NAMES),
        dtype=np.float32,
    )

    print(
        "⚠️ Default scaler is being used."
    )


# =========================================================
# MODEL INPUT DIAGNOSTICS
# =========================================================

if audio_model is not None:
    try:
        print(
            "\nAudio model inputs:"
        )

        for index, model_input in enumerate(
            audio_model.inputs
        ):
            print(
                f"  Input {index}: "
                f"name={model_input.name}, "
                f"shape={model_input.shape}"
            )

        print(
            "Audio model output shape:",
            audio_model.output_shape,
        )

        print(
            "Configured positive class:",
            POSITIVE_CLASS,
        )

        print(
            "LFCC normalization enabled:",
            NORMALIZE_LFCC,
        )

    except Exception as error:
        print(
            "Audio model input inspection failed:",
            error,
        )


# =========================================================
# GENERAL HELPERS
# =========================================================

def safe_mean(
    values: np.ndarray,
) -> float:
    if values is None:
        return 0.0

    values = np.asarray(
        values
    )

    if values.size == 0:
        return 0.0

    return float(
        np.mean(values)
    )


def safe_std(
    values: np.ndarray,
) -> float:
    if values is None:
        return 0.0

    values = np.asarray(
        values
    )

    if values.size == 0:
        return 0.0

    return float(
        np.std(values)
    )


def safe_float(
    value: Any,
    default: float = 0.0,
) -> float:
    try:
        number = float(
            np.asarray(value).reshape(-1)[0]
        )

        if not np.isfinite(
            number
        ):
            return default

        return number

    except (
        TypeError,
        ValueError,
        IndexError,
    ):
        return default


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
            "Empty audio file"
        )

    if not np.all(
        np.isfinite(audio)
    ):
        raise ValueError(
            "Audio contains NaN or infinite values"
        )

    peak = float(
        np.max(
            np.abs(audio)
        )
    )

    if peak <= 1e-8:
        raise ValueError(
            "Audio signal is silent or nearly silent"
        )

    return audio


# =========================================================
# LFCC EXTRACTION
# =========================================================

def extract_lfcc_features(
    audio: np.ndarray,
) -> np.ndarray:
    features = lfcc(
        sig=audio,
        fs=SAMPLE_RATE,
        num_ceps=NUM_CEPS,
        nfilts=NUM_FILTERS,
    )

    features = np.asarray(
        features,
        dtype=np.float32,
    )

    if features.ndim != 2:
        raise ValueError(
            "LFCC extraction returned an invalid shape: "
            f"{features.shape}"
        )

    if features.shape[1] != NUM_CEPS:
        raise ValueError(
            f"Expected {NUM_CEPS} LFCC coefficients, "
            f"received {features.shape[1]}."
        )

    if features.shape[0] < MAX_FRAMES:
        padding = (
            MAX_FRAMES
            - features.shape[0]
        )

        features = np.pad(
            features,
            (
                (0, padding),
                (0, 0),
            ),
            mode="constant",
        )

    else:
        features = features[
            :MAX_FRAMES,
            :
        ]

    if NORMALIZE_LFCC:
        feature_mean = float(
            np.mean(features)
        )

        feature_std = float(
            np.std(features)
        )

        features = (
            features
            - feature_mean
        ) / (
            feature_std
            + 1e-8
        )

    if not np.all(
        np.isfinite(features)
    ):
        raise ValueError(
            "LFCC features contain NaN or infinity"
        )

    features = np.expand_dims(
        features,
        axis=0,
    )

    features = np.expand_dims(
        features,
        axis=-1,
    )

    return features.astype(
        np.float32
    )


# =========================================================
# ACOUSTIC PARAMETER EXTRACTION
# =========================================================

def extract_audio_parameters(
    audio: np.ndarray,
) -> np.ndarray:
    try:
        spectral_centroid = (
            librosa.feature.spectral_centroid(
                y=audio,
                sr=SAMPLE_RATE,
            )[0]
        )

        spectral_bandwidth = (
            librosa.feature.spectral_bandwidth(
                y=audio,
                sr=SAMPLE_RATE,
            )[0]
        )

        spectral_rolloff = (
            librosa.feature.spectral_rolloff(
                y=audio,
                sr=SAMPLE_RATE,
            )[0]
        )

        spectral_flatness = (
            librosa.feature.spectral_flatness(
                y=audio
            )[0]
        )

        zero_crossing_rate = (
            librosa.feature.zero_crossing_rate(
                audio
            )[0]
        )

        rms_energy = (
            librosa.feature.rms(
                y=audio
            )[0]
        )

        entropy = calculate_entropy(
            audio
        )

        variance = float(
            np.var(audio)
        )

        try:
            pitch = librosa.yin(
                audio,
                fmin=50,
                fmax=500,
                sr=SAMPLE_RATE,
            )

            pitch = np.asarray(
                pitch
            )

            pitch = pitch[
                np.isfinite(pitch)
            ]

            pitch_mean = safe_mean(
                pitch
            )

            pitch_std = safe_std(
                pitch
            )

            pitch_range = (
                float(
                    np.max(pitch)
                    - np.min(pitch)
                )
                if pitch.size > 0
                else 0.0
            )

        except Exception as error:
            print(
                "Pitch extraction warning:",
                error,
            )

            pitch_mean = 0.0
            pitch_std = 0.0
            pitch_range = 0.0

        pause_ratio = float(
            np.mean(
                np.abs(audio) < 0.02
            )
        )

        amplitude_difference = np.diff(
            audio
        )

        amplitude_discontinuity = safe_mean(
            np.abs(
                amplitude_difference
            )
        )

        stft = librosa.stft(
            audio
        )

        phase = np.angle(
            stft
        )

        phase_variance = float(
            np.var(phase)
        )

        phase_difference = np.diff(
            phase,
            axis=1,
        )

        phase_discontinuity = safe_mean(
            np.abs(
                phase_difference
            )
        )

        noise_floor = float(
            np.percentile(
                np.abs(audio),
                10,
            )
        )

        energy_variation = safe_std(
            rms_energy
        )

        try:
            tempo_result = (
                librosa.beat.beat_track(
                    y=audio,
                    sr=SAMPLE_RATE,
                )[0]
            )

            tempo = safe_float(
                tempo_result,
                0.0,
            )

        except Exception as error:
            print(
                "Tempo extraction warning:",
                error,
            )

            tempo = 0.0

        chroma = (
            librosa.feature.chroma_stft(
                y=audio,
                sr=SAMPLE_RATE,
            )
        )

        parameters = np.array(
            [
                safe_mean(
                    spectral_centroid
                ),
                safe_std(
                    spectral_centroid
                ),
                safe_mean(
                    spectral_bandwidth
                ),
                safe_std(
                    spectral_bandwidth
                ),
                safe_mean(
                    spectral_rolloff
                ),
                safe_std(
                    spectral_rolloff
                ),
                safe_mean(
                    spectral_flatness
                ),
                safe_std(
                    spectral_flatness
                ),
                safe_mean(
                    zero_crossing_rate
                ),
                safe_std(
                    zero_crossing_rate
                ),
                safe_mean(
                    rms_energy
                ),
                safe_std(
                    rms_energy
                ),
                entropy,
                variance,
                pitch_mean,
                pitch_std,
                pitch_range,
                pause_ratio,
                amplitude_discontinuity,
                phase_variance,
                phase_discontinuity,
                noise_floor,
                energy_variation,
                tempo,
                safe_mean(
                    chroma
                ),
                safe_std(
                    chroma
                ),
                float(
                    len(audio)
                    / SAMPLE_RATE
                ),
            ],
            dtype=np.float32,
        )

        if parameters.shape != (
            len(PARAMETER_NAMES),
        ):
            raise ValueError(
                "Audio parameter extraction returned "
                f"{parameters.shape}; expected "
                f"{(len(PARAMETER_NAMES),)}."
            )

        if not np.all(
            np.isfinite(parameters)
        ):
            raise ValueError(
                "Raw audio parameters contain NaN or infinity"
            )

        return parameters

    except Exception as error:
        raise RuntimeError(
            "Audio parameter extraction failed: "
            f"{error}"
        ) from error


# =========================================================
# PARAMETER SCALING
# =========================================================

def scale_parameters(
    parameters: np.ndarray,
) -> np.ndarray:
    parameters = np.asarray(
        parameters,
        dtype=np.float32,
    )

    expected_shape = (
        len(PARAMETER_NAMES),
    )

    if parameters.shape != expected_shape:
        raise ValueError(
            f"Expected raw parameters with shape {expected_shape}, "
            f"received {parameters.shape}."
        )

    scaled = (
        parameters
        - SCALER_MEAN
    ) / SCALER_SCALE

    if not np.all(
        np.isfinite(scaled)
    ):
        raise ValueError(
            "Scaled audio parameters contain NaN or infinity"
        )

    return scaled.reshape(
        1,
        -1,
    ).astype(
        np.float32
    )


# =========================================================
# MODEL INPUT ORDER
# =========================================================

def determine_model_inputs(
    lfcc_features: np.ndarray,
    scaled_parameters: np.ndarray,
) -> Tuple[Any, str]:
    if audio_model is None:
        raise RuntimeError(
            "Audio model is not loaded"
        )

    model_inputs = list(
        audio_model.inputs
    )

    if len(model_inputs) != 2:
        raise ValueError(
            "Expected the audio fusion model to have two inputs, "
            f"but found {len(model_inputs)}."
        )

    first_shape = tuple(
        model_inputs[0].shape
    )

    second_shape = tuple(
        model_inputs[1].shape
    )

    first_last_dimension = (
        first_shape[-1]
        if len(first_shape) > 1
        else None
    )

    second_last_dimension = (
        second_shape[-1]
        if len(second_shape) > 1
        else None
    )

    expected_parameter_count = len(
        PARAMETER_NAMES
    )

    if (
        first_last_dimension
        == expected_parameter_count
    ):
        return (
            [
                scaled_parameters,
                lfcc_features,
            ],
            "parameters-first",
        )

    if (
        second_last_dimension
        == expected_parameter_count
    ):
        return (
            [
                lfcc_features,
                scaled_parameters,
            ],
            "lfcc-first",
        )

    # Fallback to the existing order.
    return (
        [
            lfcc_features,
            scaled_parameters,
        ],
        "fallback-lfcc-first",
    )


# =========================================================
# OUTPUT INTERPRETATION
# =========================================================

def interpret_model_output(
    model_output: float,
) -> Tuple[
    float,
    float,
]:
    model_output = float(
        np.clip(
            model_output,
            0.0,
            1.0,
        )
    )

    positive_class = (
        POSITIVE_CLASS
        .strip()
        .upper()
    )

    if positive_class == "FAKE":
        probability_fake = (
            model_output
        )

        probability_real = (
            1.0
            - model_output
        )

    elif positive_class == "REAL":
        probability_real = (
            model_output
        )

        probability_fake = (
            1.0
            - model_output
        )

    else:
        raise ValueError(
            "POSITIVE_CLASS must be either 'FAKE' or 'REAL'"
        )

    return (
        probability_real,
        probability_fake,
    )


# =========================================================
# DEBUG OUTPUT
# =========================================================

def print_audio_diagnostics(
    *,
    file_path: str,
    audio: np.ndarray,
    lfcc_features: np.ndarray,
    raw_parameters: np.ndarray,
    scaled_parameters: np.ndarray,
    raw_prediction: np.ndarray,
    model_output: float,
    probability_real: float,
    probability_fake: float,
    model_input_order: str,
) -> None:
    if not DEBUG_AUDIO_MODEL:
        return

    print(
        "\n"
        + "=" * 80
    )

    print(
        "FORGE AUDIO MODEL DIAGNOSTIC"
    )

    print(
        "=" * 80
    )

    print(
        "Audio file:",
        file_path,
    )

    print(
        "Positive class:",
        POSITIVE_CLASS,
    )

    print(
        "Model input order:",
        model_input_order,
    )

    print(
        "Audio samples:",
        audio.shape,
    )

    print(
        "Audio duration:",
        round(
            len(audio)
            / SAMPLE_RATE,
            4,
        ),
        "seconds",
    )

    print(
        "Audio min/max/mean/std:",
        float(
            np.min(audio)
        ),
        float(
            np.max(audio)
        ),
        float(
            np.mean(audio)
        ),
        float(
            np.std(audio)
        ),
    )

    print(
        "-" * 80
    )

    print(
        "LFCC shape:",
        lfcc_features.shape,
    )

    print(
        "LFCC min:",
        float(
            np.min(lfcc_features)
        ),
    )

    print(
        "LFCC max:",
        float(
            np.max(lfcc_features)
        ),
    )

    print(
        "LFCC mean:",
        float(
            np.mean(lfcc_features)
        ),
    )

    print(
        "LFCC std:",
        float(
            np.std(lfcc_features)
        ),
    )

    print(
        "-" * 80
    )

    print(
        "Raw parameter shape:",
        raw_parameters.shape,
    )

    print(
        "Raw parameters:"
    )

    for name, value in zip(
        PARAMETER_NAMES,
        raw_parameters,
    ):
        print(
            f"  {name}: {float(value):.8f}"
        )

    print(
        "-" * 80
    )

    print(
        "Scaled parameter shape:",
        scaled_parameters.shape,
    )

    print(
        "Scaled min:",
        float(
            np.min(
                scaled_parameters
            )
        ),
    )

    print(
        "Scaled max:",
        float(
            np.max(
                scaled_parameters
            )
        ),
    )

    print(
        "Scaled mean:",
        float(
            np.mean(
                scaled_parameters
            )
        ),
    )

    print(
        "Scaled std:",
        float(
            np.std(
                scaled_parameters
            )
        ),
    )

    print(
        "Scaled values outside ±5:",
        int(
            np.sum(
                np.abs(
                    scaled_parameters
                ) > 5
            )
        ),
    )

    print(
        "Scaled values outside ±10:",
        int(
            np.sum(
                np.abs(
                    scaled_parameters
                ) > 10
            )
        ),
    )

    print(
        "Scaled parameters:"
    )

    for name, value in zip(
        PARAMETER_NAMES,
        scaled_parameters.flatten(),
    ):
        print(
            f"  {name}: {float(value):.8f}"
        )

    print(
        "-" * 80
    )

    print(
        "Raw model prediction:",
        raw_prediction,
    )

    print(
        "Scalar model output:",
        model_output,
    )

    print(
        "Probability real:",
        probability_real,
    )

    print(
        "Probability fake:",
        probability_fake,
    )

    print(
        "=" * 80
        + "\n"
    )


# =========================================================
# MAIN AUDIO ANALYSIS
# =========================================================

def analyze_audio_model(
    file_path: str,
) -> Dict[str, Any]:
    if audio_model is None:
        return {
            "error": (
                "Audio model not loaded"
            )
        }

    try:
        audio = load_audio(
            file_path
        )

        lfcc_features = (
            extract_lfcc_features(
                audio
            )
        )

        raw_parameters = (
            extract_audio_parameters(
                audio
            )
        )

        scaled_parameters = (
            scale_parameters(
                raw_parameters
            )
        )

        model_inputs, input_order = (
            determine_model_inputs(
                lfcc_features,
                scaled_parameters,
            )
        )

        raw_prediction = (
            audio_model.predict(
                model_inputs,
                verbose=0,
            )
        )

        flattened_prediction = (
            np.asarray(
                raw_prediction
            ).reshape(
                -1
            )
        )

        if flattened_prediction.size == 0:
            raise ValueError(
                "Audio model returned an empty prediction"
            )

        model_output = float(
            flattened_prediction[0]
        )

        if not np.isfinite(
            model_output
        ):
            raise ValueError(
                "Audio model returned an invalid probability: "
                f"{model_output}"
            )

        probability_real, probability_fake = (
            interpret_model_output(
                model_output
            )
        )

        print_audio_diagnostics(
            file_path=file_path,
            audio=audio,
            lfcc_features=lfcc_features,
            raw_parameters=raw_parameters,
            scaled_parameters=scaled_parameters,
            raw_prediction=np.asarray(
                raw_prediction
            ),
            model_output=model_output,
            probability_real=probability_real,
            probability_fake=probability_fake,
            model_input_order=input_order,
        )

        if probability_fake >= THRESHOLD:
            prediction = "FAKE"

            confidence = (
                probability_fake
                * 100
            )

        else:
            prediction = "REAL"

            confidence = (
                probability_real
                * 100
            )

        return {
            "prediction": prediction,

            "confidence": round(
                confidence,
                2,
            ),

            "raw_model_output": round(
                model_output,
                8,
            ),

            "positive_class": (
                POSITIVE_CLASS
            ),

            "model_input_order": (
                input_order
            ),

            "raw_probability_real": round(
                probability_real,
                8,
            ),

            "raw_probability_fake": round(
                probability_fake,
                8,
            ),

            "risk_level": risk_level(
                probability_fake
                * 100
            ),

            "risk_score": round(
                probability_fake
                * 100,
                2,
            ),

            "raw_parameters": (
                raw_parameters
            ),

            "scaled_parameters": (
                scaled_parameters.flatten()
            ),

            "parameter_names": (
                PARAMETER_NAMES
            ),

            "diagnostics": {
                "lfcc_shape": list(
                    lfcc_features.shape
                ),

                "lfcc_min": float(
                    np.min(
                        lfcc_features
                    )
                ),

                "lfcc_max": float(
                    np.max(
                        lfcc_features
                    )
                ),

                "lfcc_mean": float(
                    np.mean(
                        lfcc_features
                    )
                ),

                "lfcc_std": float(
                    np.std(
                        lfcc_features
                    )
                ),

                "scaled_min": float(
                    np.min(
                        scaled_parameters
                    )
                ),

                "scaled_max": float(
                    np.max(
                        scaled_parameters
                    )
                ),

                "scaled_mean": float(
                    np.mean(
                        scaled_parameters
                    )
                ),

                "scaled_std": float(
                    np.std(
                        scaled_parameters
                    )
                ),

                "scaled_values_outside_5": int(
                    np.sum(
                        np.abs(
                            scaled_parameters
                        ) > 5
                    )
                ),

                "scaled_values_outside_10": int(
                    np.sum(
                        np.abs(
                            scaled_parameters
                        ) > 10
                    )
                ),

                "lfcc_normalized": (
                    NORMALIZE_LFCC
                ),
            },
        }

    except Exception as error:
        print(
            "❌ Audio inference failed:",
            error,
        )

        return {
            "error": (
                "Audio inference failed: "
                f"{str(error)}"
            )
        }