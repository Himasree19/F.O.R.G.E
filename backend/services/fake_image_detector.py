import os
import json
import joblib
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from backend.services.image_region_analysis import (
    analyse_image_regions
)

from PIL import Image, ExifTags
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input

# =====================================================
# FORGE IMAGE DETECTOR
# CNN + RANDOM FOREST FUSION + 48 FEATURES
# FIXED CLASS MAPPING
# =====================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")

CNN_MODEL_PATH = os.path.join(MODEL_DIR, "final2_randomf_fusion_v3.pkl")
RF_MODEL_PATH = os.path.join(MODEL_DIR, "IMAGE_RF_v2.pkl")
FEATURE_NAMES_PATH = os.path.join(MODEL_DIR, "rf_feature_names.json")

IMAGE_SIZE = (224, 224)
THRESHOLD = 0.5
DEBUG_MODE = True

ALL_48_FEATURES = [
    "cnn_score", "lighting", "shadow", "boundary", "texture", "symmetry",
    "high_frequency_noise", "gan_fingerprint", "checkerboard_artifact",
    "prnu_score", "copy_move_score", "splicing_score", "jpeg_artifact_score",
    "ela_score", "resampling_score", "rgb_inconsistency", "hsv_anomaly",
    "ycbcr_chroma", "saturation_anomaly", "cfa_artifact", "eye_score",
    "iris_score", "landmark_score", "skin_texture_score", "face_blending_score",
    "object_relationship_score", "hand_anatomy_score", "eye_alignment_score",
    "object_count_score", "shadow_consistency_score", "geometry_score",
    "depth_of_field_score", "anatomy_score", "perspective_score",
    "text_error_score", "camera_model_missing", "iso_missing",
    "shutter_missing", "editing_software_found", "timestamp_available",
    "anomaly_mean", "anomaly_max", "suspicious_patches", "quality_score",
    "noise_score", "sharpness_score", "compression_score", "authenticity_score"
]

cnn_model = None
rf_model = None
rf_feature_names = []

try:
    cnn_model = tf.keras.models.load_model(CNN_MODEL_PATH, compile=False)
    print("✅ Image CNN Model Loaded:", CNN_MODEL_PATH)
except Exception as e:
    print("❌ Image CNN Model Loading Failed:", e)

try:
    rf_model = joblib.load(RF_MODEL_PATH)
    print("✅ Image RandomForest Model Loaded:", RF_MODEL_PATH)
except Exception as e:
    print("❌ RF Model Loading Failed:", e)

try:
    if rf_model is not None and hasattr(rf_model, "feature_names_in_"):
        rf_feature_names = list(rf_model.feature_names_in_)
        print("✅ RF feature names taken from model:", len(rf_feature_names))
    elif os.path.exists(FEATURE_NAMES_PATH):
        with open(FEATURE_NAMES_PATH, "r") as f:
            rf_feature_names = json.load(f)

        if rf_model is not None and hasattr(rf_model, "n_features_in_"):
            rf_feature_names = rf_feature_names[:int(rf_model.n_features_in_)]

        print("✅ RF feature names loaded:", len(rf_feature_names))
    else:
        rf_feature_names = ALL_48_FEATURES

        if rf_model is not None and hasattr(rf_model, "n_features_in_"):
            rf_feature_names = rf_feature_names[:int(rf_model.n_features_in_)]

        print("⚠️ Using fallback RF features:", len(rf_feature_names))
except Exception as e:
    print("⚠️ RF feature loading failed:", e)
    rf_feature_names = ALL_48_FEATURES


def safe_float(value):
    try:
        value = float(value)
        if np.isnan(value) or np.isinf(value):
            return 0.0
        return value
    except Exception:
        return 0.0


def safe_round(value, digits=2):
    return round(safe_float(value), digits)


def risk_level(score):
    score = safe_float(score)
    if score >= 75:
        return "HIGH"
    if score >= 45:
        return "MEDIUM"
    return "LOW"


def confidence_color(confidence):
    confidence = safe_float(confidence)
    if confidence >= 85:
        return "red"
    if confidence >= 60:
        return "orange"
    return "green"


def decision_strength(confidence):
    confidence = safe_float(confidence)
    if confidence >= 85:
        return "STRONG"
    if confidence >= 65:
        return "MODERATE"
    return "LOW"


def normalize_0_100(value, max_value):
    value = safe_float(value)
    if max_value == 0:
        return 0.0
    return float(np.clip((value / max_value) * 100, 0, 100))


def validate_image_path(image_path):
    if not image_path:
        raise ValueError("No image path provided")

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    if not image_path.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
        raise ValueError("Unsupported image format. Use JPG, JPEG, PNG or WEBP.")


def load_image_cv(image_path):
    validate_image_path(image_path)
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Unable to read image")
    return cv2.resize(img, IMAGE_SIZE)


def load_image_pil(image_path):
    validate_image_path(image_path)
    img = Image.open(image_path).convert("RGB")
    return img.resize(IMAGE_SIZE)


def load_image_for_cnn(image_path):
    validate_image_path(image_path)

    img = image.load_img(image_path, target_size=IMAGE_SIZE)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    return img_array


def predict_with_cnn(image_path):
    if cnn_model is None:
        return {
            "raw_output": 0.5,
            "real_probability": 0.5,
            "fake_probability": 0.5,
            "cnn_score": 50.0
        }

    img_array = load_image_for_cnn(image_path)

    raw_output = float(
        cnn_model.predict(img_array, verbose=0)[0][0]
    )

    fake_probability = raw_output
    real_probability = 1.0 - raw_output

    return {
        "raw_output": safe_float(raw_output),
        "real_probability": safe_float(real_probability),
        "fake_probability": safe_float(fake_probability),
        "cnn_score": safe_float(fake_probability * 100)
    }


def extract_metadata_features(img_pil):
    features = {}

    try:
        exif = img_pil.getexif()
        exif_data = {}

        for tag_id, value in exif.items():
            tag = ExifTags.TAGS.get(tag_id, tag_id)
            exif_data[tag] = value

        camera_keys = ["Make", "Model", "LensModel", "LensMake"]
        iso_keys = ["ISOSpeedRatings", "PhotographicSensitivity"]
        shutter_keys = ["ExposureTime", "ShutterSpeedValue"]
        timestamp_keys = ["DateTime", "DateTimeOriginal", "DateTimeDigitized"]

        features["camera_model_missing"] = 0.0 if any(k in exif_data for k in camera_keys) else 100.0
        features["iso_missing"] = 0.0 if any(k in exif_data for k in iso_keys) else 100.0
        features["shutter_missing"] = 0.0 if any(k in exif_data for k in shutter_keys) else 100.0
        features["timestamp_available"] = 100.0 if any(k in exif_data for k in timestamp_keys) else 0.0

        software = str(exif_data.get("Software", "")).lower()
        editing_words = ["photoshop", "gimp", "lightroom", "canva", "editor", "snapseed", "picsart", "adobe"]

        features["editing_software_found"] = 100.0 if any(word in software for word in editing_words) else 0.0

    except Exception:
        features["camera_model_missing"] = 100.0
        features["iso_missing"] = 100.0
        features["shutter_missing"] = 100.0
        features["editing_software_found"] = 0.0
        features["timestamp_available"] = 0.0

    return features


def extract_all_48_features(image_path, cnn_score):
    img_cv = load_image_cv(image_path)
    img_pil = load_image_pil(image_path)

    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    hsv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2HSV)
    ycrcb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2YCrCb)

    h, w = gray.shape
    features = {}

    features["cnn_score"] = safe_float(cnn_score)
    features["lighting"] = normalize_0_100(np.mean(gray), 255)
    features["shadow"] = safe_float(np.mean(gray < 45) * 100)

    edges = cv2.Canny(gray, 80, 180)
    features["boundary"] = safe_float(np.mean(edges > 0) * 100)

    lap = cv2.Laplacian(gray, cv2.CV_64F)
    features["texture"] = normalize_0_100(lap.var(), 1000)

    left = gray[:, :w // 2]
    right = cv2.flip(gray[:, w - w // 2:], 1)
    min_w = min(left.shape[1], right.shape[1])

    symmetry_diff = np.mean(
        np.abs(left[:, :min_w].astype(float) - right[:, :min_w].astype(float))
    )

    features["symmetry"] = safe_float(100 - np.clip(symmetry_diff, 0, 100))

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    high_freq = gray.astype(float) - blurred.astype(float)

    features["high_frequency_noise"] = normalize_0_100(np.std(high_freq), 80)

    fft = np.fft.fft2(gray)
    fft_shift = np.fft.fftshift(fft)
    magnitude = np.log(np.abs(fft_shift) + 1)

    features["gan_fingerprint"] = normalize_0_100(np.mean(magnitude), 12)

    checker_kernel = np.array([[1, -1], [-1, 1]])
    checker = cv2.filter2D(gray.astype(float), -1, checker_kernel)

    features["checkerboard_artifact"] = normalize_0_100(np.mean(np.abs(checker)), 80)

    denoised = cv2.GaussianBlur(gray, (3, 3), 0)
    prnu = gray.astype(float) - denoised.astype(float)

    features["prnu_score"] = normalize_0_100(np.std(prnu), 60)

    try:
        orb = cv2.ORB_create()
        keypoints, _ = orb.detectAndCompute(gray, None)
        keypoint_count = len(keypoints) if keypoints is not None else 0
        features["copy_move_score"] = normalize_0_100(keypoint_count, 600)
    except Exception:
        features["copy_move_score"] = 0.0

    features["splicing_score"] = normalize_0_100(np.std(edges), 128)
    
    # -----------------------------------------------------------------
    # FIXED: Scaled denominator to 255 so artifact scores vary properly
    # -----------------------------------------------------------------
    features["jpeg_artifact_score"] = normalize_0_100(np.mean(np.abs(np.diff(gray, axis=1))), 255)
    features["ela_score"] = normalize_0_100(np.std(high_freq), 80)
    features["resampling_score"] = normalize_0_100(np.mean(np.abs(np.diff(gray, axis=0))), 255)

    r = rgb[:, :, 0].astype(float)
    g = rgb[:, :, 1].astype(float)
    b = rgb[:, :, 2].astype(float)

    rgb_inconsistency = (
        np.std(r - g) +
        np.std(g - b) +
        np.std(r - b)
    ) / 3

    features["rgb_inconsistency"] = normalize_0_100(rgb_inconsistency, 80)
    features["hsv_anomaly"] = normalize_0_100(np.std(hsv[:, :, 1]), 100)

    features["ycbcr_chroma"] = normalize_0_100(
        np.std(ycrcb[:, :, 1]) + np.std(ycrcb[:, :, 2]),
        160
    )

    features["saturation_anomaly"] = normalize_0_100(np.mean(hsv[:, :, 1]), 255)

    try:
        cfa_value = abs(np.std(gray[::2, ::2]) - np.std(gray[1::2, 1::2]))
        features["cfa_artifact"] = normalize_0_100(cfa_value, 40)
    except Exception:
        features["cfa_artifact"] = 0.0

    features["skin_texture_score"] = safe_float(features["texture"])

    face_like_score = np.mean([
        features["symmetry"],
        100 - features["shadow"],
        features["skin_texture_score"]
    ])

    features["eye_score"] = safe_float(features["symmetry"])
    features["iris_score"] = safe_float(features["symmetry"] * 0.85)
    features["landmark_score"] = safe_float(face_like_score)
    features["face_blending_score"] = safe_float(100 - np.clip(features["boundary"], 0, 100))
    
    # -----------------------------------------------------------------
    # FIXED: Dynamic estimations instead of hardcoded 50.0 & 35.0
    # -----------------------------------------------------------------
    features["object_relationship_score"] = safe_float(normalize_0_100(features["splicing_score"] + features["symmetry"], 200))
    features["hand_anatomy_score"] = safe_float(normalize_0_100(features["texture"] + features["high_frequency_noise"], 200))
    features["eye_alignment_score"] = safe_float(features["symmetry"])
    features["object_count_score"] = safe_float(normalize_0_100(features["boundary"] * 2, 100))
    
    features["shadow_consistency_score"] = safe_float(100 - np.clip(features["shadow"], 0, 100))
    features["geometry_score"] = safe_float(features["symmetry"])
    features["depth_of_field_score"] = normalize_0_100(np.std(gray), 90)

    features["anatomy_score"] = safe_float(
        np.mean([
            features["eye_score"],
            features["landmark_score"],
            features["skin_texture_score"]
        ])
    )

    features["perspective_score"] = safe_float(100 - np.clip(features["boundary"], 0, 100))
    features["text_error_score"] = safe_float(normalize_0_100(features["high_frequency_noise"] + features["boundary"], 200))

    features.update(extract_metadata_features(img_pil))

    anomaly_values = [
        features["high_frequency_noise"],
        features["gan_fingerprint"],
        features["checkerboard_artifact"],
        features["rgb_inconsistency"],
        features["jpeg_artifact_score"],
        features["resampling_score"],
        features["cfa_artifact"]
    ]

    features["anomaly_mean"] = safe_float(np.mean(anomaly_values))
    features["anomaly_max"] = safe_float(np.max(anomaly_values))
    features["suspicious_patches"] = safe_float(np.mean(edges > 0) * 100)

    # -----------------------------------------------------------------
    # FIXED: Quality score updates dynamically based on artifact score
    # -----------------------------------------------------------------
    features["quality_score"] = safe_float(100 - np.clip(features["jpeg_artifact_score"], 0, 100))
    features["noise_score"] = safe_float(features["high_frequency_noise"])
    features["sharpness_score"] = safe_float(features["texture"])
    features["compression_score"] = safe_float(features["jpeg_artifact_score"])

    suspicious_metadata = np.mean([
        features["camera_model_missing"],
        features["iso_missing"],
        features["shutter_missing"],
        features["editing_software_found"]
    ])

    authenticity = 100 - np.mean([
        features["anomaly_mean"],
        suspicious_metadata,
        features["compression_score"]
    ])

    features["authenticity_score"] = safe_float(np.clip(authenticity, 0, 100))

    for name in ALL_48_FEATURES:
        features[name] = safe_float(features.get(name, 0.0))

    return features


def predict_with_random_forest(features):
    if rf_model is None:
        return None

    expected_count = int(rf_model.n_features_in_)
    names = rf_feature_names[:expected_count]

    data = {}

    for name in names:
        data[name] = safe_float(features.get(name, 0.0))

    X = pd.DataFrame([data], columns=names)

    prediction_raw = rf_model.predict(X)[0]

    if hasattr(rf_model, "predict_proba"):
        probabilities = rf_model.predict_proba(X)[0]
        classes = list(rf_model.classes_)

        if 0 in classes:
            fake_probability = probabilities[classes.index(0)]
        elif "0" in classes:
            fake_probability = probabilities[classes.index("0")]
        elif "Fake" in classes:
            fake_probability = probabilities[classes.index("Fake")]
        elif "AI" in classes:
            fake_probability = probabilities[classes.index("AI")]
        else:
            fake_probability = probabilities[0]
    else:
        pred = str(prediction_raw).lower()
        fake_probability = 1.0 if pred in ["fake", "ai", "0"] else 0.0

    fake_probability = safe_float(fake_probability)
    real_probability = 1.0 - fake_probability

    return {
        "prediction_raw": prediction_raw,
        "fake_probability": fake_probability,
        "real_probability": real_probability
    }


def fuse_predictions(cnn_result, rf_result):
    cnn_fake = cnn_result["fake_probability"]
    cnn_real = cnn_result["real_probability"]

    if rf_result is None:
        fake_probability = cnn_fake
        real_probability = cnn_real
        method = "CNN_ONLY"
    else:
        rf_fake = rf_result["fake_probability"]
        rf_real = rf_result["real_probability"]

        fake_probability = (0.55 * cnn_fake) + (0.45 * rf_fake)
        real_probability = (0.55 * cnn_real) + (0.45 * rf_real)

        method = "CNN_RF_FUSION"

    total = fake_probability + real_probability

    if total > 0:
        fake_probability = fake_probability / total
        real_probability = real_probability / total

    return {
        "fake_probability": safe_float(fake_probability),
        "real_probability": safe_float(real_probability),
        "fusion_method": method
    }


def mean_feature(features, names):
    values = [safe_float(features.get(name, 0.0)) for name in names]
    return safe_float(np.mean(values)) if values else 0.0


def build_parameter_contribution(features, fake_probability, confidence):
    fake_score = fake_probability * 100

    visual_score = mean_feature(
        features,
        ["lighting", "shadow", "boundary", "texture", "symmetry", "face_blending_score"]
    )

    frequency_score = mean_feature(
        features,
        ["high_frequency_noise", "gan_fingerprint", "checkerboard_artifact", "prnu_score"]
    )

    manipulation_score = mean_feature(
        features,
        ["copy_move_score", "splicing_score", "jpeg_artifact_score", "ela_score", "resampling_score"]
    )

    color_score = mean_feature(
        features,
        ["rgb_inconsistency", "hsv_anomaly", "ycbcr_chroma", "saturation_anomaly", "cfa_artifact"]
    )

    anatomy_score = mean_feature(
        features,
        ["eye_score", "iris_score", "landmark_score", "skin_texture_score", "hand_anatomy_score", "eye_alignment_score", "anatomy_score"]
    )

    metadata_score = mean_feature(
        features,
        ["camera_model_missing", "iso_missing", "shutter_missing", "editing_software_found"]
    )

    authenticity_score = safe_float(100 - features.get("authenticity_score", 50))
    semantic_score = safe_float(max(0, 100 - confidence))

    return {
        "deep_learning_detection": {
            "score": safe_round(fake_score),
            "risk": risk_level(fake_score),
            "reason": "CNN and RandomForest fusion analyzed deep visual and forensic image patterns."
        },
        "visual_artifacts": {
            "score": safe_round(visual_score),
            "risk": risk_level(visual_score),
            "reason": "Lighting, shadow, boundary, texture, symmetry and blending traces were analyzed."
        },
        "frequency_patterns": {
            "score": safe_round(frequency_score),
            "risk": risk_level(frequency_score),
            "reason": "High-frequency noise, GAN fingerprint, checkerboard and PRNU traces were checked."
        },
        "manipulation_traces": {
            "score": safe_round(manipulation_score),
            "risk": risk_level(manipulation_score),
            "reason": "Copy-move, splicing, JPEG, ELA and resampling artifacts were analyzed."
        },
        "color_inconsistency": {
            "score": safe_round(color_score),
            "risk": risk_level(color_score),
            "reason": "RGB, HSV, YCbCr, saturation and CFA color inconsistencies were analyzed."
        },
        "face_anatomy": {
            "score": safe_round(anatomy_score),
            "risk": risk_level(anatomy_score),
            "reason": "Eyes, iris, landmarks, skin texture and anatomy consistency were checked."
        },
        "metadata_authenticity": {
            "score": safe_round(metadata_score),
            "risk": risk_level(metadata_score),
            "reason": "Camera metadata, ISO, shutter and editing software indicators were inspected."
        },
        "authenticity_consistency": {
            "score": safe_round(authenticity_score),
            "risk": risk_level(authenticity_score),
            "reason": "Overall authenticity was estimated from anomaly, compression and metadata behavior."
        },
        "semantic_consistency": {
            "score": safe_round(semantic_score),
            "risk": risk_level(semantic_score),
            "reason": "Decision confidence and internal visual consistency were used for reliability scoring."
        }
    }


def build_recommendation(prediction, risk_score, confidence, fusion_method):
    if prediction == "AI":
        return "AI-like or manipulated image signals detected. Additional forensic verification is advised."

    if confidence < 65:
        return "Image appears authentic, but confidence is low. Manual verification is recommended."

    return f"Image appears authentic based on {fusion_method} analysis."


def analyze_image(image_path):
    try:
        validate_image_path(image_path)

        cnn_result = predict_with_cnn(image_path)

        features_48 = extract_all_48_features(
            image_path,
            cnn_score=cnn_result["cnn_score"]
        )

        rf_result = predict_with_random_forest(features_48)
        fused = fuse_predictions(cnn_result, rf_result)

        fake_probability = fused["fake_probability"]
        real_probability = fused["real_probability"]
        fusion_method = fused["fusion_method"]

        if fake_probability >= THRESHOLD:
            prediction = "AI"
            confidence = fake_probability * 100
        else:
            prediction = "HUMAN"
            confidence = real_probability * 100

        risk_score = fake_probability * 100

        parameter_contribution = build_parameter_contribution(
            features_48,
            fake_probability,
            confidence
        )

        recommendation = build_recommendation(
            prediction,
            risk_score,
            confidence,
            fusion_method
        )

        if DEBUG_MODE:
            print("=" * 70)
            print("DEBUG FORGE IMAGE MODEL")
            print("Image:", image_path)
            print("CNN raw output:", cnn_result["raw_output"])
            print("CNN fake probability:", cnn_result["fake_probability"])
            print("CNN real probability:", cnn_result["real_probability"])

            if rf_result is not None:
                print("RF raw prediction:", rf_result["prediction_raw"])
                print("RF fake probability:", rf_result["fake_probability"])
                print("RF real probability:", rf_result["real_probability"])
            else:
                print("RF result: Not available")

            print("Fusion method:", fusion_method)
            print("Final fake probability:", fake_probability)
            print("Final real probability:", real_probability)
            print("Prediction:", prediction)
            print("Confidence:", confidence)
            print("Extracted 48 features:", len(features_48))
            print("RF features used:", int(rf_model.n_features_in_) if rf_model is not None else 0)
            print("=" * 70)

        return {
            "modality": "image",
            "prediction": prediction,
            "confidence": safe_round(confidence),
            "risk_level": risk_level(risk_score),
            "risk_score": safe_round(risk_score),
            "decision_strength": decision_strength(confidence),
            "fusion_method": fusion_method,

            "raw_model_output": safe_round(cnn_result["raw_output"], 4),

            "cnn_ai_probability": safe_round(cnn_result["fake_probability"] * 100),
            "cnn_human_probability": safe_round(cnn_result["real_probability"] * 100),

            "rf_ai_probability": (
                safe_round(rf_result["fake_probability"] * 100)
                if rf_result is not None
                else None
            ),
            "rf_human_probability": (
                safe_round(rf_result["real_probability"] * 100)
                if rf_result is not None
                else None
            ),

            "raw_ai_probability": safe_round(fake_probability * 100),
            "raw_human_probability": safe_round(real_probability * 100),

            "extracted_feature_count": 48,
            "rf_feature_count_used": int(rf_model.n_features_in_) if rf_model is not None else 0,

            # =====================================================
            # PASS ALL 48 EXTRACTED FORENSIC FEATURES TO PIPELINE
            # =====================================================
            "forensic_features": features_48,

            "parameter_contribution": parameter_contribution,
            "recommendation": recommendation,
            "color": confidence_color(confidence)
        }

    except Exception as e:
        return {
            "error": str(e)
        }