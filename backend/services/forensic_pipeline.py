from __future__ import annotations

import re
import string
from typing import Dict

import numpy as np
from nltk import pos_tag, word_tokenize
from textstat import flesch_reading_ease

from backend.services.text_model_registry import (
    get_sbert_model,
    ngram_vectorizer,
    shap_explainer,
    text_model,
    tfidf_vectorizer,
)
from backend.xai.parameter_reasoning import (
    ngram_reason,
    semantic_reason,
    stylometric_reason,
    tfidf_reason,
)
from backend.xai.sentence_highlighter import analyze_document


# ==========================================
# STYLOMETRIC FEATURE EXTRACTION
# ==========================================

def extract_stylometric(text: str) -> list[float]:
    try:
        words = word_tokenize(text)

        sentences = re.split(
            r"[.!?]",
            text,
        )

        sentence_lengths = [
            len(sentence.split())
            for sentence in sentences
            if sentence.strip()
        ]

        sentence_variance = (
            float(np.var(sentence_lengths))
            if sentence_lengths
            else 0.0
        )

        word_frequency = float(len(words))

        punctuation_count = float(
            sum(
                1
                for character in text
                if character in string.punctuation
            )
        )

        tagged_words = pos_tag(words)

        noun_count = float(
            sum(
                1
                for _, tag in tagged_words
                if "NN" in tag
            )
        )

        verb_count = float(
            sum(
                1
                for _, tag in tagged_words
                if "VB" in tag
            )
        )

        readability = float(
            flesch_reading_ease(text)
        )

        return [
            sentence_variance,
            word_frequency,
            punctuation_count,
            noun_count,
            verb_count,
            readability,
        ]

    except Exception as error:
        print(
            "Stylometric extraction warning:",
            error,
        )

        return [
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ]


# ==========================================
# PARAMETER SCORE NORMALIZATION
# ==========================================

def normalize_to_total(
    scores: Dict[str, float],
) -> Dict[str, float]:
    total = float(
        sum(scores.values())
    )

    if total <= 0:
        return {
            key: 25.0
            for key in scores
        }

    return {
        key: round(
            (float(value) / total) * 100.0,
            2,
        )
        for key, value in scores.items()
    }


def cap_and_redistribute(
    scores: Dict[str, float],
    max_cap: float = 40.0,
) -> Dict[str, float]:
    capped: Dict[str, float] = {}
    overflow = 0.0
    under_cap_keys: list[str] = []

    for key, value in scores.items():
        numeric_value = float(value)

        if numeric_value > max_cap:
            capped[key] = max_cap
            overflow += numeric_value - max_cap
        else:
            capped[key] = numeric_value
            under_cap_keys.append(key)

    if overflow > 0 and under_cap_keys:
        addition = overflow / len(
            under_cap_keys
        )

        for key in under_cap_keys:
            capped[key] += addition

    return normalize_to_total(
        capped
    )


def get_balanced_parameter_scores(
    stylometric_value: float,
    tfidf_value: float,
    ngram_value: float,
    semantic_value: float,
) -> Dict[str, float]:
    raw_scores = {
        "stylometric": float(
            stylometric_value
        ),
        "tfidf": float(
            tfidf_value
        ),
        "ngram": float(
            ngram_value
        ),
        "semantic": float(
            semantic_value
        ),
    }

    normalized = normalize_to_total(
        raw_scores
    )

    return cap_and_redistribute(
        normalized,
        max_cap=40.0,
    )


# ==========================================
# SHAP OUTPUT NORMALIZATION
# ==========================================

def _extract_shap_vector(
    raw_shap: object,
) -> np.ndarray:
    if isinstance(raw_shap, list):
        array = np.asarray(
            raw_shap[-1]
        )
    else:
        array = np.asarray(
            raw_shap
        )

    if array.ndim == 3:
        # Typical shape:
        # (samples, features, classes)
        array = array[
            0,
            :,
            -1,
        ]

    elif array.ndim == 2:
        # Typical shape:
        # (samples, features)
        array = array[0]

    elif array.ndim == 1:
        pass

    else:
        array = array.reshape(-1)

    return np.abs(
        array.astype(float)
    )


# ==========================================
# MAIN TEXT FORENSIC PIPELINE
# ==========================================

def analyze_text(
    text: str,
) -> dict:
    normalized_text = (
        text or ""
    ).strip()

    if not normalized_text:
        raise ValueError(
            "Text input cannot be empty."
        )

    stylometric_features = np.asarray(
        [
            extract_stylometric(
                normalized_text
            )
        ],
        dtype=np.float32,
    )

    tfidf_features = (
        tfidf_vectorizer
        .transform(
            [normalized_text]
        )
        .toarray()
    )

    ngram_features = (
        ngram_vectorizer
        .transform(
            [normalized_text]
        )
        .toarray()
    )

    semantic_embeddings = (
        get_sbert_model()
        .encode(
            [normalized_text],
            convert_to_numpy=True,
            show_progress_bar=False,
        )
    )

    feature_matrix = np.hstack(
        [
            stylometric_features,
            tfidf_features,
            ngram_features,
            semantic_embeddings,
        ]
    )

    raw_prediction = int(
        text_model.predict(
            feature_matrix
        )[0]
    )

    class_probabilities = (
        text_model.predict_proba(
            feature_matrix
        )[0]
    )

    predicted_probability = float(
        np.max(
            class_probabilities
        )
    )

    confidence = round(
        predicted_probability * 100.0,
        2,
    )

    ai_probability = round(
        float(
            class_probabilities[1]
        ) * 100.0,
        2,
    )

    human_probability = round(
        float(
            class_probabilities[0]
        ) * 100.0,
        2,
    )

    # ======================================
    # SHAP CONTRIBUTION ANALYSIS
    # ======================================

    raw_shap_values = (
        shap_explainer.shap_values(
            feature_matrix
        )
    )

    shap_vector = _extract_shap_vector(
        raw_shap_values
    )

    shap_total = float(
        np.sum(
            shap_vector
        )
    )

    if shap_total <= 0:
        scaled_shap = np.zeros_like(
            shap_vector
        )
    else:
        scaled_shap = (
            shap_vector
            / shap_total
        ) * 100.0

    stylometric_length = (
        stylometric_features.shape[1]
    )

    tfidf_length = (
        tfidf_features.shape[1]
    )

    ngram_length = (
        ngram_features.shape[1]
    )

    tfidf_start = (
        stylometric_length
    )

    ngram_start = (
        tfidf_start
        + tfidf_length
    )

    semantic_start = (
        ngram_start
        + ngram_length
    )

    stylometric_value = float(
        np.sum(
            scaled_shap[
                :stylometric_length
            ]
        )
    )

    tfidf_value = float(
        np.sum(
            scaled_shap[
                tfidf_start:
                ngram_start
            ]
        )
    )

    ngram_value = float(
        np.sum(
            scaled_shap[
                ngram_start:
                semantic_start
            ]
        )
    )

    semantic_value = float(
        np.sum(
            scaled_shap[
                semantic_start:
            ]
        )
    )

    balanced_scores = (
        get_balanced_parameter_scores(
            stylometric_value,
            tfidf_value,
            ngram_value,
            semantic_value,
        )
    )

    parameter_contribution = {
        "stylometric": (
            stylometric_reason(
                balanced_scores[
                    "stylometric"
                ]
            )
        ),
        "tfidf": (
            tfidf_reason(
                balanced_scores[
                    "tfidf"
                ]
            )
        ),
        "ngram": (
            ngram_reason(
                balanced_scores[
                    "ngram"
                ]
            )
        ),
        "semantic": (
            semantic_reason(
                balanced_scores[
                    "semantic"
                ]
            )
        ),
    }

    highlighted_document = (
        analyze_document(
            normalized_text
        )
    )

    prediction_label = (
        "AI"
        if raw_prediction == 1
        else "Human"
    )

    risk_level = (
        "HIGH"
        if confidence >= 75
        else "MEDIUM"
        if confidence >= 45
        else "LOW"
    )

    return {
        "prediction": prediction_label,
        "confidence": confidence,
        "ai_probability": ai_probability,
        "human_probability": (
            human_probability
        ),
        "risk_score": confidence,
        "risk_level": risk_level,
        "parameter_contribution": (
            parameter_contribution
        ),
        "highlighted_document": (
            highlighted_document
        ),
        "full_document": (
            highlighted_document
        ),
    }