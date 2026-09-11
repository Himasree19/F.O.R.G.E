from __future__ import annotations

import re
import string

import numpy as np
from nltk import pos_tag, word_tokenize
from textstat import flesch_reading_ease

from backend.services.text_model_registry import (
    get_sbert_model,
    ngram_vectorizer,
    text_model,
    tfidf_vectorizer,
)


# ==========================================
# SENTENCE RISK CLASSIFICATION
# ==========================================

def classify_sentence(
    score: float,
) -> dict:
    numeric_score = float(
        score
    )

    if numeric_score >= 85:
        return {
            "risk": "VERY HIGH",
            "color": "red",
        }

    if numeric_score >= 60:
        return {
            "risk": "HIGH",
            "color": "orange",
        }

    if numeric_score >= 40:
        return {
            "risk": "MEDIUM",
            "color": "yellow",
        }

    return {
        "risk": "LOW",
        "color": "green",
    }


# ==========================================
# STYLOMETRIC EXTRACTION
# ==========================================

def extract_stylometric(
    text: str,
) -> list[float]:
    try:
        words = word_tokenize(
            text
        )

        sentence_parts = re.split(
            r"[.!?]",
            text,
        )

        sentence_lengths = [
            len(sentence.split())
            for sentence in sentence_parts
            if sentence.strip()
        ]

        sentence_variance = (
            float(
                np.var(
                    sentence_lengths
                )
            )
            if sentence_lengths
            else 0.0
        )

        word_frequency = float(
            len(words)
        )

        punctuation_count = float(
            sum(
                1
                for character in text
                if character
                in string.punctuation
            )
        )

        tagged_words = pos_tag(
            words
        )

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
            flesch_reading_ease(
                text
            )
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
            "Sentence stylometric warning:",
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
# EXPLANATION GENERATION
# ==========================================

def build_sentence_reason(
    ai_score: float,
) -> str:
    if ai_score >= 85:
        return (
            "Strong machine-generated linguistic "
            "patterns were detected, including highly "
            "regular sentence organization, predictable "
            "lexical selection, and reduced stylistic "
            "variation."
        )

    if ai_score >= 60:
        return (
            "The sentence contains multiple AI-like "
            "characteristics, including comparatively "
            "uniform structure and statistically "
            "predictable phrasing."
        )

    if ai_score >= 40:
        return (
            "Moderate synthetic-language indicators "
            "were detected; however, the available "
            "evidence is not independently conclusive."
        )

    return (
        "The sentence preserves natural human-like "
        "variation in vocabulary, phrasing, and "
        "sentence construction."
    )


# ==========================================
# DOCUMENT SENTENCE ANALYSIS
# ==========================================

def analyze_document(
    text: str,
) -> list[dict]:
    normalized_text = (
        text or ""
    ).strip()

    if not normalized_text:
        return []

    sentences = re.split(
        r"(?<=[.!?])\s+",
        normalized_text,
    )

    results: list[dict] = []

    sbert_model = get_sbert_model()

    for sentence_index, sentence in enumerate(
        sentences
    ):
        normalized_sentence = (
            sentence.strip()
        )

        if len(normalized_sentence) < 5:
            continue

        try:
            stylometric_features = (
                np.asarray(
                    [
                        extract_stylometric(
                            normalized_sentence
                        )
                    ],
                    dtype=np.float32,
                )
            )

            tfidf_features = (
                tfidf_vectorizer
                .transform(
                    [normalized_sentence]
                )
                .toarray()
            )

            ngram_features = (
                ngram_vectorizer
                .transform(
                    [normalized_sentence]
                )
                .toarray()
            )

            semantic_embeddings = (
                sbert_model.encode(
                    [
                        normalized_sentence
                    ],
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

            probability = (
                text_model.predict_proba(
                    feature_matrix
                )[0]
            )

            ai_score = round(
                float(
                    probability[1]
                ) * 100.0,
                2,
            )

            human_score = round(
                float(
                    probability[0]
                ) * 100.0,
                2,
            )

            classification = (
                classify_sentence(
                    ai_score
                )
            )

            results.append(
                {
                    "index": sentence_index,
                    "sentence": (
                        normalized_sentence
                    ),
                    "score": ai_score,
                    "ai_probability": (
                        ai_score
                    ),
                    "human_probability": (
                        human_score
                    ),
                    "risk": (
                        classification[
                            "risk"
                        ]
                    ),
                    "color": (
                        classification[
                            "color"
                        ]
                    ),
                    "reason": (
                        build_sentence_reason(
                            ai_score
                        )
                    ),
                }
            )

        except Exception as error:
            print(
                "Sentence analysis error:",
                normalized_sentence[:80],
                error,
            )

            results.append(
                {
                    "index": sentence_index,
                    "sentence": (
                        normalized_sentence
                    ),
                    "score": 0.0,
                    "ai_probability": 0.0,
                    "human_probability": 0.0,
                    "risk": "UNKNOWN",
                    "color": "gray",
                    "reason": (
                        "The sentence could not be "
                        "evaluated because an internal "
                        "feature extraction error occurred."
                    ),
                    "error": str(
                        error
                    ),
                }
            )

    return results