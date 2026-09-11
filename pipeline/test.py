# ==========================================
# TESTING + SHAP + XAI PIPELINE
# ==========================================

import os
import sys
import json
import re
import string

import numpy as np
import pandas as pd
import joblib
import shap
import nltk

from nltk import word_tokenize, pos_tag
from textstat import flesch_reading_ease

from sentence_transformers import SentenceTransformer

from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay
)

import matplotlib.pyplot as plt

# ==========================================
# FIX IMPORT PATH
# ==========================================

sys.path.append(
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            ".."
        )
    )
)

# ==========================================
# IMPORT XAI MODULES
# ==========================================

import sys
import os

sys.path.append(
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            ".."
        )
    )
)

# ==========================================
# XAI IMPORTS
# ==========================================

from xai.parameter_reasoning import (
    stylometric_reason,
    tfidf_reason,
    ngram_reason,
    semantic_reason
)

from xai.sentence_highlighter import (
    analyze_document
)

# ==========================================
# LOAD MODELS
# ==========================================

print("\nLoading Models...")

model = joblib.load(
    "models/final_text_model.pkl"
)

tfidf = joblib.load(
    "models/tf_idf.pkl"
)

ngram = joblib.load(
    "models/n_gram.pkl"
)

sbert_model = SentenceTransformer(
    'all-MiniLM-L6-v2'
)

print("Models Loaded Successfully")

# ==========================================
# LOAD TEST DATA
# ==========================================

TEST_PATH = "dataset/test"

fake_path = os.path.join(
    TEST_PATH,
    "Fake.csv"
)

real_path = os.path.join(
    TEST_PATH,
    "Real.csv"
)

fake_df = pd.read_csv(fake_path)
real_df = pd.read_csv(real_path)

# ----------------------------------
# RENAME COLUMN IF NEEDED
# ----------------------------------

if "sentence" in fake_df.columns:
    fake_df = fake_df.rename(
        columns={"sentence": "text"}
    )

if "sentence" in real_df.columns:
    real_df = real_df.rename(
        columns={"sentence": "text"}
    )

# ----------------------------------
# LABELS
# ----------------------------------

fake_df["label"] = 1
real_df["label"] = 0

# ----------------------------------
# COMBINE DATA
# ----------------------------------

df = pd.concat([
    fake_df[["text", "label"]],
    real_df[["text", "label"]]
], ignore_index=True)

df = df.dropna()

df = df.drop_duplicates(
    subset=["text"]
)

print("\nTotal Test Samples:", len(df))

texts = df["text"].astype(str)

y_true = df["label"]

# ==========================================
# STYLOMETRIC FEATURE EXTRACTION
# ==========================================

def extract_stylometric(text):

    try:

        words = word_tokenize(text)

        sentences = re.split(
            r'[.!?]',
            text
        )

        sent_lengths = [
            len(s.split())
            for s in sentences
            if s.strip() != ""
        ]

        sent_var = (
            np.var(sent_lengths)
            if len(sent_lengths) > 0
            else 0
        )

        word_freq = len(words)

        punct_count = sum(
            1 for c in text
            if c in string.punctuation
        )

        pos_tags = pos_tag(words)

        noun_count = sum(
            1 for w, t in pos_tags
            if "NN" in t
        )

        verb_count = sum(
            1 for w, t in pos_tags
            if "VB" in t
        )

        readability = (
            flesch_reading_ease(text)
        )

        return [
            float(sent_var),
            float(word_freq),
            float(punct_count),
            float(noun_count),
            float(verb_count),
            float(readability)
        ]

    except Exception as e:

        print("Stylometric Error:", e)

        return [0, 0, 0, 0, 0, 0]

# ==========================================
# FEATURE EXTRACTION
# ==========================================

print("\nExtracting Stylometric Features...")

stylometric = np.vstack([
    extract_stylometric(t)
    for t in texts
])

print("Stylometric Shape:", stylometric.shape)

# ----------------------------------
# TF-IDF
# ----------------------------------

print("\nGenerating TF-IDF Features...")

tfidf_features = tfidf.transform(
    texts
).toarray()

print("TF-IDF Shape:", tfidf_features.shape)

# ----------------------------------
# NGRAM
# ----------------------------------

print("\nGenerating N-Gram Features...")

ngram_features = ngram.transform(
    texts
).toarray()

print("N-Gram Shape:", ngram_features.shape)

# ----------------------------------
# SBERT EMBEDDINGS
# ----------------------------------

print("\nGenerating Semantic Embeddings...")

embeddings = sbert_model.encode(
    texts.tolist()
)

print("Embeddings Shape:", embeddings.shape)

# ==========================================
# COMBINE FEATURES
# ==========================================

X = np.hstack([
    stylometric,
    tfidf_features,
    ngram_features,
    embeddings
])

print("\nFinal Feature Shape:", X.shape)

# ==========================================
# PREDICTIONS
# ==========================================

print("\nGenerating Predictions...")

pred = model.predict(X)

prob = model.predict_proba(X)

print("Predictions Completed")

# ==========================================
# SHAP EXPLAINABILITY
# ==========================================

print("\nGenerating SHAP Values...")

explainer = shap.TreeExplainer(model)

shap_values = explainer.shap_values(X)[1]

print("SHAP Analysis Completed")

# ==========================================
# FEATURE LENGTHS
# ==========================================

stylometric_len = stylometric.shape[1]

tfidf_len = tfidf_features.shape[1]

ngram_len = ngram_features.shape[1]

# ==========================================
# GENERATE RESULTS
# ==========================================

results = []

for idx in range(len(texts)):

    confidence = float(
        np.max(prob[idx])
    )

    # ----------------------------------
    # SHAP SCORES
    # ----------------------------------

    shap_val = np.abs(
        shap_values[idx]
    )

    total = np.sum(shap_val)

    if total == 0:

        scaled = np.zeros_like(
            shap_val
        )

    else:

        scaled = (
            shap_val / total
        ) * 100

    # ----------------------------------
    # SPLIT FEATURE CONTRIBUTIONS
    # ----------------------------------

    stylometric_val = np.sum(
        scaled[:stylometric_len]
    )

    tfidf_val = np.sum(
        scaled[
            stylometric_len:
            stylometric_len + tfidf_len
        ]
    )

    ngram_val = np.sum(
        scaled[
            stylometric_len + tfidf_len:
            stylometric_len + tfidf_len + ngram_len
        ]
    )

    semantic_val = np.sum(
        scaled[
            stylometric_len + tfidf_len + ngram_len:
        ]
    )

    # ==========================================
    # PARAMETER REASONING
    # ==========================================

    parameter_contribution = {

        "stylometric":
            stylometric_reason(
                stylometric_val
            ),

        "tfidf":
            tfidf_reason(
                tfidf_val
            ),

        "ngram":
            ngram_reason(
                ngram_val
            ),

        "semantic":
            semantic_reason(
                semantic_val
            )
    }

    # ==========================================
    # SENTENCE LEVEL ANALYSIS
    # ==========================================

    suspicious_sentences = (
        analyze_document(
            texts.iloc[idx]
        )
    )

    # ==========================================
    # FINAL RESULT
    # ==========================================

    results.append({

        "id": idx + 1,

        "text": texts.iloc[idx],

        "actual":
            "AI"
            if y_true.iloc[idx] == 1
            else "Human",

        "prediction":
            "AI"
            if pred[idx] == 1
            else "Human",

        "confidence":
            round(confidence * 100, 2),

        "parameter_contribution":
            parameter_contribution,

        "suspicious_sentences":
            suspicious_sentences
    })

# ==========================================
# SAVE JSON OUTPUT
# ==========================================

with open(
    "results.json",
    "w"
) as f:

    json.dump(
        results,
        f,
        indent=4
    )

print("\n✅ JSON OUTPUT GENERATED")

print("\nSaved File: results.json")

# ==========================================
# CONFUSION MATRIX
# ==========================================

cm = confusion_matrix(
    y_true,
    pred
)

print("\nConfusion Matrix:")
print(cm)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Human", "AI"]
)

disp.plot(cmap="Blues")

plt.title(
    "Testing Confusion Matrix"
)

plt.show()

print("\n✅ TEST PIPELINE COMPLETED")