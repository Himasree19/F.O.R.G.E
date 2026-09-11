# ==========================================
# VALIDATION PIPELINE
# ==========================================

import os
import json
import numpy as np
import pandas as pd
import joblib
import re
import string

import nltk

from nltk import word_tokenize, pos_tag
from textstat import flesch_reading_ease

from sentence_transformers import SentenceTransformer

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

import matplotlib.pyplot as plt

# ----------------------------------
# LOAD MODELS
# ----------------------------------

model = joblib.load("models/final_text_model.pkl")
tfidf = joblib.load("models/tf_idf.pkl")
ngram = joblib.load("models/n_gram.pkl")

sbert_model = SentenceTransformer('all-MiniLM-L6-v2')

# ----------------------------------
# LOAD VALIDATION DATA
# ----------------------------------

VAL_PATH = "dataset/validation"

fake_path = os.path.join(VAL_PATH, "Fake.csv")
real_path = os.path.join(VAL_PATH, "Real.csv")

fake_df = pd.read_csv(fake_path)
real_df = pd.read_csv(real_path)

if "sentence" in fake_df.columns:
    fake_df = fake_df.rename(columns={"sentence": "text"})

if "sentence" in real_df.columns:
    real_df = real_df.rename(columns={"sentence": "text"})

fake_df["label"] = 1
real_df["label"] = 0

df = pd.concat([
    fake_df[["text", "label"]],
    real_df[["text", "label"]]
], ignore_index=True)

df = df.dropna().drop_duplicates(subset=["text"])

texts = df["text"].astype(str)
y_true = df["label"]

print("Validation Samples:", len(df))

# ----------------------------------
# STYLOMETRIC
# ----------------------------------

def extract_stylometric(text):

    try:

        words = word_tokenize(text)

        sentences = re.split(r'[.!?]', text)

        sent_lengths = [
            len(s.split())
            for s in sentences
            if s.strip()
        ]

        sent_var = np.var(sent_lengths) if sent_lengths else 0

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

        readability = flesch_reading_ease(text)

        return [
            sent_var,
            word_freq,
            punct_count,
            noun_count,
            verb_count,
            readability
        ]

    except:
        return [0, 0, 0, 0, 0, 0]

# ----------------------------------
# FEATURE EXTRACTION
# ----------------------------------

stylometric = np.vstack([
    extract_stylometric(t)
    for t in texts
])

tfidf_features = tfidf.transform(texts).toarray()

ngram_features = ngram.transform(texts).toarray()

embeddings = sbert_model.encode(texts.tolist())

X = np.hstack([
    stylometric,
    tfidf_features,
    ngram_features,
    embeddings
])

# ----------------------------------
# PREDICTION
# ----------------------------------

pred = model.predict(X)
prob = model.predict_proba(X)

# ----------------------------------
# JSON OUTPUT
# ----------------------------------

results = []

for idx in range(len(texts)):

    confidence = float(np.max(prob[idx]))

    results.append({

        "id": idx + 1,

        "sentence": texts.iloc[idx],

        "actual":
            "AI" if y_true.iloc[idx] == 1
            else "Human",

        "prediction":
            "AI" if pred[idx] == 1
            else "Human",

        "confidence": confidence
    })

# ----------------------------------
# SAVE JSON
# ----------------------------------

with open("validation_results.json", "w") as f:
    json.dump(results, f, indent=4)

print("\n✅ Validation JSON Saved")

# ----------------------------------
# METRICS
# ----------------------------------

print("\nAccuracy:", accuracy_score(y_true, pred))

print("\nClassification Report:\n")
print(classification_report(y_true, pred))

cm = confusion_matrix(y_true, pred)

disp = ConfusionMatrixDisplay(
    cm,
    display_labels=["Human", "AI"]
)

disp.plot(cmap="Blues")

plt.title("Validation Confusion Matrix")

plt.show()