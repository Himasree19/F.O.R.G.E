# ==========================================
# TRAINING PIPELINE
# ==========================================

import os
import re
import string
import joblib
import nltk
import numpy as np
import pandas as pd

from nltk import word_tokenize, pos_tag
from textstat import flesch_reading_ease

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sentence_transformers import SentenceTransformer

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

# ----------------------------------
# DOWNLOAD NLTK
# ----------------------------------

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

# ----------------------------------
# CREATE MODELS FOLDER
# ----------------------------------

os.makedirs("models", exist_ok=True)

# ----------------------------------
# LOAD DATA
# ----------------------------------

TRAIN_PATH = "dataset/train"

fake_path = os.path.join(TRAIN_PATH, "Fake.csv")
real_path = os.path.join(TRAIN_PATH, "Real.csv")

fake_df = pd.read_csv(fake_path)
real_df = pd.read_csv(real_path)

# rename if needed
if "sentence" in fake_df.columns:
    fake_df = fake_df.rename(columns={"sentence": "text"})

if "sentence" in real_df.columns:
    real_df = real_df.rename(columns={"sentence": "text"})

# labels
fake_df["label"] = 1
real_df["label"] = 0

# combine
df = pd.concat([
    fake_df[["text", "label"]],
    real_df[["text", "label"]]
], ignore_index=True)

df = df.dropna().drop_duplicates(subset=["text"])

print("Training Samples:", len(df))

texts = df["text"].astype(str)
labels = df["label"]

# ----------------------------------
# STYLOMETRIC FEATURES
# ----------------------------------

def extract_stylometric(text):

    try:
        words = word_tokenize(text)

        sentences = re.split(r'[.!?]', text)

        sent_lengths = [
            len(s.split())
            for s in sentences
            if s.strip() != ""
        ]

        sent_var = np.var(sent_lengths) if len(sent_lengths) > 0 else 0

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
            float(sent_var),
            float(word_freq),
            float(punct_count),
            float(noun_count),
            float(verb_count),
            float(readability)
        ]

    except:
        return [0, 0, 0, 0, 0, 0]

# ----------------------------------
# FEATURE EXTRACTION
# ----------------------------------

print("\nExtracting Stylometric Features...")
stylometric = np.vstack([
    extract_stylometric(t)
    for t in texts
])

print("Stylometric Shape:", stylometric.shape)

# TF-IDF
print("\nExtracting TF-IDF Features...")

tfidf = TfidfVectorizer(
    max_features=5000,
    stop_words="english"
)

tfidf_features = tfidf.fit_transform(texts).toarray()

print("TF-IDF Shape:", tfidf_features.shape)

# NGRAM
print("\nExtracting N-Gram Features...")

ngram = CountVectorizer(
    ngram_range=(2, 3),
    max_features=3000
)

ngram_features = ngram.fit_transform(texts).toarray()

print("N-Gram Shape:", ngram_features.shape)

# SEMANTIC EMBEDDINGS
print("\nGenerating SBERT Embeddings...")

sbert_model = SentenceTransformer('all-MiniLM-L6-v2')

embeddings = sbert_model.encode(texts.tolist())

print("Embeddings Shape:", embeddings.shape)

# ----------------------------------
# COMBINE FEATURES
# ----------------------------------

X = np.hstack([
    stylometric,
    tfidf_features,
    ngram_features,
    embeddings
])

y = labels

print("\nFinal Feature Shape:", X.shape)

# ----------------------------------
# TRAIN TEST SPLIT
# ----------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ----------------------------------
# TRAIN MODEL
# ----------------------------------

print("\nTraining Random Forest Model...")

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# ----------------------------------
# EVALUATE
# ----------------------------------

pred = model.predict(X_test)

accuracy = accuracy_score(y_test, pred)

print("\nAccuracy:", accuracy)

print("\nClassification Report:\n")
print(classification_report(y_test, pred))

# ----------------------------------
# SAVE MODELS
# ----------------------------------

print("\nSaving Models...")

joblib.dump(model, "models/final_text_model.pkl")
joblib.dump(tfidf, "models/tf_idf.pkl")
joblib.dump(ngram, "models/n_gram.pkl")

print("\n✅ Training Completed")