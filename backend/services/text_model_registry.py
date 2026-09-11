from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import joblib
import shap


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MODEL_DIR = PROJECT_ROOT / "models"

TEXT_MODEL_PATH = MODEL_DIR / "final_text_model.pkl"
TFIDF_PATH = MODEL_DIR / "tf_idf.pkl"
NGRAM_PATH = MODEL_DIR / "n_gram.pkl"

SBERT_MODEL_NAME = "all-MiniLM-L6-v2"


print("Loading shared text forensic models...")

text_model = joblib.load(TEXT_MODEL_PATH)
tfidf_vectorizer = joblib.load(TFIDF_PATH)
ngram_vectorizer = joblib.load(NGRAM_PATH)
shap_explainer = shap.TreeExplainer(text_model)

print("Shared text forensic models loaded.")


_sbert_model: Any = None
_sbert_lock = threading.Lock()


def get_sbert_model() -> Any:
    """
    Lazily load Sentence-BERT once.

    Lazy initialization prevents PyTorch, Transformers and TensorFlow
    from initializing simultaneously during FastAPI application import.
    """
    global _sbert_model

    if _sbert_model is not None:
        return _sbert_model

    with _sbert_lock:
        if _sbert_model is None:
            print("Loading Sentence-BERT model lazily...")

            from sentence_transformers import SentenceTransformer

            _sbert_model = SentenceTransformer(
                SBERT_MODEL_NAME,
                device="cpu",
            )

            print("Sentence-BERT model loaded successfully.")

    return _sbert_model