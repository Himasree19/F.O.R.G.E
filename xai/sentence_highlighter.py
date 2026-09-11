# ==========================================
# SENTENCE LEVEL ANALYZER
# ==========================================

import re

from xai.parameter_reasoning import (
    semantic_reason,
    stylometric_reason
)

# ==========================================
# SPLIT SENTENCES
# ==========================================

def split_sentences(text):

    sentences = re.split(r'(?<=[.!?])\s+', text)

    return [
        s.strip()
        for s in sentences
        if len(s.strip()) > 10
    ]

# ==========================================
# ANALYZE SENTENCE
# ==========================================

def analyze_sentence(sentence):

    word_count = len(sentence.split())

    punctuation = sentence.count(",")

    avg_word_len = (
        sum(len(w) for w in sentence.split())
        / max(word_count, 1)
    )

    # ----------------------------------
    # SIMPLE RISK HEURISTICS
    # ----------------------------------

    score = 0

    if word_count > 25:
        score += 30

    if avg_word_len > 5:
        score += 25

    if punctuation > 3:
        score += 20

    if "furthermore" in sentence.lower():
        score += 15

    if "moreover" in sentence.lower():
        score += 15

    if "therefore" in sentence.lower():
        score += 15

    # cap
    score = min(score, 100)

    # ----------------------------------
    # GENERATE REASONING
    # ----------------------------------

    reasoning = semantic_reason(score)

    return {

        "sentence": sentence,

        "score": score,

        "risk": reasoning["risk"],

        "reason": reasoning["reason"]
    }

# ==========================================
# DOCUMENT ANALYSIS
# ==========================================

def analyze_document(text):

    sentences = split_sentences(text)

    results = []

    for s in sentences:

        result = analyze_sentence(s)

        if result["score"] >= 35:

            results.append(result)

    return results