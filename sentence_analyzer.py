import re


class SentenceAnalyzer:

    def __init__(self, model, vectorizer, cleaner):

        self.model = model
        self.vectorizer = vectorizer
        self.cleaner = cleaner

    def split_sentences(self, text):

        sentences = re.split(r'(?<=[.!?])\s+', text)

        return [s.strip() for s in sentences if s.strip()]

    def analyze_sentences(self, text):

        sentences = self.split_sentences(text)

        suspicious_results = []

        for sentence in sentences:

            cleaned = self.cleaner(sentence)

            vector = self.vectorizer.transform([cleaned])

            probabilities = self.model.predict_proba(vector)[0]

            ai_probability = probabilities[1]

            reason = self.generate_reason(sentence, ai_probability)

            suspicious_results.append({

                "sentence": sentence,

                "ai_score": round(float(ai_probability), 4),

                "reason": reason
            })

        # sort by suspiciousness
        suspicious_results = sorted(
            suspicious_results,
            key=lambda x: x["ai_score"],
            reverse=True
        )

        return suspicious_results

    def generate_reason(self, sentence, score):

        sentence_length = len(sentence.split())

        if score > 0.90:
            return "Strong AI-like sentence structure detected"

        elif score > 0.75:
            return "Highly formal and statistically predictable phrasing"

        elif sentence_length > 25:
            return "Long uniform sentence structure"

        elif "furthermore" in sentence.lower():
            return "Common AI transitional phrase detected"

        else:
            return "Moderate AI-like writing patterns"