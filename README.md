# F.O.R.G.E

## Forensic Observation & Recognition Gateway for Emerging Generative Exploits

F.O.R.G.E is a multimodal AI-forensics platform designed to detect synthetic and manipulated content across Text, Images, and Audio while providing Explainable AI (XAI) reasoning for every prediction.

The system combines machine learning, deep learning, forensic feature extraction, visual heatmaps, and explainable evidence generation to help identify AI-generated or manipulated media.

---

## Key Features

### Text Forensics

* AI-generated text detection
* Stylometric analysis
* TF-IDF feature extraction
* N-Gram pattern analysis
* SBERT semantic embeddings
* SHAP explainability
* Sentence-level heatmaps
* PDF, DOCX and TXT support

### Image Forensics

* CNN-based forgery detection
* Random Forest fusion engine
* 48 forensic image features
* GAN artifact detection
* Metadata analysis
* Face consistency analysis
* Heatmap visualization
* Explainable image reasoning

### Audio Forensics

* CNN-BiLSTM deepfake voice detection
* LFCC feature extraction
* Spectrogram analysis
* Waveform visualization
* Acoustic anomaly detection
* Explainable audio reasoning

---

## Explainable AI (XAI)

F.O.R.G.E does not provide only a classification result.

Every prediction includes:

* Parameter contribution analysis
* Confidence estimation
* Risk scoring
* Sentence-level explanation
* Image forensic heatmaps
* Audio anomaly localization

This enables transparent and interpretable forensic decisions.

---

## Technology Stack

### Frontend

* React.js
* CSS3
* Responsive Dashboard UI

### Backend

* FastAPI
* Python

### Machine Learning

* Scikit-Learn
* TensorFlow / Keras
* Random Forest
* CNN
* CNN-BiLSTM
* Sentence Transformers
* SHAP

### Computer Vision

* OpenCV
* NumPy
* PIL

---

## Project Architecture

```text
Frontend (React)
        │
        ▼
FastAPI Backend
        │
 ┌──────┼──────┐
 ▼      ▼      ▼
Text  Image  Audio
Model Model  Model
 │      │      │
 ▼      ▼      ▼
 XAI   Heatmap Spectrogram
 │      │      │
 └──────┼──────┘
        ▼
 Investigation Report
```

---

## Project Structure

```text
F.O.R.G.E
│
├── backend
│   ├── services
│   ├── xai
│   ├── reports
│   └── uploads
│
├── frontend
│   ├── src
│   ├── public
│   └── assets
│
├── models
│   ├── text_models
│   ├── image_models
│   └── audio_models
│
└── README.md
```

---

## Research Objective

The objective of F.O.R.G.E is to develop an explainable multimodal forensic framework capable of identifying AI-generated media while providing transparent reasoning for investigative, academic, and cybersecurity applications.

---

## Future Enhancements

* Video Deepfake Detection
* Real-Time Browser Extension
* Blockchain Evidence Verification
* Cloud Deployment
* Multi-language Text Forensics
* Advanced Adversarial Robustness

---

## Author

Ashish Ranjan

B.Tech Computer Science

Multimodal Deepfake Detection & Explainable AI Research

---

## License

MIT License
