FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV TF_CPP_MIN_LOG_LEVEL=2
ENV USE_TF=0
ENV TOKENIZERS_PARALLELISM=false

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ffmpeg \
    libsndfile1 \
    libgl1 \
    libglib2.0-0 \
    tesseract-ocr \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt /app/backend/requirements.txt

RUN python -m pip install --upgrade pip setuptools wheel \
    && python -m pip install -r /app/backend/requirements.txt

COPY backend /app/backend
COPY models /app/models
COPY xai /app/xai
COPY pipeline /app/pipeline
COPY features /app/features
COPY preprocessing /app/preprocessing

RUN mkdir -p \
    /app/uploads/audio \
    /app/uploads/heatmaps \
    /app/backend/uploads \
    /app/backend/uploads/heatmaps \
    /app/backend/audio_visuals/waveforms \
    /app/backend/audio_visuals/heatmaps \
    /app/backend/audio_visuals/spectrograms \
    /app/backend/charts \
    /app/backend/reports \
    /app/reports \
    /app/outputs

EXPOSE 8000
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]