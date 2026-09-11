import os
import uuid
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

VISUAL_DIR = os.path.join(BASE_DIR, "audio_visuals")
WAVEFORM_DIR = os.path.join(VISUAL_DIR, "waveforms")
SPECTROGRAM_DIR = os.path.join(VISUAL_DIR, "spectrograms")
HEATMAP_DIR = os.path.join(VISUAL_DIR, "heatmaps")

os.makedirs(WAVEFORM_DIR, exist_ok=True)
os.makedirs(SPECTROGRAM_DIR, exist_ok=True)
os.makedirs(HEATMAP_DIR, exist_ok=True)

SAMPLE_RATE = 16000


def load_audio(file_path):
    audio, sr = librosa.load(file_path, sr=SAMPLE_RATE, mono=True)
    return audio, sr


def generate_waveform(file_path):
    audio, sr = load_audio(file_path)

    name = f"{uuid.uuid4()}_waveform.png"
    save_path = os.path.join(WAVEFORM_DIR, name)

    plt.figure(figsize=(12, 4))
    librosa.display.waveshow(audio, sr=sr)
    plt.title("Audio Waveform")
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

    return f"/audio-visuals/waveforms/{name}"


def generate_spectrogram(file_path):
    audio, sr = load_audio(file_path)

    name = f"{uuid.uuid4()}_spectrogram.png"
    save_path = os.path.join(SPECTROGRAM_DIR, name)

    stft = librosa.stft(audio)
    db = librosa.amplitude_to_db(np.abs(stft), ref=np.max)

    plt.figure(figsize=(12, 5))
    librosa.display.specshow(db, sr=sr, x_axis="time", y_axis="hz")
    plt.colorbar(format="%+2.0f dB")
    plt.title("Audio Spectrogram")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

    return f"/audio-visuals/spectrograms/{name}"


def generate_audio_heatmap(file_path):
    audio, sr = load_audio(file_path)

    name = f"{uuid.uuid4()}_audio_heatmap.png"
    save_path = os.path.join(HEATMAP_DIR, name)

    mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128)
    mel_db = librosa.power_to_db(mel, ref=np.max)

    plt.figure(figsize=(12, 5))
    librosa.display.specshow(mel_db, sr=sr, x_axis="time", y_axis="mel")
    plt.colorbar(format="%+2.0f dB")
    plt.title("Audio XAI Heatmap")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

    return f"/audio-visuals/heatmaps/{name}"


def generate_audio_visuals(file_path):
    return {
        "waveform": generate_waveform(file_path),
        "spectrogram": generate_spectrogram(file_path),
        "audio_heatmap": generate_audio_heatmap(file_path)
    }