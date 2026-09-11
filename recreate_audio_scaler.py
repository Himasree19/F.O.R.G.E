import os
import numpy as np
import librosa
from sklearn.preprocessing import StandardScaler

DATASET_PATH = "audio_ai/trainning"

REAL_DIR = os.path.join(DATASET_PATH, "real")
FAKE_DIR = os.path.join(DATASET_PATH, "fake")

SAVE_DIR = "backend/models"

os.makedirs(SAVE_DIR, exist_ok=True)

SAMPLE_RATE = 16000


def safe_mean(x):
    return float(np.mean(x)) if len(x) > 0 else 0.0


def safe_std(x):
    return float(np.std(x)) if len(x) > 0 else 0.0


def calculate_entropy(signal):
    hist, _ = np.histogram(signal, bins=50, density=True)
    hist = hist + 1e-8
    return float(-np.sum(hist * np.log2(hist)))


def extract_audio_parameters(audio):
    spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=SAMPLE_RATE)[0]
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=SAMPLE_RATE)[0]
    spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=SAMPLE_RATE)[0]
    spectral_flatness = librosa.feature.spectral_flatness(y=audio)[0]

    zcr = librosa.feature.zero_crossing_rate(audio)[0]
    rms = librosa.feature.rms(y=audio)[0]

    entropy = calculate_entropy(audio)
    variance = float(np.var(audio))

    try:
        pitch = librosa.yin(audio, fmin=50, fmax=500, sr=SAMPLE_RATE)
        pitch = pitch[np.isfinite(pitch)]
        pitch_mean = safe_mean(pitch)
        pitch_std = safe_std(pitch)
        pitch_range = float(np.max(pitch) - np.min(pitch)) if len(pitch) > 0 else 0.0
    except Exception:
        pitch_mean = 0.0
        pitch_std = 0.0
        pitch_range = 0.0

    pause_ratio = float(np.mean(np.abs(audio) < 0.02))

    amplitude_diff = np.diff(audio)
    amplitude_discontinuity = float(np.mean(np.abs(amplitude_diff)))

    stft = librosa.stft(audio)
    phase = np.angle(stft)

    phase_variance = float(np.var(phase))
    phase_discontinuity = float(np.mean(np.abs(np.diff(phase, axis=1)))

    )

    noise_floor = float(np.percentile(np.abs(audio), 10))
    energy_variation = safe_std(rms)

    try:
        tempo, _ = librosa.beat.beat_track(y=audio, sr=SAMPLE_RATE)
        tempo = float(tempo)
    except Exception:
        tempo = 0.0

    chroma = librosa.feature.chroma_stft(y=audio, sr=SAMPLE_RATE)

    return np.array([
        safe_mean(spectral_centroid),
        safe_std(spectral_centroid),

        safe_mean(spectral_bandwidth),
        safe_std(spectral_bandwidth),

        safe_mean(spectral_rolloff),
        safe_std(spectral_rolloff),

        safe_mean(spectral_flatness),
        safe_std(spectral_flatness),

        safe_mean(zcr),
        safe_std(zcr),

        safe_mean(rms),
        safe_std(rms),

        entropy,
        variance,

        pitch_mean,
        pitch_std,
        pitch_range,

        pause_ratio,
        amplitude_discontinuity,

        phase_variance,
        phase_discontinuity,

        noise_floor,
        energy_variation,

        tempo,

        float(np.mean(chroma)),
        float(np.std(chroma)),

        float(len(audio) / SAMPLE_RATE)
    ], dtype=np.float32)


all_params = []

for folder in [REAL_DIR, FAKE_DIR]:
    for filename in os.listdir(folder):
        if filename.lower().endswith((".wav", ".flac")):
            path = os.path.join(folder, filename)

            try:
                audio, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)
                params = extract_audio_parameters(audio)
                all_params.append(params)

            except Exception as e:
                print("Error:", path, e)

print("Total parameter samples:", len(all_params))

X_params = np.array(all_params, dtype=np.float32)

scaler = StandardScaler()
scaler.fit(X_params)

np.save(os.path.join(SAVE_DIR, "scaler_mean.npy"), scaler.mean_)
np.save(os.path.join(SAVE_DIR, "scaler_scale.npy"), scaler.scale_)

print("✅ scaler_mean.npy saved")
print("✅ scaler_scale.npy saved")
print("Saved in:", SAVE_DIR)
