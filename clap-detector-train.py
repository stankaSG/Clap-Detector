import sounddevice as sd
import wavio
import librosa
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
import joblib


def record_audio(filename, duration = 2, samplerate=44100):
    print(f"Recording {filename}... Clap or make noise.")
    audio = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype='float32')
    sd.wait()
    wavio.write(filename, audio, samplerate, sampwidth=3)
    print(f"Saved as {filename}")

# record_audio(filename="audios/clap13.wav")
# record_audio(filename="audios/noise13.wav")

def extract_features(filename):
    #     Loading audio and extract MFCCs
    y, sr = librosa.load(filename, sr=44100)

    intervals = librosa.effects.split(y, top_db=8)
    if len(intervals) > 0:
        y_trimmed = np.concatenate([y[start:end] for start, end in intervals])
    else:
        y_trimmed = y

    # Is it long enought ?
    if len(y_trimmed) < 500:
        print(f"Warning: Skipping {filename}, audio is too short!")
        return None

    y_trimmed = librosa.util.normalize(y_trimmed)

    y_harmonic, y_percussive = librosa.effects.hpss(y_trimmed)
    mfccs = np.mean(librosa.feature.mfcc(y=y_percussive, sr=sr, n_mfcc=40), axis=1)
    mel = np.mean(librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128), axis=1)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr), axis=1)
    onset_strength = np.mean(librosa.onset.onset_strength(y=y_percussive, sr=sr))
    contrast = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr), axis=1)
    rms = np.mean(librosa.feature.rms(y=y))
    flatness = np.mean(librosa.feature.spectral_flatness(y=y), axis=1)
    zero_crossing_rate = np.mean(librosa.feature.zero_crossing_rate(y))

    return np.hstack([mfccs, mel, tempo, chroma, onset_strength, contrast, rms, flatness, zero_crossing_rate])

# clap_features=extract_features("audios/clap2.wav")
# noise_features=extract_features("audios/noise4.wav")
#
# print(f"Clap Features:", clap_features)
# print(f"Noise Features:", noise_features)

# /-------Model Training--------/
# Data preparation
X = [extract_features(f"audios/clap{i}.wav") for i in range(1, 51)] + \
    [extract_features(f"audios/noise{i}.wav") for i in range(1, 51)] + \
    [extract_features(f"audios/voice{i}.wav") for i in range(1,51)]
y = [0] * 50 + [1] * 50 + [2] * 50

# Division to train and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Neuron net training
model = MLPClassifier(hidden_layer_sizes=(128, 64),
                      solver="adam",
                      activation="relu",
                      max_iter=5000,
                      alpha=0.0005)
model.fit(X_train, y_train)

# Accuracy testing
accuracy= model.score(X_test, y_test)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Save model
joblib.dump(model, "clap_detector.pkl")
print("Model was saved as clap_detector.pkl")
