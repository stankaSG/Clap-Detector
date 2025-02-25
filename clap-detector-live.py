import sounddevice as sd
import librosa
import numpy as np
import joblib
import subprocess
import wavio

model = joblib.load("clap_detector.pkl")

def record_audio(filename, duration =2 , samplerate=44100):
    print(f"Recording...")
    audio = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype='float32')
    sd.wait()
    wavio.write(filename, audio, samplerate, sampwidth=3)


def extract_features(filename):
    y, sr = librosa.load(filename, sr=44100)

    intervals = librosa.effects.split(y, top_db=8)
    if len(intervals) > 0:
        y_trimmed = np.concatenate([y[start:end] for start, end in intervals])
    else:
        y_trimmed = y

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

def live_detection():
    # recording audio from micro and detect
    while True:
        record_audio("live.wav", duration=2)
        features = extract_features("live.wav")

        if features is None:
            print("No valid audio detected, retrying...")
            continue

        prediction = model.predict([features])[0]

        probabilities = model.predict_proba([features])[0]
        print(f"Predicted class: {prediction}")
        print(f"Prediction probabilities: {probabilities}")

        if prediction == 0:
            print("Clap detected...")
            subprocess.run("spotify.exe", shell=True)
            break

live_detection()
