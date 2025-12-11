import pickle
import librosa
import numpy as np

SAMPLE_RATE = 16000
N_MFCC = 40
class_names = ['MY_YES','ANOTHER_YES','NOT_YS','NOISE']

try:
    with open('audio_model.pkl', 'rb') as file:
        svc_model = pickle.load(file)
    with open('audio_scaler.pkl', 'rb') as file:
        scaler = pickle.load(file)
except FileNotFoundError:
    raise FileNotFoundError("Model files not found. Please ensure 'audio_model.pkl' and 'audio_scaler.pkl' are in the same directory.")

def extract_features(path, sample_rate=SAMPLE_RATE, n_mfcc=N_MFCC):
    voice, sr = librosa.load(path, sr=sample_rate, res_type='kaiser_fast')
    mfccs = librosa.feature.mfcc(y=voice, sr=sr, n_mfcc=n_mfcc)
    mfccs_processed = np.mean(mfccs.T, axis=0)
    return mfccs_processed

def predict_audio(path):
    """
    Fungsi untuk memprediksi suara dari file path.
    Input: path (str)
    Output: predicted_class_name (str), confidence (float)
    """
    features = extract_features(path)
    features_scaled = scaler.transform([features])
    pred_idx = svc_model.predict(features_scaled)[0]
    proba = svc_model.predict_proba(features_scaled)[0]

    predicted_class_name = class_names[pred_idx]
    confidence = np.max(proba)

    return predicted_class_name, confidence