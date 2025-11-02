import streamlit as st
import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras.models import load_model

# -------------------- Load trained model --------------------
MODEL_PATH =  r"C:\Users\Amrutha\Downloads\for-2sec\final_crnn_bilstm.h5"
model = load_model(MODEL_PATH)

# -------------------- Preprocessing function --------------------
def preprocess_audio(audio_path):
    y, sr = librosa.load(audio_path, sr=16000, duration=2.0, mono=True)

    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    log_mel = librosa.power_to_db(mel, ref=np.max)
    log_mel = (log_mel - log_mel.min()) / (log_mel.max() - log_mel.min())

    # Resize using NumPy if needed
    if log_mel.shape[1] != 63:
        log_mel = np.interp(
            np.linspace(0, log_mel.shape[1], 63),
            np.arange(log_mel.shape[1]),
            log_mel,
        )

    log_mel = log_mel[:, :63]  # trim excess (safe-guard)
    return log_mel[np.newaxis, :, :, np.newaxis].astype(np.float32)

# -------------------- Streamlit UI --------------------
st.title("🎙️ DeepFake Audio Detector")
st.write("Upload a **2-second WAV audio file** to check whether it is REAL or FAKE.")

uploaded_file = st.file_uploader("Upload a `.wav` file", type=["wav"])

if uploaded_file is not None:
    st.audio(uploaded_file, format="audio/wav")

    with open("temp.wav", "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.write("🔄 Processing audio...")
    data = preprocess_audio("temp.wav")
    pred = model.predict(data)[0][0]

    label = "🟥 FAKE AUDIO" if pred > 0.5 else "🟩 REAL AUDIO"
    confidence = float(pred if pred > 0.5 else 1 - pred)

    st.subheader(f"Result: **{label}**")
    st.write(f"Confidence: **{confidence:.4f}**")

    if pred > 0.5:
        st.warning("⚠️ This audio is likely synthetically generated.")
    else:
        st.success("✅ This audio is likely real.")

st.write("---")
st.caption("Model: CRNN + BiLSTM | Dataset: Fake or Real (2 sec audios)")
