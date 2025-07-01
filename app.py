import streamlit as st
import torch
import numpy as np
import soundfile as sf
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import tempfile
from audiorecorder import audiorecorder

st.set_page_config(page_title="Vietnamese ASR with Whisper", layout="centered")

@st.cache_resource
def load_model():
    processor = WhisperProcessor.from_pretrained("./whisper-vi-finetuned")
    model = WhisperForConditionalGeneration.from_pretrained("./whisper-vi-finetuned")
    model.eval()
    return processor, model

processor, model = load_model()

def transcribe(audio, sampling_rate):
    if sampling_rate != 16000:
        import librosa
        audio = librosa.resample(audio, orig_sr=sampling_rate, target_sr=16000)
    input_features = processor.feature_extractor(audio, sampling_rate=16000, return_tensors="pt").input_features
    with torch.no_grad():
        predicted_ids = model.generate(input_features)
    transcription = processor.tokenizer.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    return transcription

st.title("🎤 Vietnamese ASR (Whisper)")
st.write("Upload an audio file (16kHz WAV) and get the Vietnamese transcription, or record audio below.")

# Audio Recorder Section
st.header("Audio Recorder")
audio = audiorecorder("Click to record", "Click to stop recording")

if len(audio) > 0:
    st.audio(audio.export().read())
    audio.export("audio.wav", format="wav")
    st.write(f"Frame rate: {audio.frame_rate}, Frame width: {audio.frame_width}, Duration: {audio.duration_seconds} seconds")
    if st.button("Transcribe Recorded Audio"):
        with st.spinner("Transcribing..."):
            audio_data = np.array(audio.get_array_of_samples()).astype(np.float32) / (2**15)
            sr = audio.frame_rate
            text = transcribe(audio_data, sr)
        st.success("Transcription:")
        st.write(text)

# File uploader section
uploaded_file = st.file_uploader("Upload a WAV file (16kHz)", type=["wav"])
if uploaded_file is not None:
    audio, sr = sf.read(uploaded_file)
    st.audio(uploaded_file, format="audio/wav")
    if st.button("Transcribe Uploaded Audio"):
        with st.spinner("Transcribing..."):
            text = transcribe(audio, sr)
        st.success("Transcription:")
        st.write(text)
