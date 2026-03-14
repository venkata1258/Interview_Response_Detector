import streamlit as st
import numpy as np
import pickle
import speech_recognition as sr
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences

# ========================
# Page Config
# ========================
st.set_page_config(page_title="Interview Response Analyzer", page_icon="🎯")

st.title("🎯 Interview Response Analyzer")
st.write("Record your voice or type your answer to analyze the response quality.")

# ========================
# Load Model & Tokenizer
# ========================
@st.cache_resource
def load_artifacts():
    # Ensure these files are in your project folder
    model = load_model("emotion_model.keras")
    with open("preprocessor.pkl", "rb") as f:
        data = pickle.load(f)
    return model, data["tokenizer"], data["max_len"]

try:
    model, tokenizer, max_len = load_artifacts()
except Exception as e:
    st.error(f"Error loading models: {e}")
    st.stop()

# ========================
# Logic: Prediction
# ========================
def predict_answer(text):
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=max_len)
    prediction = model.predict(padded)
    
    index = np.argmax(prediction)
    confidence = float(np.max(prediction))
    labels = ["Short Answer", "Medium Answer", "Long Answer"]
    
    return labels[index], confidence

# ========================
# Logic: Speech Processing
# ========================
def process_audio(audio_file):
    recognizer = sr.Recognizer()
    try:
        with sr.AudioFile(audio_file) as source:
            audio_data = recognizer.record(source)
            # Using Google's free web API
            text = recognizer.recognize_google(audio_data)
            return text
    except sr.UnknownValueError:
        return "Error: Could not understand audio."
    except sr.RequestError:
        return "Error: Speech service is down."
    except Exception as e:
        return f"Error: {e}"

# ========================
# User Interface
# ========================

# 1. Audio Input (The FFmpeg-free way)
st.subheader("Step 1: Record or Type")
recorded_audio = st.audio_input("Record your interview answer")

# Initialize session state for the text
if "final_text" not in st.session_state:
    st.session_state.final_text = ""

# If audio is recorded, process it immediately
if recorded_audio:
    with st.spinner("Transcribing..."):
        transcript = process_audio(recorded_audio)
        if "Error" not in transcript:
            st.session_state.final_text = transcript
            st.success("Transcription complete!")
        else:
            st.error(transcript)

# 2. Text Input (Allows manual editing of transcription)
user_text = st.text_area(
    "Edit or type your response here:", 
    value=st.session_state.final_text,
    height=150
)

# 3. Analyze Button
if st.button("🔍 Analyze Answer"):
    if not user_text.strip():
        st.warning("Please provide an answer first.")
    else:
        label, confidence = predict_answer(user_text)

        st.divider()
        st.subheader("Analysis Results")
        
        c1, c2 = st.columns(2)
        c1.metric("Evaluation", label)
        c2.metric("Model Confidence", f"{confidence*100:.1f}%")
        
        st.progress(confidence)

        if label == "Short Answer":
            st.info("💡 Tip: Try to elaborate more using the STAR method (Situation, Task, Action, Result).")
        elif label == "Medium Answer":
            st.success("✅ Good balance! Your response is concise yet informative.")
        else:
            st.success("🌟 Excellent! You've provided a comprehensive and detailed answer.")
