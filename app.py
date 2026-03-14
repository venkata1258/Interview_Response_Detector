import streamlit as st
import numpy as np
import pickle
import speech_recognition as sr
import tempfile
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from streamlit_mic_recorder import mic_recorder
from pydub import AudioSegment

# ==============================
# Page Configuration
# ==============================
st.set_page_config(
    page_title="Interview Response Detector",
    page_icon="🎯",
    layout="wide"
)

st.title("🎯 AI Interview Response Analyzer")
st.write("Speak or type your interview answer and analyze its quality.")

# ==============================
# Load Model + Preprocessor
# ==============================
@st.cache_resource
def load_artifacts():
    try:
        model = load_model("emotion_model.keras")

        with open("preprocessor.pkl", "rb") as f:
            data = pickle.load(f)

        tokenizer = data["tokenizer"]
        max_len = data["max_len"]

        return model, tokenizer, max_len

    except Exception as e:
        st.error("⚠ Model loading failed")
        st.write(e)
        return None, None, None


model, tokenizer, max_len = load_artifacts()

# ==============================
# Prediction Function
# ==============================
def predict_answer(text):

    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=max_len)

    prediction = model.predict(padded)

    pred_index = np.argmax(prediction)
    confidence = float(np.max(prediction))

    labels = ["Short Answer", "Medium Answer", "Long Answer"]

    return labels[pred_index], confidence


# ==============================
# Speech → Text
# ==============================
def speech_to_text(audio_bytes):

    recognizer = sr.Recognizer()

    with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as temp_audio:
        temp_audio.write(audio_bytes)
        webm_path = temp_audio.name

    wav_path = webm_path.replace(".webm", ".wav")

    try:
        audio = AudioSegment.from_file(webm_path)
        audio.export(wav_path, format="wav")

        with sr.AudioFile(wav_path) as source:
            audio_data = recognizer.record(source)

        text = recognizer.recognize_google(audio_data)

        return text

    except sr.UnknownValueError:
        st.warning("⚠ Speech not clear. Try again.")
        return None

    except sr.RequestError:
        st.error("Speech recognition service unavailable.")
        return None

    except Exception as e:
        st.error(f"Audio processing error: {e}")
        return None


# ==============================
# Session State
# ==============================
if "speech_text" not in st.session_state:
    st.session_state.speech_text = ""

# ==============================
# Input UI
# ==============================
col1, col2 = st.columns([8,1])

with col1:
    user_input = st.text_input(
        "Type your answer",
        value=st.session_state.speech_text,
        label_visibility="collapsed",
        placeholder="Type your interview answer here..."
    )

with col2:
    audio = mic_recorder(start_prompt="🎤", stop_prompt="⏹", just_once=True)

# ==============================
# Handle Voice
# ==============================
if audio:

    st.info("Processing voice...")

    detected_text = speech_to_text(audio["bytes"])

    if detected_text:
        st.session_state.speech_text = detected_text
        st.success("Voice recognized successfully!")
        st.write("**Speech Text:**", detected_text)

# ==============================
# Analyze Button
# ==============================
if st.button("🔍 Analyze Answer"):

    final_text = user_input.strip()

    if final_text == "":
        st.warning("⚠ Please type or record an answer.")

    elif model is None:
        st.error("Model not loaded.")

    else:

        with st.spinner("Analyzing response..."):

            label, confidence = predict_answer(final_text)

        st.subheader("📊 Prediction Result")

        col1, col2 = st.columns(2)

        col1.metric("Answer Type", label)
        col2.metric("Confidence", f"{confidence*100:.2f}%")

        st.progress(confidence)

        # Feedback
        if label == "Short Answer":
            st.info("💡 Try explaining your answer with more details.")

        elif label == "Medium Answer":
            st.success("👍 Good answer length and clarity.")

        else:
            st.success("🔥 Excellent detailed response!")
