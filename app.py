import streamlit as st
import numpy as np
import pickle
import speech_recognition as sr
import tempfile
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from streamlit_mic_recorder import mic_recorder
from pydub import AudioSegment

# ===== Page Config =====
st.set_page_config(page_title="Interview Response Detector", page_icon="🎯", layout="wide")

# ===== Load Model and Tokenizer =====
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
        st.error("Error loading model or preprocessing files.")
        st.write(e)
        return None, None, None


model, tokenizer, max_len = load_artifacts()


# ===== Prediction Function =====
def predict_answer(text):

    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=max_len)

    prediction = model.predict(padded)

    pred_index = np.argmax(prediction)
    confidence = float(np.max(prediction))

    labels = ["Short Answer", "Medium Answer", "Long Answer"]

    return labels[pred_index], confidence


# ===== Speech to Text =====
def speech_to_text(audio_bytes):

    recognizer = sr.Recognizer()

    with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as temp_webm:
        temp_webm.write(audio_bytes)
        webm_path = temp_webm.name

    wav_path = webm_path.replace(".webm", ".wav")

    try:
        audio = AudioSegment.from_file(webm_path)
        audio.export(wav_path, format="wav")

        with sr.AudioFile(wav_path) as source:
            audio_data = recognizer.record(source)

        text = recognizer.recognize_google(audio_data)
        return text

    except:
        return None


# ===== Session State =====
if "speech_text" not in st.session_state:
    st.session_state.speech_text = ""


# ===== UI =====
st.title("🎯 Interview Response Detector")
st.write("Type or speak your interview answer.")

st.divider()

# ===== Input Layout =====
col1, col2 = st.columns([8,1])

with col1:
    user_input = st.text_input(
        "Your Answer",
        value=st.session_state.speech_text,
        label_visibility="collapsed"
    )

with col2:
    audio = mic_recorder(start_prompt="🎤", stop_prompt="⏹", just_once=True)


# ===== Handle Voice =====
if audio:

    detected_text = speech_to_text(audio["bytes"])

    if detected_text:
        st.session_state.speech_text = detected_text
        st.success("Voice detected")
        st.write("Speech Text:", detected_text)
    else:
        st.error("Speech could not be recognized")


# ===== Prediction =====
if st.button("🔍 Analyze Answer"):

    final_text = user_input.strip()

    if final_text == "":
        st.warning("⚠ Please type or record an answer")

    elif model is None:
        st.error("Model is not loaded. Please check deployment files.")

    else:

        with st.spinner("Analyzing response..."):

            label, confidence = predict_answer(final_text)

        st.subheader("📊 Prediction Result")

        col1, col2 = st.columns(2)

        col1.metric("Answer Type", label)
        col2.metric("Confidence", f"{confidence*100:.2f}%")

        st.progress(confidence)

        if label == "Short Answer":
            st.info("💡 Try giving a more detailed response.")

        elif label == "Medium Answer":
            st.success("👍 Good balance of clarity and detail.")

        else:
            st.success("🔥 Excellent answer!")