import streamlit as st
import numpy as np
import pickle
import speech_recognition as sr
import tempfile
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from streamlit_mic_recorder import mic_recorder
from pydub import AudioSegment

# ========================
# Page Config
# ========================
st.set_page_config(page_title="Interview Response Analyzer", page_icon="🎯")

st.title("🎯 Interview Response Analyzer")
st.write("Speak or type your interview answer to analyze it.")

# ========================
# Load Model
# ========================
@st.cache_resource
def load_artifacts():

    model = load_model("emotion_model.keras")

    with open("preprocessor.pkl", "rb") as f:
        data = pickle.load(f)

    tokenizer = data["tokenizer"]
    max_len = data["max_len"]

    return model, tokenizer, max_len


model, tokenizer, max_len = load_artifacts()

# ========================
# Prediction
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
# Speech → Text
# ========================
def speech_to_text(audio_bytes):

    recognizer = sr.Recognizer()

    with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as temp_audio:
        temp_audio.write(audio_bytes)
        webm_path = temp_audio.name

    wav_path = webm_path.replace(".webm", ".wav")

    try:

        # Convert WEBM → WAV
        audio = AudioSegment.from_file(webm_path)
        audio.export(wav_path, format="wav")

        with sr.AudioFile(wav_path) as source:

            recognizer.adjust_for_ambient_noise(source)
            audio_data = recognizer.record(source)

        text = recognizer.recognize_google(audio_data)

        return text

    except sr.UnknownValueError:
        st.warning("Speech not clear. Please speak again.")
        return None

    except sr.RequestError:
        st.error("Speech recognition service unavailable.")
        return None

    except Exception as e:
        st.error(f"Audio processing error: {e}")
        return None


# ========================
# Session State
# ========================
if "speech_text" not in st.session_state:
    st.session_state.speech_text = ""


# ========================
# Input UI
# ========================
col1, col2 = st.columns([8,1])

with col1:
    user_input = st.text_input(
        "Your Answer",
        value=st.session_state.speech_text,
        label_visibility="collapsed",
        placeholder="Type your interview answer..."
    )

with col2:
    audio = mic_recorder(start_prompt="🎤 Start", stop_prompt="⏹ Stop", just_once=False)


# ========================
# Handle Voice
# ========================
if audio:

    detected_text = speech_to_text(audio["bytes"])

    if detected_text:
        st.session_state.speech_text = detected_text
        st.success("Voice recognized!")
        st.write("Speech Text:", detected_text)
    else:
        st.warning("Speech could not be recognized.")


# ========================
# Analyze
# ========================
if st.button("🔍 Analyze Answer"):

    final_text = user_input.strip()

    if final_text == "":
        st.warning("Please type or record an answer.")

    else:

        label, confidence = predict_answer(final_text)

        st.subheader("Prediction Result")

        col1, col2 = st.columns(2)

        col1.metric("Answer Type", label)
        col2.metric("Confidence", f"{confidence*100:.2f}%")

        st.progress(confidence)

        if label == "Short Answer":
            st.info("Try giving a more detailed response.")

        elif label == "Medium Answer":
            st.success("Good balance of clarity and detail.")

        else:
            st.success("Excellent answer!")