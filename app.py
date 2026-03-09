
import streamlit as st
import numpy as np
import pickle
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences

st.set_page_config(page_title="Interview Answer Detector", page_icon="🎯", layout="wide")

@st.cache_resource
def load_files():
    model = load_model("emotion_model.keras")

    with open("preprocessor.pkl", "rb") as f:
        data = pickle.load(f)

    tokenizer = data["tokenizer"]
    max_len = data["max_len"]

    return model, tokenizer, max_len


model, tokenizer, max_len = load_files()

def predict_answer(text):

    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=max_len)

    prediction = model.predict(padded)

    pred_index = np.argmax(prediction)
    confidence = float(np.max(prediction))

    labels = ["Short Answer", "Medium Answer", "Long Answer"]
    label = labels[pred_index]

    return label, confidence


st.title("🎯 Interview Answer Detector")

st.markdown("""
This **AI-powered application** analyzes interview responses and classifies them into:

- **Short Answer**
- **Medium Answer**
- **Long Answer**
""")

st.divider()

st.subheader("✍️ Enter Interview Answer")

user_input = st.text_area(
    "Type your answer here:",
    height=150
)

if st.button("🔍 Detect Answer Type"):

    if user_input.strip() == "":
        st.warning("⚠ Please enter some text.")

    else:
        label, confidence = predict_answer(user_input)

        st.divider()
        st.subheader("📊 Prediction Result")

        col1, col2 = st.columns(2)

        col1.metric("Answer Type", label)
        col2.metric("Confidence", f"{confidence*100:.2f}%")

        st.progress(confidence)

        st.success("Prediction completed!")