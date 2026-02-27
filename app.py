import streamlit as st
import tensorflow as tf
import pickle
import json
import re
import os
from tensorflow.keras.preprocessing.sequence import pad_sequences

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# ---------- PAGE CONFIG ----------
st.set_page_config(
    page_title="Disaster Tweet Classifier",
    page_icon="assets/logo.webp",
    layout="centered"
)

# ---------- THEME TOGGLE ----------
theme = st.toggle("🌙 Dark Mode")

if theme:
    bg_color = "#0E1117"
    text_color = "white"
    card_color = "#161B22"
else:
    bg_color = "#FFFFFF"
    text_color = "#000000"
    card_color = "#F9F9F9"

st.markdown(f"""
<style>
body {{
    background-color: {bg_color};
    color: {text_color};
}}

.card {{
    background-color: {card_color};
    padding: 25px;
    border-radius: 15px;
    box-shadow: 0 8px 20px rgba(0,0,0,0.1);
    margin-top: 20px;
}}

.result {{
    padding: 20px;
    border-radius: 12px;
    font-size: 18px;
    font-weight: 600;
    text-align: center;
    animation: fadeIn 0.6s ease-in-out;
}}

@keyframes fadeIn {{
    from {{opacity: 0; transform: translateY(10px);}}
    to {{opacity: 1; transform: translateY(0);}}
}}

.stButton button {{
    border-radius: 10px;
    height: 3em;
    font-weight: 600;
}}
</style>
""", unsafe_allow_html=True)

# ---------- CLEAN FUNCTION ----------
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# ---------- LOAD MODEL ----------
@st.cache_resource
def load_assets():
    model = tf.keras.models.load_model("model/disaster_lstm_model.keras")

    with open("model/tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)

    with open("model/model_config.json", "r") as f:
        config = json.load(f)

    return model, tokenizer, config["max_len"]

model, tokenizer, max_len = load_assets()

# ---------- HEADER ----------

st.title("Disaster Tweet Classifier")
st.caption("AI-powered detection of real disaster events from social media")

st.markdown("---")

# ---------- INPUT SECTION ----------
st.markdown('<div class="card">', unsafe_allow_html=True)

st.subheader("Enter Tweet")
user_input = st.text_area(
    "",
    height=120,
    placeholder="Type a tweet here..."
)

predict_btn = st.button("🔍 Analyze Tweet", use_container_width=True)

st.markdown('</div>', unsafe_allow_html=True)

# ---------- PREDICTION ----------
if predict_btn:

    if user_input.strip() == "":
        st.warning("Please enter a tweet.")
    else:
        cleaned = clean_text(user_input)

        seq = tokenizer.texts_to_sequences([cleaned])
        padded = pad_sequences(
            seq,
            maxlen=max_len,
            padding="post",
            truncating="post"
        )

        prediction = model.predict(padded, verbose=0)[0][0]
        confidence = float(prediction)

        st.markdown('<div class="card">', unsafe_allow_html=True)

        st.subheader("Prediction Result")

        if confidence > 0.5:
            st.markdown(
                f'<div class="result" style="background-color:#FF4B4B;color:white;">🔥 Disaster Tweet Detected</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f'<div class="result" style="background-color:#2ECC71;color:white;">✅ Not a Disaster Tweet</div>',
                unsafe_allow_html=True
            )

        st.write(f"Confidence Score: {confidence:.2f}")
        st.progress(confidence)

        st.markdown('</div>', unsafe_allow_html=True)