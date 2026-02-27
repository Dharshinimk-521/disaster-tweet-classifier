#lets u test from terminal without frontend-streamlit
import tensorflow as tf
import pickle
import json
import re
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ---------- CLEAN ----------
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# ---------- LOAD ----------
model = tf.keras.models.load_model("model/disaster_lstm_model.keras")

with open("model/tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

with open("model/model_config.json", "r") as f:
    config = json.load(f)

max_len = config["max_len"]

# ---------- PREDICT FUNCTION ----------
def predict_disaster(text):
    cleaned = clean_text(text)
    seq = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(
        seq,
        maxlen=max_len,
        padding="post",
        truncating="post"
    )

    prediction = model.predict(padded, verbose=0)[0][0]

    print(f"\nInput: {text}")
    print(f"Probability: {prediction:.4f}")

    if prediction > 0.5:
        print("🔥 Disaster Tweet")
    else:
        print("✅ Not Disaster")

# ---------- CLI LOOP ----------
while True:
    text = input("\nEnter tweet (or type 'exit'): ")

    if text.lower() == "exit":
        break

    predict_disaster(text)