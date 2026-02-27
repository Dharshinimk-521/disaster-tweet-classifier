import pandas as pd
import numpy as np
import tensorflow as tf
import re
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
df=pd.read_csv("train.csv")
df.head()
"""Layer roles:

Embedding → Meaning

LSTM → Context memory

Dropout → Regularization

Dense + ReLU → Learn complex patterns

Dense + Sigmoid → Output probability"""
import re

def clean_text(text):
    text = text.lower()                        # lowercase
    text = re.sub(r"http\S+", "", text)       # remove URLs
    text = re.sub(r"@\w+", "", text)          # remove mentions
    text = re.sub(r"[^a-z\s]", "", text)      # keep only letters
    text = re.sub(r"\s+", " ", text).strip()  # remove extra spaces
    return text

df["clean_text"] = df["text"].apply(clean_text)
#tokenization: convert the text to integers sequence
# Tokenization
max_words = 15000
max_len = 80 #if not sam elength then,TensorFlow cannot stack them into a matrix.

tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
tokenizer.fit_on_texts(df["clean_text"])
#padding : make the seq length same before feeding to lstm cuz it expects input as (batch_size, sequence_length)(no.of_rows_of_text,length =50=max_length)

sequences = tokenizer.texts_to_sequences(df["clean_text"])
X = pad_sequences(
    sequences,       # list of tokenized tweets (numbers)
    maxlen=max_len,  # enforce length = 50
    padding="post",  # add zeros at the END of shorter sequences
    truncating="post" # cut extra words from the END if too long
)
y = df["target"].values # Extract target labels (0 or 1)
X_train, X_test, y_train, y_test = train_test_split(
    X,                  # padded input sequences
    y,                  # labels
    test_size=0.2,      # 20% data used for testing
    random_state=42,# ensures same split every time (reproducibility)
    stratify=y
)
#build model- bidirectional lstm to undertsand the context better
model = Sequential([
    Embedding(max_words, 256, input_length=max_len),

    Bidirectional(LSTM(128, return_sequences=True)),
    Dropout(0.3),

    Bidirectional(LSTM(64)),
    Dropout(0.3),

    Dense(64, activation="relu"),
    Dropout(0.3),

    Dense(1, activation="sigmoid")
])

#compile model
model.compile(
    loss="binary_crossentropy",  # Correct loss for binary classification
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), # Adaptive learning rate optimizer
    metrics=["accuracy"]         # Track accuracy during training
)
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

# 🔹 Define EarlyStopping
early_stop = EarlyStopping(
    monitor="val_loss",
    patience=3,              # wait 3 epochs before stopping
    restore_best_weights=True,
    verbose=1
)

# 🔹 Compute class weights (handles imbalance)
class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(y_train),
    y=y_train
)

class_weights = dict(enumerate(class_weights))
print("Class Weights:", class_weights)

# 🔹 Train Model
history = model.fit(
    X_train,
    y_train,
    epochs=15,
    batch_size=32,
    validation_data=(X_test, y_test),
    callbacks=[early_stop],
    class_weight=class_weights,
    verbose=1
)
# Evaluate
y_pred = (model.predict(X_test) > 0.5).astype("int32").flatten()

print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))


# Prediction function
def predict_disaster(text):
    cleaned = clean_text(text)
    seq = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(seq, maxlen=max_len, padding="post")
    
    prediction = model.predict(padded, verbose=0)[0][0]
    print(f"Probability of Disaster: {prediction:.4f}")
    if prediction > 0.5:
        print("🔥 Disaster Tweet")
    else:
        print("✅ Not Disaster")
        
predict_disaster("There is a huge fire in downtown area!")
predict_disaster("Massive earthquake destroys several buildings")
predict_disaster("I love watching Netflix on weekends")
predict_disaster("Flood waters rising rapidly after heavy rain")
predict_disaster("My cat is sleeping peacefully")

# Save model
model.save("disaster_lstm_model.keras")

# Save tokenizer
import pickle
with open("tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)

# Save config
import json
with open("model_config.json", "w") as f:
    json.dump({"max_len": max_len}, f)

print("✅ Model, tokenizer, and config saved successfully!")