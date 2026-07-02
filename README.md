
# Disaster Tweet Classification System
Production-ready LSTM-based NLP web application for real-time disaster tweet detection.

---

## Overview

This project is a deep learning-based Natural Language Processing (NLP) system that classifies tweets as:

- **Disaster-related**
- **Non-disaster-related**

The application uses a trained LSTM (Long Short-Term Memory) neural network and provides a clean web interface built with Streamlit for real-time inference.

---

## Dataset

The model was trained on a labeled disaster tweet dataset containing real-world Twitter messages.

### Dataset Characteristics

- Total Samples: ~7,600 tweets
- Classes:
  - 1 → Disaster
  - 0 → Not Disaster
- Data Type: Short informal social media text
- Balanced binary classification problem

### Preprocessing Steps

- Lowercasing text
- Removing URLs
- Removing mentions (@user)
- Removing special characters and punctuation
- Tokenization using Keras Tokenizer
- Sequence padding to fixed maximum length

The dataset was split into:

- 80% Training Data
- 20% Testing Data

---

## Model Architecture

Embedding Layer  
→ LSTM Layer  
→ Dense Layer  
→ Sigmoid Output  

### Configuration

- Architecture: Embedding → LSTM → Dense → Sigmoid
- Task Type: Binary Classification
- Loss Function: Binary Crossentropy
- Optimizer: Adam
- Evaluation Metric: Accuracy

---

## Model Performance

The model was evaluated on the unseen test dataset.

### Test Accuracy

**~80–85% accuracy** (depending on training run)

### Evaluation Metrics

Example classification report:

- Precision: Measures correctness of positive predictions
- Recall: Measures how many actual disasters were correctly identified
- F1-Score: Harmonic mean of precision and recall
- Confusion Matrix used for error analysis

Typical Performance:

- Precision: ~0.83
- Recall: ~0.81
- F1-Score: ~0.82

These metrics indicate strong performance for short, noisy social media text classification.

---

## Key Features

- LSTM-based deep learning model (TensorFlow/Keras)
- Real-time tweet classification
- Confidence score visualization
- Clean SaaS-style UI with dark/light mode
- Modular project structure
- Saved model, tokenizer, and configuration
- Ready for deployment

---

## System Architecture

User Input  
→ Text Cleaning  
→ Tokenization  
→ Padding  
→ LSTM Model  
→ Sigmoid Output  
→ UI Display with Confidence Score  

---

## Project Structure

```

disaster-tweet-classifier/
│
├── app.py
├── predict.py
├── requirements.txt
├── README.md
│
├── model/
│   ├── disaster_lstm_model.keras
│   ├── tokenizer.pkl
│   └── model_config.json
│
└── assets/
    └── logo.webp

```

---

## Installation Guide

### 1. Clone Repository

```

git clone https://github.com/Dharshinimk-521/disaster-tweet-classifier.git
cd disaster-tweet-classifier

```

### 2. Create Virtual Environment

Windows:
```

python -m venv venv
venv\Scripts\activate

```

Mac/Linux:
```

python3 -m venv venv
source venv/bin/activate

```

### 3. Install Dependencies

```

pip install -r requirements.txt

```

---

## Running the Application

```

streamlit run app.py

```

Access locally at:

```

[http://localhost:8501](http://localhost:8501)

```

---

## Example Predictions

Disaster:
- "Earthquake destroys buildings in downtown area"
- "Wildfire spreading rapidly due to strong winds"

Non-Disaster:
- "This exam was a disaster lol"
- "I love rainy days and coffee"

---

## Model Saving Workflow

After training:

```python
model.save("disaster_lstm_model.keras")

import pickle
with open("tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)

import json
with open("model_config.json", "w") as f:
    json.dump({"max_len": max_len}, f)
```

---

## Deployment Options

* Streamlit Cloud
* Render
* Railway
* Docker containerization
* AWS / GCP / Azure

---

## License

MIT License



