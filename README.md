# Sentiment BiLSTM Model (Yelp Reviews)

This project trains and tests a BiLSTM-based sentiment analysis model using the Yelp Academic Dataset.  
It includes preprocessing scripts, tokenizer creation, GloVe embeddings, model training, and deployment.

---

## Repository Structure

```
.
├── utils/                        # helper scripts and optional functions
├── deployment.py                 # script for loading model and making predictions
├── yelp_sentiment_training_v2.1_BiLSTM.py   # main training script
├── yelp_sentiment_model_v2.1_BiLSTM.h5      # trained model weights
├── tokenizer_v2.pkl              # saved tokenizer
├── embedding_matrix.npy          # embedding matrix used in training
├── requirements.txt              # package dependencies
└── README.md
```

---

## How to Run

### 1. Create environment
```bash
python3 -m venv .venv
source .venv/bin/activate     # on Mac/Linux
# .venv\Scripts\activate      # on Windows
```

### 2. Install dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Train the model
```bash
python yelp_sentiment_training_v2.1_BiLSTM.py
```

### 4. Run predictions
```bash
python deployment.py
```

---

## Notes
- The dataset used is `yelp_academic_dataset_review.json` from Yelp Open Dataset.
- Pretrained word vectors: `glove.6B.100d.txt` (GloVe 100-dimensional version).
- The model outputs predicted sentiment scores from 1 to 5 stars.

---

## Requirements

Key packages:
- TensorFlow / Keras
- NumPy / Pandas
- Streamlit (optional, for web app)
- GloVe embeddings (external download)

---

## Author
**Charles Liu**
