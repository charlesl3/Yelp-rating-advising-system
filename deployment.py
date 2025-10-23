import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import pandas as pd


# ==========================
# 1 Load model and tokenizer
# ==========================
@st.cache_resource
def load_sentiment_model():
    model = load_model("sentiment_model_v2.1_BiLSTM_best.h5")
    with open("tokenizer_v2.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    return model, tokenizer

model, tokenizer = load_sentiment_model()

# Use same MAX_LEN as training
MAX_LEN = 292 

# ==========================
# 2 Streamlit UI
# ==========================
st.title("🌟 Yelp Review Advisor (BiLSTM + GloVe)")
st.write("Type a Yelp-style review and see the suggested star rating (1–5).")

user_input = st.text_area("📝 Enter your review text:", height=150)

if st.button("Predict Sentiment"):
    if not user_input.strip():
        st.warning("Please type something before predicting.")
    else:
        seq = tokenizer.texts_to_sequences([user_input])
        padded = pad_sequences(seq, maxlen=MAX_LEN, padding='post', truncating='post')

        preds = model.predict(padded)
        pred_class = np.argmax(preds, axis=1)[0] + 1  # convert 0-based → 1–5 stars

        star_display = "⭐" * pred_class + "☆" * (5 - pred_class)

        st.markdown(f"### 🌠 Suggested Rating: {star_display}  ({pred_class} stars)")
        st.write("#### Probability Distribution:")
        probs_df = pd.DataFrame({
            "Star Rating (1–5)": [1, 2, 3, 4, 5],
            "Probability": preds[0]
        })
        st.bar_chart(probs_df.set_index("Star Rating (1–5)"))
