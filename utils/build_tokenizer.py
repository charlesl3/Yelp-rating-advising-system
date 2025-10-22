"""
build_tokenizer.py
-------------------
Fits a Keras Tokenizer on Yelp text data and saves it for later reuse.

Usage:
    python build_tokenizer.py
"""

import pandas as pd
import pickle
from tensorflow.keras.preprocessing.text import Tokenizer

def create_and_save_tokenizer(csv_path: str, tokenizer_path: str, max_vocab: int = 20000):
    """
    Create a Keras Tokenizer and save it as a pickle file.

    Args:
        csv_path (str): Path to cleaned CSV containing 'text' column
        tokenizer_path (str): Output path for saved tokenizer
        max_vocab (int): Maximum vocabulary size to retain
    """
    print(f"📖 Reading data from {csv_path} ...")
    df = pd.read_csv(csv_path)

    tokenizer = Tokenizer(num_words=max_vocab, oov_token="<UNK>", lower=True)
    tokenizer.fit_on_texts(df["text"].tolist())

    with open(tokenizer_path, "wb") as f:
        pickle.dump(tokenizer, f)

    print(f"✅ Tokenizer fitted on {len(tokenizer.word_index):,} words")
    print(f"💾 Tokenizer saved to {tokenizer_path}")

if __name__ == "__main__":
    create_and_save_tokenizer("yelp_reviews_cleaned.csv", "tokenizer_v2.pkl")
