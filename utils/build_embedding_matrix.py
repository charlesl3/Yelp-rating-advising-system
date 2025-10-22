"""
build_embedding_matrix.py
---------------------------
Builds an embedding matrix aligning tokenizer vocabulary
with pre-trained GloVe word vectors.

Usage:
    python build_embedding_matrix.py
"""

import numpy as np
import pickle

def build_embedding_matrix(tokenizer_path: str, glove_path: str, output_path: str,
                           max_vocab: int = 20000, embedding_dim: int = 100):
    """
    Create embedding matrix from tokenizer and GloVe file.

    Args:
        tokenizer_path (str): Path to saved tokenizer pickle file
        glove_path (str): Path to GloVe embeddings (e.g., glove.6B.100d.txt)
        output_path (str): Where to save embedding matrix (.npy)
        max_vocab (int): Maximum vocab size to use
        embedding_dim (int): Dimensionality of GloVe embeddings
    """
    print(f"📦 Loading tokenizer from {tokenizer_path} ...")
    with open(tokenizer_path, "rb") as f:
        tokenizer = pickle.load(f)

    print(f"📚 Loading GloVe embeddings from {glove_path} ...")
    embedding_index = {}
    with open(glove_path, encoding="utf8") as f:
        for line in f:
            values = line.split()
            word = values[0]
            coeffs = np.asarray(values[1:], dtype="float32")
            embedding_index[word] = coeffs

    print(f"✅ Loaded {len(embedding_index):,} GloVe word vectors")

    num_words = min(max_vocab, len(tokenizer.word_index) + 1)
    embedding_matrix = np.zeros((num_words, embedding_dim))

    found = 0
    for word, i in tokenizer.word_index.items():
        if i >= max_vocab:
            continue
        vector = embedding_index.get(word)
        if vector is not None:
            embedding_matrix[i] = vector
            found += 1

    np.save(output_path, embedding_matrix)
    print(f"✅ Embedding matrix saved to {output_path}")
    print(f"→ Shape: {embedding_matrix.shape}, Found embeddings for {found:,} words")

if __name__ == "__main__":
    build_embedding_matrix("tokenizer_v2.pkl", "glove.6B.100d.txt", "embedding_matrix.npy")
