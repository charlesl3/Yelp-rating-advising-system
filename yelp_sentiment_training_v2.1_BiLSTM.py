# ===============================================
# SENTIMENT MODEL v2.1 — BiLSTM
# ===============================================

import numpy as np
import pickle
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dropout, Dense, Bidirectional
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# ==============================
# STEP 1: LOAD PRE-SAVED DATA
# ==============================

print("Loading saved preprocessed data and embeddings...")

# Load padded data
X_train_padded = np.load("X_train_padded.npy")
X_val_padded   = np.load("X_val_padded.npy")
X_test_padded  = np.load("X_test_padded.npy")
y_train        = np.load("y_train.npy")
y_val          = np.load("y_val.npy")
y_test         = np.load("y_test.npy")

# Load embedding matrix
embedding_matrix = np.load("embedding_matrix.npy")

# Load tokenizer
with open("tokenizer_v2.pkl", "rb") as f:
    tokenizer = pickle.load(f)

MAX_LEN = X_train_padded.shape[1]
vocab_size = embedding_matrix.shape[0]
embedding_dim = embedding_matrix.shape[1]

print(f"Loaded successfully: X_train {X_train_padded.shape}, embedding {embedding_matrix.shape}")


# ==============================
# STEP 2: BUILD BiLSTM MODEL
# ==============================

print("Building the 2-layer BiLSTM model ...")

model = Sequential([
    Embedding(
        input_dim=vocab_size,
        output_dim=embedding_dim,
        weights=[embedding_matrix],
        input_length=MAX_LEN,
        trainable=False
    ),
    Bidirectional(LSTM(128, return_sequences=True)),   # from both directions
    Dropout(0.5),
    Bidirectional(LSTM(64)),
    Dropout(0.5),
    Dense(5, activation='softmax')
])

model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=Adam(learning_rate=1e-3),
    metrics=['accuracy']
)

model.build(input_shape=(None, MAX_LEN))
model.summary()


# ==============================
# STEP 3: TRAIN & SAVE
# ==============================

checkpoint = ModelCheckpoint(
    "sentiment_model_v2.1_BiLSTM_best.h5",
    monitor="val_accuracy",
    save_best_only=True,
    mode="max",
    verbose=1
)
early_stop = EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True,
    verbose=1
)

print("Starting BiLSTM training ...\n")
history = model.fit(
    X_train_padded, y_train,
    validation_data=(X_val_padded, y_val),
    epochs=10,
    batch_size=256,
    callbacks=[checkpoint, early_stop],
    verbose=1
)

val_loss, val_acc = model.evaluate(X_val_padded, y_val, verbose=1)
print(f"Validation Accuracy: {val_acc:.4f} | Validation Loss: {val_loss:.4f}")

model.save("sentiment_model_v2.1_BiLSTM_final.h5")
print("Final trained model saved as sentiment_model_v2.1_BiLSTM_final.h5")


# ==============================
# STEP 4: VISUALIZE RESULTS
# ==============================

plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title("BiLSTM Training vs Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

print("All training artifacts saved successfully!")
