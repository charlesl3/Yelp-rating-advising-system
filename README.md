# Sentiment Analysis with BiLSTM + GloVe  
A Streamlit-powered app that predicts Yelp review star ratings (1–5) using a Bidirectional LSTM model trained on 600k real reviews.  

## 🚀 Demo
- Model: BiLSTM with pre-trained GloVe embeddings  
- Interface: Streamlit web app  
- Frameworks: TensorFlow, Keras, NumPy, Pandas  
- Dataset: Yelp Open Dataset  

## 📂 Files
- app.py — Streamlit interface  
- tokenizer_v2.pkl — Tokenizer for preprocessing  
- sentiment_model_v2.1_BiLSTM_best.h5 — Trained model  
- requirements.txt — Package list for deployment  

## 💡 How to Run Locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## 🌐 Next Step
Will be deployed on Streamlit Cloud soon so anyone can test the review predictions interactively.

