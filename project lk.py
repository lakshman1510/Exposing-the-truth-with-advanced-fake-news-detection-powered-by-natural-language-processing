"""
Advanced Fake News Detection using Natural Language Processing (NLP)  
Libraries used: Pandas, NumPy, Scikit-learn, NLTK, Spacy, TensorFlow/Keras

Set up virtual environment and install required libraries if needed:
pip install pandas numpy scikit-learn nltk spacy tensorflow

Note: Run "python -m spacy download en_core_web_sm" once before running this script.

Dataset:
Assumes a CSV file with columns ['text', 'label'], where 'label' = 0 for real, 1 for fake.
You can replace dataset path with your own dataset.

"""

import pandas as pd
import numpy as np
import re
import nltk
import spacy
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping

# Download required NLTK data
nltk.download('stopwords')

# Load spacy English model
nlp = spacy.load("en_core_web_sm")

# Preprocessing functions
stop_words = set(stopwords.words('english'))

def clean_text(text):
    # Lowercase
    text = text.lower()
    # Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)
    # Remove special characters and digits except spaces
    text = re.sub(r'[^a-z\s]', '', text)
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def spacy_lemmatizer(text):
    doc = nlp(text)
    lemmas = [token.lemma_ for token in doc if token.text not in stop_words and not token.is_punct]
    return " ".join(lemmas)

def preprocess_text(text):
    text = clean_text(text)
    text = spacy_lemmatizer(text)
    return text

def load_and_preprocess_data(csv_path):
    df = pd.read_csv(csv_path)
    # Basic check for required columns
    if 'text' not in df.columns or 'label' not in df.columns:
        raise ValueError("Dataset must have 'text' and 'label' columns.")
    df.dropna(subset=['text', 'label'], inplace=True)
    df['clean_text'] = df['text'].apply(preprocess_text)
    return df

def prepare_tokenizer(texts, max_words=10000):
    tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
    tokenizer.fit_on_texts(texts)
    return tokenizer

def encode_texts(tokenizer, texts, max_len=200):
    sequences = tokenizer.texts_to_sequences(texts)
    padded = pad_sequences(sequences, maxlen=max_len, padding='post', truncating='post')
    return padded

def build_model(vocab_size, embedding_dim=64, input_length=200):
    model = Sequential()
    model.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=input_length))
    model.add(Bidirectional(LSTM(64, return_sequences=True)))
    model.add(Dropout(0.5))
    model.add(Bidirectional(LSTM(32)))
    model.add(Dropout(0.5))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))  # Binary classification
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def main():
    # Configuration
    csv_path = "fake_news_dataset.csv"  # Replace with your dataset path
    max_words = 10000
    max_len = 200
    embedding_dim = 64
    batch_size = 64
    epochs = 10
    validation_split = 0.2

    print("Loading and preprocessing data...")
    df = load_and_preprocess_data(csv_path)

    print("Preparing tokenizer and encoding texts...")
    tokenizer = prepare_tokenizer(df['clean_text'], max_words=max_words)
    X = encode_texts(tokenizer, df['clean_text'], max_len=max_len)
    y = df['label'].values

    print("Splitting dataset into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=validation_split, stratify=y, random_state=42)

    print(f"Building model with vocab size {max_words}...")
    model = build_model(vocab_size=max_words, embedding_dim=embedding_dim, input_length=max_len)
    model.summary()

    print("Training model...")
    early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    history = model.fit(X_train, y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        validation_data=(X_test, y_test),
                        callbacks=[early_stop],
                        verbose=2)

    print("Evaluating model...")
    y_pred_prob = model.predict(X_test)
    y_pred = (y_pred_prob > 0.5).astype("int32")

    acc = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {acc:.4f}")

    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred, digits=4))

    # Save tokenizer and model if needed
    # tokenizer_json = tokenizer.to_json()
    # with open("tokenizer.json", "w") as f:
    #     f.write(tokenizer_json)
    # model.save("fake_news_detection_model.h5")

    # Utility: Predict on new texts
    def predict_fake_news(text_list):
        preprocessed = [preprocess_text(txt) for txt in text_list]
        seq = encode_texts(tokenizer, preprocessed, max_len=max_len)
        preds = model.predict(seq)
        return ["Fake" if p > 0.5 else "Real" for p in preds.flatten()]

    # Example prediction
    example_texts = [
        "Breaking news: Scientists discover cure for cancer!",
        "This celebrity was caught in a scandalous event that shocked everyone."
    ]
    print("\nExample Predictions:")
    preds = predict_fake_news(example_texts)
    for txt, pred in zip(example_texts, preds):
        print(f"Text: {txt}\nPrediction: {pred}\n")

if __name__ == "__main__":
    main()

