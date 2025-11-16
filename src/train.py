# src/train.py

import argparse
import yaml
import pathlib
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier


def train(config_path: str):
    """
    Real training function.
    Loads data, trains a TF-IDF + Logistic Regression model, and saves the artifacts.
    """
    # Read the configuration from the provided path
    with open(config_path) as config_file:
        config = yaml.safe_load(config_file)

    print("--- Real Training Started ---")

    # --- 1. Load Data ---
    # DVC automatically handles pulling the data if it's not present locally
    print("Step 1: Loading data from data/train.csv...")
    data_path = pathlib.Path("data/train.csv")
    df = pd.read_csv(data_path)

    # Define features (X) and labels (y)
    X = df['comment_text']
    label_columns = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    y = df[label_columns]
    print(f"Data loaded. Shape: {df.shape}")

    # --- 2. Train TF-IDF Vectorizer ---
    print("Step 2: Training TF-IDF Vectorizer...")
    # Using a simple TF-IDF vectorizer as a baseline
    vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
    X_tfidf = vectorizer.fit_transform(X)
    print("TF-IDF Vectorizer trained.")

    # --- 3. Train Model ---
    print("Step 3: Training OneVsRest Logistic Regression model...")
    # Using OneVsRestClassifier to handle the multi-label problem
    # 'class_weight="balanced"' helps with the imbalanced dataset
    base_model = LogisticRegression(solver='liblinear', class_weight='balanced')
    model = OneVsRestClassifier(base_model)
    model.fit(X_tfidf, y)
    print("Model trained.")

    # --- 4. Save Artifacts ---
    print("Step 4: Saving artifacts...")
    artifacts_dir = pathlib.Path("artifacts")
    artifacts_dir.mkdir(exist_ok=True)  # Create artifacts dir if it doesn't exist

    # Save the trained vectorizer and the model
    joblib.dump(vectorizer, artifacts_dir / "vectorizer.pkl")
    joblib.dump(model, artifacts_dir / "model.pkl")
    print("Artifacts saved to /artifacts folder.")

    print("--- Real Training Complete ---")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Real training script for toxic comment classification.")
    parser.add_argument('--config-path', type=str, required=True, help='Path to the model configuration file.')
    args = parser.parse_args()

    train(config_path=args.config_path)