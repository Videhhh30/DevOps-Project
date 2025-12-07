"""Small experiment script to improve phishing URL detection accuracy.
Tries TF-IDF char n-grams + URL features with several classifiers
and reports evaluation metrics.
"""
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pickle
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from url_features import URLFeatureExtractor


def load_dataset(path='data/dataset.csv'):
    df = pd.read_csv(path)
    return df


def add_diverse_legit(df):
    # reuse a short list of diverse legit URLs
    diverse = [
        "https://google.com", "https://github.com", "https://wikipedia.org",
        "https://microsoft.com", "https://apple.com", "https://amazon.com",
        "https://stanford.edu", "https://mit.edu", "https://example.org"
    ]
    diverse_df = pd.DataFrame({'url': diverse, 'label': [0]*len(diverse)})
    return pd.concat([df, diverse_df], ignore_index=True)


def extract_features(urls, vectorizer=None, fit_vectorizer=False):
    # Character-level TF-IDF
    if vectorizer is None:
        vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(3,5), max_features=5000, lowercase=True)

    if fit_vectorizer:
        X_text = vectorizer.fit_transform(urls)
    else:
        X_text = vectorizer.transform(urls)

    url_extractor = URLFeatureExtractor()
    X_url = np.array([url_extractor.extract_features(u) for u in urls])

    scaler = StandardScaler()
    X_url_scaled = scaler.fit_transform(X_url)

    # Combine sparse X_text with dense X_url_scaled
    X_text_dense = X_text.toarray()
    X = np.hstack([X_text_dense, X_url_scaled])

    return X, vectorizer, scaler


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    return {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1': f1_score(y_test, y_pred, zero_division=0)
    }


def main():
    print("Loading dataset...")
    df = load_dataset('data/dataset.csv')
    df = add_diverse_legit(df)

    urls = df['url'].values
    labels = df['label'].values

    X, vectorizer, scaler = extract_features(urls, fit_vectorizer=True)

    X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42, stratify=labels)

    print(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")

    # Try Logistic Regression (fast, regularized)
    print("\nTraining Logistic Regression...")
    lr = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
    lr.fit(X_train, y_train)
    lr_metrics = evaluate_model(lr, X_test, y_test)
    print("Logistic Regression:", lr_metrics)

    # Try Random Forest
    print("\nTraining Random Forest...")
    rf = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_metrics = evaluate_model(rf, X_test, y_test)
    print("Random Forest:", rf_metrics)

    # Save best model (choose by f1)
    best = ('lr', lr, lr_metrics)
    if rf_metrics['f1'] > lr_metrics['f1']:
        best = ('rf', rf, rf_metrics)

    name, model, metrics = best
    print(f"\nBest model: {name} with F1={metrics['f1']:.4f}")

    # Save model and vectorizer/scaler
    out_dir = 'models'
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f'improved_model_{name}.pkl'), 'wb') as f:
        pickle.dump({'model': model, 'vectorizer': vectorizer, 'scaler': scaler}, f)

    print("Saved improved model and artifacts to models/")
    print("Evaluation summary:")
    print("  LogisticRegression:", lr_metrics)
    print("  RandomForest:     ", rf_metrics)


if __name__ == '__main__':
    main()
