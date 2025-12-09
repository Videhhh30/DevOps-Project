"""
Enhanced CNN-inspired Model for Phishing URL Detection
Combines character n-grams with intelligent URL features
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import pickle
import sys
import os

# Import URL feature extractor
sys.path.insert(0, os.path.dirname(__file__))
from url_features import URLFeatureExtractor


class SimpleCNNClassifier:
    """
    Enhanced CNN-inspired classifier combining:
    1. Character n-grams (pattern detection)
    2. Intelligent URL features (domain, security, structure)
    """
    
    def __init__(self):
        """Initialize the enhanced classifier"""
        # Character n-gram vectorizer (simulates CNN filters)
        self.vectorizer = CountVectorizer(
            analyzer='char',
            ngram_range=(2, 4),      # Reduced range for efficiency
            max_features=3000,        # Reduced since we have URL features too
            lowercase=True,
            min_df=2
        )
        
        # URL feature extractor
        self.url_extractor = URLFeatureExtractor()
        
        # Feature scaler for URL features
        self.scaler = StandardScaler()
        
        # Enhanced neural network architecture
        self.model = MLPClassifier(
            hidden_layer_sizes=(256, 128, 64, 32),  # Deeper network
            activation='relu',
            solver='adam',
            batch_size=64,                          # Larger batches
            learning_rate_init=0.0005,              # Lower learning rate
            alpha=0.0001,                           # Much lower regularization
            max_iter=100,                           # More iterations
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=15,                    # More patience
            random_state=42,
            verbose=True
        )
        
    def extract_combined_features(self, urls):
        """
        Extract combined features: n-grams + URL features
        
        Args:
            urls: List of URL strings
            
        Returns:
            Combined feature matrix
        """
        # Extract character n-grams
        ngram_features = self.vectorizer.transform(urls).toarray()
        
        # Extract URL features
        url_features = np.array([self.url_extractor.extract_features(url) for url in urls])
        
        # Scale URL features
        url_features_scaled = self.scaler.transform(url_features)
        
        # Combine features
        combined = np.hstack([ngram_features, url_features_scaled])
        
        return combined
    
    def train(self, urls, labels):
        """
        Train the enhanced model
        
        Args:
            urls: List of URL strings
            labels: List of labels (0=legitimate, 1=phishing)
        """
        print("\n" + "="*70)
        print("ENHANCED CNN MODEL - FEATURE EXTRACTION")
        print("="*70)
        
        # Extract character n-grams
        print("\n1. Extracting character n-gram patterns...")
        ngram_features = self.vectorizer.fit_transform(urls).toarray()
        print(f"   N-gram features: {ngram_features.shape[1]}")
        
        # Extract URL features
        print("\n2. Extracting intelligent URL features...")
        url_features = np.array([self.url_extractor.extract_features(url) for url in urls])
        print(f"   URL features: {url_features.shape[1]}")
        
        # Fit scaler and transform
        url_features_scaled = self.scaler.fit_transform(url_features)
        
        # Combine features
        X = np.hstack([ngram_features, url_features_scaled])
        print(f"\n3. Combined feature matrix: {X.shape}")
        print(f"   Total features: {X.shape[1]}")
        
        # Train model
        print("\n" + "="*70)
        print("TRAINING ENHANCED NEURAL NETWORK")
        print("="*70)
        print("Architecture: Input -> 256 -> 128 -> 64 -> 32 -> Output")
        print("Features: Character n-grams + URL intelligence")
        print("Regularization: L2 (alpha=0.0001) + Early stopping")
        print("="*70)
        
        self.model.fit(X, labels)
        
        return self.model
    
    def predict(self, urls):
        """Predict phishing probability"""
        X = self.extract_combined_features(urls)
        return self.model.predict(X)
    
    def predict_proba(self, urls):
        """Predict probabilities"""
        X = self.extract_combined_features(urls)
        return self.model.predict_proba(X)
    
    def evaluate(self, urls, labels):
        """Evaluate model performance"""
        y_pred = self.predict(urls)
        
        # Calculate confusion matrix
        tn, fp, fn, tp = confusion_matrix(labels, y_pred).ravel()
        
        metrics = {
            'accuracy': accuracy_score(labels, y_pred),
            'precision': precision_score(labels, y_pred, zero_division=0),
            'recall': recall_score(labels, y_pred, zero_division=0),
            'f1_score': f1_score(labels, y_pred, zero_division=0),
            'true_negatives': tn,
            'false_positives': fp,
            'false_negatives': fn,
            'true_positives': tp,
            'false_positive_rate': fp / (fp + tn) if (fp + tn) > 0 else 0
        }
        
        return metrics
    
    def save_model(self, model_path='models/simple_cnn_model.pkl'):
        """Save model, vectorizer, scaler, and feature extractor"""
        with open(model_path, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'vectorizer': self.vectorizer,
                'scaler': self.scaler,
                'url_extractor': self.url_extractor
            }, f)
        print(f"\nModel saved to {model_path}")
    
    def load_model(self, model_path='models/simple_cnn_model.pkl'):
        """Load model, vectorizer, scaler, and feature extractor"""
        with open(model_path, 'rb') as f:
            data = pickle.load(f)
            self.model = data['model']
            self.vectorizer = data['vectorizer']
            self.scaler = data.get('scaler', StandardScaler())
            self.url_extractor = data.get('url_extractor', URLFeatureExtractor())
        print(f"Model loaded from {model_path}")


def add_diverse_legitimate_urls():
    """Add diverse legitimate URLs for better training"""
    legitimate_urls = [
        # Educational institutions
        "https://msrit.com", "https://msrit.com/results", "https://msrit.com/admissions",
        "https://mit.edu", "https://stanford.edu", "https://harvard.edu",
        "https://iitb.ac.in", "https://iisc.ac.in",
        
        # Government
        "https://gov.in", "https://uidai.gov.in", "https://india.gov.in",
        "https://usa.gov", "https://gov.uk",
        
        # Popular services
        "https://google.com", "https://google.com/search",
        "https://youtube.com", "https://youtube.com/watch",
        "https://github.com", "https://github.com/user/repo",
        "https://stackoverflow.com", "https://stackoverflow.com/questions",
        "https://wikipedia.org", "https://en.wikipedia.org/wiki/Main_Page",
        
        # E-commerce
        "https://amazon.com", "https://amazon.in/products",
        "https://flipkart.com", "https://flipkart.com/deals",
        "https://ebay.com",
        
        # News
        "https://bbc.com", "https://bbc.com/news",
        "https://cnn.com", "https://cnn.com/world",
        "https://nytimes.com",
        
        # Social media
        "https://facebook.com", "https://twitter.com", "https://linkedin.com",
        "https://instagram.com", "https://reddit.com",
        
        # Tech companies
        "https://microsoft.com", "https://apple.com", "https://google.com",
        "https://amazon.com", "https://netflix.com",
        
        # Various TLDs
        "https://example.org", "https://example.net", "https://example.edu",
        "https://example.co.uk", "https://example.co.in"
    ]
    
    return legitimate_urls


def train_simple_cnn(dataset_path='data/dataset.csv'):
    """
    Train enhanced CNN model with better generalization
    """
    print("="*70)
    print("TRAINING ENHANCED CNN MODEL FOR PHISHING DETECTION")
    print("="*70)
    print("\nEnhancements:")
    print("  [+] Character n-grams (pattern detection)")
    print("  [+] Intelligent URL features (35+ features)")
    print("  [+] Deeper neural network (4 hidden layers)")
    print("  [+] Better regularization")
    print("  [+] Diverse legitimate URLs")
    print("="*70)
    
    # Load dataset
    print(f"\nLoading dataset from {dataset_path}...")
    df = pd.read_csv(dataset_path)
    
    print(f"Original dataset: {len(df)} samples")
    print(f"  Legitimate: {(df['label'] == 0).sum()}")
    print(f"  Phishing:   {(df['label'] == 1).sum()}")
    
    # Add diverse legitimate URLs
    print("\nAdding diverse legitimate URLs...")
    diverse_legit = add_diverse_legitimate_urls()
    diverse_df = pd.DataFrame({
        'url': diverse_legit,
        'label': [0] * len(diverse_legit)
    })
    df = pd.concat([df, diverse_df], ignore_index=True)
    
    print(f"Enhanced dataset: {len(df)} samples")
    print(f"  Legitimate: {(df['label'] == 0).sum()}")
    print(f"  Phishing:   {(df['label'] == 1).sum()}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        df['url'].values,
        df['label'].values,
        test_size=0.2,
        random_state=42,
        stratify=df['label'].values
    )
    
    print(f"\nTraining samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    
    # Create and train model
    cnn = SimpleCNNClassifier()
    cnn.train(X_train, y_train)
    
    # Evaluate
    print("\n" + "="*70)
    print("MODEL EVALUATION")
    print("="*70)
    
    metrics = cnn.evaluate(X_test, y_test)
    
    print(f"\nTest Set Performance:")
    print(f"  Accuracy:  {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1-Score:  {metrics['f1_score']:.4f}")
    print(f"\nConfusion Matrix:")
    print(f"  True Negatives:  {metrics['true_negatives']:4d} (Correct Legitimate)")
    print(f"  False Positives: {metrics['false_positives']:4d} (Legit → Phishing)")
    print(f"  False Negatives: {metrics['false_negatives']:4d} (Phishing → Legit)")
    print(f"  True Positives:  {metrics['true_positives']:4d} (Correct Phishing)")
    print(f"\nFalse Positive Rate: {metrics['false_positive_rate']:.2%}")
    
    # Test on specific URLs
    print("\n" + "="*70)
    print("TESTING ON SPECIFIC URLs")
    print("="*70)
    
    test_urls = [
        "https://msrit.com/results",
        "https://www.google.com",
        "https://github.com/user/repo",
        "http://paypal-secure-login.tk",
        "https://paypa1.com/verify",
        "http://g00gle.com"
    ]
    
    predictions = cnn.predict(test_urls)
    probabilities = cnn.predict_proba(test_urls)
    
    print("\nPredictions:")
    for url, pred, prob in zip(test_urls, predictions, probabilities):
        label = "PHISHING" if pred == 1 else "LEGITIMATE"
        confidence = prob[1] if pred == 1 else prob[0]
        print(f"  {label:12} ({confidence:6.2%}) | {url}")
    
    # Save model
    cnn.save_model()
    
    print("\n" + "="*70)
    print("✓ Enhanced CNN model training completed successfully!")
    print("="*70)
    
    return cnn, metrics


if __name__ == "__main__":
    # Train enhanced model
    cnn, metrics = train_simple_cnn('data/dataset.csv')
