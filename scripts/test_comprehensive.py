#!/usr/bin/env python3
"""
Quick URL Checker - Test multiple URLs
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from simple_cnn import SimpleCNNClassifier

def test_urls():
    """Test model on various URLs"""
    print("="*70)
    print("COMPREHENSIVE URL TESTING")
    print("="*70)
    
    # Load model
    model = SimpleCNNClassifier()
    model.load_model('models/simple_cnn_model.pkl')
    
    # Test URLs
    test_cases = {
        "Legitimate Educational": [
            "https://msrit.com/results",
            "https://mit.edu",
            "https://stanford.edu/admissions",
            "https://iitb.ac.in"
        ],
        "Legitimate Popular Sites": [
            "https://google.com",
            "https://youtube.com/watch",
            "https://github.com/user/repo",
            "https://stackoverflow.com/questions",
            "https://amazon.in/products"
        ],
        "Legitimate Government": [
            "https://gov.in",
            "https://uidai.gov.in",
            "https://india.gov.in"
        ],
        "Phishing - Typosquatting": [
            "https://paypa1.com",
            "http://g00gle.com",
            "https://whatsaap.com",
            "http://amaz0n.com"
        ],
        "Phishing - Suspicious TLDs": [
            "http://paypal-secure-login.tk",
            "https://account-verify.ml",
            "http://secure-banking.ga"
        ],
        "Phishing - IP Addresses": [
            "http://192.168.1.1/admin",
            "http://10.0.0.1/login"
        ]
    }
    
    for category, urls in test_cases.items():
        print(f"\n{category}")
        print("-"*70)
        
        predictions = model.predict(urls)
        probabilities = model.predict_proba(urls)
        
        for url, pred, prob in zip(urls, predictions, probabilities):
            label = "PHISHING" if pred == 1 else "LEGITIMATE"
            confidence = prob[1] if pred == 1 else prob[0]
            
            # Color coding
            if label == "LEGITIMATE":
                symbol = "✓"
            else:
                symbol = "⚠"
            
            print(f"  {symbol} {label:12} ({confidence:6.2%}) | {url}")
    
    print("\n" + "="*70)
    print("Testing completed!")
    print("="*70)

if __name__ == "__main__":
    test_urls()
