#!/usr/bin/env python3
"""
Interactive URL Checker
Input any URL and see if it's phishing or legitimate
"""

import sys
import os
sys.path.insert(0, 'src')

from simple_cnn import SimpleCNNClassifier
import pickle

def print_banner():
    print("\n" + "="*70)
    print("🔍 PHISHING URL CHECKER - Interactive Mode")
    print("="*70)
    print("Enter URLs to check if they're phishing or legitimate")
    print("Type 'quit' or 'exit' to stop")
    print("="*70 + "\n")

def load_model():
    """Load the trained CNN model"""
    try:
        model = SimpleCNNClassifier()
        model.load_model('models/simple_cnn_model.pkl')
        print("✅ Model loaded successfully!\n")
        return model
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("\nPlease train the model first:")
        print("python3 main.py --mode train --dataset data/dataset.csv\n")
        sys.exit(1)

def check_url(model, url):
    """Check if a URL is phishing or legitimate"""
    try:
        # Get prediction
        prediction = model.predict([url])[0]
        probabilities = model.predict_proba([url])[0]
        
        # Format output
        print("\n" + "-"*70)
        print(f"🔗 URL: {url}")
        print("-"*70)
        
        if prediction == 1:
            print("⚠️  Prediction: PHISHING")
            print(f"🔴 Confidence: {probabilities[1]*100:.2f}%")
            print("\n⚠️  WARNING: This URL appears to be a PHISHING attempt!")
            print("   Do NOT enter personal information or passwords!")
        else:
            print("✅ Prediction: LEGITIMATE")
            print(f"🟢 Confidence: {probabilities[0]*100:.2f}%")
            print("\n✓ This URL appears to be legitimate")
        
        print("\n📊 Probabilities:")
        print(f"   Legitimate: {probabilities[0]*100:.2f}%")
        print(f"   Phishing:   {probabilities[1]*100:.2f}%")
        print("-"*70 + "\n")
        
    except Exception as e:
        print(f"\n❌ Error checking URL: {e}\n")

def main():
    print_banner()
    
    # Load model
    model = load_model()
    
    # Interactive loop
    while True:
        try:
            # Get URL from user
            url = input("Enter URL to check (or 'quit' to exit): ").strip()
            
            # Check for exit
            if url.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Thanks for using the Phishing URL Checker!")
                print("="*70 + "\n")
                break
            
            # Skip empty input
            if not url:
                continue
            
            # Check the URL
            check_url(model, url)
            
        except KeyboardInterrupt:
            print("\n\n👋 Thanks for using the Phishing URL Checker!")
            print("="*70 + "\n")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}\n")

if __name__ == "__main__":
    main()
