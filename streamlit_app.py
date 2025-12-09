import streamlit as st
import pandas as pd
import numpy as np
import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from simple_cnn import SimpleCNNClassifier
from network_simulation import PhishingSpreadSimulator
from visualization import Visualizer
from heuristics import TyposquattingDetector

# Page configuration
st.set_page_config(
    page_title="Phishing URL Detector",
    page_icon="🛡️",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #424242;
        margin-bottom: 1rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #1E88E5;
        color: white;
    }
    .result-box {
        padding: 20px;
        border-radius: 10px;
        margin-top: 20px;
        text-align: center;
    }
    .safe {
        background-color: #E8F5E9;
        border: 2px solid #4CAF50;
        color: #2E7D32;
    }
    .phishing {
        background-color: #FFEBEE;
        border: 2px solid #F44336;
        color: #C62828;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    """Load the trained model"""
    try:
        model = SimpleCNNClassifier()
        # Check if model exists, if not, we might need to train or warn
        if os.path.exists('models/simple_cnn_model.pkl'):
            model.load_model('models/simple_cnn_model.pkl')
            return model
        else:
            return None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def main():
    # Header
    st.markdown('<div class="main-header">🛡️ Phishing URL Detection & Network Simulation</div>', unsafe_allow_html=True)
    st.markdown("---")

    # Sidebar
    st.sidebar.title("Configuration")
    st.sidebar.info("This tool uses a CNN-inspired model to detect phishing URLs and simulates their spread through a social network.")
    
    # Simulation settings in sidebar
    st.sidebar.subheader("Simulation Settings")
    network_type = st.sidebar.selectbox(
        "Network Type",
        ['barabasi', 'random', 'watts_strogatz'],
        index=0
    )
    num_nodes = st.sidebar.slider("Number of Nodes", 50, 500, 150)
    base_infection_rate = st.sidebar.slider(
        "Base Infection Rate", 
        0.02, 0.10, 0.05, step=0.01,
        help="Probability of infection spread per contact. 0.05-0.07 produces realistic 10-30% infection rates."
    )

    # Main Content
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown('<div class="sub-header">Analyze URL</div>', unsafe_allow_html=True)
        url_input = st.text_input("Enter a URL to check:", placeholder="http://example.com")
        
        analyze_btn = st.button("Analyze URL", type="primary")

    if analyze_btn and url_input:
        # URL Normalization
        # Handle malformed protocols (e.g., https:/example.com)
        if url_input.startswith('http:/') and not url_input.startswith('http://'):
            url_input = url_input.replace('http:/', 'http://', 1)
        elif url_input.startswith('https:/') and not url_input.startswith('https://'):
            url_input = url_input.replace('https:/', 'https://', 1)
            
        if not url_input.startswith(('http://', 'https://')):
            url_input = 'http://' + url_input
            
        # 1. Heuristic Check (Typosquatting)
        detector = TyposquattingDetector()
        is_typo, target, dist = detector.check(url_input)
        
        if is_typo:
            st.markdown(f"""
            <div class="result-box phishing">
                <h2>⚠️ PHISHING DETECTED</h2>
                <p><strong>Reason:</strong> Potential Typosquatting Detected</p>
                <p>This URL looks suspiciously like <strong>{target}</strong>.</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Set values for simulation
            is_phishing = True
            probabilities = [0.0, 1.0]
            confidence = 1.0
            
            # Visualization Column (Mock for heuristic)
            with col2:
                st.markdown("### Prediction Confidence")
                prob_df = pd.DataFrame({
                    'Type': ['Legitimate', 'Phishing'],
                    'Probability': [0.0, 1.0]
                })
                st.bar_chart(prob_df.set_index('Type'))
                
        else:
            # 2. Model Prediction
            model = load_model()
            
            if model is None:
                st.error("⚠️ Model not found! Please run `python main.py --mode train --dataset data/dataset.csv` first.")
                is_phishing = False  # Default to safe if no model
            else:
                with st.spinner("Analyzing URL..."):
                    # Prediction
                    prediction = model.predict([url_input])[0]
                    probabilities = model.predict_proba([url_input])[0]
                    
                    is_phishing = prediction == 1
                    confidence = probabilities[1] if is_phishing else probabilities[0]
                    
                    # Display Result
                    if is_phishing:
                        st.markdown("""
                        <div class="result-box phishing">
                            <h2>⚠️ PHISHING DETECTED</h2>
                            <p>This URL appears to be malicious!</p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown("""
                        <div class="result-box safe">
                            <h2>✅ LEGITIMATE URL</h2>
                            <p>This URL appears to be safe.</p>
                        </div>
                        """, unsafe_allow_html=True)

                    # Visualization Column
                    with col2:
                        st.markdown("### Prediction Confidence")
                        # Create a simple bar chart for probabilities
                        prob_df = pd.DataFrame({
                            'Type': ['Legitimate', 'Phishing'],
                            'Probability': [probabilities[0], probabilities[1]]
                        })
                        st.bar_chart(prob_df.set_index('Type'))

        # If Phishing, run simulation
        if is_phishing:
            st.markdown("---")
            st.markdown('<div class="sub-header">📉 Phishing Spread Simulation</div>', unsafe_allow_html=True)
            st.info(f"Simulating spread in a {network_type} network with {num_nodes} nodes based on detection confidence.")
            
            with st.spinner("Running network simulation..."):
                # Use the base infection rate from the slider directly
                simulator = PhishingSpreadSimulator(num_nodes, network_type, base_infection_rate)
                results = simulator.run_simulation(max_steps=50)
                
                # Display metrics
                m1, m2, m3 = st.columns(3)
                m1.metric("Final Infected Nodes", f"{results['final_infected_count']}")
                m2.metric("Infection Percentage", f"{results['infection_percentage']:.1f}%")
                m3.metric("Steps to Stabilize", f"{results['steps_to_stabilize']}")
                
                # Show simulation summary
                st.info("💡 The simulation shows how quickly phishing URLs can spread through social networks, highlighting the importance of early detection and user awareness.")

if __name__ == "__main__":
    main()
