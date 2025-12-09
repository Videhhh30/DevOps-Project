#!/bin/bash
# Entrypoint script for Phishing URL Detection Docker container
# Automatically trains the model if it doesn't exist, then starts Streamlit

set -e

echo "=========================================="
echo "Phishing URL Detection System Startup"
echo "=========================================="

# Check if model exists
MODEL_PATH="/app/models/simple_cnn_model.pkl"
DATASET_PATH="/app/data/dataset.csv"

if [ ! -f "$MODEL_PATH" ]; then
    echo ""
    echo "[INFO] Model not found at $MODEL_PATH"
    echo "[INFO] Starting automatic model training..."
    echo ""
    
    # Check if dataset exists
    if [ ! -f "$DATASET_PATH" ]; then
        echo "[ERROR] Dataset not found at $DATASET_PATH"
        echo "[ERROR] Cannot train model without dataset."
        echo "[ERROR] Please ensure data/dataset.csv is present."
        exit 1
    fi
    
    # Run training
    python main.py --mode train --dataset "$DATASET_PATH"
    
    if [ -f "$MODEL_PATH" ]; then
        echo ""
        echo "[SUCCESS] Model trained successfully!"
        echo "[INFO] Model saved to $MODEL_PATH"
    else
        echo "[ERROR] Model training failed. Model file not created."
        exit 1
    fi
else
    echo "[INFO] Model found at $MODEL_PATH"
    echo "[INFO] Skipping training, starting Streamlit app..."
fi

echo ""
echo "=========================================="
echo "Starting Streamlit Application"
echo "=========================================="
echo ""

# Start Streamlit app
exec streamlit run streamlit_app.py \
    --server.port=8501 \
    --server.address=0.0.0.0 \
    --logger.level=info
