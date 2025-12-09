# Phishing URL Detection and Social Network Spread Simulation

![GHCR Publish](https://github.com/Videhhh30/DevOps-Project/actions/workflows/publish-ghcr.yml/badge.svg)
![GHCR Tags](https://github.com/Videhhh30/DevOps-Project/actions/workflows/publish-ghcr-tags.yml/badge.svg)
![Docker Hub (master)](https://github.com/Videhhh30/DevOps-Project/actions/workflows/publish-dockerhub-master.yml/badge.svg)
![Docker Hub (tags)](https://github.com/Videhhh30/DevOps-Project/actions/workflows/publish-dockerhub-tags.yml/badge.svg)

A comprehensive machine learning system that detects phishing URLs and simulates their spread through social networks using a Streamlit web interface.

## 🎯 Features

- **🛡️ ML-Based Phishing Detection**: CNN-inspired classifier with 99%+ accuracy
- **🔍 Heuristic Detection**: Typosquatting detection for common domains
- **📊 Network Simulation**: Model phishing URL propagation through real social networks
- **🌐 Interactive Web UI**: User-friendly Streamlit interface
- **📈 Visualizations**: Network graphs, spread timelines, and confidence metrics
- **🔬 Real Facebook Network**: Uses Stanford SNAP dataset with 4,039 users

## 📁 Project Structure

```
phishing-detection/
├── README.md                      # This file
├── requirements.txt               # Python dependencies
├── streamlit_app.py              # Web interface (main app)
├── main.py                        # CLI interface
├── demo.py                        # Demo script
│
├── src/                           # Source code modules
│   ├── simple_cnn.py             # CNN-inspired classifier
│   ├── network_simulation.py     # Social network simulation
│   ├── visualization.py          # Plotting and graphs
│   ├── heuristics.py             # Typosquatting detection
│   ├── dataset_handler.py        # Data loading and processing
│   ├── augmentation.py           # Dataset augmentation
│   └── facebook_network_loader.py # Real network loader
│
├── data/                          # Datasets
│   ├── dataset.csv               # Training dataset
│   ├── phishing_urls.csv         # Phishing URL samples
│   └── facebook_combined.txt     # Real Facebook network (4,039 users)
│
├── models/                        # Trained ML models
│   └── simple_cnn_model.pkl      # Trained classifier
│
├── outputs/                       # Generated visualizations
│
├── tests/                         # Test files
│   └── test_all.py               # Comprehensive tests
│
├── scripts/                       # Utility scripts
│   └── check_url.py              # Quick URL checker
│
└── docs/                          # Documentation
    ├── cnn_explanation.md        # Model architecture details
    ├── project_explanation.md    # Project overview for presentations
    ├── facebook_network.md       # Real network dataset info
    ├── technical_details.md      # Technical implementation
    ├── test_results.md           # Testing and validation
    ├── improvements.md           # Future enhancements
    └── troubleshooting.md        # Common issues and fixes
```

## 🚀 Quick Start

### Installation

1. **Clone or download the project**

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

### Running the Web App

**Launch the Streamlit interface:**
```bash
streamlit run streamlit_app.py
```

Then open your browser to `http://localhost:8501`

### Using the CLI

**Train the model:**
```bash
python main.py --mode train --dataset data/dataset.csv
```

**Check a URL:**
```bash
python main.py --mode predict --url "http://suspicious-site.com"
```

**Run network simulation:**
```bash
python main.py --mode simulate --infection-rate 0.3 --network-type barabasi
```

**Quick URL check script:**
```bash
python scripts/check_url.py
```

## 🐳 Docker Deployment

### Prerequisites
- [Docker](https://docs.docker.com/get-docker/) installed on your system
- [Docker Compose](https://docs.docker.com/compose/install/) (optional, but recommended)

### Quick Start with Docker

**1. Build and run the container using docker-compose (easiest):**
```bash
docker-compose up --build
```
Then open your browser to `http://localhost:8501`

**2. Or build and run manually:**
```bash
# Build the Docker image
docker build -t phishing-detector:latest .

# Run the Streamlit app
docker run -p 8501:8501 phishing-detector:latest

# Open http://localhost:8501 in your browser
```

**3. Run CLI commands in Docker:**
```bash
# Train the model
docker run phishing-detector:latest python main.py --mode train --dataset data/dataset.csv

# Predict a URL
docker run phishing-detector:latest python main.py --mode predict --url "http://example.com"

# Run network simulation
docker run phishing-detector:latest python main.py --mode simulate --infection-rate 0.3
```

### Docker Compose Services

The `docker-compose.yml` includes:
- **phishing-detector** (port 8501): Main Streamlit app with health checks
- **jupyter** (port 8888): Jupyter Lab for interactive exploration (optional)

Start both services:
```bash
docker-compose up
```

Access:
- Streamlit: http://localhost:8501
- Jupyter Lab: http://localhost:8888

### Volumes and Data Persistence

Data and models are mounted as volumes:
- `./data` → `/app/data` (dataset files)
- `./models` → `/app/models` (trained models)
- `./logs` → `/app/logs` (application logs)

This allows you to:
- Add new datasets without rebuilding the image
- Share trained models between containers
- Persist logs and outputs

### Troubleshooting Docker

**Port already in use?**
```bash
# Use a different port
docker run -p 9501:8501 phishing-detector:latest

# For docker-compose, edit docker-compose.yml ports
```

**Check container logs:**
```bash
docker logs phishing-url-detector
docker logs phishing-jupyter
```

**Stop containers:**
```bash
docker-compose down
```

**Remove everything (images, volumes, containers):**
```bash
docker-compose down -v
docker image rm phishing-detector:latest
```

## 📊 How It Works

### 1. Phishing Detection
- **Heuristic Check**: First checks for typosquatting (e.g., "paypa1.com" vs "paypal.com")
- **ML Classification**: CNN-inspired model analyzes URL features
- **Confidence Score**: Provides probability-based predictions

### 2. Network Simulation
When a phishing URL is detected:
- Simulates spread through social network (150-4,039 nodes)
- Uses SIR-like infection model
- Supports multiple network types:
  - **Barabási-Albert** (scale-free)
  - **Watts-Strogatz** (small-world)
  - **Erdős-Rényi** (random)
  - **Real Facebook Network** (4,039 users from Stanford SNAP)

### 3. Visualization
- **Prediction Confidence**: Bar chart showing probabilities
- **Spread Timeline**: Infection growth over time
- **Network Graph**: Visual representation of infected vs susceptible nodes

## 📈 Model Performance

- **Accuracy**: 99.3%
- **Precision**: 99.1%
- **Recall**: 99.5%
- **F1-Score**: 99.3%

## 🔬 Network Simulation Results

Using the real Facebook network (4,039 users):
- **Infection Rate 0.3**: Reaches 99% of network in ~10 steps
- **Critical Nodes**: User 107 has 1,045 connections (super-spreader)
- **Realistic Clustering**: 0.61 coefficient (friends-of-friends effect)

## 📚 Documentation

Detailed documentation is available in the `docs/` directory:
- **[CNN Explanation](docs/cnn_explanation.md)**: Model architecture and design
- **[Project Explanation](docs/project_explanation.md)**: For presentations and demos
- **[Facebook Network Guide](docs/facebook_network.md)**: Real network dataset details
- **[Technical Details](docs/technical_details.md)**: Implementation specifics
- **[Test Results](docs/test_results.md)**: Validation and testing
- **[Troubleshooting](docs/troubleshooting.md)**: Common issues

## 🧪 Testing

Run comprehensive tests:
```bash
python tests/test_all.py
```

## 📦 Requirements

- Python 3.8+
- pandas, numpy, scikit-learn
- networkx (graph operations)
- matplotlib, seaborn (visualization)
- streamlit (web interface)

## 📦 Pull Prebuilt Docker Image (GHCR)

I published a GitHub Actions workflow that automatically builds and pushes a prebuilt image to GitHub Container Registry (GHCR) on each push to `master`.

To pull and run the prebuilt image on any machine (no build required):

```bash
# Pull the image (replace OWNER if different)
docker pull ghcr.io/Videhhh30/devops-project:latest

# Run the container
docker run -d --name phishing-detector -p 8501:8501 ghcr.io/Videhhh30/devops-project:latest

# Open http://localhost:8501
```

If you prefer Docker Hub instead, I can add a workflow to push there as well — tell me and I will add it.

## 🎓 Academic Use

This project demonstrates:
- Machine learning for cybersecurity
- Social network analysis
- Real-world dataset application (Stanford SNAP)
- Interactive visualization
- Full-stack development (ML + Web UI)

## 📝 Dataset

The system uses:
- **Training Data**: CSV with URL and label columns
- **Facebook Network**: Stanford SNAP dataset (4,039 users, 88,234 connections)
- **Phishing Samples**: Curated phishing URL collection

## 🔧 Configuration

Adjust simulation parameters in the Streamlit sidebar:
- **Network Type**: Choose topology (barabasi, random, watts_strogatz)
- **Number of Nodes**: 50-500 users
- **Infection Rate**: 0.1-1.0 (probability of spread)

## 📄 License

Educational and academic use.

## 👨‍💻 Author

Created as an academic project for phishing detection and network security analysis.

---

**For detailed setup and usage instructions, see the [docs/](docs/) directory.**
