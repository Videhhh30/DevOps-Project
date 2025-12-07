# Contributing to DEVOPS-PROJECT

Thank you for your interest in contributing to the Phishing URL Detection & Network Simulation System! 🎉

## 🚀 Getting Started

### Prerequisites
- Python 3.8 or higher
- Git installed on your machine
- GitHub account

### Setup Development Environment

1. **Clone the repository**:
   ```bash
   git clone https://github.com/YOUR_USERNAME/DEVOPS-PROJECT.git
   cd DEVOPS-PROJECT
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify installation**:
   ```bash
   # Run tests
   python -m pytest tests/
   
   # Start the web app
   streamlit run streamlit_app.py
   ```

---

## 🌿 Development Workflow

### 1. Create a Feature Branch

Always create a new branch for your work:

```bash
# Update main branch
git checkout main
git pull origin main

# Create and switch to a new branch
git checkout -b feature/your-feature-name
```

**Branch naming conventions**:
- `feature/feature-name` - New features
- `bugfix/bug-description` - Bug fixes
- `docs/topic` - Documentation updates
- `test/test-name` - Test additions
- `refactor/component` - Code refactoring

### 2. Make Your Changes

- Write clean, readable code
- Follow existing code style
- Add comments for complex logic
- Update documentation if needed

### 3. Test Your Changes

```bash
# Run all tests
python -m pytest tests/

# Run specific test file
python test_infection_rates.py

# Test the web app
streamlit run streamlit_app.py
```

### 4. Commit Your Changes

```bash
# Stage your changes
git add .

# Commit with a descriptive message
git commit -m "Add: Brief description of your changes"
```

**Commit message format**:
```
Type: Brief description (50 chars or less)

Detailed explanation if needed (wrap at 72 chars)

Examples:
- Add: New visualization for infection timeline
- Fix: Correct infection rate calculation in simulator
- Update: Improve README installation instructions
- Refactor: Simplify CNN model architecture
- Test: Add unit tests for URL feature extraction
- Docs: Add API documentation for network simulation
```

### 5. Push to GitHub

```bash
git push -u origin feature/your-feature-name
```

### 6. Create a Pull Request

1. Go to the repository on GitHub
2. Click **"Compare & pull request"**
3. Fill in the PR template:
   - **Title**: Clear, descriptive title
   - **Description**: What changes were made and why
   - **Related Issues**: Link any related issues
4. Request review from team members
5. Click **"Create pull request"**

---

## 📋 Pull Request Guidelines

### PR Checklist

Before submitting a PR, ensure:

- [ ] Code follows the project's style guidelines
- [ ] All tests pass (`python -m pytest tests/`)
- [ ] New tests added for new features
- [ ] Documentation updated (if applicable)
- [ ] Commit messages are clear and descriptive
- [ ] No merge conflicts with main branch
- [ ] Code has been reviewed locally

### PR Template

```markdown
## Description
Brief description of what this PR does

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Performance improvement
- [ ] Code refactoring

## Changes Made
- Change 1
- Change 2
- Change 3

## Testing
Describe how you tested these changes

## Screenshots (if applicable)
Add screenshots for UI changes

## Related Issues
Closes #issue_number
```

---

## 🎨 Code Style Guidelines

### Python Style

Follow PEP 8 guidelines:

```python
# Good
def calculate_infection_rate(num_infected, total_nodes):
    """
    Calculate the infection percentage.
    
    Args:
        num_infected (int): Number of infected nodes
        total_nodes (int): Total number of nodes
        
    Returns:
        float: Infection percentage
    """
    if total_nodes == 0:
        return 0.0
    return (num_infected / total_nodes) * 100

# Use descriptive variable names
infection_rate = 0.05
network_type = 'barabasi'

# Add docstrings to functions and classes
# Use type hints where appropriate
# Keep functions focused and small
```

### File Organization

```python
# 1. Standard library imports
import os
import sys
from typing import List, Dict

# 2. Third-party imports
import numpy as np
import pandas as pd
import networkx as nx

# 3. Local imports
from src.simple_cnn import SimpleCNNClassifier
from src.network_simulation import PhishingSpreadSimulator
```

---

## 🧪 Testing Guidelines

### Writing Tests

```python
# tests/test_network_simulation.py
import pytest
from src.network_simulation import PhishingSpreadSimulator

def test_simulator_initialization():
    """Test that simulator initializes correctly"""
    simulator = PhishingSpreadSimulator(
        num_nodes=100,
        network_type='barabasi',
        infection_rate=0.05
    )
    assert simulator.num_nodes == 100
    assert simulator.infection_rate == 0.05

def test_infection_spread():
    """Test that infection spreads correctly"""
    simulator = PhishingSpreadSimulator(100, 'barabasi', 0.05)
    results = simulator.run_simulation(max_steps=10)
    assert results['final_infected_count'] > 0
    assert results['infection_percentage'] > 0
```

### Running Tests

```bash
# Run all tests
python -m pytest tests/

# Run with verbose output
python -m pytest tests/ -v

# Run specific test file
python -m pytest tests/test_network_simulation.py

# Run specific test function
python -m pytest tests/test_network_simulation.py::test_infection_spread
```

---

## 📁 Project Structure

```
DEVOPS-PROJECT/
├── src/                          # Core source code
│   ├── simple_cnn.py            # CNN classifier
│   ├── network_simulation.py    # Phishing spread simulation
│   ├── heuristics.py            # Typosquatting detection
│   ├── url_features.py          # URL feature extraction
│   ├── visualization.py         # Visualization utilities
│   ├── dataset_handler.py       # Dataset management
│   ├── facebook_network_loader.py  # Social network loader
│   └── augmentation.py          # Data augmentation
│
├── data/                         # Data files
│   ├── dataset.csv              # Training dataset
│   ├── facebook_combined.txt    # Social network data
│   └── sample_urls.txt          # Sample URLs
│
├── models/                       # Trained models
│   └── *.pkl                    # Model files
│
├── docs/                         # Documentation
│   ├── project_explanation.md
│   ├── technical_details.md
│   └── ...
│
├── tests/                        # Test files
│   └── test_all.py
│
├── scripts/                      # Utility scripts
│   ├── check_url.py
│   └── test_comprehensive.py
│
├── streamlit_app.py             # Web UI application
├── main.py                      # Training/testing script
├── requirements.txt             # Python dependencies
├── README.md                    # Project overview
└── CONTRIBUTING.md              # This file
```

---

## 🐛 Reporting Bugs

### Before Reporting

1. Check if the bug has already been reported in Issues
2. Try to reproduce the bug with the latest code
3. Gather relevant information (error messages, screenshots, etc.)

### Bug Report Template

```markdown
## Bug Description
Clear description of the bug

## Steps to Reproduce
1. Step 1
2. Step 2
3. Step 3

## Expected Behavior
What you expected to happen

## Actual Behavior
What actually happened

## Environment
- OS: [e.g., macOS 13.0]
- Python version: [e.g., 3.9.7]
- Browser: [e.g., Chrome 120]

## Screenshots
Add screenshots if applicable

## Additional Context
Any other relevant information
```

---

## 💡 Suggesting Features

### Feature Request Template

```markdown
## Feature Description
Clear description of the proposed feature

## Problem It Solves
What problem does this feature address?

## Proposed Solution
How would you implement this feature?

## Alternatives Considered
What other solutions did you consider?

## Additional Context
Any mockups, examples, or references
```

---

## 🔍 Code Review Process

### For Reviewers

- Be constructive and respectful
- Explain the "why" behind suggestions
- Approve PRs that meet quality standards
- Request changes if needed

### For Contributors

- Be open to feedback
- Respond to review comments
- Make requested changes promptly
- Ask questions if unclear

---

## 🎯 Areas for Contribution

### High Priority
- [ ] Improve CNN model accuracy
- [ ] Add more network types for simulation
- [ ] Enhance UI/UX of Streamlit app
- [ ] Add comprehensive unit tests
- [ ] Optimize performance

### Medium Priority
- [ ] Add more visualization options
- [ ] Implement real-time URL checking API
- [ ] Add export functionality for results
- [ ] Create mobile-responsive UI
- [ ] Add internationalization (i18n)

### Documentation
- [ ] Improve API documentation
- [ ] Add video tutorials
- [ ] Create deployment guide
- [ ] Write architecture overview
- [ ] Add troubleshooting guide

---

## 📞 Getting Help

- **Questions**: Open a GitHub Discussion or Issue
- **Chat**: [Add your communication channel if any]
- **Email**: [Add contact email if applicable]

---

## 📜 License

By contributing, you agree that your contributions will be licensed under the same license as the project.

---

## 🙏 Thank You!

Your contributions make this project better for everyone. We appreciate your time and effort! 🎉
