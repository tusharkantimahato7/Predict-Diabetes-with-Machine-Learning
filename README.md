# Diabetes ML Project

A simple machine learning project for predicting diabetes using Logistic Regression.

## Project Structure
- `data/` - Raw and processed datasets
- `src/` - Source code modules (data, features, models, evaluation, utils)
- `notebooks/` - Jupyter notebook for analysis
- `tests/` - Simple test scripts
- `main.py` - Main pipeline script

## Setup

1. Create virtual environment:
```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
   pip install -r requirements.txt
```

## Usage

Run the complete pipeline:
```bash
python main.py
```

Run tests:
```bash
python tests/test_pipeline.py
```

Open notebook:
```bash
jupyter notebook notebooks/diabetes_analysis.ipynb
```

## Model
- Algorithm: Logistic Regression
- Features: 8 (Pregnancies, Glucose, Blood Pressure, etc.)
- Target: Binary (Diabetes: Yes/No)

## Results
Model achieves ~70-80% accuracy on test set (depends on random split).
```
**What it does:** Project documentation with setup and usage instructions.

---

### `.gitignore`
```
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
venv/
env/
*.egg-info/
dist/
build/

# Jupyter
.ipynb_checkpoints/

# IDE
.vscode/settings.json
.idea/

# Data
*.csv.gz
*.pickle

# OS
.DS_Store
Thumbs.db