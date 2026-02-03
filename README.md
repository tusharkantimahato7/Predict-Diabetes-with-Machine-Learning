# 🩺 Diabetes Prediction ML Project

A clean, simple, and complete Machine Learning project for predicting diabetes using Logistic Regression. Built with Python and scikit-learn.

---

## 📋 Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Dataset](#dataset)
- [Model Details](#model-details)
- [Results](#results)
- [Testing](#testing)
- [Contributing](#contributing)

---

## 🎯 Overview

This project demonstrates a complete machine learning pipeline for diabetes prediction:
- Data loading and preprocessing
- Feature scaling with StandardScaler
- Model training with Logistic Regression
- Performance evaluation with metrics
- Modular, maintainable code structure

Perfect for academic projects, portfolio, or learning ML fundamentals.

---

## ✨ Features

- ✅ Clean, modular code structure
- ✅ Complete ML pipeline from data to prediction
- ✅ Jupyter notebook for interactive analysis
- ✅ VS Code configuration included
- ✅ Unit tests for core functionality
- ✅ Easy to understand and extend

---

## 📁 Project Structure
```
diabetes-ml-project/
│
├── .vscode/                    # VS Code configuration
│   ├── settings.json           # Editor settings
│   ├── launch.json             # Debug configuration
│   └── extensions.json         # Recommended extensions
│
├── data/
│   ├── raw/                    # Original dataset
│   │   └── diabetes.csv
│   └── processed/              # Cleaned dataset
│       └── cleaned_diabetes.csv
│
├── notebooks/
│   └── diabetes_analysis.ipynb # Jupyter notebook for analysis
│
├── src/                        # Source code
│   ├── __init__.py
│   │
│   ├── data/                   # Data handling
│   │   ├── __init__.py
│   │   └── data_loader.py
│   │
│   ├── features/               # Feature engineering
│   │   ├── __init__.py
│   │   └── preprocessing.py
│   │
│   ├── models/                 # ML models
│   │   ├── __init__.py
│   │   └── model.py
│   │
│   ├── evaluation/             # Model evaluation
│   │   ├── __init__.py
│   │   └── evaluation.py
│   │
│   └── utils/                  # Helper functions
│       ├── __init__.py
│       └── helpers.py
│
├── tests/
│   └── test_pipeline.py        # Unit tests
│
├── main.py                     # Main pipeline script
├── requirements.txt            # Python dependencies
├── README.md                   # This file
└── .gitignore                  # Git ignore rules
```

---

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager
- Git (optional)

### Step 1: Clone or Download
```bash
# Clone with git
git clone https://github.com/tusharkantimaahato7/diabetes-ml-project.git
cd diabetes-ml-project

# Or download and extract ZIP
```

### Step 2: Create Virtual Environment
```bash
# On macOS/Linux
python3 -m venv venv
source venv/bin/activate

# On Windows
python -m venv venv
venv\Scripts\activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 💻 Usage

### Run Complete Pipeline
```bash
python main.py
```

**Output:**
```
### Diabetes Prediction ML Pipeline


[1] Loading data...
Loaded 10 rows from data/raw/diabetes.csv
...
Accuracy: 0.67
```

### Run in VS Code
1. Open project: `code .`
2. Select Python interpreter: `Ctrl+Shift+P` → "Python: Select Interpreter" → Choose `venv`
3. Press `F5` to debug or run `main.py`

### Use Jupyter Notebook
```bash
jupyter notebook notebooks/diabetes_analysis.ipynb
```
Then run cells sequentially for interactive analysis.

---

## 📊 Dataset

**Source:** Sample diabetes dataset (Pima Indians Diabetes Database style)

**Features (8):**
- Pregnancies: Number of pregnancies
- Glucose: Plasma glucose concentration
- BloodPressure: Diastolic blood pressure (mm Hg)
- SkinThickness: Triceps skin fold thickness (mm)
- Insulin: 2-Hour serum insulin (mu U/ml)
- BMI: Body mass index (weight in kg/(height in m)^2)
- DiabetesPedigreeFunction: Diabetes pedigree function
- Age: Age in years

**Target:**
- Outcome: 0 (No diabetes) or 1 (Has diabetes)

**Sample Size:** 10 rows (small for demonstration; can be expanded)

---

## 🤖 Model Details

**Algorithm:** Logistic Regression
- Simple, interpretable binary classifier
- Max iterations: 200
- Random state: 42 (for reproducibility)

**Preprocessing:**
- Train-test split: 70/30
- Feature scaling: StandardScaler (zero mean, unit variance)

**Evaluation Metrics:**
- Accuracy
- Precision, Recall, F1-Score
- Confusion Matrix

---

## 📈 Results

**Expected Performance:**
- Accuracy: ~60-70% (on small sample)
- Note: Performance varies with random split

**Example Output:**
```
Accuracy: 0.67

Classification Report:
              precision    recall  f1-score   support

           0       0.50      0.50      0.50         2
           1       0.50      0.50      0.50         1

    accuracy                           0.67         3

Confusion Matrix:
[[1 1]
 [0 1]]
```

---

## 🧪 Testing

Run unit tests to verify pipeline:
```bash
python tests/test_pipeline.py
```

**Expected Output:**
```
Loaded 10 rows from data/raw/diabetes.csv
✓ Data loading test passed
Loaded 10 rows from data/raw/diabetes.csv
✓ Feature split test passed

All tests passed!
```

---

## 🛠️ Extending the Project

**Add more data:**
- Replace `data/raw/diabetes.csv` with a larger dataset
- Run `python main.py` again

**Try different models:**
- Edit `src/models/model.py`
- Replace `LogisticRegression` with `RandomForestClassifier`, `SVM`, etc.

**Add more features:**
- Edit `src/features/preprocessing.py`
- Add feature engineering functions

**Improve evaluation:**
- Edit `src/evaluation/evaluation.py`
- Add ROC curves, cross-validation, etc.

---

## 📝 Requirements
```
pandas==2.0.3
numpy==1.24.3
scikit-learn==1.3.0
matplotlib==3.7.2
seaborn==0.12.2
jupyter==1.0.0
```

---

## 🤝 Contributing

This is an academic/learning project. Feel free to:
1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

---

## 📄 License

This project is open source and available for educational purposes.

---

## 👤 Author

- Email:papaimahato022gmail.com
- GitHub: [@tusharkantimahato7](https://github.com/tusharkantimahato7)

---

## 🙏 Acknowledgments

- Dataset inspired by Pima Indians Diabetes Database
- Built with scikit-learn and pandas
- VS Code configuration for optimal development

---

## ❓ FAQ

**Q: Why is the accuracy low?**  
A: The sample dataset has only 10 rows for demonstration. Use a larger dataset for better results.

**Q: Can I use a different model?**  
A: Yes! Edit `src/models/model.py` and replace LogisticRegression with any sklearn classifier.

**Q: How do I add more data?**  
A: Add more rows to `data/raw/diabetes.csv` following the same format, then run the pipeline again.

**Q: The code won't run. What should I do?**  
A: Ensure you activated the virtual environment and installed requirements. Check Python version is 3.8+.

---

**Happy Learning! 🎓**
