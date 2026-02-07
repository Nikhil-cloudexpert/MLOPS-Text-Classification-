# End-to-End MLOps Pipeline for SMS Spam Classification

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![MLflow](https://img.shields.io/badge/MLflow-Experiment%20Tracking-green)
![DVC](https://img.shields.io/badge/DVC-Pipeline%20Automation-orange)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML%20Framework-yellow)

## 📋 Project Overview

This project implements a **production-grade MLOps pipeline** for text classification using classical machine learning techniques. The pipeline demonstrates industry best practices for:

- ✅ **Automated ML Workflows** using DVC
- ✅ **Experiment Tracking** with MLflow
- ✅ **Version Control** for data, code, and models
- ✅ **Reproducibility** and parameterization
- ✅ **Modular Architecture** with clean separation of concerns

### Problem Statement

Binary classification of SMS messages as **spam** or **ham** (legitimate) using the UCI SMS Spam Collection dataset.

### Why This Matters

This pipeline showcases how to build maintainable, reproducible ML systems that can be easily modified, versioned, and deployed - critical skills for any MLOps engineer.

---

## 🛠️ Technologies & Tools

| Category | Tools |
|----------|-------|
| **Language** | Python 3.8+ |
| **ML Framework** | Scikit-learn |
| **Pipeline Automation** | DVC (Data Version Control) |
| **Experiment Tracking** | MLflow |
| **NLP** | NLTK, TF-IDF |
| **Version Control** | Git |
| **Configuration** | YAML |

---

## 📁 Project Structure
```
MLOps-Text-Classification/
│
├── data/
│   ├── raw/                      # Raw dataset (tracked by DVC)
│   │   └── data.csv
│   ├── processed/                # Processed data (tracked by DVC)
│   │   ├── processed.csv
│   │   ├── X_train.pkl
│   │   ├── X_test.pkl
│   │   ├── y_train.pkl
│   │   └── y_test.pkl
│
├── models/                       # Trained models (tracked by DVC)
│   ├── model.pkl
│   └── vectorizer.pkl
│
├── src/                          # Source code (modular pipeline stages)
│   ├── data_loader.py           # Stage 1: Data loading
│   ├── preprocess.py            # Stage 2: Text preprocessing
│   ├── features.py              # Stage 3: Feature engineering
│   ├── train.py                 # Stage 4: Model training
│   └── evaluate.py              # Stage 5: Model evaluation
│
├── mlruns/                       # MLflow tracking data
├── params.yaml                   # Centralized configuration
├── dvc.yaml                      # DVC pipeline definition
├── dvc.lock                      # DVC pipeline lock file
├── requirements.txt              # Python dependencies
├── metrics.json                  # Evaluation metrics
├── README.md                     # Project documentation
└── .gitignore                    # Git ignore rules
```

---

## 🚀 Setup Instructions

### Prerequisites

- Python 3.8 or higher
- Git
- pip

### 1. Clone the Repository
```bash
git clone <repository-url>
cd MLOps-Text-Classification
```

### 2. Create Virtual Environment
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Linux/Mac:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Initialize DVC
```bash
dvc init
```

---

## 📊 Pipeline Execution

### Run Complete Pipeline

Execute the entire ML pipeline with a single command:
```bash
dvc repro
```

This will automatically:
1. Download the dataset
2. Preprocess text data
3. Engineer features (TF-IDF)
4. Train the model
5. Evaluate performance

### Run Individual Stages

You can also run specific stages:
```bash
# Run only data loading
dvc repro data_loader

# Run up to preprocessing
dvc repro preprocess

# Run training only (after dependencies are met)
dvc repro train
```

### Check Pipeline Status
```bash
# View pipeline DAG
dvc dag

# Check pipeline status
dvc status
```

---

## 🔧 Configuration

All pipeline parameters are centralized in `params.yaml`. Modify this file to experiment with different configurations:

### Key Parameters
```yaml
# Feature Engineering
features:
  max_features: 3000        # Maximum TF-IDF features
  ngram_range: [1, 2]       # Unigrams and bigrams
  
# Model Selection
model:
  algorithm: logistic_regression  # or 'naive_bayes'
  
# Train-Test Split
features:
  train_test_split:
    test_size: 0.2
    random_state: 42
```

After modifying `params.yaml`, simply run:
```bash
dvc repro
```

DVC will automatically detect changes and re-run only the affected stages.

---

## 📈 Experiment Tracking with MLflow

### View Experiments

Start the MLflow UI:
```bash
mlflow ui
```

Then open your browser to: `http://localhost:5000`

### What's Tracked

- **Parameters**: Model hyperparameters, feature settings
- **Metrics**: Accuracy, Precision, Recall, F1-Score (train & test)
- **Artifacts**: Trained models, vectorizers
- **Metadata**: Run timestamps, git commit hash

### Compare Experiments

1. Navigate to the MLflow UI
2. Select multiple runs
3. Click "Compare" to view side-by-side metrics
4. Visualize performance differences

---

## 📊 Pipeline Stages Explained

### Stage 1: Data Loading (`data_loader.py`)
- Downloads SMS Spam Collection dataset
- Saves raw CSV to `data/raw/data.csv`
- No preprocessing applied

### Stage 2: Preprocessing (`preprocess.py`)
- Text cleaning (lowercase, punctuation removal)
- Stopword removal using NLTK
- Handles missing values
- Label encoding (spam=1, ham=0)
- Output: `data/processed/processed.csv`

### Stage 3: Feature Engineering (`features.py`)
- TF-IDF vectorization with configurable parameters
- Train-test split (80/20 stratified)
- Saves feature matrices and vectorizer
- Output: `X_train.pkl`, `X_test.pkl`, `y_train.pkl`, `y_test.pkl`, `vectorizer.pkl`

### Stage 4: Model Training (`train.py`)
- Trains Logistic Regression or Naive Bayes
- Logs parameters and metrics to MLflow
- Saves trained model
- Output: `models/model.pkl`

### Stage 5: Evaluation (`evaluate.py`)
- Evaluates on test set
- Computes comprehensive metrics
- Logs to MLflow and saves to `metrics.json`
- Outputs confusion matrix and classification report

---

## 🔄 Reproducibility

### Version Control Strategy

1. **Code**: Tracked by Git
2. **Data & Models**: Tracked by DVC
3. **Parameters**: Defined in `params.yaml`
4. **Experiments**: Logged in MLflow

### Reproduce Experiments

To reproduce exact results:
```bash
# Checkout specific git commit
git checkout <commit-hash>

# Pull corresponding data/models
dvc pull

# Reproduce pipeline
dvc repro
```

### Share Your Pipeline
```bash
# Add remote storage for DVC (e.g., AWS S3, Google Drive)
dvc remote add -d myremote s3://mybucket/path

# Push data and models
dvc push

# Others can now pull your data
dvc pull
```

---

## 📊 Expected Results

With default parameters, you should achieve approximately:

- **Test Accuracy**: ~96-98%
- **Precision**: ~96-99%
- **Recall**: ~88-92%
- **F1-Score**: ~92-95%

Results may vary slightly due to randomization in train-test split.

---

## 🎯 Key MLOps Principles Demonstrated

1. **Pipeline Automation**: One command (`dvc repro`) runs the entire workflow
2. **Parameterization**: All tunable values in `params.yaml`
3. **Modularity**: Each stage is an independent Python module
4. **Experiment Tracking**: MLflow logs all runs for comparison
5. **Reproducibility**: DVC ensures consistent results across environments
6. **Version Control**: Git + DVC track code, data, and models
7. **Dependency Management**: DVC automatically handles stage dependencies

---

## 🔍 Troubleshooting

### Common Issues

**Issue**: `ModuleNotFoundError: No module named 'nltk'`
```bash
pip install -r requirements.txt
```

**Issue**: NLTK data not found
```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
```

**Issue**: DVC pipeline fails
```bash
# Clean DVC cache and re-run
dvc clean -f
dvc repro
```

**Issue**: MLflow UI not starting
```bash
# Ensure you're in the project directory
cd MLOps-Text-Classification
mlflow ui --port 5000
```

---

## 📚 Further Enhancements

This pipeline can be extended with:

- [ ] Hyperparameter tuning (GridSearchCV)
- [ ] Cross-validation
- [ ] Advanced feature engineering (word embeddings)
- [ ] Model deployment (Flask API, Docker)
- [ ] CI/CD integration (GitHub Actions)
- [ ] Cloud deployment (AWS SageMaker, Azure ML)
- [ ] Monitoring and drift detection

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit changes (`git commit -am 'Add new feature'`)
4. Push to branch (`git push origin feature/improvement`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License.

---

## 👨‍💻 Author

**MLOps Engineer | NLP Practitioner**

For questions or feedback, please open an issue in the repository.

---

## 🙏 Acknowledgments

- **Dataset**: UCI Machine Learning Repository - SMS Spam Collection
- **Tools**: DVC, MLflow, Scikit-learn, NLTK
- **Inspiration**: MLOps best practices from academic and industry sources

---

## 📞 Support

If you encounter any issues or have questions:

1. Check the [Troubleshooting](#-troubleshooting) section
2. Review [DVC Documentation](https://dvc.org/doc)
3. Review [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
4. Open an issue in this repository

---

**Happy MLOps Engineering! 🚀**