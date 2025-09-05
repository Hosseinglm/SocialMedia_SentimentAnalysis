# SocialMedia Sentiment Analysis - Comprehensive Documentation

## Table of Contents

1. [Project Overview](#project-overview)
2. [Architecture Documentation](#architecture-documentation)
3. [Installation Guide](#installation-guide)
4. [API Documentation](#api-documentation)
5. [Usage Examples](#usage-examples)
6. [Configuration Guide](#configuration-guide)
7. [Data Models](#data-models)
8. [Development Guide](#development-guide)
9. [Deployment Instructions](#deployment-instructions)
10. [Troubleshooting](#troubleshooting)

---

## 1. Project Overview

### Purpose
The SocialMedia Sentiment Analysis project is a comprehensive machine learning pipeline designed to analyze sentiment in Twitter data. It implements state-of-the-art natural language processing techniques combined with synthetic data augmentation to address class imbalance issues commonly found in sentiment analysis datasets.

### Objectives
- **Automated Sentiment Classification**: Classify Twitter posts into sentiment categories (Positive, Negative, Neutral, etc.)
- **Class Imbalance Mitigation**: Use synthetic data generation techniques (SMOTE, TVAE) to improve model performance on underrepresented classes
- **Model Comparison**: Evaluate multiple machine learning algorithms (Multinomial Naive Bayes, Random Forest, XGBoost) across different feature representations
- **Reproducible Pipeline**: Provide a complete, automated pipeline from raw data to trained models with comprehensive evaluation

### Key Features
- **Multi-Algorithm Support**: Implements three different classification algorithms with cross-validation
- **Advanced Feature Engineering**: Creates both Bag-of-Words and TF-IDF feature representations with n-gram support
- **Synthetic Data Generation**: Integrates SMOTE for oversampling and TVAE for tabular synthetic data generation
- **Comprehensive Evaluation**: Provides detailed performance metrics, confusion matrices, and error analysis
- **Automated Pipeline**: Single-command execution with quick test mode for rapid iteration
- **Extensible Architecture**: Modular design allows easy addition of new algorithms and features

### System Capabilities
- Process Twitter CSV datasets with automatic schema unification
- Clean and preprocess text data with advanced NLP techniques
- Generate synthetic samples to balance class distributions
- Train and evaluate multiple models with cross-validation
- Produce comprehensive reports with visualizations
- Save trained models with metadata for production use

---

## 2. Architecture Documentation

### System Design

The system follows a modular pipeline architecture with six main components:

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Data Loader   │───▶│ Text Processor  │───▶│Feature Engineer │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│Model Evaluator  │◄───│ Model Trainer   │◄───│Synthetic Generator│
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### Component Relationships

#### Core Pipeline Flow
1. **DataLoader** → Loads and unifies training/validation datasets
2. **TextPreprocessor** → Cleans and preprocesses text data
3. **FeatureEngineer** → Creates BoW and TF-IDF feature matrices
4. **ModelTrainer** → Trains multiple algorithms with cross-validation
5. **SyntheticDataGenerator** → Generates synthetic samples for class balancing
6. **ModelEvaluator** → Evaluates models and generates comprehensive reports

#### Data Flow Diagram

```
Raw CSV Files
     │
     ▼
┌─────────────────┐
│ Data Loading &  │ ── Raw Tweets CSV
│ Unification     │ ── Clean Tweets CSV
└─────────────────┘
     │
     ▼
┌─────────────────┐
│ Text Processing │ ── Feature Extraction
│ & Cleaning      │ ── Lemmatization
└─────────────────┘
     │
     ▼
┌─────────────────┐
│ Feature         │ ── BoW Features (NPZ)
│ Engineering     │ ── TF-IDF Features (NPZ)
└─────────────────┘
     │
     ▼
┌─────────────────┐
│ Model Training  │ ── Trained Models (PKL)
│ & Evaluation    │ ── Model Metadata (JSON)
└─────────────────┘
     │
     ▼
┌─────────────────┐
│ Synthetic Data  │ ── SMOTE Augmentation
│ Generation      │ ── TVAE Synthetic Data
└─────────────────┘
     │
     ▼
┌─────────────────┐
│ Final Evaluation│ ── Confusion Matrices
│ & Reporting     │ ── Performance Reports
└─────────────────┘
```

### Technology Stack
- **Core Language**: Python 3.12+
- **ML Framework**: scikit-learn, XGBoost
- **NLP Processing**: NLTK, WordCloud
- **Synthetic Data**: SDV (TVAE), imbalanced-learn (SMOTE)
- **Data Processing**: pandas, numpy
- **Visualization**: matplotlib, seaborn
- **Testing**: pytest
- **Development**: Jupyter notebooks

---

## 3. Installation Guide

### Prerequisites
- Python 3.12 or higher
- pip package manager
- Git (for cloning repository)
- Minimum 4GB RAM (8GB recommended for large datasets)
- 2GB free disk space for datasets and models

### Step-by-Step Installation

#### 1. Clone the Repository
```bash
git clone <repository-url>
cd SocialMedia_SentimentAnalysis
```

#### 2. Create Virtual Environment (Recommended)
```bash
# Using venv
python -m venv sentiment_env
source sentiment_env/bin/activate  # On Windows: sentiment_env\Scripts\activate

# Or using conda
conda create -n sentiment_env python=3.12
conda activate sentiment_env
```

#### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

#### 4. Download NLTK Data
```bash
python -c "import nltk; nltk.download('all')"
```

#### 5. Verify Installation
```bash
# Run tests to verify setup
pytest tests/ -v

# Quick pipeline test
python run_pipeline.py --quick-test
```

### Environment Configuration

#### Required Environment Variables
No environment variables are required for basic operation. All configuration is handled through command-line arguments and default parameters.

#### Optional Configuration
- `PYTHONPATH`: Add project root to Python path if running modules individually
- `OMP_NUM_THREADS`: Control parallel processing threads (default: auto-detect)

### Initial Setup

#### 1. Prepare Data Directory
```bash
mkdir -p datasets
# Place your training data files:
# - datasets/twitter_training.csv
# - datasets/twitter_validation.csv
```

#### 2. Create Output Directories
```bash
mkdir -p features models reports/eda reports/evaluation
```

#### 3. Test Installation
```bash
# Run quick test to verify everything works
python run_pipeline.py --quick-test --max-samples 100
```

### Troubleshooting Installation

#### Common Issues
1. **NLTK Download Fails**: Run `python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"`
2. **XGBoost Installation Issues**: Try `conda install xgboost` instead of pip
3. **Memory Issues**: Reduce `max_samples` in quick test mode
4. **SDV Installation**: Ensure you have `pip install sdv>=1.0.0`

---

## 4. API Documentation

### Core Modules

#### DataLoader Class

**Purpose**: Handles loading and unifying Twitter data from multiple CSV sources.

**Constructor**:
```python
DataLoader()
```

**Methods**:

##### `load_and_unify_data(train_path, val_path)`
Loads training and validation CSV files and creates unified schema.

**Parameters**:
- `train_path` (str): Path to training CSV file
- `val_path` (str): Path to validation CSV file

**Returns**:
- `pandas.DataFrame`: Unified dataset with standardized schema

**Example**:
```python
from src.data_loader import DataLoader

loader = DataLoader()
data = loader.load_and_unify_data(
    'datasets/twitter_training.csv',
    'datasets/twitter_validation.csv'
)
```

##### `save_raw_data(df, output_path)`
Saves unified raw data to CSV file.

**Parameters**:
- `df` (pandas.DataFrame): Data to save
- `output_path` (str): Output file path

#### TextPreprocessor Class

**Purpose**: Handles text cleaning, preprocessing, and feature extraction.

**Constructor**:
```python
TextPreprocessor()
```

**Methods**:

##### `clean_dataset(df)`
Cleans and preprocesses entire dataset.

**Parameters**:
- `df` (pandas.DataFrame): Raw dataset

**Returns**:
- `pandas.DataFrame`: Cleaned dataset with additional features

**Features Added**:
- `clean_tweet`: Processed and lemmatized text
- `tweet_len`: Original tweet character length
- `num_hashtags`: Number of hashtags
- `num_mentions`: Number of @mentions

##### `save_clean_data(df, output_path)`
Saves cleaned data to CSV file.

#### FeatureEngineer Class

**Purpose**: Creates feature matrices for machine learning models.

**Constructor**:
```python
FeatureEngineer()
```

**Methods**:

##### `create_train_test_split(df, test_size=0.2, random_state=42)`
Creates stratified train-test split.

**Parameters**:
- `df` (pandas.DataFrame): Cleaned dataset
- `test_size` (float): Proportion of test data (default: 0.2)
- `random_state` (int): Random seed (default: 42)

**Returns**:
- `tuple`: (X_train, X_test, y_train, y_test)

##### `create_bow_features(X_train, X_test, max_features=5000, min_df=2)`
Creates Bag-of-Words feature matrices.

**Parameters**:
- `X_train` (pandas.Series): Training text data
- `X_test` (pandas.Series): Test text data
- `max_features` (int): Maximum number of features (default: 5000)
- `min_df` (int): Minimum document frequency (default: 2)

**Returns**:
- `dict`: Dictionary with 'train', 'test', and 'vectorizer' keys

##### `create_tfidf_features(X_train, X_test, max_features=5000, min_df=2)`
Creates TF-IDF feature matrices with n-gram support.

**Parameters**:
- Same as `create_bow_features`

**Returns**:
- `dict`: Dictionary with 'train', 'test', and 'vectorizer' keys

#### ModelTrainer Class

**Purpose**: Trains and evaluates multiple machine learning models.

**Constructor**:
```python
ModelTrainer()
```

**Methods**:

##### `train_all_models(bow_features, tfidf_features, y_train, y_test)`
Trains all models on different feature sets with cross-validation.

**Parameters**:
- `bow_features` (dict): BoW feature dictionary
- `tfidf_features` (dict): TF-IDF feature dictionary
- `y_train` (pandas.Series): Training labels
- `y_test` (pandas.Series): Test labels

**Returns**:
- `dict`: Results dictionary with model performance metrics

##### `get_best_model(results)`
Identifies best performing model based on F1-macro score.

**Parameters**:
- `results` (dict): Results from `train_all_models`

**Returns**:
- `str`: Best model key identifier

##### `save_best_model(model, filepath)`
Saves trained model to pickle file.

#### SyntheticDataGenerator Class

**Purpose**: Generates synthetic data for class imbalance mitigation.

**Constructor**:
```python
SyntheticDataGenerator()
```

**Methods**:

##### `apply_smote(X, y, random_state=42)`
Applies SMOTE oversampling to balance classes.

**Parameters**:
- `X` (scipy.sparse matrix): Feature matrix
- `y` (pandas.Series): Target labels
- `random_state` (int): Random seed

**Returns**:
- `tuple`: (X_resampled, y_resampled)

##### `generate_tvae_data(df, n_samples=1000)`
Generates synthetic tabular data using TVAE.

**Parameters**:
- `df` (pandas.DataFrame): Original dataset
- `n_samples` (int): Number of synthetic samples to generate

**Returns**:
- `pandas.DataFrame`: Synthetic data

#### ModelEvaluator Class

**Purpose**: Comprehensive model evaluation and reporting.

**Constructor**:
```python
ModelEvaluator()
```

**Methods**:

##### `evaluate_model(model, X_test, y_test, model_name=None)`
Performs comprehensive model evaluation.

**Parameters**:
- `model`: Trained model object
- `X_test`: Test features
- `y_test`: Test labels
- `model_name` (str, optional): Model name for tracking

**Returns**:
- `dict`: Evaluation metrics including accuracy, F1-scores, precision, recall

##### `plot_confusion_matrix(y_true, y_pred, title="Confusion Matrix")`
Creates and saves confusion matrix visualization.

##### `generate_classification_report(y_true, y_pred)`
Generates detailed per-class performance metrics.

---

## 5. Usage Examples

### Basic Pipeline Execution

#### Full Pipeline
```bash
# Run complete pipeline with all features
python run_pipeline.py
```

#### Quick Test Mode
```bash
# Fast execution on subset of data (completes in <3 minutes)
python run_pipeline.py --quick-test

# Custom sample size
python run_pipeline.py --quick-test --max-samples 500
```

### Programmatic Usage

#### Basic Sentiment Analysis
```python
from src import DataLoader, TextPreprocessor, FeatureEngineer, ModelTrainer

# 1. Load and preprocess data
loader = DataLoader()
data = loader.load_and_unify_data('train.csv', 'val.csv')

preprocessor = TextPreprocessor()
clean_data = preprocessor.clean_dataset(data)

# 2. Create features
engineer = FeatureEngineer()
X_train, X_test, y_train, y_test = engineer.create_train_test_split(clean_data)
bow_features = engineer.create_bow_features(X_train, X_test)
tfidf_features = engineer.create_tfidf_features(X_train, X_test)

# 3. Train models
trainer = ModelTrainer()
results = trainer.train_all_models(bow_features, tfidf_features, y_train, y_test)

# 4. Get best model
best_model_key = trainer.get_best_model(results)
best_model = trainer.models[best_model_key]

print(f"Best model: {best_model_key}")
print(f"Accuracy: {results[best_model_key]['accuracy']:.4f}")
```

#### Custom Data Analysis
```python
import pandas as pd
from src import TextPreprocessor, FeatureEngineer

# Load your custom data
df = pd.read_csv('your_data.csv')

# Ensure required columns exist
required_columns = ['tweet_text', 'sentiment_label']
assert all(col in df.columns for col in required_columns)

# Process data
preprocessor = TextPreprocessor()
clean_df = preprocessor.clean_dataset(df)

# Create features for prediction
engineer = FeatureEngineer()
# Note: You'll need to fit vectorizers on training data first
```

#### Synthetic Data Generation
```python
from src import SyntheticDataGenerator

# Initialize generator
synthetic_gen = SyntheticDataGenerator()

# Apply SMOTE to feature matrix
X_balanced, y_balanced = synthetic_gen.apply_smote(X_train, y_train)

# Generate TVAE synthetic data
synthetic_data = synthetic_gen.generate_tvae_data(clean_data, n_samples=1000)
synthetic_gen.save_synthetic_data(synthetic_data, 'synthetic_data.csv')
```

### Jupyter Notebook Analysis

#### Interactive Exploration
```bash
# Start Jupyter notebook
jupyter notebook main.ipynb
```

The main notebook provides:
- Step-by-step pipeline execution
- Exploratory data analysis with visualizations
- Model comparison and evaluation
- Interactive parameter tuning

#### Custom Analysis Notebook
```python
# In Jupyter cell
%load_ext autoreload
%autoreload 2

import sys
sys.path.append('src')

from data_loader import DataLoader
from preprocessor import TextPreprocessor
# ... continue with analysis
```

### Command Line Interface

#### Available Arguments
```bash
python run_pipeline.py --help
```

**Options**:
- `--quick-test`: Run on subset of data for rapid testing
- `--max-samples N`: Maximum samples to use in quick test mode (default: 1000)

#### Examples
```bash
# Production run
python run_pipeline.py

# Development/testing
python run_pipeline.py --quick-test --max-samples 200

# Verify installation
python run_pipeline.py --quick-test --max-samples 50
```

---

## 6. Configuration Guide

### Model Parameters

#### Feature Engineering Configuration
```python
# BoW/TF-IDF Parameters
MAX_FEATURES = 5000        # Maximum vocabulary size
MIN_DF = 2                 # Minimum document frequency
MAX_DF = 0.95             # Maximum document frequency
NGRAM_RANGE = (1, 2)      # N-gram range for TF-IDF

# Train-test split
TEST_SIZE = 0.2           # Proportion of test data
RANDOM_STATE = 42         # Random seed for reproducibility
```

#### Model Training Configuration
```python
# Random Forest
N_ESTIMATORS = 100        # Number of trees
N_JOBS = -1              # Parallel processing (-1 = all cores)
CV_FOLDS = 5             # Cross-validation folds

# XGBoost
EVAL_METRIC = 'mlogloss'  # Evaluation metric
VERBOSITY = 0            # Logging level

# Multinomial Naive Bayes
# Uses default scikit-learn parameters
```

#### Synthetic Data Configuration
```python
# SMOTE Parameters
K_NEIGHBORS = 5          # Number of neighbors (auto-adjusted for small classes)
SMOTE_RANDOM_STATE = 42  # Random seed

# TVAE Parameters
TVAE_EPOCHS = 50         # Training epochs
TVAE_VERBOSE = False     # Disable training output
MAX_SYNTHETIC_SAMPLES = 2000  # Maximum synthetic samples per run
```

### File Paths and Directories

#### Input Data Paths
```python
TRAIN_DATA_PATH = 'datasets/twitter_training.csv'
VAL_DATA_PATH = 'datasets/twitter_validation.csv'
```

#### Output Directories
```python
DATASETS_DIR = 'datasets/'
FEATURES_DIR = 'features/'
MODELS_DIR = 'models/'
REPORTS_DIR = 'reports/'
EDA_DIR = 'reports/eda/'
EVALUATION_DIR = 'reports/evaluation/'
```

#### Generated Files
```python
# Data files
RAW_TWEETS = 'datasets/raw_tweets.csv'
CLEAN_TWEETS = 'datasets/clean_tweets.csv'
SYNTHETIC_TVAE = 'datasets/synthetic_tvae_data.csv'

# Feature files
BOW_FEATURES = 'features/bow.npz'
TFIDF_FEATURES = 'features/tfidf.npz'
BOW_VOCAB = 'features/bow_vocab.pkl'
TFIDF_VOCAB = 'features/tfidf_vocab.pkl'

# Model files
BEST_MODEL = 'models/best_model_{algorithm}_{features}.pkl'
MODEL_CARD = 'models/model_card.json'

# Report files
CONFUSION_MATRICES = 'reports/evaluation/confusion_matrices.png'
EDA_OVERVIEW = 'reports/eda/overview_analysis.png'
SENTIMENT_WORDCLOUDS = 'reports/eda/sentiment_wordclouds.png'
```

### Performance Tuning

#### Memory Optimization
```python
# Reduce memory usage
MAX_FEATURES = 2000       # Reduce vocabulary size
QUICK_TEST_SAMPLES = 500  # Use smaller sample for testing

# Disable parallel processing if memory constrained
N_JOBS = 1
```

#### Speed Optimization
```python
# Increase parallel processing
N_JOBS = -1               # Use all available cores
OMP_NUM_THREADS = 8       # Control OpenMP threads

# Skip time-intensive operations in quick test
SKIP_TVAE_IN_QUICK_TEST = True
REDUCE_CV_FOLDS = 3       # Reduce cross-validation folds
```

### Environment Variables

#### Optional Environment Configuration
```bash
# Control parallel processing
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

# Python path (if needed)
export PYTHONPATH="${PYTHONPATH}:/path/to/project"

# Disable warnings (optional)
export PYTHONWARNINGS="ignore"
```

### Customization Options

#### Adding New Models
To add a new model to the pipeline:

1. **Modify ModelTrainer**:
```python
# In src/model_trainer.py
model_configs = {
    'MultinomialNB': lambda: MultinomialNB(),
    'RandomForest': lambda: RandomForestClassifier(n_estimators=100, random_state=42),
    'XGBoost': lambda: xgb.XGBClassifier(random_state=42),
    'YourNewModel': lambda: YourModelClass(parameters)  # Add here
}
```

2. **Handle Special Cases**:
```python
# Add any special handling needed for your model
if model_name == 'YourNewModel':
    # Special preprocessing or prediction handling
    pass
```

#### Custom Feature Engineering
```python
# In src/feature_engineer.py
def create_custom_features(self, X_train, X_test):
    """Add your custom feature engineering here"""
    # Implement custom vectorization or feature extraction
    pass
```

#### Custom Evaluation Metrics
```python
# In src/evaluator.py
def custom_evaluation_metric(self, y_true, y_pred):
    """Add custom evaluation metrics"""
    # Implement domain-specific metrics
    pass
```

---

## 7. Data Models

### Input Data Schema

#### Required CSV Format
The pipeline expects CSV files with the following structure:

**Training/Validation Files** (`twitter_training.csv`, `twitter_validation.csv`):
```csv
tweet_id,brand,sentiment_label,tweet_text
1,Apple,Positive,"Love the new iPhone! Amazing camera quality."
2,Google,Negative,"Search results are getting worse lately."
3,Microsoft,Neutral,"Just updated Windows, seems okay."
```

**Column Descriptions**:
- `tweet_id` (string): Unique identifier for each tweet
- `brand` (string): Brand or topic category being discussed
- `sentiment_label` (string): Sentiment classification (Positive, Negative, Neutral, etc.)
- `tweet_text` (string): Raw tweet content with original formatting

### Processed Data Schema

#### Unified Raw Data (`raw_tweets.csv`)
After loading and unification:
```csv
tweet_id,author,date,tweet_text,sentiment_label,brand,source
1,Apple,2023-03-15,"Love the new iPhone!",Positive,Apple,train
2,Google,2023-04-22,"Search results are bad",Negative,Google,validation
```

**Additional Columns**:
- `author` (string): Author information (derived from brand)
- `date` (datetime): Synthetic date for temporal analysis
- `source` (string): Data source identifier ('train' or 'validation')

#### Clean Data Schema (`clean_tweets.csv`)
After text preprocessing:
```csv
tweet_id,author,date,tweet_text,sentiment_label,brand,source,clean_tweet_basic,clean_tweet,tweet_len,num_hashtags,num_mentions
1,Apple,2023-03-15,"Love the new iPhone!",Positive,Apple,train,"love new iphone","love new iphone",20,0,0
```

**Processing Features**:
- `clean_tweet_basic` (string): Basic cleaned text (lowercase, URLs removed, etc.)
- `clean_tweet` (string): Fully processed text (lemmatized, stopwords removed)
- `tweet_len` (int): Original tweet character length
- `num_hashtags` (int): Number of hashtags in original tweet
- `num_mentions` (int): Number of @mentions in original tweet

### Feature Matrices

#### Bag-of-Words Features (`bow.npz`)
- **Format**: Compressed NumPy sparse matrix
- **Shape**: (n_samples, max_features)
- **Type**: Binary features (0 or 1)
- **Vocabulary**: Saved separately in `bow_vocab.pkl`

#### TF-IDF Features (`tfidf.npz`)
- **Format**: Compressed NumPy sparse matrix
- **Shape**: (n_samples, max_features)
- **Type**: Float values (TF-IDF scores)
- **N-grams**: Unigrams and bigrams (1,2)
- **Vocabulary**: Saved separately in `tfidf_vocab.pkl`

### Model Artifacts

#### Trained Models (`best_model_*.pkl`)
- **Format**: Pickle serialized scikit-learn/XGBoost models
- **Naming**: `best_model_{Algorithm}_{FeatureType}.pkl`
- **Examples**:
  - `best_model_RandomForest_BoW.pkl`
  - `best_model_XGBoost_TF-IDF.pkl`

#### Model Metadata (`model_card.json`)
```json
{
  "model_name": "RandomForest",
  "features": "BoW",
  "version": "1.0",
  "created_date": "2023-12-01T10:30:00",
  "accuracy": 0.8542,
  "f1_macro": 0.8234,
  "model_key": "RandomForest_BoW",
  "cv_mean": 0.8456,
  "cv_std": 0.0123
}
```

### Synthetic Data Schema

#### TVAE Synthetic Data (`synthetic_tvae_data.csv`)
```csv
sentiment_label,tweet_len,num_hashtags,num_mentions,brand
Positive,45,1,0,Apple
Negative,32,0,2,Google
```

**Generated Features**:
- Maintains original categorical and numerical distributions
- Preserves correlations between features
- Quality validated using Kolmogorov-Smirnov tests

### Entity Relationships

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Raw Tweets    │────▶│  Clean Tweets   │────▶│ Feature Matrices│
│                 │     │                 │     │                 │
│ - tweet_id (PK) │     │ + clean_tweet   │     │ - BoW Features  │
│ - brand         │     │ + tweet_len     │     │ - TF-IDF        │
│ - sentiment     │     │ + num_hashtags  │     │                 │
│ - tweet_text    │     │ + num_mentions  │     │                 │
└─────────────────┘     └─────────────────┘     └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│ Synthetic Data  │     │ Trained Models  │     │ Evaluation      │
│                 │     │                 │     │ Reports         │
│ - TVAE Data     │     │ - Model Files   │     │ - Metrics       │
│ - SMOTE Data    │     │ - Metadata      │     │ - Visualizations│
└─────────────────┘     └─────────────────┘     └─────────────────┘
```

---

## 8. Development Guide

### Contributing Guidelines

#### Getting Started
1. **Fork the Repository**: Create your own fork of the project
2. **Clone Locally**: `git clone <your-fork-url>`
3. **Create Branch**: `git checkout -b feature/your-feature-name`
4. **Set Up Environment**: Follow installation guide
5. **Run Tests**: Ensure all tests pass before making changes

#### Development Workflow
1. **Create Feature Branch**: Always work on feature branches
2. **Write Tests**: Add tests for new functionality
3. **Follow Coding Standards**: Use consistent style and documentation
4. **Test Thoroughly**: Run full test suite and pipeline
5. **Submit Pull Request**: Include clear description of changes

### Coding Standards

#### Python Style Guide
- **PEP 8 Compliance**: Follow Python style guidelines
- **Line Length**: Maximum 100 characters
- **Imports**: Group imports (standard library, third-party, local)
- **Docstrings**: Use Google-style docstrings for all functions and classes

#### Example Function Documentation
```python
def create_bow_features(self, X_train, X_test, max_features=5000, min_df=2):
    """
    Create Bag-of-Words features from text data.

    Args:
        X_train (pandas.Series): Training text data
        X_test (pandas.Series): Test text data
        max_features (int): Maximum number of features to extract
        min_df (int): Minimum document frequency for feature inclusion

    Returns:
        dict: Dictionary containing:
            - 'train': Training feature matrix (scipy.sparse)
            - 'test': Test feature matrix (scipy.sparse)
            - 'vectorizer': Fitted CountVectorizer object

    Raises:
        ValueError: If input data is empty or invalid

    Example:
        >>> engineer = FeatureEngineer()
        >>> features = engineer.create_bow_features(X_train, X_test)
        >>> print(features['train'].shape)
        (1000, 5000)
    """
```

#### Class Documentation
```python
class ModelTrainer:
    """
    Handles training and evaluation of multiple machine learning models.

    This class provides functionality to train Multinomial Naive Bayes,
    Random Forest, and XGBoost classifiers on both BoW and TF-IDF features,
    with cross-validation and comprehensive evaluation.

    Attributes:
        models (dict): Dictionary storing trained model instances
        label_encoder (LabelEncoder): Encoder for XGBoost compatibility

    Example:
        >>> trainer = ModelTrainer()
        >>> results = trainer.train_all_models(bow_features, tfidf_features, y_train, y_test)
        >>> best_model = trainer.get_best_model(results)
    """
```

### Testing Procedures

#### Test Structure
```
tests/
├── test_pipeline.py          # Main test suite
├── test_data_loader.py       # Data loading tests
├── test_preprocessor.py      # Text processing tests
├── test_feature_engineer.py  # Feature engineering tests
├── test_model_trainer.py     # Model training tests
├── test_synthetic_generator.py # Synthetic data tests
└── test_evaluator.py         # Evaluation tests
```

#### Running Tests
```bash
# Run all tests
pytest tests/ -v

# Run specific test class
pytest tests/test_pipeline.py::TestDataLoader -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run quick tests only
pytest tests/ -m "not slow" -v
```

#### Writing Tests
```python
import pytest
import pandas as pd
from src.data_loader import DataLoader

class TestDataLoader:
    """Test data loading functionality"""

    def test_data_loader_init(self):
        """Test DataLoader initialization"""
        loader = DataLoader()
        assert loader.raw_data is None

    def test_load_and_unify_data(self):
        """Test data loading and unification"""
        # Create test data
        test_data = pd.DataFrame({
            'tweet_id': [1, 2],
            'brand': ['Apple', 'Google'],
            'sentiment_label': ['Positive', 'Negative'],
            'tweet_text': ['Great product!', 'Poor service.']
        })

        # Test loading logic
        loader = DataLoader()
        # ... test implementation
```

#### Test Categories
- **Unit Tests**: Test individual functions and methods
- **Integration Tests**: Test component interactions
- **Pipeline Tests**: Test end-to-end pipeline execution
- **Performance Tests**: Test execution time and memory usage

### Code Quality Tools

#### Linting and Formatting
```bash
# Install development tools
pip install flake8 black isort mypy

# Format code
black src/ tests/

# Sort imports
isort src/ tests/

# Lint code
flake8 src/ tests/

# Type checking
mypy src/
```

#### Pre-commit Hooks
```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/psf/black
    rev: 22.3.0
    hooks:
      - id: black
  - repo: https://github.com/pycqa/isort
    rev: 5.10.1
    hooks:
      - id: isort
  - repo: https://github.com/pycqa/flake8
    rev: 4.0.1
    hooks:
      - id: flake8
```

### Adding New Features

#### Adding a New Model
1. **Modify ModelTrainer**:
```python
# In src/model_trainer.py
from sklearn.svm import SVC

model_configs = {
    'MultinomialNB': lambda: MultinomialNB(),
    'RandomForest': lambda: RandomForestClassifier(n_estimators=100, random_state=42),
    'XGBoost': lambda: xgb.XGBClassifier(random_state=42),
    'SVM': lambda: SVC(random_state=42)  # New model
}
```

2. **Add Special Handling** (if needed):
```python
# Handle model-specific requirements
if model_name == 'SVM':
    # SVM requires dense matrices
    X_train_dense = X_train.toarray()
    X_test_dense = X_test.toarray()
    model.fit(X_train_dense, y_train)
    y_pred = model.predict(X_test_dense)
```

3. **Update Tests**:
```python
def test_svm_training(self):
    """Test SVM model training"""
    # Add test for new model
    pass
```

#### Adding New Features
1. **Extend FeatureEngineer**:
```python
def create_word_embeddings(self, X_train, X_test, embedding_dim=100):
    """Create word embedding features"""
    # Implement embedding logic
    pass
```

2. **Update Pipeline**:
```python
# In run_pipeline.py
embedding_features = engineer.create_word_embeddings(X_train, X_test)
```

3. **Add Configuration**:
```python
# Add new parameters to configuration section
EMBEDDING_DIM = 100
EMBEDDING_MODEL = 'word2vec'
```

### Documentation Standards

#### Module Documentation
- Every module should have a clear docstring explaining its purpose
- Include usage examples in module docstrings
- Document all public functions and classes

#### API Documentation
- Use type hints for all function parameters and returns
- Include parameter descriptions and examples
- Document exceptions that may be raised

#### README Updates
- Update README.md when adding new features
- Include new dependencies in requirements.txt
- Update usage examples if interface changes

### Performance Considerations

#### Memory Management
- Use sparse matrices for large feature sets
- Implement batch processing for large datasets
- Monitor memory usage during development

#### Optimization Guidelines
- Profile code to identify bottlenecks
- Use vectorized operations where possible
- Implement parallel processing for CPU-intensive tasks
- Consider caching for expensive computations

#### Scalability
- Design components to handle varying dataset sizes
- Implement progress tracking for long-running operations
- Consider distributed processing for very large datasets

---

## 9. Deployment Instructions

### Production Deployment

#### Environment Setup

##### 1. Production Server Requirements
```bash
# Minimum system requirements
- CPU: 4+ cores (8+ recommended)
- RAM: 8GB minimum (16GB+ recommended)
- Storage: 10GB free space
- OS: Ubuntu 20.04+ / CentOS 8+ / Amazon Linux 2
- Python: 3.12+
```

##### 2. Production Installation
```bash
# Create production user
sudo useradd -m -s /bin/bash sentiment_app
sudo usermod -aG sudo sentiment_app

# Switch to production user
sudo su - sentiment_app

# Clone repository
git clone <repository-url> /opt/sentiment_analysis
cd /opt/sentiment_analysis

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install production dependencies
pip install -r requirements.txt

# Download NLTK data
python -c "import nltk; nltk.download('all')"
```

##### 3. System Service Setup
```bash
# Create systemd service file
sudo tee /etc/systemd/system/sentiment-analysis.service << EOF
[Unit]
Description=Sentiment Analysis Pipeline
After=network.target

[Service]
Type=simple
User=sentiment_app
WorkingDirectory=/opt/sentiment_analysis
Environment=PATH=/opt/sentiment_analysis/venv/bin
ExecStart=/opt/sentiment_analysis/venv/bin/python run_pipeline.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# Enable and start service
sudo systemctl daemon-reload
sudo systemctl enable sentiment-analysis
sudo systemctl start sentiment-analysis
```

#### Configuration Management

##### Production Configuration
```python
# config/production.py
import os

class ProductionConfig:
    # Data paths
    DATA_DIR = os.environ.get('SENTIMENT_DATA_DIR', '/data/sentiment')
    MODEL_DIR = os.environ.get('SENTIMENT_MODEL_DIR', '/models/sentiment')

    # Performance settings
    MAX_FEATURES = int(os.environ.get('MAX_FEATURES', '10000'))
    N_JOBS = int(os.environ.get('N_JOBS', '-1'))

    # Logging
    LOG_LEVEL = os.environ.get('LOG_LEVEL', 'INFO')
    LOG_FILE = os.environ.get('LOG_FILE', '/var/log/sentiment/app.log')

    # Memory limits
    MAX_MEMORY_GB = int(os.environ.get('MAX_MEMORY_GB', '8'))
```

##### Environment Variables
```bash
# /etc/environment or .env file
SENTIMENT_DATA_DIR=/data/sentiment
SENTIMENT_MODEL_DIR=/models/sentiment
MAX_FEATURES=10000
N_JOBS=8
LOG_LEVEL=INFO
LOG_FILE=/var/log/sentiment/app.log
MAX_MEMORY_GB=16
```

#### Containerization

##### Dockerfile
```dockerfile
FROM python:3.12-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Create app user
RUN useradd -m -s /bin/bash sentiment_app

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download NLTK data
RUN python -c "import nltk; nltk.download('all')"

# Copy application code
COPY . .

# Change ownership to app user
RUN chown -R sentiment_app:sentiment_app /app

# Switch to app user
USER sentiment_app

# Expose port (if adding web interface)
EXPOSE 8000

# Run application
CMD ["python", "run_pipeline.py"]
```

##### Docker Compose
```yaml
# docker-compose.yml
version: '3.8'

services:
  sentiment-analysis:
    build: .
    container_name: sentiment_app
    volumes:
      - ./data:/app/datasets
      - ./models:/app/models
      - ./reports:/app/reports
    environment:
      - MAX_FEATURES=10000
      - N_JOBS=4
      - LOG_LEVEL=INFO
    restart: unless-stopped

  # Optional: Add monitoring
  prometheus:
    image: prom/prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
```

##### Deployment Commands
```bash
# Build and deploy
docker-compose build
docker-compose up -d

# Check status
docker-compose ps
docker-compose logs sentiment-analysis

# Update deployment
docker-compose pull
docker-compose up -d --force-recreate
```

### Scaling Considerations

#### Horizontal Scaling

##### Load Balancing
```bash
# nginx configuration for multiple instances
upstream sentiment_backend {
    server sentiment_app_1:8000;
    server sentiment_app_2:8000;
    server sentiment_app_3:8000;
}

server {
    listen 80;
    location / {
        proxy_pass http://sentiment_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

##### Kubernetes Deployment
```yaml
# k8s-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: sentiment-analysis
spec:
  replicas: 3
  selector:
    matchLabels:
      app: sentiment-analysis
  template:
    metadata:
      labels:
        app: sentiment-analysis
    spec:
      containers:
      - name: sentiment-app
        image: sentiment-analysis:latest
        resources:
          requests:
            memory: "2Gi"
            cpu: "1"
          limits:
            memory: "4Gi"
            cpu: "2"
        env:
        - name: MAX_FEATURES
          value: "10000"
        - name: N_JOBS
          value: "2"
```

#### Vertical Scaling

##### Resource Optimization
```python
# Optimize for larger datasets
def optimize_for_scale(dataset_size):
    if dataset_size > 1000000:  # 1M+ samples
        return {
            'max_features': 20000,
            'n_jobs': -1,
            'batch_size': 10000,
            'use_sparse_matrices': True
        }
    elif dataset_size > 100000:  # 100K+ samples
        return {
            'max_features': 10000,
            'n_jobs': 8,
            'batch_size': 5000,
            'use_sparse_matrices': True
        }
    else:
        return {
            'max_features': 5000,
            'n_jobs': 4,
            'batch_size': 1000,
            'use_sparse_matrices': False
        }
```

### Monitoring Setup

#### Application Monitoring

##### Logging Configuration
```python
# logging_config.py
import logging
import logging.handlers

def setup_logging(log_level='INFO', log_file='/var/log/sentiment/app.log'):
    """Configure application logging"""

    # Create logger
    logger = logging.getLogger('sentiment_analysis')
    logger.setLevel(getattr(logging, log_level))

    # Create formatters
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # File handler with rotation
    file_handler = logging.handlers.RotatingFileHandler(
        log_file, maxBytes=10*1024*1024, backupCount=5
    )
    file_handler.setFormatter(formatter)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger
```

##### Performance Metrics
```python
# metrics.py
import time
import psutil
from functools import wraps

def monitor_performance(func):
    """Decorator to monitor function performance"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB

        result = func(*args, **kwargs)

        end_time = time.time()
        end_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB

        logger.info(f"{func.__name__} - Execution time: {end_time - start_time:.2f}s")
        logger.info(f"{func.__name__} - Memory usage: {end_memory - start_memory:.2f}MB")

        return result
    return wrapper
```

#### System Monitoring

##### Health Check Endpoint
```python
# health_check.py
import json
import os
from datetime import datetime

def health_check():
    """Generate health check report"""

    health_status = {
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'version': '1.0.0',
        'checks': {
            'disk_space': check_disk_space(),
            'memory': check_memory(),
            'dependencies': check_dependencies(),
            'models': check_models()
        }
    }

    # Determine overall status
    if any(check['status'] == 'unhealthy' for check in health_status['checks'].values()):
        health_status['status'] = 'unhealthy'

    return health_status

def check_disk_space(threshold_gb=5):
    """Check available disk space"""
    statvfs = os.statvfs('.')
    free_gb = (statvfs.f_frsize * statvfs.f_bavail) / (1024**3)

    return {
        'status': 'healthy' if free_gb > threshold_gb else 'unhealthy',
        'free_space_gb': round(free_gb, 2),
        'threshold_gb': threshold_gb
    }
```

### Backup and Recovery

#### Data Backup Strategy
```bash
#!/bin/bash
# backup_script.sh

BACKUP_DIR="/backup/sentiment_analysis"
DATE=$(date +%Y%m%d_%H%M%S)

# Create backup directory
mkdir -p "$BACKUP_DIR/$DATE"

# Backup models
tar -czf "$BACKUP_DIR/$DATE/models.tar.gz" models/

# Backup processed data
tar -czf "$BACKUP_DIR/$DATE/datasets.tar.gz" datasets/

# Backup configuration
cp -r config/ "$BACKUP_DIR/$DATE/"

# Cleanup old backups (keep last 7 days)
find "$BACKUP_DIR" -type d -mtime +7 -exec rm -rf {} \;

echo "Backup completed: $BACKUP_DIR/$DATE"
```

#### Recovery Procedures
```bash
#!/bin/bash
# recovery_script.sh

BACKUP_DATE=$1
BACKUP_DIR="/backup/sentiment_analysis/$BACKUP_DATE"

if [ ! -d "$BACKUP_DIR" ]; then
    echo "Backup directory not found: $BACKUP_DIR"
    exit 1
fi

# Stop service
sudo systemctl stop sentiment-analysis

# Restore models
tar -xzf "$BACKUP_DIR/models.tar.gz" -C /

# Restore data
tar -xzf "$BACKUP_DIR/datasets.tar.gz" -C /

# Restore configuration
cp -r "$BACKUP_DIR/config/" /opt/sentiment_analysis/

# Start service
sudo systemctl start sentiment-analysis

echo "Recovery completed from backup: $BACKUP_DATE"
```

---

## 10. Troubleshooting

### Common Issues and Solutions

#### Installation Issues

##### 1. NLTK Download Failures
**Problem**: NLTK data download fails or times out
```
[nltk_data] Error loading punkt: <urlopen error [Errno 11001]>
```

**Solutions**:
```bash
# Method 1: Manual download
python -c "
import nltk
nltk.download('punkt', download_dir='/usr/local/share/nltk_data')
nltk.download('stopwords', download_dir='/usr/local/share/nltk_data')
nltk.download('wordnet', download_dir='/usr/local/share/nltk_data')
"

# Method 2: Offline installation
wget https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/tokenizers/punkt.zip
# Extract to nltk_data directory

# Method 3: Use alternative mirror
python -c "
import nltk
nltk.set_proxy('http://proxy.example.com:8080')  # If behind proxy
nltk.download('all')
"
```

##### 2. XGBoost Installation Problems
**Problem**: XGBoost fails to install or import
```
ImportError: No module named 'xgboost'
```

**Solutions**:
```bash
# Method 1: Use conda
conda install -c conda-forge xgboost

# Method 2: Install from source
pip install xgboost --no-binary xgboost

# Method 3: Use pre-compiled wheel
pip install xgboost --find-links https://download.pytorch.org/whl/torch_stable.html
```

##### 3. SDV (TVAE) Installation Issues
**Problem**: SDV package installation fails
```
ERROR: Failed building wheel for sdv
```

**Solutions**:
```bash
# Install build dependencies
sudo apt-get install build-essential python3-dev

# Install specific version
pip install sdv==1.0.0

# Alternative: Skip TVAE functionality
# Set SKIP_TVAE=True in configuration
```

#### Runtime Issues

##### 1. Memory Errors
**Problem**: Out of memory during feature engineering or model training
```
MemoryError: Unable to allocate array
```

**Solutions**:
```python
# Reduce feature dimensions
MAX_FEATURES = 2000  # Instead of 5000

# Use quick test mode
python run_pipeline.py --quick-test --max-samples 500

# Enable sparse matrices
USE_SPARSE_MATRICES = True

# Reduce batch size
BATCH_SIZE = 1000
```

##### 2. SMOTE Failures
**Problem**: SMOTE fails with small class sizes
```
ValueError: Expected n_neighbors <= n_samples, but n_samples = 2, n_neighbors = 5
```

**Solutions**:
```python
# Automatic k_neighbors adjustment (already implemented)
k_neighbors = min(5, min_class_count - 1)

# Skip SMOTE for very small classes
if min_class_count < 3:
    print("Skipping SMOTE due to insufficient samples")
    return X, y

# Use alternative oversampling
from imblearn.over_sampling import RandomOverSampler
ros = RandomOverSampler(random_state=42)
X_resampled, y_resampled = ros.fit_resample(X, y)
```

##### 3. Model Training Failures
**Problem**: Model training fails or produces poor results
```
ValueError: Input contains NaN, infinity or a value too large
```

**Solutions**:
```python
# Check for NaN values
print(f"NaN values in features: {np.isnan(X_train.data).sum()}")

# Handle infinite values
X_train.data = np.nan_to_num(X_train.data, nan=0.0, posinf=1.0, neginf=-1.0)

# Normalize features if needed
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler(with_mean=False)  # For sparse matrices
X_train_scaled = scaler.fit_transform(X_train)
```

#### Data Issues

##### 1. Empty or Invalid Text
**Problem**: Text preprocessing results in empty strings
```
Warning: Removed 150 empty tweets after cleaning
```

**Solutions**:
```python
# Adjust preprocessing parameters
MIN_WORD_LENGTH = 2  # Instead of 3
REMOVE_STOPWORDS = False  # Keep stopwords for very short texts

# Add text length validation
def validate_text_length(df, min_length=10):
    valid_mask = df['tweet_text'].str.len() >= min_length
    return df[valid_mask]

# Use alternative cleaning strategy
def gentle_clean_text(text):
    """Less aggressive text cleaning"""
    text = text.lower()
    text = re.sub(r'http\S+', '', text)  # Remove URLs only
    text = re.sub(r'[^\w\s]', ' ', text)  # Keep alphanumeric and spaces
    return ' '.join(text.split())
```

##### 2. Class Imbalance Issues
**Problem**: Severe class imbalance affects model performance
```
Class distribution: {'Positive': 5000, 'Negative': 100, 'Neutral': 50}
```

**Solutions**:
```python
# Use stratified sampling
from sklearn.utils import resample

def balance_classes(df, max_samples_per_class=1000):
    balanced_dfs = []
    for class_label in df['sentiment_label'].unique():
        class_df = df[df['sentiment_label'] == class_label]
        if len(class_df) > max_samples_per_class:
            class_df = resample(class_df, n_samples=max_samples_per_class, random_state=42)
        balanced_dfs.append(class_df)
    return pd.concat(balanced_dfs, ignore_index=True)

# Use class weights
from sklearn.utils.class_weight import compute_class_weight

class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
model = RandomForestClassifier(class_weight='balanced')
```

##### 3. Encoding Issues
**Problem**: Text encoding problems with special characters
```
UnicodeDecodeError: 'utf-8' codec can't decode byte
```

**Solutions**:
```python
# Read CSV with encoding specification
df = pd.read_csv(filepath, encoding='utf-8', errors='ignore')

# Alternative encodings
encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
for encoding in encodings:
    try:
        df = pd.read_csv(filepath, encoding=encoding)
        break
    except UnicodeDecodeError:
        continue

# Clean problematic characters
def clean_encoding(text):
    if isinstance(text, str):
        return text.encode('utf-8', errors='ignore').decode('utf-8')
    return text
```

#### Performance Issues

##### 1. Slow Execution
**Problem**: Pipeline takes too long to execute
```
Pipeline execution time: 45 minutes (expected: <10 minutes)
```

**Solutions**:
```python
# Enable parallel processing
N_JOBS = -1  # Use all available cores

# Reduce feature dimensions
MAX_FEATURES = 3000  # Instead of 5000

# Use quick test mode for development
python run_pipeline.py --quick-test

# Profile code to identify bottlenecks
import cProfile
cProfile.run('run_pipeline()', 'profile_output.prof')
```

##### 2. High Memory Usage
**Problem**: Memory usage exceeds available RAM
```
Process killed (OOM - Out of Memory)
```

**Solutions**:
```python
# Monitor memory usage
import psutil

def check_memory():
    memory = psutil.virtual_memory()
    print(f"Memory usage: {memory.percent}%")
    if memory.percent > 90:
        print("Warning: High memory usage!")

# Use memory-efficient data types
df = df.astype({
    'tweet_len': 'int16',
    'num_hashtags': 'int8',
    'num_mentions': 'int8'
})

# Process data in chunks
def process_in_chunks(df, chunk_size=1000):
    for i in range(0, len(df), chunk_size):
        chunk = df.iloc[i:i+chunk_size]
        yield process_chunk(chunk)
```

### Error Messages and Solutions

#### Common Error Patterns

##### 1. Feature Engineering Errors
```python
# Error: "ValueError: empty vocabulary"
# Solution: Check text preprocessing and minimum document frequency
if len(clean_texts) == 0:
    raise ValueError("No valid text data after preprocessing")

# Error: "IndexError: list index out of range"
# Solution: Validate data shapes and indices
assert len(X_train) > 0, "Training data is empty"
assert len(y_train) == len(X_train), "Feature and label counts don't match"
```

##### 2. Model Training Errors
```python
# Error: "ValueError: Unknown label type"
# Solution: Ensure labels are properly formatted
y_train = y_train.astype(str)  # Convert to string labels

# Error: "NotFittedError: This model has not been fitted yet"
# Solution: Check model training completion
if not hasattr(model, 'classes_'):
    raise ValueError("Model must be fitted before prediction")
```

##### 3. File I/O Errors
```python
# Error: "FileNotFoundError: No such file or directory"
# Solution: Create directories and check paths
os.makedirs(os.path.dirname(filepath), exist_ok=True)

# Error: "PermissionError: Access denied"
# Solution: Check file permissions
os.chmod(filepath, 0o644)
```

### Debugging Tools and Techniques

#### 1. Logging and Debugging
```python
import logging

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Add debug prints
logger.debug(f"Data shape: {df.shape}")
logger.debug(f"Class distribution: {y.value_counts()}")
```

#### 2. Data Validation
```python
def validate_pipeline_data(df):
    """Comprehensive data validation"""
    checks = {
        'non_empty': len(df) > 0,
        'required_columns': all(col in df.columns for col in ['tweet_text', 'sentiment_label']),
        'no_null_text': not df['tweet_text'].isnull().any(),
        'valid_labels': df['sentiment_label'].notna().all()
    }

    failed_checks = [check for check, passed in checks.items() if not passed]
    if failed_checks:
        raise ValueError(f"Data validation failed: {failed_checks}")

    return True
```

#### 3. Performance Profiling
```python
import time
from functools import wraps

def profile_execution(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"{func.__name__} took {end - start:.2f} seconds")
        return result
    return wrapper

# Use decorator on slow functions
@profile_execution
def slow_function():
    # Function implementation
    pass
```

### Getting Help

#### 1. Check Logs
```bash
# Application logs
tail -f /var/log/sentiment/app.log

# System logs
journalctl -u sentiment-analysis -f

# Docker logs
docker-compose logs -f sentiment-analysis
```

#### 2. Run Diagnostics
```bash
# Test installation
python -c "import src; print('Import successful')"

# Check dependencies
pip check

# Verify NLTK data
python -c "import nltk; nltk.data.find('tokenizers/punkt')"

# Test pipeline components
pytest tests/test_pipeline.py::TestDataLoader -v
```

#### 3. Community Support
- **GitHub Issues**: Report bugs and request features
- **Documentation**: Check this comprehensive guide
- **Stack Overflow**: Search for similar issues with tags: `sentiment-analysis`, `scikit-learn`, `nltk`
- **Python Community**: Python Discord, Reddit r/MachineLearning

#### 4. Professional Support
For production deployments requiring professional support:
- Performance optimization consulting
- Custom feature development
- Enterprise deployment assistance
- Training and workshops

---

## Conclusion

This comprehensive documentation provides everything needed to understand, install, configure, develop, deploy, and troubleshoot the SocialMedia Sentiment Analysis system. The modular architecture and extensive documentation make it suitable for both research and production use cases.

For additional support or questions not covered in this documentation, please refer to the project repository or contact the development team.

