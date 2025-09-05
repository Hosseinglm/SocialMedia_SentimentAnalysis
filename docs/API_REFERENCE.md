# API Reference

## Overview

This document provides detailed API reference for all classes and methods in the SocialMedia Sentiment Analysis system.

## Core Modules

### src.data_loader

#### DataLoader

```python
class DataLoader:
    """Handles loading and unifying Twitter data from multiple sources"""
```

##### Methods

###### `__init__()`
```python
def __init__(self)
```
Initialize DataLoader instance.

**Returns**: None

---

###### `load_and_unify_data(train_path, val_path)`
```python
def load_and_unify_data(self, train_path: str, val_path: str) -> pd.DataFrame
```
Load training and validation CSV files and unify schema.

**Parameters**:
- `train_path` (str): Path to training CSV file
- `val_path` (str): Path to validation CSV file

**Returns**:
- `pandas.DataFrame`: Unified dataset with standardized schema

**Raises**:
- `FileNotFoundError`: If input files don't exist
- `ValueError`: If CSV format is invalid

**Example**:
```python
loader = DataLoader()
data = loader.load_and_unify_data('train.csv', 'val.csv')
print(f"Loaded {len(data)} samples")
```

---

###### `save_raw_data(df, output_path)`
```python
def save_raw_data(self, df: pd.DataFrame, output_path: str) -> None
```
Save raw unified data to CSV.

**Parameters**:
- `df` (pandas.DataFrame): Data to save
- `output_path` (str): Output file path

**Returns**: None

---

### src.preprocessor

#### TextPreprocessor

```python
class TextPreprocessor:
    """Handles text cleaning, preprocessing, and feature extraction"""
```

##### Methods

###### `__init__()`
```python
def __init__(self)
```
Initialize TextPreprocessor with NLTK components.

**Downloads**: Required NLTK data (punkt, stopwords, wordnet)

---

###### `clean_dataset(df)`
```python
def clean_dataset(self, df: pd.DataFrame) -> pd.DataFrame
```
Clean and preprocess entire dataset.

**Parameters**:
- `df` (pandas.DataFrame): Raw dataset with 'tweet_text' column

**Returns**:
- `pandas.DataFrame`: Cleaned dataset with additional features

**Added Columns**:
- `clean_tweet_basic`: Basic cleaned text
- `clean_tweet`: Lemmatized and processed text
- `tweet_len`: Original tweet character length
- `num_hashtags`: Number of hashtags in original tweet
- `num_mentions`: Number of @mentions in original tweet

**Example**:
```python
preprocessor = TextPreprocessor()
clean_data = preprocessor.clean_dataset(raw_data)
print(f"Added features: {clean_data.columns.tolist()}")
```

---

###### `save_clean_data(df, output_path)`
```python
def save_clean_data(self, df: pd.DataFrame, output_path: str) -> None
```
Save cleaned data to CSV file.

---

### src.feature_engineer

#### FeatureEngineer

```python
class FeatureEngineer:
    """Handles feature engineering for text data"""
```

##### Methods

###### `create_train_test_split(df, test_size=0.2, random_state=42)`
```python
def create_train_test_split(self, df: pd.DataFrame, test_size: float = 0.2, 
                           random_state: int = 42) -> tuple
```
Create stratified train-test split.

**Parameters**:
- `df` (pandas.DataFrame): Cleaned dataset
- `test_size` (float): Proportion of test data (default: 0.2)
- `random_state` (int): Random seed (default: 42)

**Returns**:
- `tuple`: (X_train, X_test, y_train, y_test)

---

###### `create_bow_features(X_train, X_test, max_features=5000, min_df=2)`
```python
def create_bow_features(self, X_train: pd.Series, X_test: pd.Series, 
                       max_features: int = 5000, min_df: int = 2) -> dict
```
Create Bag-of-Words features.

**Parameters**:
- `X_train` (pandas.Series): Training text data
- `X_test` (pandas.Series): Test text data
- `max_features` (int): Maximum number of features (default: 5000)
- `min_df` (int): Minimum document frequency (default: 2)

**Returns**:
- `dict`: Dictionary with keys:
  - `'train'`: Training feature matrix (scipy.sparse)
  - `'test'`: Test feature matrix (scipy.sparse)
  - `'vectorizer'`: Fitted CountVectorizer object

**Configuration**:
- Binary features (0 or 1)
- max_df=0.95 (ignore terms in >95% of documents)

---

###### `create_tfidf_features(X_train, X_test, max_features=5000, min_df=2)`
```python
def create_tfidf_features(self, X_train: pd.Series, X_test: pd.Series,
                         max_features: int = 5000, min_df: int = 2) -> dict
```
Create TF-IDF features with n-gram support.

**Parameters**:
- Same as `create_bow_features`

**Returns**:
- `dict`: Same structure as BoW features

**Configuration**:
- N-gram range: (1, 2) - unigrams and bigrams
- TF-IDF normalization applied

---

###### `save_features(bow_features, tfidf_features, features_dir='features')`
```python
def save_features(self, bow_features: dict, tfidf_features: dict, 
                 features_dir: str = 'features') -> None
```
Save feature matrices and vocabularies to disk.

**Saved Files**:
- `bow.npz`: Compressed BoW matrices
- `tfidf.npz`: Compressed TF-IDF matrices
- `bow_vocab.pkl`: BoW vocabulary
- `tfidf_vocab.pkl`: TF-IDF vocabulary

---

### src.model_trainer

#### ModelTrainer

```python
class ModelTrainer:
    """Handles training and evaluation of multiple machine learning models"""
```

##### Attributes

- `models` (dict): Dictionary storing trained model instances
- `label_encoder` (LabelEncoder): Encoder for XGBoost compatibility

##### Methods

###### `train_all_models(bow_features, tfidf_features, y_train, y_test)`
```python
def train_all_models(self, bow_features: dict, tfidf_features: dict,
                    y_train: pd.Series, y_test: pd.Series) -> dict
```
Train all models on different feature sets with cross-validation.

**Models Trained**:
- Multinomial Naive Bayes
- Random Forest (100 estimators)
- XGBoost

**Feature Sets**:
- Bag-of-Words
- TF-IDF

**Returns**:
- `dict`: Results with keys like 'RandomForest_BoW', 'XGBoost_TF-IDF', etc.

**Result Structure**:
```python
{
    'model': 'RandomForest',
    'features': 'BoW',
    'accuracy': 0.8542,
    'f1_macro': 0.8234,
    'f1_weighted': 0.8456,
    'precision_macro': 0.8123,
    'recall_macro': 0.8345,
    'cv_scores': array([0.84, 0.85, 0.86, 0.84, 0.85])
}
```

---

###### `get_best_model(results)`
```python
def get_best_model(self, results: dict) -> str
```
Identify best performing model based on F1-macro score.

**Parameters**:
- `results` (dict): Results from `train_all_models`

**Returns**:
- `str`: Best model key identifier (e.g., 'RandomForest_BoW')

---

###### `retrain_on_augmented_data(X_aug, y_aug, X_test, y_test, model_type)`
```python
def retrain_on_augmented_data(self, X_aug, y_aug, X_test, y_test, 
                             model_type: str) -> dict
```
Retrain model on SMOTE-augmented data.

**Parameters**:
- `X_aug`: Augmented feature matrix
- `y_aug`: Augmented labels
- `X_test`: Test features
- `y_test`: Test labels
- `model_type` (str): Model type ('MultinomialNB', 'RandomForest', 'XGBoost')

**Returns**:
- `dict`: Evaluation metrics on augmented model

---

###### `save_best_model(model, filepath)`
```python
def save_best_model(self, model, filepath: str) -> None
```
Save trained model to pickle file.

---

###### `save_model_card(result, model_key, filepath='models/model_card.json')`
```python
def save_model_card(self, result: dict, model_key: str, 
                   filepath: str = 'models/model_card.json') -> None
```
Save model metadata and performance metrics.

**Model Card Structure**:
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

---

### src.synthetic_generator

#### SyntheticDataGenerator

```python
class SyntheticDataGenerator:
    """Handles synthetic data generation for class imbalance"""
```

##### Methods

###### `analyze_class_imbalance(y)`
```python
def analyze_class_imbalance(self, y: pd.Series) -> dict
```
Analyze class imbalance in the dataset.

**Returns**:
```python
{
    'class_counts': {'Positive': 5000, 'Negative': 1000, 'Neutral': 500},
    'majority_count': 5000,
    'minority_classes': ['Neutral'],
    'imbalance_ratio': 10.0
}
```

---

###### `apply_smote(X, y, random_state=42)`
```python
def apply_smote(self, X, y, random_state: int = 42) -> tuple
```
Apply SMOTE oversampling to balance classes.

**Parameters**:
- `X`: Feature matrix (scipy.sparse)
- `y` (pandas.Series): Target labels
- `random_state` (int): Random seed

**Returns**:
- `tuple`: (X_resampled, y_resampled)

**Features**:
- Automatic k_neighbors adjustment for small classes
- Handles sparse matrices
- Preserves feature matrix format

---

###### `generate_tvae_data(df, n_samples=1000)`
```python
def generate_tvae_data(self, df: pd.DataFrame, n_samples: int = 1000) -> pd.DataFrame
```
Generate synthetic tabular data using TVAE.

**Parameters**:
- `df` (pandas.DataFrame): Original dataset
- `n_samples` (int): Number of synthetic samples to generate

**Returns**:
- `pandas.DataFrame`: Synthetic data

**TVAE Configuration**:
- Epochs: 50
- Columns used: ['sentiment_label', 'tweet_len', 'num_hashtags', 'num_mentions', 'brand']
- Quality validation included

---

### src.evaluator

#### ModelEvaluator

```python
class ModelEvaluator:
    """Comprehensive model evaluation and reporting"""
```

##### Methods

###### `evaluate_model(model, X_test, y_test, model_name=None, label_encoder=None)`
```python
def evaluate_model(self, model, X_test, y_test, model_name: str = None,
                  label_encoder = None) -> dict
```
Comprehensive model evaluation.

**Returns**:
```python
{
    'accuracy': 0.8542,
    'f1_macro': 0.8234,
    'f1_weighted': 0.8456,
    'precision_macro': 0.8123,
    'recall_macro': 0.8345,
    'predictions': array(['Positive', 'Negative', ...])
}
```

---

###### `plot_confusion_matrix(y_true, y_pred, title="Confusion Matrix")`
```python
def plot_confusion_matrix(self, y_true, y_pred, title: str = "Confusion Matrix") -> None
```
Create and display confusion matrix visualization.

---

###### `generate_classification_report(y_true, y_pred)`
```python
def generate_classification_report(self, y_true, y_pred) -> str
```
Generate detailed per-class performance metrics.

**Returns**: Formatted classification report string

---

## Usage Patterns

### Basic Pipeline Usage
```python
from src import DataLoader, TextPreprocessor, FeatureEngineer, ModelTrainer

# Load data
loader = DataLoader()
data = loader.load_and_unify_data('train.csv', 'val.csv')

# Preprocess
preprocessor = TextPreprocessor()
clean_data = preprocessor.clean_dataset(data)

# Engineer features
engineer = FeatureEngineer()
X_train, X_test, y_train, y_test = engineer.create_train_test_split(clean_data)
bow_features = engineer.create_bow_features(X_train, X_test)
tfidf_features = engineer.create_tfidf_features(X_train, X_test)

# Train models
trainer = ModelTrainer()
results = trainer.train_all_models(bow_features, tfidf_features, y_train, y_test)
```

### Custom Model Integration
```python
# Extend ModelTrainer for custom models
class CustomModelTrainer(ModelTrainer):
    def __init__(self):
        super().__init__()
        self.custom_models = {
            'SVM': lambda: SVC(random_state=42),
            'LogisticRegression': lambda: LogisticRegression(random_state=42)
        }
    
    def train_custom_model(self, X_train, X_test, y_train, y_test, model_name):
        model = self.custom_models[model_name]()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        return self.evaluate_predictions(y_test, y_pred)
```

### Error Handling
```python
try:
    results = trainer.train_all_models(bow_features, tfidf_features, y_train, y_test)
except ValueError as e:
    print(f"Training failed: {e}")
    # Handle specific error cases
except MemoryError:
    print("Insufficient memory, reducing feature dimensions")
    # Reduce max_features and retry
```
