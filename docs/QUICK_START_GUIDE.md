# Quick Start Guide

## Overview

This guide will get you up and running with the SocialMedia Sentiment Analysis system in under 10 minutes.

## Prerequisites

- Python 3.12 or higher
- 4GB RAM minimum (8GB recommended)
- 2GB free disk space

## Installation

### 1. Clone Repository
```bash
git clone <repository-url>
cd SocialMedia_SentimentAnalysis
```

### 2. Install Dependencies
```bash
# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install packages
pip install -r requirements.txt

# Download NLTK data
python -c "import nltk; nltk.download('all')"
```

### 3. Verify Installation
```bash
# Quick test (completes in ~2 minutes)
python run_pipeline.py --quick-test --max-samples 100
```

## Basic Usage

### Option 1: Command Line (Recommended for First Time)

#### Quick Test
```bash
# Fast execution on sample data
python run_pipeline.py --quick-test
```

#### Full Pipeline
```bash
# Complete analysis (requires your data files)
python run_pipeline.py
```

### Option 2: Jupyter Notebook

```bash
# Start Jupyter
jupyter notebook main.ipynb
```

Follow the notebook cells for interactive analysis.

### Option 3: Python Script

```python
from src import DataLoader, TextPreprocessor, FeatureEngineer, ModelTrainer

# 1. Load your data
loader = DataLoader()
data = loader.load_and_unify_data('datasets/twitter_training.csv', 
                                  'datasets/twitter_validation.csv')

# 2. Clean text
preprocessor = TextPreprocessor()
clean_data = preprocessor.clean_dataset(data)

# 3. Create features
engineer = FeatureEngineer()
X_train, X_test, y_train, y_test = engineer.create_train_test_split(clean_data)
bow_features = engineer.create_bow_features(X_train, X_test)

# 4. Train model
trainer = ModelTrainer()
results = trainer.train_all_models(bow_features, {}, y_train, y_test)

# 5. Get results
best_model = trainer.get_best_model(results)
print(f"Best model: {best_model}")
print(f"Accuracy: {results[best_model]['accuracy']:.3f}")
```

## Data Format

### Input Files Required

Place these files in the `datasets/` directory:

**twitter_training.csv**:
```csv
tweet_id,brand,sentiment_label,tweet_text
1,Apple,Positive,"Love the new iPhone! Great camera."
2,Google,Negative,"Search results are getting worse."
3,Microsoft,Neutral,"Windows update installed successfully."
```

**twitter_validation.csv**:
```csv
tweet_id,brand,sentiment_label,tweet_text
4,Apple,Positive,"Amazing build quality on MacBook."
5,Google,Negative,"Gmail is down again."
6,Microsoft,Neutral,"Office 365 works as expected."
```

### Column Requirements

- `tweet_id`: Unique identifier (string/number)
- `brand`: Brand or topic category (string)
- `sentiment_label`: Sentiment class (string: Positive, Negative, Neutral, etc.)
- `tweet_text`: Raw tweet content (string)

## Understanding Output

### Generated Files

After running the pipeline, you'll find:

```
datasets/
├── raw_tweets.csv          # Unified input data
├── clean_tweets.csv        # Processed text data
└── synthetic_tvae_data.csv # Generated synthetic data

features/
├── bow.npz                 # Bag-of-Words features
├── tfidf.npz              # TF-IDF features
├── bow_vocab.pkl          # BoW vocabulary
└── tfidf_vocab.pkl        # TF-IDF vocabulary

models/
├── best_model_*.pkl       # Trained model file
└── model_card.json        # Model metadata

reports/
├── eda/
│   ├── overview_analysis.png     # Data overview plots
│   └── sentiment_wordclouds.png  # Word clouds by sentiment
└── evaluation/
    └── confusion_matrices.png    # Model performance
```

### Key Metrics

The pipeline outputs several important metrics:

```
PIPELINE SUMMARY
================
Total samples processed: 15,000
Best model: RandomForest_BoW
Original accuracy: 0.8542
Augmented accuracy: 0.8687
Improvement: +0.0145
```

**What this means**:
- **Total samples**: Number of tweets processed
- **Best model**: Algorithm and feature combination with highest F1-score
- **Original accuracy**: Performance before synthetic data augmentation
- **Augmented accuracy**: Performance after SMOTE balancing
- **Improvement**: Accuracy gain from synthetic data

## Common Use Cases

### 1. Analyze Your Own Twitter Data

```python
# Replace with your data files
data = loader.load_and_unify_data('your_training.csv', 'your_validation.csv')
```

### 2. Quick Model Comparison

```bash
# Compare models quickly
python run_pipeline.py --quick-test --max-samples 500
```

### 3. Production Model Training

```bash
# Full training for production use
python run_pipeline.py
```

### 4. Custom Analysis

```python
# Load pre-trained model
import pickle
with open('models/best_model_RandomForest_BoW.pkl', 'rb') as f:
    model = pickle.load(f)

# Predict new text
new_text = ["This product is amazing!"]
# ... preprocess and vectorize new_text
prediction = model.predict(vectorized_text)
```

## Troubleshooting

### Common Issues

#### 1. Installation Problems
```bash
# If NLTK download fails
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"

# If XGBoost fails
conda install xgboost  # Alternative to pip
```

#### 2. Memory Issues
```bash
# Reduce memory usage
python run_pipeline.py --quick-test --max-samples 200
```

#### 3. No Data Files
```bash
# Create sample data for testing
mkdir -p datasets
# Add your CSV files to datasets/ directory
```

#### 4. Empty Results
- Check that your CSV files have the required columns
- Ensure tweet_text column contains actual text data
- Verify sentiment_label column has valid categories

### Getting Help

1. **Check logs**: Look for error messages in the console output
2. **Run tests**: `pytest tests/ -v` to verify installation
3. **Reduce data size**: Use `--quick-test` mode for debugging
4. **Check file format**: Ensure CSV files match the required schema

## Next Steps

### Explore Advanced Features

1. **Custom Models**: Add your own algorithms to ModelTrainer
2. **Feature Engineering**: Experiment with different text preprocessing
3. **Hyperparameter Tuning**: Optimize model parameters
4. **Deployment**: Set up production pipeline

### Learn More

- Read the [Comprehensive Documentation](COMPREHENSIVE_DOCUMENTATION.md) for detailed information
- Check the [API Reference](API_REFERENCE.md) for programming details
- Explore the Jupyter notebook for interactive analysis
- Review test files for usage examples

### Performance Tips

1. **Use Quick Test**: Always start with `--quick-test` for development
2. **Monitor Memory**: Watch RAM usage with large datasets
3. **Parallel Processing**: The system automatically uses multiple CPU cores
4. **Feature Reduction**: Reduce `max_features` if memory is limited

## Example Workflow

Here's a complete example workflow:

```bash
# 1. Setup
git clone <repo-url>
cd SocialMedia_SentimentAnalysis
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 2. Prepare data
mkdir -p datasets
# Copy your twitter_training.csv and twitter_validation.csv to datasets/

# 3. Quick test
python run_pipeline.py --quick-test

# 4. Full analysis
python run_pipeline.py

# 5. Check results
ls -la models/
ls -la reports/
```

Expected completion times:
- Quick test: 2-3 minutes
- Full pipeline: 10-30 minutes (depending on data size)

## Success Criteria

You'll know the system is working correctly when:

1. ✅ Quick test completes without errors
2. ✅ Model files are generated in `models/` directory
3. ✅ Accuracy improvement is shown (target: ≥3 percentage points)
4. ✅ Visualizations are created in `reports/` directory
5. ✅ All tests pass: `pytest tests/ -v`

## Support

If you encounter issues:

1. Check the [Troubleshooting section](COMPREHENSIVE_DOCUMENTATION.md#troubleshooting) in the main documentation
2. Run the test suite: `pytest tests/ -v`
3. Try quick test mode: `python run_pipeline.py --quick-test --max-samples 50`
4. Review the error messages and logs
5. Check that your data format matches the requirements

For additional help, refer to the comprehensive documentation or create an issue in the project repository.
