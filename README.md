# Twitter Sentiment Analytics & Synthetic Boost

A complete, reproducible sentiment analysis pipeline for Twitter data with synthetic data augmentation capabilities. This project implements a comprehensive machine learning pipeline that processes Twitter data, trains multiple classifiers, and uses advanced synthetic data generation techniques to address class imbalance.

## Project Overview

This pipeline replicates and adapts the architecture from "Web & Social Media Sentiment Analytics" for Twitter data, featuring:

- **Data Processing**: Clean and preprocess Twitter text data
- **Feature Engineering**: Create BoW, TF-IDF, and optional embedding features  
- **Model Training**: Train and compare Multinomial NB, Random Forest, and XGBoost classifiers
- **Synthetic Data**: Generate synthetic samples using SMOTE and TVAE techniques
- **Evaluation**: Comprehensive model evaluation with confusion matrices and performance metrics

## Quick Start

### Prerequisites

- Python 3.12+
- Required packages (see `requirements.txt`)

### Installation

1. Clone or download the project:
```bash
cd Web-SocialM-Sentiment-Analytics
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the complete pipeline:
```bash
python run_pipeline.py
```

### Quick Test Mode

For rapid testing on a subset of data (completes in <3 minutes):
```bash
python run_pipeline.py --quick-test
```

## Project Structure

```
Web-SocialM-Sentiment-Analytics/
├── datasets/                          # Data files
│   ├── twitter_training.csv          # Original training data
│   ├── twitter_validation.csv        # Original validation data
│   ├── raw_tweets.csv                # Unified raw data
│   ├── clean_tweets.csv              # Processed clean data
│   └── synthetic_tvae_data.csv       # Generated synthetic data
├── features/                          # Feature matrices
│   ├── bow.npz                       # Bag-of-Words features
│   ├── tfidf.npz                     # TF-IDF features
│   ├── bow_vocab.pkl                 # BoW vocabulary
│   └── tfidf_vocab.pkl               # TF-IDF vocabulary
├── models/                           # Trained models
│   ├── best_model_*.pkl              # Best performing model
│   └── model_card.json               # Model metadata
├── reports/                          # Analysis results
│   ├── eda/                          # Exploratory data analysis
│   │   ├── overview_analysis.png     # Dataset overview plots
│   │   └── sentiment_wordclouds.png  # Word clouds by sentiment
│   └── evaluation/                   # Model evaluation
│       ├── confusion_matrices.png    # Confusion matrices
│       └── deep_learning_training.png # DL training curves
├── src/                              # Source code modules
│   ├── data_loader.py                # Data loading and unification
│   ├── preprocessor.py               # Text cleaning and preprocessing
│   ├── feature_engineer.py           # Feature engineering
│   ├── model_trainer.py              # Model training and evaluation
│   ├── synthetic_generator.py        # Synthetic data generation
│   └── evaluator.py                  # Model evaluation utilities
├── tests/                            # Test suite
│   ├── main.ipynb                    # Main demonstration notebook
│   └── test_pipeline.py              # Automated tests
├── run_pipeline.py                   # Main execution script
├── requirements.txt                  # Python dependencies
└── README.md                         # This file
```

## Pipeline Components

### 1. Data Loading & Unification
- Loads training and validation CSV files
- Unifies schema with columns: `tweet_id`, `author`, `date`, `tweet_text`, `sentiment_label`
- Saves immutable raw dataset

### 2. Text Preprocessing
- **Cleaning**: Lowercase, remove URLs, @handles, hashtags (keep words), expand contractions
- **Feature extraction**: Tweet length, number of hashtags, number of mentions
- **Lemmatization**: WordNet lemmatizer with stopword removal
- **Output**: Clean dataset with processed text

### 3. Feature Engineering
- **Bag-of-Words**: Binary matrix with 5000 max features
- **TF-IDF**: Term frequency-inverse document frequency with bigrams
- **Train-Test Split**: 80-20 stratified split for cross-model comparison
- **Storage**: Compressed matrices (.npz) and vocabularies (.pkl)

### 4. Model Training
Multiple classifiers trained and evaluated:

| Model | Features | Evaluation Method |
|-------|----------|------------------|
| Multinomial Naive Bayes | BoW, TF-IDF | Accuracy, F1-macro |
| Random Forest | BoW, TF-IDF | 10-fold CV, Accuracy, F1-macro |
| XGBoost | BoW, TF-IDF | Early stopping, Accuracy, F1-macro |

### 5. Synthetic Data Generation
- **SMOTE**: Oversampling for immediate class balance
- **TVAE**: Generate 2000+ synthetic samples matching full schema
- **Quality Validation**: Kolmogorov-Smirnov tests and correlation analysis
- **Augmentation**: Retrain best model on augmented dataset

### 6. Deep Learning (Optional)
- Simple Keras Sequential MLP on TF-IDF features
- Monitor validation accuracy for 20 epochs
- Visualization of training curves

## Model Performance

The pipeline automatically selects the best performing model based on F1-macro score and provides:

- **Confusion matrices** for each model/feature combination
- **Classification reports** with per-class precision, recall, F1-score
- **Cross-validation results** for Random Forest models
- **Improvement tracking** after synthetic data augmentation

### Acceptance Criteria
- **Target**: ≥3 percentage point accuracy improvement OR clear class imbalance reduction  
- **Quick Test**: Complete pipeline execution on 1000 samples in <3 minutes  
- **Testing**: All tests in `pytest tests/` pass

## Testing

Run the automated test suite:
```bash
pytest tests/ -v
```

Test individual components:
```bash
python -m pytest tests/test_pipeline.py::TestDataLoader -v
```

## Data Schema

### Input Data Format
The pipeline expects CSV files with columns:
- `tweet_id`: Unique identifier
- `brand/topic`: Brand or topic category  
- `sentiment_label`: Sentiment class (Positive, Negative, Neutral, etc.)
- `tweet_text`: Raw tweet content

### Output Data Schema
Processed data includes additional features:
- `author`: Author/brand information
- `date`: Synthetic date for temporal analysis
- `clean_tweet`: Processed and lemmatized text
- `tweet_len`: Original tweet character length
- `num_hashtags`: Number of hashtags in original tweet
- `num_mentions`: Number of @mentions in original tweet

## Configuration

### Model Parameters
- **BoW/TF-IDF**: 5000 max features, min_df=2, max_df=0.95
- **Random Forest**: 100 estimators, 5-fold CV
- **XGBoost**: Early stopping, mlogloss evaluation
- **SMOTE**: k_neighbors=5 (auto-adjusted for small classes)

### Synthetic Data Parameters
- **TVAE**: 50 epochs, tabular synthesis
- **Quality Threshold**: Kolmogorov-Smirnov p-value analysis
- **Sample Size**: Up to 2000 synthetic samples per run

## Troubleshooting

### Common Issues

1. **Memory Issues**: Reduce max_features in feature engineering
2. **SMOTE Fails**: Check for classes with <5 samples
3. **TVAE Installation**: Use `pip install sdv>=1.0.0`
4. **NLTK Data**: Run `python -c "import nltk; nltk.download('all')"`

### Performance Optimization

- Use `--quick-test` for rapid iteration
- Reduce `max_features` for faster processing
- Skip TVAE generation in quick test mode
- Use `n_jobs=-1` for parallel processing where available

## Dependencies

Core libraries:
- **Data**: pandas, numpy
- **ML**: scikit-learn, xgboost, imbalanced-learn
- **NLP**: nltk, wordcloud
- **Visualization**: matplotlib, seaborn
- **Synthetic Data**: sdv (TVAE)
- **Optional**: tensorflow (deep learning)

See `requirements.txt` for complete dependency list with versions.

## Usage Examples

### Basic Pipeline Execution
```bash
# Full pipeline
python run_pipeline.py

# Quick test with 500 samples
python run_pipeline.py --quick-test --max-samples 500
```

### Jupyter Notebook Analysis
```bash
jupyter notebook tests/main.ipynb
```

### Custom Data Analysis
```python
from src import DataLoader, TextPreprocessor, ModelTrainer

# Load your own data
loader = DataLoader()
data = loader.load_and_unify_data('your_train.csv', 'your_val.csv')

# Process and train
preprocessor = TextPreprocessor()
clean_data = preprocessor.clean_dataset(data)
# ... continue with pipeline
```

## Data and Model Files

**Note**: To keep the repository size manageable, datasets, trained models, and generated features are not included in Git. These files will be generated when you run the pipeline.

### Regenerating Data and Models

1. **Place your training data** in the `datasets/` directory:
   - `twitter_training.csv` - Training dataset
   - `twitter_validation.csv` - Validation dataset

2. **Run the pipeline** to generate all files:
   ```bash
   python run_pipeline.py
   ```

3. **Generated files** will be created in:
   - `datasets/` - Cleaned and processed data
   - `features/` - Feature matrices (BoW, TF-IDF)
   - `models/` - Trained model files and metadata
   - `reports/` - Visualizations and evaluation plots

### File Size Note

Trained models can be very large (100MB+). If you need to share models:
- Use Git LFS for large files
- Or upload models to cloud storage
- Or retrain models from the pipeline

## License

This project is provided for educational and research purposes. Please ensure compliance with data usage policies for any Twitter data used.

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Run tests to verify setup: `pytest tests/ -v`
3. Use quick test mode for debugging: `python run_pipeline.py --quick-test`

---

This pipeline provides a robust foundation for Twitter sentiment analysis with state-of-the-art synthetic data augmentation techniques.
