"""
Test suite for Twitter Sentiment Analytics Pipeline
"""

import pytest
import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from data_loader import DataLoader
from preprocessor import TextPreprocessor
from feature_engineer import FeatureEngineer
from model_trainer import ModelTrainer
from synthetic_generator import SyntheticDataGenerator
from evaluator import ModelEvaluator


class TestDataLoader:
    """Test data loading functionality"""
    
    def test_data_loader_init(self):
        """Test DataLoader initialization"""
        loader = DataLoader()
        assert loader.raw_data is None
    
    def test_create_unified_schema(self):
        """Test schema unification"""
        loader = DataLoader()
        
        # Create sample data
        sample_data = pd.DataFrame({
            'tweet_id': [1, 2, 3],
            'brand': ['Brand1', 'Brand2', 'Brand3'],
            'sentiment_label': ['Positive', 'Negative', 'Neutral'],
            'tweet_text': ['Good product', 'Bad service', 'Okay experience'],
            'source': ['train', 'train', 'validation']
        })
        
        unified = loader._create_unified_schema(sample_data)
        
        # Check required columns exist
        required_columns = ['tweet_id', 'author', 'date', 'tweet_text', 'sentiment_label', 'brand', 'source']
        for col in required_columns:
            assert col in unified.columns
        
        # Check author column is populated
        assert not unified['author'].isna().any()
        assert not unified['date'].isna().any()


class TestTextPreprocessor:
    """Test text preprocessing functionality"""
    
    def test_preprocessor_init(self):
        """Test TextPreprocessor initialization"""
        preprocessor = TextPreprocessor()
        assert preprocessor.lemmatizer is not None
        assert len(preprocessor.stop_words) > 0
    
    def test_clean_text(self):
        """Test text cleaning function"""
        preprocessor = TextPreprocessor()
        
        # Test basic cleaning
        text = "This is a GREAT product! Check it out at http://example.com #awesome @company"
        cleaned = preprocessor._clean_text(text)
        
        assert cleaned.islower()
        assert 'http' not in cleaned
        assert '@' not in cleaned
        assert '#' not in cleaned
        assert 'awesome' in cleaned  # hashtag word should remain
    
    def test_extract_features(self):
        """Test feature extraction"""
        preprocessor = TextPreprocessor()
        
        text = "Love this product! #great @company Visit http://example.com"
        length, hashtags, mentions = preprocessor._extract_features(text)
        
        assert length > 0
        assert hashtags == 1
        assert mentions == 1
    
    def test_lemmatize_text(self):
        """Test text lemmatization"""
        preprocessor = TextPreprocessor()
        
        text = "running dogs are jumping quickly"
        lemmatized = preprocessor._lemmatize_text(text)
        
        # Should contain lemmatized forms
        assert len(lemmatized) > 0
        assert isinstance(lemmatized, str)


class TestFeatureEngineer:
    """Test feature engineering functionality"""
    
    def test_feature_engineer_init(self):
        """Test FeatureEngineer initialization"""
        engineer = FeatureEngineer()
        assert engineer.bow_vectorizer is None
        assert engineer.tfidf_vectorizer is None
    
    def test_create_train_test_split(self):
        """Test train-test split creation"""
        engineer = FeatureEngineer()
        
        # Create sample data with enough instances per class
        df = pd.DataFrame({
            'clean_tweet': ['good product', 'bad service', 'okay experience', 'great item',
                          'good product', 'bad service', 'okay experience', 'great item'],
            'sentiment_label': ['Positive', 'Negative', 'Neutral', 'Positive',
                              'Positive', 'Negative', 'Neutral', 'Positive']
        })
        
        X_train, X_test, y_train, y_test = engineer.create_train_test_split(df, test_size=0.5, random_state=42)
        
        assert len(X_train) == 2
        assert len(X_test) == 2
        assert len(y_train) == 2
        assert len(y_test) == 2
    
    def test_bow_features(self):
        """Test BoW feature creation"""
        engineer = FeatureEngineer()
        
        X_train = pd.Series(['good product', 'bad service', 'excellent quality', 'poor value'])
        X_test = pd.Series(['okay product'])
        
        bow_features = engineer.create_bow_features(X_train, X_test, max_features=100, min_df=1)
        
        assert 'train' in bow_features
        assert 'test' in bow_features
        assert 'vectorizer' in bow_features
        assert bow_features['train'].shape[0] == 2
        assert bow_features['test'].shape[0] == 1
    
    def test_tfidf_features(self):
        """Test TF-IDF feature creation"""
        engineer = FeatureEngineer()
        
        X_train = pd.Series(['good product', 'bad service', 'excellent quality', 'poor value'])
        X_test = pd.Series(['okay product'])
        
        tfidf_features = engineer.create_tfidf_features(X_train, X_test, max_features=100, min_df=1)
        
        assert 'train' in tfidf_features
        assert 'test' in tfidf_features
        assert 'vectorizer' in tfidf_features
        assert tfidf_features['train'].shape[0] == 2
        assert tfidf_features['test'].shape[0] == 1


class TestModelTrainer:
    """Test model training functionality"""
    
    def test_model_trainer_init(self):
        """Test ModelTrainer initialization"""
        trainer = ModelTrainer()
        assert len(trainer.models) == 0
        assert len(trainer.results) == 0
    
    def test_get_best_model(self):
        """Test best model selection"""
        trainer = ModelTrainer()
        
        # Mock results
        results = {
            'model1': {'f1_macro': 0.7, 'accuracy': 0.75},
            'model2': {'f1_macro': 0.8, 'accuracy': 0.8},
            'model3': {'f1_macro': 0.6, 'accuracy': 0.65}
        }
        
        best_key = trainer.get_best_model(results)
        assert best_key == 'model2'


class TestSyntheticDataGenerator:
    """Test synthetic data generation functionality"""
    
    def test_synthetic_generator_init(self):
        """Test SyntheticDataGenerator initialization"""
        generator = SyntheticDataGenerator()
        assert generator.smote is None
        assert generator.tvae is None
    
    def test_analyze_class_imbalance(self):
        """Test class imbalance analysis"""
        generator = SyntheticDataGenerator()
        
        # Create imbalanced dataset
        y = ['Positive'] * 100 + ['Negative'] * 20 + ['Neutral'] * 5
        
        analysis = generator.analyze_class_imbalance(y)
        
        assert 'class_counts' in analysis
        assert 'minority_classes' in analysis
        assert 'imbalance_ratio' in analysis
        assert len(analysis['minority_classes']) > 0  # Should detect minorities


class TestModelEvaluator:
    """Test model evaluation functionality"""
    
    def test_evaluator_init(self):
        """Test ModelEvaluator initialization"""
        evaluator = ModelEvaluator()
        assert len(evaluator.evaluation_results) == 0
    
    def test_compare_models(self):
        """Test model comparison"""
        evaluator = ModelEvaluator()
        
        results = {
            'model1': {'accuracy': 0.8, 'f1_macro': 0.75},
            'model2': {'accuracy': 0.85, 'f1_macro': 0.8}
        }
        
        comparison = evaluator.compare_models(results)
        
        assert isinstance(comparison, pd.DataFrame)
        assert len(comparison) == 2
        assert 'Model' in comparison.columns
        assert 'Accuracy' in comparison.columns
        assert 'F1-Macro' in comparison.columns
    
    def test_calculate_class_performance(self):
        """Test per-class performance calculation"""
        evaluator = ModelEvaluator()
        
        y_true = np.array(['A', 'B', 'A', 'B', 'A'])
        y_pred = np.array(['A', 'B', 'A', 'A', 'A'])
        
        class_perf = evaluator.calculate_class_performance(y_true, y_pred)
        
        assert isinstance(class_perf, pd.DataFrame)
        assert 'Class' in class_perf.columns
        assert 'Precision' in class_perf.columns
        assert 'Recall' in class_perf.columns
        assert 'F1-Score' in class_perf.columns


class TestIntegration:
    """Integration tests for the complete pipeline"""
    
    def test_pipeline_integration(self):
        """Test basic pipeline integration"""
        # Create sample data
        sample_data = pd.DataFrame({
            'tweet_id': range(50),  # Increased sample size
            'brand': ['Brand1'] * 25 + ['Brand2'] * 25,
            'sentiment_label': ['Positive'] * 17 + ['Negative'] * 17 + ['Neutral'] * 16,
            'tweet_text': [f'Sample tweet {i} with some good content here' for i in range(50)],
            'source': ['train'] * 40 + ['validation'] * 10
        })
        
        # Test preprocessing
        preprocessor = TextPreprocessor()
        clean_data = preprocessor.clean_dataset(sample_data)
        
        assert len(clean_data) <= len(sample_data)  # May remove empty tweets
        assert 'clean_tweet' in clean_data.columns
        assert 'tweet_len' in clean_data.columns
        
        # Test feature engineering
        engineer = FeatureEngineer()
        X_train, X_test, y_train, y_test = engineer.create_train_test_split(clean_data, test_size=0.3)
        
        bow_features = engineer.create_bow_features(X_train, X_test, max_features=50, min_df=1)
        tfidf_features = engineer.create_tfidf_features(X_train, X_test, max_features=50, min_df=1)
        
        assert bow_features['train'].shape[0] == len(X_train)
        assert tfidf_features['train'].shape[0] == len(X_train)
        
        print("✅ Integration test passed!")


# Pytest configuration
def pytest_configure():
    """Configure pytest"""
    # Suppress warnings for cleaner output
    import warnings
    warnings.filterwarnings("ignore")


if __name__ == "__main__":
    # Run tests directly
    pytest.main([__file__, '-v'])
