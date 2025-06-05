"""
Feature engineering module for creating BoW, TF-IDF, and other features
"""

import numpy as np
import pandas as pd
import pickle
import os
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split


class FeatureEngineer:
    """Handles feature engineering for text data"""
    
    def __init__(self):
        self.bow_vectorizer = None
        self.tfidf_vectorizer = None
        
    def create_train_test_split(self, df, test_size=0.2, random_state=42):
        """
        Create stratified train-test split
        
        Args:
            df: Cleaned DataFrame
            test_size: Proportion of test data
            random_state: Random seed
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        X = df['clean_tweet']
        y = df['sentiment_label']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        print(f"   Train samples: {len(X_train)}")
        print(f"   Test samples: {len(X_test)}")
        
        return X_train, X_test, y_train, y_test
    
    def create_bow_features(self, X_train, X_test, max_features=5000, min_df=2):
        """
        Create Bag-of-Words features
        
        Args:
            X_train: Training text data
            X_test: Test text data
            max_features: Maximum number of features
            min_df: Minimum document frequency
            
        Returns:
            Dictionary with train and test matrices
        """
        print("   Creating Bag-of-Words features...")
        
        self.bow_vectorizer = CountVectorizer(
            binary=True, 
            max_features=max_features, 
            min_df=min_df, 
            max_df=0.95
        )
        
        X_train_bow = self.bow_vectorizer.fit_transform(X_train)
        X_test_bow = self.bow_vectorizer.transform(X_test)
        
        print(f"   BoW matrix shape: {X_train_bow.shape}")
        
        return {
            'train': X_train_bow,
            'test': X_test_bow,
            'vectorizer': self.bow_vectorizer
        }
    
    def create_tfidf_features(self, X_train, X_test, max_features=5000, min_df=2):
        """
        Create TF-IDF features
        
        Args:
            X_train: Training text data
            X_test: Test text data
            max_features: Maximum number of features
            min_df: Minimum document frequency
            
        Returns:
            Dictionary with train and test matrices
        """
        print("   Creating TF-IDF features...")
        
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=max_features,
            min_df=min_df,
            max_df=0.95,
            ngram_range=(1, 2)
        )
        
        X_train_tfidf = self.tfidf_vectorizer.fit_transform(X_train)
        X_test_tfidf = self.tfidf_vectorizer.transform(X_test)
        
        print(f"   TF-IDF matrix shape: {X_train_tfidf.shape}")
        
        return {
            'train': X_train_tfidf,
            'test': X_test_tfidf,
            'vectorizer': self.tfidf_vectorizer
        }
    
    def save_features(self, bow_features, tfidf_features, features_dir='features'):
        """Save feature matrices and vocabularies"""
        os.makedirs(features_dir, exist_ok=True)
        
        # Save BoW features
        np.savez_compressed(
            os.path.join(features_dir, 'bow.npz'),
            train=bow_features['train'],
            test=bow_features['test']
        )
        
        with open(os.path.join(features_dir, 'bow_vocab.pkl'), 'wb') as f:
            pickle.dump(bow_features['vectorizer'].vocabulary_, f)
        
        # Save TF-IDF features
        np.savez_compressed(
            os.path.join(features_dir, 'tfidf.npz'),
            train=tfidf_features['train'],
            test=tfidf_features['test']
        )
        
        with open(os.path.join(features_dir, 'tfidf_vocab.pkl'), 'wb') as f:
            pickle.dump(tfidf_features['vectorizer'].vocabulary_, f)
        
        print(f"   Features saved to {features_dir}/")
    
    def load_features(self, features_dir='features'):
        """Load saved feature matrices"""
        bow_data = np.load(os.path.join(features_dir, 'bow.npz'), allow_pickle=True)
        tfidf_data = np.load(os.path.join(features_dir, 'tfidf.npz'), allow_pickle=True)
        
        with open(os.path.join(features_dir, 'bow_vocab.pkl'), 'rb') as f:
            bow_vocab = pickle.load(f)
        
        with open(os.path.join(features_dir, 'tfidf_vocab.pkl'), 'rb') as f:
            tfidf_vocab = pickle.load(f)
        
        bow_features = {
            'train': bow_data['train'],
            'test': bow_data['test'],
            'vocabulary': bow_vocab
        }
        
        tfidf_features = {
            'train': tfidf_data['train'],
            'test': tfidf_data['test'],
            'vocabulary': tfidf_vocab
        }
        
        return bow_features, tfidf_features
    
    def create_sentence_embeddings(self, X_train, X_test):
        """
        Create sentence embeddings (placeholder for future implementation)
        
        Args:
            X_train: Training text data
            X_test: Test text data
            
        Returns:
            Dictionary with train and test embeddings
        """
        # Placeholder for sentence transformer embeddings
        # This would require sentence_transformers library
        print("   Sentence embeddings not implemented yet")
        return None
    
    def get_feature_stats(self, bow_features, tfidf_features):
        """Get statistics about the created features"""
        stats = {
            'bow_features': bow_features['train'].shape[1],
            'tfidf_features': tfidf_features['train'].shape[1],
            'bow_sparsity': 1 - (bow_features['train'].nnz / np.prod(bow_features['train'].shape)),
            'tfidf_sparsity': 1 - (tfidf_features['train'].nnz / np.prod(tfidf_features['train'].shape)),
            'train_samples': bow_features['train'].shape[0],
            'test_samples': bow_features['test'].shape[0]
        }
        
        return stats
