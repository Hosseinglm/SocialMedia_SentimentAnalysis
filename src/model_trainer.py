"""
Model training and evaluation module
"""

import numpy as np
import pandas as pd
import pickle
import json
import os
from datetime import datetime
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb


class ModelTrainer:
    """Handles training and evaluation of multiple models"""
    
    def __init__(self):
        self.models = {}
        self.results = {}
        self.label_encoder = LabelEncoder()
        
    def train_all_models(self, bow_features, tfidf_features, y_train, y_test):
        """
        Train all models on different feature sets
        
        Args:
            bow_features: BoW feature dictionary
            tfidf_features: TF-IDF feature dictionary
            y_train: Training labels
            y_test: Test labels
            
        Returns:
            Dictionary with all results
        """
        print("   Training models on different feature sets...")
        
        # Encode labels for XGBoost
        y_train_encoded = self.label_encoder.fit_transform(y_train)
        y_test_encoded = self.label_encoder.transform(y_test)
        
        # Model configurations
        model_configs = {
            'MultinomialNB': lambda: MultinomialNB(),
            'RandomForest': lambda: RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
            'XGBoost': lambda: xgb.XGBClassifier(random_state=42, eval_metric='mlogloss', verbosity=0)
        }
        
        feature_sets = {
            'BoW': bow_features,
            'TF-IDF': tfidf_features
        }
        
        # Cross-validation setup
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        results = {}
        
        for feature_name, features in feature_sets.items():
            print(f"\\n   Feature Set: {feature_name}")
            
            X_train, X_test = features['train'], features['test']
            
            for model_name, model_factory in model_configs.items():
                print(f"     Training {model_name}...")
                
                # Create fresh model instance
                model = model_factory()
                
                # Use encoded labels for XGBoost, original for others
                if model_name == 'XGBoost':
                    model.fit(X_train, y_train_encoded)
                    y_pred_encoded = model.predict(X_test)
                    y_pred = self.label_encoder.inverse_transform(y_pred_encoded)
                else:
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                
                # Metrics
                accuracy = accuracy_score(y_test, y_pred)
                f1_macro = f1_score(y_test, y_pred, average='macro')
                
                # Cross-validation for RandomForest
                cv_scores = None
                if model_name == 'RandomForest':
                    # Create a fresh model for CV
                    cv_model = model_factory()
                    cv_scores = cross_val_score(cv_model, X_train, y_train, cv=cv, scoring='accuracy')
                elif model_name == 'XGBoost':
                    # Create a fresh model for CV
                    cv_model = model_factory()
                    cv_scores = cross_val_score(cv_model, X_train, y_train_encoded, cv=cv, scoring='accuracy')
                
                # Store results
                key = f"{model_name}_{feature_name}"
                results[key] = {
                    'model': model_name,
                    'features': feature_name,
                    'accuracy': accuracy,
                    'f1_macro': f1_macro,
                    'cv_scores': cv_scores,
                    'predictions': y_pred
                }
                
                # Store model
                self.models[key] = model
                
                print(f"       Accuracy: {accuracy:.4f}, F1-macro: {f1_macro:.4f}")
                
                if cv_scores is not None:
                    print(f"       CV: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        self.results = results
        return results
    
    def get_best_model(self, results):
        """Get the best performing model based on F1-macro score"""
        best_key = max(results.keys(), key=lambda k: results[k]['f1_macro'])
        print(f"\\n   Best model: {best_key}")
        print(f"   F1-macro: {results[best_key]['f1_macro']:.4f}")
        print(f"   Accuracy: {results[best_key]['accuracy']:.4f}")
        
        return best_key
    
    def retrain_on_augmented_data(self, X_aug, y_aug, X_test, y_test, model_type):
        """
        Retrain best model type on augmented data
        
        Args:
            X_aug: Augmented training features
            y_aug: Augmented training labels
            X_test: Test features
            y_test: Test labels
            model_type: Type of model to retrain
            
        Returns:
            Trained model on augmented data
        """
        print(f"   Retraining {model_type} on augmented data...")
        
        # Initialize model
        if model_type == 'MultinomialNB':
            model = MultinomialNB()
            model.fit(X_aug, y_aug)
            y_pred = model.predict(X_test)
        elif model_type == 'RandomForest':
            model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
            model.fit(X_aug, y_aug)
            y_pred = model.predict(X_test)
        else:  # XGBoost
            model = xgb.XGBClassifier(random_state=42, eval_metric='mlogloss', verbosity=0)
            # Encode labels for XGBoost
            y_aug_encoded = self.label_encoder.transform(y_aug)
            model.fit(X_aug, y_aug_encoded)
            y_pred_encoded = model.predict(X_test)
            y_pred = self.label_encoder.inverse_transform(y_pred_encoded)
        accuracy = accuracy_score(y_test, y_pred)
        f1_macro = f1_score(y_test, y_pred, average='macro')
        
        print(f"   Augmented model - Accuracy: {accuracy:.4f}, F1-macro: {f1_macro:.4f}")
        
        return model
    
    def save_best_model(self, model, filepath):
        """Save the best model to disk"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(model, f)
        print(f"   Model saved to {filepath}")
    
    def save_model_card(self, result, model_key, filepath='models/model_card.json'):
        """Save model metadata and performance metrics"""
        model_card = {
            'model_name': result['model'],
            'features': result['features'],
            'version': '1.0',
            'created_date': datetime.now().isoformat(),
            'accuracy': float(result['accuracy']),
            'f1_macro': float(result['f1_macro']),
            'model_key': model_key
        }
        
        # Add CV scores if available
        if result['cv_scores'] is not None:
            model_card['cv_mean'] = float(result['cv_scores'].mean())
            model_card['cv_std'] = float(result['cv_scores'].std())
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(model_card, f, indent=2)
        
        print(f"   Model card saved to {filepath}")
    
    def load_model(self, filepath):
        """Load a saved model"""
        with open(filepath, 'rb') as f:
            model = pickle.load(f)
        return model
    
    def get_model_comparison(self):
        """Get comparison of all trained models"""
        if not self.results:
            return "No models trained yet"
        
        comparison = []
        for key, result in self.results.items():
            comparison.append({
                'model_key': key,
                'model': result['model'],
                'features': result['features'],
                'accuracy': result['accuracy'],
                'f1_macro': result['f1_macro']
            })
        
        # Sort by F1-macro score
        comparison.sort(key=lambda x: x['f1_macro'], reverse=True)
        
        return comparison
