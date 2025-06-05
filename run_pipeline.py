#!/usr/bin/env python3
"""
Twitter Sentiment Analytics Pipeline
Main execution script with quick test capability
"""

import pandas as pd
import numpy as np
import argparse
import time
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

# Import custom modules
from src.data_loader import DataLoader
from src.preprocessor import TextPreprocessor
from src.feature_engineer import FeatureEngineer
from src.model_trainer import ModelTrainer
from src.synthetic_generator import SyntheticDataGenerator
from src.evaluator import ModelEvaluator


def run_pipeline(quick_test=False, max_samples=1000):
    """Run the complete sentiment analysis pipeline"""
    
    start_time = time.time()
    print("Starting Twitter Sentiment Analytics Pipeline")
    print("=" * 60)
    
    if quick_test:
        print(f"QUICK TEST MODE: Using max {max_samples} samples")
    
    try:
        # 1. Data Loading
        print("\nStep 1: Loading Data...")
        loader = DataLoader()
        tweets_raw = loader.load_and_unify_data(
            'datasets/twitter_training.csv',
            'datasets/twitter_validation.csv'
        )
        
        if quick_test:
            tweets_raw = tweets_raw.sample(n=min(max_samples, len(tweets_raw)), random_state=42)
            print(f"   Using {len(tweets_raw)} samples for quick test")
        
        loader.save_raw_data(tweets_raw, 'datasets/raw_tweets.csv')
        
        # 2. Preprocessing
        print("\nStep 2: Data Cleaning & Preprocessing...")
        preprocessor = TextPreprocessor()
        clean_tweets = preprocessor.clean_dataset(tweets_raw)
        preprocessor.save_clean_data(clean_tweets, 'datasets/clean_tweets.csv')
        
        # 3. Feature Engineering
        print("\nStep 3: Feature Engineering...")
        feature_engineer = FeatureEngineer()
        X_train, X_test, y_train, y_test = feature_engineer.create_train_test_split(clean_tweets)
        
        # Create feature matrices
        bow_features = feature_engineer.create_bow_features(X_train, X_test)
        tfidf_features = feature_engineer.create_tfidf_features(X_train, X_test)
        feature_engineer.save_features(bow_features, tfidf_features)
        
        # 4. Model Training
        print("\nStep 4: Model Training & Evaluation...")
        trainer = ModelTrainer()
        results = trainer.train_all_models(bow_features, tfidf_features, y_train, y_test)
        
        best_model_key = trainer.get_best_model(results)
        best_model = trainer.models[best_model_key]
        
        # 5. Synthetic Data Generation
        print("\nStep 5: Synthetic Data Generation...")
        synthetic_gen = SyntheticDataGenerator()
        
        # SMOTE augmentation
        if 'TF-IDF' in best_model_key:
            X_aug, y_aug = synthetic_gen.apply_smote(tfidf_features['train'], y_train)
        else:
            X_aug, y_aug = synthetic_gen.apply_smote(bow_features['train'], y_train)
        
        # TVAE generation (optional)
        if not quick_test:  # Skip TVAE in quick test due to time
            try:
                synthetic_data = synthetic_gen.generate_tvae_data(clean_tweets, n_samples=1000)
                synthetic_gen.save_synthetic_data(synthetic_data, 'datasets/synthetic_tvae_data.csv')
            except Exception as e:
                print(f"   Warning: TVAE generation failed: {e}")
        
        # 6. Model Retraining on Augmented Data
        print("\nStep 6: Retraining on Augmented Data...")
        model_type = results[best_model_key]['model']
        augmented_model = trainer.retrain_on_augmented_data(
            X_aug, y_aug, 
            tfidf_features['test'] if 'TF-IDF' in best_model_key else bow_features['test'],
            y_test, model_type
        )
        
        # 7. Final Evaluation
        print("\nStep 7: Final Evaluation...")
        evaluator = ModelEvaluator()
        
        # Select correct features based on model key
        test_features = tfidf_features['test'] if 'TF-IDF' in best_model_key else bow_features['test']
        
        # Compare original vs augmented performance
        label_encoder = trainer.label_encoder if 'XGBoost' in best_model_key else None
        
        original_metrics = evaluator.evaluate_model(
            best_model, 
            test_features,
            y_test,
            label_encoder=label_encoder
        )
        
        augmented_metrics = evaluator.evaluate_model(
            augmented_model,
            test_features,
            y_test,
            label_encoder=label_encoder
        )
        
        # Save results
        trainer.save_best_model(best_model, f'models/best_model_{best_model_key}.pkl')
        trainer.save_model_card(results[best_model_key], best_model_key)
        
        # Print summary
        print("\nPIPELINE SUMMARY")
        print("=" * 40)
        print(f"Total samples processed: {len(clean_tweets):,}")
        print(f"Best model: {best_model_key}")
        print(f"Original accuracy: {original_metrics['accuracy']:.4f}")
        print(f"Augmented accuracy: {augmented_metrics['accuracy']:.4f}")
        print(f"Improvement: {augmented_metrics['accuracy'] - original_metrics['accuracy']:+.4f}")
        
        # Check acceptance criteria
        improvement = augmented_metrics['accuracy'] - original_metrics['accuracy']
        if improvement >= 0.03:
            print("ACCEPTANCE CRITERIA MET: ≥3pp accuracy improvement!")
        else:
            print(f"Progress: {improvement/0.03*100:.1f}% towards 3pp target")
        
        elapsed_time = time.time() - start_time
        print(f"\nTotal execution time: {elapsed_time:.1f} seconds")
        
        if quick_test and elapsed_time < 180:  # 3 minutes
            print("QUICK TEST PASSED: Completed in under 3 minutes")
        
        return True
        
    except Exception as e:
        print(f"Pipeline failed: {e}")
        return False


def main():
    """Main function with command line argument parsing"""
    parser = argparse.ArgumentParser(description='Twitter Sentiment Analytics Pipeline')
    parser.add_argument('--quick-test', action='store_true', 
                       help='Run pipeline on subset of data for quick testing')
    parser.add_argument('--max-samples', type=int, default=1000,
                       help='Maximum samples to use in quick test mode')
    
    args = parser.parse_args()
    
    # Create necessary directories
    os.makedirs('datasets', exist_ok=True)
    os.makedirs('features', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    os.makedirs('reports/eda', exist_ok=True)
    os.makedirs('reports/evaluation', exist_ok=True)
    
    success = run_pipeline(quick_test=args.quick_test, max_samples=args.max_samples)
    
    if success:
        print("\nPipeline completed successfully!")
        sys.exit(0)
    else:
        print("\nPipeline failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
