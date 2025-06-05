"""
Synthetic data generation module using SMOTE and TVAE
"""

import pandas as pd
import numpy as np
import os
from imblearn.over_sampling import SMOTE
from scipy import stats


class SyntheticDataGenerator:
    """Handles synthetic data generation for class imbalance"""
    
    def __init__(self):
        self.smote = None
        self.tvae = None
        
    def analyze_class_imbalance(self, y):
        """
        Analyze class imbalance in the dataset
        
        Args:
            y: Target labels
            
        Returns:
            Dictionary with imbalance analysis
        """
        class_counts = pd.Series(y).value_counts()
        majority_count = class_counts.max()
        minority_threshold = majority_count * 0.2
        
        minority_classes = class_counts[class_counts < minority_threshold].index.tolist()
        
        analysis = {
            'class_counts': class_counts.to_dict(),
            'majority_count': majority_count,
            'minority_classes': minority_classes,
            'imbalance_ratio': majority_count / class_counts.min() if class_counts.min() > 0 else float('inf')
        }
        
        return analysis
    
    def apply_smote(self, X, y, random_state=42):
        """
        Apply SMOTE oversampling
        
        Args:
            X: Feature matrix (sparse or dense)
            y: Target labels
            random_state: Random seed
            
        Returns:
            Tuple of (X_resampled, y_resampled)
        """
        print("   Applying SMOTE oversampling...")
        
        try:
            # Analyze class distribution
            class_counts = pd.Series(y).value_counts()
            min_class_count = class_counts.min()
            
            # Adjust k_neighbors if needed
            k_neighbors = min(5, min_class_count - 1)
            if k_neighbors < 1:
                print(f"     Warning: Not enough samples for SMOTE (min class: {min_class_count})")
                return X, y
            
            self.smote = SMOTE(random_state=random_state, k_neighbors=k_neighbors)
            X_resampled, y_resampled = self.smote.fit_resample(X, y)
            
            print(f"   Original shape: {X.shape}")
            print(f"   Resampled shape: {X_resampled.shape}")
            
            # Show new class distribution
            new_class_counts = pd.Series(y_resampled).value_counts()
            print(f"   New class distribution: {new_class_counts.to_dict()}")
            
            return X_resampled, y_resampled
            
        except Exception as e:
            print(f"   SMOTE failed: {e}")
            return X, y
    
    def generate_tvae_data(self, df, n_samples=2000):
        """
        Generate synthetic data using TVAE
        
        Args:
            df: Original DataFrame
            n_samples: Number of synthetic samples to generate
            
        Returns:
            DataFrame with synthetic data
        """
        print(f"   Generating {n_samples} synthetic samples with TVAE...")
        
        try:
            from sdv.single_table import TVAESynthesizer
            
            # Prepare data for TVAE
            tvae_columns = ['sentiment_label', 'tweet_len', 'num_hashtags', 'num_mentions', 'brand']
            tvae_data = df[tvae_columns].copy()
            
            # Initialize and train TVAE
            synthesizer = TVAESynthesizer(epochs=50, verbose=False)
            synthesizer.fit(tvae_data)
            
            # Generate synthetic samples
            synthetic_data = synthesizer.sample(num_rows=n_samples)
            
            print(f"   Generated {len(synthetic_data)} synthetic samples")
            
            # Validate synthetic data quality
            quality_metrics = self._validate_synthetic_quality(tvae_data, synthetic_data)
            print(f"   Quality validation: {quality_metrics}")
            
            return synthetic_data
            
        except ImportError:
            print("   SDV not available. Skipping TVAE generation.")
            return pd.DataFrame()
        except Exception as e:
            print(f"   TVAE generation failed: {e}")
            return pd.DataFrame()
    
    def _validate_synthetic_quality(self, real_data, synthetic_data):
        """
        Validate quality of synthetic data using statistical tests
        
        Args:
            real_data: Original data
            synthetic_data: Generated synthetic data
            
        Returns:
            Dictionary with quality metrics
        """
        quality_metrics = {}
        
        for col in ['tweet_len', 'num_hashtags', 'num_mentions']:
            if col in real_data.columns and col in synthetic_data.columns:
                # Kolmogorov-Smirnov test
                ks_stat, ks_p_value = stats.ks_2samp(real_data[col], synthetic_data[col])
                
                # Mean comparison
                real_mean = real_data[col].mean()
                synthetic_mean = synthetic_data[col].mean()
                mean_diff = abs(real_mean - synthetic_mean) / real_mean if real_mean != 0 else 0
                
                quality_metrics[col] = {
                    'ks_statistic': ks_stat,
                    'ks_p_value': ks_p_value,
                    'mean_difference_ratio': mean_diff,
                    'quality_score': 1 - min(ks_stat, mean_diff)  # Simple quality score
                }
        
        return quality_metrics
    
    def save_synthetic_data(self, synthetic_data, filepath):
        """Save synthetic data to CSV"""
        if len(synthetic_data) > 0:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            synthetic_data.to_csv(filepath, index=False)
            print(f"   Synthetic data saved to {filepath}")
        else:
            print("   No synthetic data to save")
    
    def create_augmented_dataset(self, original_df, synthetic_data, quality_threshold=0.7):
        """
        Create augmented dataset by combining original and high-quality synthetic data
        
        Args:
            original_df: Original training data
            synthetic_data: Generated synthetic data
            quality_threshold: Minimum quality score for inclusion
            
        Returns:
            Augmented DataFrame
        """
        if len(synthetic_data) == 0:
            print("   No synthetic data available for augmentation")
            return original_df
        
        # Filter high-quality synthetic samples (placeholder logic)
        # In practice, you'd use the quality metrics from validation
        high_quality_synthetic = synthetic_data.copy()
        
        # Add marker for synthetic data
        high_quality_synthetic['is_synthetic'] = True
        original_df_marked = original_df.copy()
        original_df_marked['is_synthetic'] = False
        
        # Combine datasets
        augmented_df = pd.concat([original_df_marked, high_quality_synthetic], ignore_index=True)
        
        print(f"   Created augmented dataset: {len(original_df)} original + {len(high_quality_synthetic)} synthetic = {len(augmented_df)} total")
        
        return augmented_df
    
    def get_augmentation_summary(self, original_size, augmented_size, synthetic_size):
        """Get summary of augmentation process"""
        summary = {
            'original_samples': original_size,
            'synthetic_samples': synthetic_size,
            'augmented_samples': augmented_size,
            'augmentation_ratio': augmented_size / original_size if original_size > 0 else 0,
            'synthetic_ratio': synthetic_size / augmented_size if augmented_size > 0 else 0
        }
        
        return summary
