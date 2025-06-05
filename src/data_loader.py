"""
Data loading and unification module
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta


class DataLoader:
    """Handles loading and unifying Twitter data from multiple sources"""
    
    def __init__(self):
        self.raw_data = None
    
    def load_and_unify_data(self, train_path, val_path):
        """
        Load training and validation CSV files and unify schema
        
        Args:
            train_path: Path to training CSV
            val_path: Path to validation CSV
            
        Returns:
            pandas.DataFrame: Unified dataset with standardized schema
        """
        print(f"   Loading training data from {train_path}")
        train_df = pd.read_csv(train_path, names=['tweet_id', 'brand', 'sentiment_label', 'tweet_text'])
        
        print(f"   Loading validation data from {val_path}")
        val_df = pd.read_csv(val_path, names=['tweet_id', 'brand', 'sentiment_label', 'tweet_text'])
        
        # Add source tracking
        train_df['source'] = 'train'
        val_df['source'] = 'validation'
        
        # Combine datasets
        combined_df = pd.concat([train_df, val_df], ignore_index=True)
        
        # Create unified schema
        unified_df = self._create_unified_schema(combined_df)
        
        print(f"   Combined dataset shape: {unified_df.shape}")
        print(f"   Sentiment distribution: {unified_df['sentiment_label'].value_counts().to_dict()}")
        
        self.raw_data = unified_df
        return unified_df
    
    def _create_unified_schema(self, df):
        """Create unified schema with required columns"""
        # Add author column (using brand as proxy)
        df['author'] = df['brand']
        
        # Add synthetic date column (for demo purposes)
        base_date = datetime(2023, 1, 1)
        date_range = 365  # days
        df['date'] = [
            base_date + timedelta(days=np.random.randint(0, date_range))
            for _ in range(len(df))
        ]
        
        # Reorder columns to match required schema
        column_order = ['tweet_id', 'author', 'date', 'tweet_text', 'sentiment_label', 'brand', 'source']
        df = df[column_order]
        
        return df
    
    def save_raw_data(self, df, output_path):
        """Save raw unified data to CSV"""
        df.to_csv(output_path, index=False)
        print(f"   Raw data saved to {output_path}")
    
    def get_data_summary(self):
        """Get summary statistics of loaded data"""
        if self.raw_data is None:
            return "No data loaded"
        
        summary = {
            'total_samples': len(self.raw_data),
            'sentiment_distribution': self.raw_data['sentiment_label'].value_counts().to_dict(),
            'unique_brands': self.raw_data['brand'].nunique(),
            'date_range': {
                'start': self.raw_data['date'].min(),
                'end': self.raw_data['date'].max()
            },
            'source_distribution': self.raw_data['source'].value_counts().to_dict()
        }
        
        return summary
