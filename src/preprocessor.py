"""
Text preprocessing and cleaning module
"""

import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


class TextPreprocessor:
    """Handles text cleaning and preprocessing for Twitter data"""
    
    def __init__(self):
        # Download required NLTK data
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords', quiet=True)
        
        try:
            nltk.data.find('corpora/wordnet')
        except LookupError:
            nltk.download('wordnet', quiet=True)
            
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
    def clean_dataset(self, df):
        """
        Clean entire dataset
        
        Args:
            df: DataFrame with tweet data
            
        Returns:
            DataFrame: Cleaned dataset with additional features
        """
        print("   Extracting additional features...")
        clean_df = df.copy()
        
        # Extract features before cleaning
        feature_data = clean_df['tweet_text'].apply(self._extract_features)
        clean_df[['tweet_len', 'num_hashtags', 'num_mentions']] = pd.DataFrame(
            feature_data.tolist(), index=clean_df.index
        )
        
        # Clean text
        print("   Cleaning tweet text...")
        clean_df['clean_tweet_basic'] = clean_df['tweet_text'].apply(self._clean_text)
        
        # Lemmatize
        print("   Lemmatizing text...")
        clean_df['clean_tweet'] = clean_df['clean_tweet_basic'].apply(self._lemmatize_text)
        
        # Remove empty tweets after cleaning
        initial_count = len(clean_df)
        clean_df = clean_df[clean_df['clean_tweet'].str.strip() != '']
        final_count = len(clean_df)
        
        removed_count = initial_count - final_count
        if removed_count > 0:
            print(f"   Removed {removed_count} empty tweets after cleaning")
        
        return clean_df
    
    def _clean_text(self, text):
        """Clean individual tweet text"""
        if pd.isna(text):
            return ""
        
        text = str(text).lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove @mentions but keep the word after @
        text = re.sub(r'@\w+', '', text)
        
        # Remove hashtags but keep the word
        text = re.sub(r'#(\w+)', r'\1', text)
        
        # Expand contractions
        contractions = {
            "n't": " not", "'re": " are", "'ve": " have", "'ll": " will",
            "'d": " would", "'m": " am", "'s": " is"
        }
        for contraction, expansion in contractions.items():
            text = text.replace(contraction, expansion)
        
        # Remove digits and punctuation
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def _extract_features(self, text):
        """Extract features from original tweet text"""
        if pd.isna(text):
            return 0, 0, 0
        
        text = str(text)
        tweet_len = len(text)
        num_hashtags = len(re.findall(r'#\w+', text))
        num_mentions = len(re.findall(r'@\w+', text))
        
        return tweet_len, num_hashtags, num_mentions
    
    def _lemmatize_text(self, text):
        """Lemmatize text and remove stopwords"""
        if pd.isna(text) or text == "":
            return ""
        
        words = text.split()
        lemmatized = [
            self.lemmatizer.lemmatize(word) 
            for word in words 
            if word not in self.stop_words and len(word) > 2
        ]
        
        return ' '.join(lemmatized)
    
    def save_clean_data(self, df, output_path):
        """Save cleaned data to CSV"""
        df.to_csv(output_path, index=False)
        print(f"   Clean data saved to {output_path}")
    
    def get_cleaning_stats(self, original_df, cleaned_df):
        """Get statistics about the cleaning process"""
        stats = {
            'original_samples': len(original_df),
            'cleaned_samples': len(cleaned_df),
            'removed_samples': len(original_df) - len(cleaned_df),
            'avg_original_length': original_df['tweet_text'].str.len().mean(),
            'avg_cleaned_length': cleaned_df['clean_tweet'].str.len().mean(),
            'empty_after_cleaning': (cleaned_df['clean_tweet'].str.strip() == '').sum()
        }
        
        return stats
