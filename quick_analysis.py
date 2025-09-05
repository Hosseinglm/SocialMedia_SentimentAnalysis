#!/usr/bin/env python3
"""
Quick Sentiment Analysis for Augment Intelligence Agent
"""

import pandas as pd
import numpy as np
from collections import Counter

def analyze_sentiment_data():
    """Analyze the sentiment dataset and provide executive insights"""
    
    # Load the clean dataset
    df = pd.read_csv('datasets/clean_tweets.csv')
    
    print('=' * 60)
    print('🎯 AUGMENT SENTIMENT INTELLIGENCE REPORT')
    print('=' * 60)
    print(f'📊 Dataset Overview: {len(df):,} total social media posts analyzed')
    print(f'📅 Date Range: {df["date"].min()} to {df["date"].max()}')
    print()
    
    # Sentiment Distribution Analysis
    sentiment_counts = df['sentiment_label'].value_counts()
    total_posts = len(df)
    
    print('📈 SENTIMENT DISTRIBUTION ANALYSIS:')
    print('-' * 40)
    for sentiment, count in sentiment_counts.items():
        percentage = (count / total_posts) * 100
        print(f'  {sentiment:12s}: {count:6,} posts ({percentage:5.1f}%)')
    
    # Calculate sentiment health score
    positive_ratio = sentiment_counts.get('Positive', 0) / total_posts
    negative_ratio = sentiment_counts.get('Negative', 0) / total_posts
    sentiment_health = (positive_ratio - negative_ratio) * 100
    
    print(f'\n💡 Sentiment Health Score: {sentiment_health:+.1f}%')
    if sentiment_health > 10:
        print('   ✅ HEALTHY: Positive sentiment dominates')
    elif sentiment_health > -10:
        print('   ⚠️  BALANCED: Mixed sentiment landscape')
    else:
        print('   🚨 CRITICAL: Negative sentiment dominates')
    print()
    
    # Brand Analysis
    brand_counts = df['brand'].value_counts()
    print(f'🏢 BRAND LANDSCAPE: {len(brand_counts)} brands monitored')
    print('-' * 40)
    print('Top 10 Most Mentioned Brands:')
    for i, (brand, count) in enumerate(brand_counts.head(10).items(), 1):
        percentage = (count / total_posts) * 100
        print(f'  {i:2d}. {brand:25s}: {count:5,} posts ({percentage:4.1f}%)')
    print()
    
    # Brand Sentiment Analysis
    print('🎯 BRAND SENTIMENT BREAKDOWN (Top 5 Brands):')
    print('-' * 50)
    for brand in brand_counts.head(5).index:
        brand_data = df[df['brand'] == brand]
        brand_sentiment = brand_data['sentiment_label'].value_counts()
        total_brand = len(brand_data)
        
        pos_pct = (brand_sentiment.get('Positive', 0) / total_brand) * 100
        neg_pct = (brand_sentiment.get('Negative', 0) / total_brand) * 100
        neu_pct = (brand_sentiment.get('Neutral', 0) / total_brand) * 100
        
        print(f'  {brand:20s}: Pos {pos_pct:4.1f}% | Neg {neg_pct:4.1f}% | Neu {neu_pct:4.1f}%')
    print()
    
    # Temporal Analysis
    df['date'] = pd.to_datetime(df['date'])
    df['month'] = df['date'].dt.to_period('M')
    monthly_counts = df['month'].value_counts().sort_index()
    
    print('📅 TEMPORAL ACTIVITY ANALYSIS:')
    print('-' * 40)
    for month, count in monthly_counts.items():
        percentage = (count / total_posts) * 100
        print(f'  {month}: {count:5,} posts ({percentage:4.1f}%)')
    print()
    
    # Engagement Metrics
    print('📊 ENGAGEMENT & CONTENT METRICS:')
    print('-' * 40)
    print(f'  Average post length: {df["tweet_len"].mean():.0f} characters')
    print(f'  Posts with hashtags: {(df["num_hashtags"] > 0).sum():,} ({(df["num_hashtags"] > 0).mean()*100:.1f}%)')
    print(f'  Posts with mentions: {(df["num_mentions"] > 0).sum():,} ({(df["num_mentions"] > 0).mean()*100:.1f}%)')
    print(f'  Avg hashtags per post: {df["num_hashtags"].mean():.2f}')
    print(f'  Avg mentions per post: {df["num_mentions"].mean():.2f}')
    print()
    
    # Data Quality Assessment
    print('🔍 DATA QUALITY ASSESSMENT:')
    print('-' * 40)
    train_count = (df['source'] == 'train').sum()
    val_count = (df['source'] == 'validation').sum()
    clean_coverage = (df['clean_tweet'].notna() & (df['clean_tweet'] != '')).sum()
    
    print(f'  Training data: {train_count:,} posts ({train_count/total_posts*100:.1f}%)')
    print(f'  Validation data: {val_count:,} posts ({val_count/total_posts*100:.1f}%)')
    print(f'  Clean text coverage: {clean_coverage:,} posts ({clean_coverage/total_posts*100:.1f}%)')
    print()
    
    # Key Insights & Recommendations
    print('💡 KEY INSIGHTS & RECOMMENDATIONS:')
    print('-' * 40)
    
    # Find most negative brand
    brand_neg_scores = {}
    for brand in brand_counts.head(10).index:
        brand_data = df[df['brand'] == brand]
        neg_ratio = (brand_data['sentiment_label'] == 'Negative').mean()
        brand_neg_scores[brand] = neg_ratio
    
    worst_brand = max(brand_neg_scores.items(), key=lambda x: x[1])
    best_brand = min(brand_neg_scores.items(), key=lambda x: x[1])
    
    print(f'  🚨 Highest negative sentiment: {worst_brand[0]} ({worst_brand[1]*100:.1f}%)')
    print(f'  ✅ Lowest negative sentiment: {best_brand[0]} ({best_brand[1]*100:.1f}%)')
    
    # Volume insights
    peak_month = monthly_counts.idxmax()
    low_month = monthly_counts.idxmin()
    print(f'  📈 Peak activity month: {peak_month} ({monthly_counts[peak_month]:,} posts)')
    print(f'  📉 Lowest activity month: {low_month} ({monthly_counts[low_month]:,} posts)')
    
    print()
    print('🎯 RECOMMENDED ACTIONS:')
    print('-' * 40)
    print(f'  1. Investigate negative sentiment drivers for {worst_brand[0]}')
    print(f'  2. Analyze {best_brand[0]} success strategies for replication')
    print(f'  3. Deep dive into {peak_month} activity spike causes')
    print(f'  4. Monitor sentiment trends across all {len(brand_counts)} brands')
    print()
    
    return {
        'total_posts': total_posts,
        'sentiment_distribution': sentiment_counts.to_dict(),
        'sentiment_health_score': sentiment_health,
        'top_brands': brand_counts.head(10).to_dict(),
        'worst_sentiment_brand': worst_brand,
        'best_sentiment_brand': best_brand,
        'peak_month': str(peak_month),
        'data_quality_score': clean_coverage/total_posts
    }

if __name__ == "__main__":
    results = analyze_sentiment_data()
