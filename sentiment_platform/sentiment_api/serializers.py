"""
Django REST Framework serializers for Sentiment Intelligence Platform
"""
from rest_framework import serializers
from django.db.models import Count, Q
from .models import Brand, SentimentTweet, SentimentAnalysis, AlertRule, Alert


class BrandSerializer(serializers.ModelSerializer):
    """Serializer for Brand model"""
    total_mentions = serializers.ReadOnlyField()
    sentiment_distribution = serializers.ReadOnlyField()
    
    class Meta:
        model = Brand
        fields = ['id', 'name', 'display_name', 'category', 'total_mentions', 
                 'sentiment_distribution', 'created_at']


class SentimentTweetSerializer(serializers.ModelSerializer):
    """Serializer for SentimentTweet model"""
    brand_name = serializers.CharField(source='brand.name', read_only=True)
    engagement_score = serializers.ReadOnlyField()
    is_high_engagement = serializers.ReadOnlyField()
    
    class Meta:
        model = SentimentTweet
        fields = ['id', 'tweet_id', 'brand', 'brand_name', 'sentiment_label', 
                 'tweet_text', 'clean_tweet', 'author', 'date', 'source',
                 'tweet_len', 'num_hashtags', 'num_mentions', 'engagement_score',
                 'is_high_engagement', 'created_at']


class SentimentTweetListSerializer(serializers.ModelSerializer):
    """Lightweight serializer for tweet lists"""
    brand_name = serializers.CharField(source='brand.name', read_only=True)
    
    class Meta:
        model = SentimentTweet
        fields = ['id', 'tweet_id', 'brand_name', 'sentiment_label', 
                 'date', 'tweet_len', 'num_hashtags', 'num_mentions']


class SentimentAnalysisSerializer(serializers.ModelSerializer):
    """Serializer for SentimentAnalysis model"""
    brands_data = BrandSerializer(source='brands', many=True, read_only=True)
    
    class Meta:
        model = SentimentAnalysis
        fields = ['id', 'analysis_type', 'title', 'description', 'brands', 
                 'brands_data', 'date_from', 'date_to', 'results', 'insights',
                 'recommendations', 'created_at', 'created_by']


class AlertRuleSerializer(serializers.ModelSerializer):
    """Serializer for AlertRule model"""
    brand_name = serializers.CharField(source='brand.name', read_only=True)
    
    class Meta:
        model = AlertRule
        fields = ['id', 'name', 'alert_type', 'brand', 'brand_name',
                 'threshold_value', 'time_window_hours', 'is_active',
                 'last_triggered', 'created_at']


class AlertSerializer(serializers.ModelSerializer):
    """Serializer for Alert model"""
    rule_name = serializers.CharField(source='rule.name', read_only=True)
    brand_name = serializers.CharField(source='rule.brand.name', read_only=True)
    
    class Meta:
        model = Alert
        fields = ['id', 'rule', 'rule_name', 'brand_name', 'severity', 'status',
                 'title', 'message', 'data', 'triggered_at', 'acknowledged_at',
                 'resolved_at']


class SentimentStatsSerializer(serializers.Serializer):
    """Serializer for sentiment statistics"""
    total_tweets = serializers.IntegerField()
    sentiment_distribution = serializers.DictField()
    top_brands = serializers.ListField()
    date_range = serializers.DictField()
    engagement_stats = serializers.DictField()


class BrandComparisonSerializer(serializers.Serializer):
    """Serializer for brand comparison data"""
    brand_name = serializers.CharField()
    total_mentions = serializers.IntegerField()
    sentiment_breakdown = serializers.DictField()
    sentiment_score = serializers.FloatField()
    engagement_avg = serializers.FloatField()
    trend_direction = serializers.CharField()


class TrendAnalysisSerializer(serializers.Serializer):
    """Serializer for trend analysis data"""
    period = serializers.CharField()
    sentiment_trends = serializers.DictField()
    volume_trends = serializers.DictField()
    key_insights = serializers.ListField()
    anomalies = serializers.ListField()


class KeywordAnalysisSerializer(serializers.Serializer):
    """Serializer for keyword analysis"""
    positive_keywords = serializers.ListField()
    negative_keywords = serializers.ListField()
    trending_hashtags = serializers.ListField()
    mention_patterns = serializers.DictField()


class CrisisDetectionSerializer(serializers.Serializer):
    """Serializer for crisis detection results"""
    risk_level = serializers.CharField()
    risk_score = serializers.FloatField()
    affected_brands = serializers.ListField()
    warning_signals = serializers.ListField()
    recommended_actions = serializers.ListField()
    monitoring_suggestions = serializers.ListField()
