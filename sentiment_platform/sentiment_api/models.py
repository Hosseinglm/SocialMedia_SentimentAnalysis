"""
Django models for Sentiment Intelligence Platform
"""
from django.db import models
from django.utils import timezone
from django.core.validators import MinValueValidator, MaxValueValidator


class Brand(models.Model):
    """Brand model for organizing sentiment data"""
    name = models.CharField(max_length=100, unique=True)
    display_name = models.CharField(max_length=100, blank=True)
    category = models.CharField(max_length=50, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        ordering = ['name']
    
    def __str__(self):
        return self.display_name or self.name
    
    @property
    def total_mentions(self):
        return self.tweets.count()
    
    @property
    def sentiment_distribution(self):
        """Get sentiment distribution for this brand"""
        tweets = self.tweets.all()
        total = tweets.count()
        if total == 0:
            return {}
        
        distribution = {}
        for sentiment in ['Positive', 'Negative', 'Neutral', 'Irrelevant']:
            count = tweets.filter(sentiment_label=sentiment).count()
            distribution[sentiment] = {
                'count': count,
                'percentage': round((count / total) * 100, 1)
            }
        return distribution


class SentimentTweet(models.Model):
    """Main model for storing tweet sentiment data"""
    
    SENTIMENT_CHOICES = [
        ('Positive', 'Positive'),
        ('Negative', 'Negative'),
        ('Neutral', 'Neutral'),
        ('Irrelevant', 'Irrelevant'),
    ]
    
    SOURCE_CHOICES = [
        ('train', 'Training'),
        ('validation', 'Validation'),
    ]
    
    # Core fields
    tweet_id = models.CharField(max_length=50, unique=True)
    brand = models.ForeignKey(Brand, on_delete=models.CASCADE, related_name='tweets')
    sentiment_label = models.CharField(max_length=20, choices=SENTIMENT_CHOICES)
    tweet_text = models.TextField()
    clean_tweet = models.TextField(blank=True)
    
    # Metadata
    author = models.CharField(max_length=100, blank=True)
    date = models.DateTimeField()
    source = models.CharField(max_length=20, choices=SOURCE_CHOICES)
    
    # Engagement metrics
    tweet_len = models.PositiveIntegerField(validators=[MinValueValidator(0)])
    num_hashtags = models.PositiveIntegerField(default=0, validators=[MinValueValidator(0)])
    num_mentions = models.PositiveIntegerField(default=0, validators=[MinValueValidator(0)])
    
    # System fields
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        ordering = ['-date']
        indexes = [
            models.Index(fields=['brand', 'sentiment_label']),
            models.Index(fields=['date']),
            models.Index(fields=['sentiment_label']),
            models.Index(fields=['source']),
        ]
    
    def __str__(self):
        return f"{self.brand.name} - {self.sentiment_label} - {self.tweet_id}"
    
    @property
    def engagement_score(self):
        """Calculate engagement score based on hashtags and mentions"""
        return self.num_hashtags + self.num_mentions
    
    @property
    def is_high_engagement(self):
        """Check if tweet has high engagement (>2 hashtags or mentions)"""
        return self.engagement_score > 2


class SentimentAnalysis(models.Model):
    """Model for storing analysis results and insights"""
    
    ANALYSIS_TYPES = [
        ('brand_overview', 'Brand Overview'),
        ('trend_analysis', 'Trend Analysis'),
        ('comparative', 'Comparative Analysis'),
        ('crisis_detection', 'Crisis Detection'),
    ]
    
    analysis_type = models.CharField(max_length=30, choices=ANALYSIS_TYPES)
    title = models.CharField(max_length=200)
    description = models.TextField(blank=True)
    
    # Analysis parameters
    brands = models.ManyToManyField(Brand, blank=True)
    date_from = models.DateTimeField(null=True, blank=True)
    date_to = models.DateTimeField(null=True, blank=True)
    
    # Results
    results = models.JSONField(default=dict)
    insights = models.JSONField(default=list)
    recommendations = models.JSONField(default=list)
    
    # Metadata
    created_at = models.DateTimeField(auto_now_add=True)
    created_by = models.CharField(max_length=100, default='system')
    
    class Meta:
        ordering = ['-created_at']
    
    def __str__(self):
        return f"{self.get_analysis_type_display()} - {self.title}"


class AlertRule(models.Model):
    """Model for sentiment monitoring alerts"""
    
    ALERT_TYPES = [
        ('sentiment_threshold', 'Sentiment Threshold'),
        ('volume_spike', 'Volume Spike'),
        ('negative_trend', 'Negative Trend'),
    ]
    
    name = models.CharField(max_length=100)
    alert_type = models.CharField(max_length=30, choices=ALERT_TYPES)
    brand = models.ForeignKey(Brand, on_delete=models.CASCADE, related_name='alert_rules')
    
    # Threshold settings
    threshold_value = models.FloatField()
    time_window_hours = models.PositiveIntegerField(default=24)
    
    # Status
    is_active = models.BooleanField(default=True)
    last_triggered = models.DateTimeField(null=True, blank=True)
    
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        ordering = ['brand', 'name']
    
    def __str__(self):
        return f"{self.brand.name} - {self.name}"


class Alert(models.Model):
    """Model for storing triggered alerts"""
    
    SEVERITY_LEVELS = [
        ('low', 'Low'),
        ('medium', 'Medium'),
        ('high', 'High'),
        ('critical', 'Critical'),
    ]
    
    STATUS_CHOICES = [
        ('open', 'Open'),
        ('acknowledged', 'Acknowledged'),
        ('resolved', 'Resolved'),
    ]
    
    rule = models.ForeignKey(AlertRule, on_delete=models.CASCADE, related_name='alerts')
    severity = models.CharField(max_length=20, choices=SEVERITY_LEVELS)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='open')
    
    title = models.CharField(max_length=200)
    message = models.TextField()
    data = models.JSONField(default=dict)
    
    triggered_at = models.DateTimeField(auto_now_add=True)
    acknowledged_at = models.DateTimeField(null=True, blank=True)
    resolved_at = models.DateTimeField(null=True, blank=True)
    
    class Meta:
        ordering = ['-triggered_at']
    
    def __str__(self):
        return f"{self.rule.brand.name} - {self.title}"
