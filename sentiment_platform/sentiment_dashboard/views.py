"""
Dashboard views for Sentiment Intelligence Platform
"""
from django.shortcuts import render
from django.http import JsonResponse
from django.db.models import Count, Avg, Q
from django.utils import timezone
from datetime import timedelta
import json

from sentiment_api.models import Brand, SentimentTweet, Alert


def dashboard_home(request):
    """Main dashboard view"""
    # Get basic statistics
    total_tweets = SentimentTweet.objects.count()
    total_brands = Brand.objects.count()
    
    # Recent activity (last 7 days)
    week_ago = timezone.now() - timedelta(days=7)
    recent_tweets = SentimentTweet.objects.filter(date__gte=week_ago)
    
    # Sentiment distribution
    sentiment_counts = SentimentTweet.objects.values('sentiment_label').annotate(
        count=Count('id')
    )
    
    # Top brands by mentions
    top_brands = SentimentTweet.objects.values('brand__name').annotate(
        count=Count('id')
    ).order_by('-count')[:10]
    
    # Active alerts
    active_alerts = Alert.objects.filter(status='open').count()
    
    context = {
        'total_tweets': total_tweets,
        'total_brands': total_brands,
        'recent_tweets_count': recent_tweets.count(),
        'sentiment_distribution': list(sentiment_counts),
        'top_brands': list(top_brands),
        'active_alerts': active_alerts,
    }
    
    return render(request, 'dashboard/home.html', context)


def brand_analysis(request):
    """Brand analysis dashboard"""
    brands = Brand.objects.all()
    
    # Get brand with most mentions
    if brands.exists():
        top_brand = max(brands, key=lambda b: b.total_mentions)
        
        # Get sentiment breakdown for top brand
        top_brand_tweets = top_brand.tweets.all()
        sentiment_breakdown = top_brand_tweets.values('sentiment_label').annotate(
            count=Count('id')
        )
    else:
        top_brand = None
        sentiment_breakdown = []
    
    context = {
        'brands': brands,
        'top_brand': top_brand,
        'sentiment_breakdown': list(sentiment_breakdown),
    }
    
    return render(request, 'dashboard/brand_analysis.html', context)


def trend_analysis(request):
    """Trend analysis dashboard"""
    # Get recent trends (last 30 days)
    month_ago = timezone.now() - timedelta(days=30)
    recent_tweets = SentimentTweet.objects.filter(date__gte=month_ago)
    
    # Group by date and sentiment
    daily_trends = recent_tweets.extra(
        select={'day': 'DATE(date)'}
    ).values('day', 'sentiment_label').annotate(
        count=Count('id')
    ).order_by('day')
    
    context = {
        'daily_trends': list(daily_trends),
        'date_range': {
            'start': month_ago.date(),
            'end': timezone.now().date()
        }
    }
    
    return render(request, 'dashboard/trend_analysis.html', context)


def alerts_dashboard(request):
    """Alerts and monitoring dashboard"""
    all_alerts = Alert.objects.all().order_by('-triggered_at')
    alerts = all_alerts[:50]

    # Alert statistics
    alert_stats = {
        'total': all_alerts.count(),
        'open': all_alerts.filter(status='open').count(),
        'critical': all_alerts.filter(severity='critical').count(),
    }
    
    context = {
        'alerts': alerts,
        'alert_stats': alert_stats,
    }
    
    return render(request, 'dashboard/alerts.html', context)


def api_explorer(request):
    """API explorer interface"""
    api_endpoints = [
        {
            'name': 'Sentiment Statistics',
            'url': '/api/stats/',
            'method': 'GET',
            'description': 'Get overall sentiment statistics'
        },
        {
            'name': 'Brand Comparison',
            'url': '/api/brands/comparison/',
            'method': 'GET',
            'description': 'Compare sentiment across brands'
        },
        {
            'name': 'Trend Analysis',
            'url': '/api/trends/',
            'method': 'GET',
            'description': 'Get sentiment trends over time'
        },
        {
            'name': 'Crisis Detection',
            'url': '/api/crisis/',
            'method': 'GET',
            'description': 'Detect potential sentiment crises'
        },
    ]
    
    context = {
        'api_endpoints': api_endpoints,
    }
    
    return render(request, 'dashboard/api_explorer.html', context)


# AJAX endpoints for dashboard data
def dashboard_data(request):
    """AJAX endpoint for dashboard data"""
    data_type = request.GET.get('type', 'overview')
    
    if data_type == 'overview':
        # Overview statistics
        total_tweets = SentimentTweet.objects.count()
        sentiment_counts = SentimentTweet.objects.values('sentiment_label').annotate(
            count=Count('id')
        )
        
        data = {
            'total_tweets': total_tweets,
            'sentiment_distribution': list(sentiment_counts),
            'timestamp': timezone.now().isoformat()
        }
    
    elif data_type == 'brand_sentiment':
        brand_name = request.GET.get('brand')
        if brand_name:
            try:
                brand = Brand.objects.get(name=brand_name)
                sentiment_data = brand.tweets.values('sentiment_label').annotate(
                    count=Count('id')
                )
                data = {
                    'brand': brand_name,
                    'sentiment_data': list(sentiment_data)
                }
            except Brand.DoesNotExist:
                data = {'error': 'Brand not found'}
        else:
            data = {'error': 'Brand name required'}
    
    elif data_type == 'recent_activity':
        # Recent activity in last 24 hours
        day_ago = timezone.now() - timedelta(days=1)
        recent_tweets = SentimentTweet.objects.filter(date__gte=day_ago)
        
        hourly_activity = recent_tweets.extra(
            select={'hour': 'EXTRACT(hour FROM date)'}
        ).values('hour').annotate(count=Count('id')).order_by('hour')
        
        data = {
            'hourly_activity': list(hourly_activity),
            'total_recent': recent_tweets.count()
        }
    
    else:
        data = {'error': 'Invalid data type'}
    
    return JsonResponse(data)


def brand_detail_data(request, brand_id):
    """AJAX endpoint for detailed brand data"""
    try:
        brand = Brand.objects.get(id=brand_id)
        
        # Get sentiment breakdown
        sentiment_breakdown = brand.tweets.values('sentiment_label').annotate(
            count=Count('id')
        )
        
        # Get recent tweets
        recent_tweets = brand.tweets.order_by('-date')[:10]
        
        # Get engagement metrics
        engagement_stats = brand.tweets.aggregate(
            avg_length=Avg('tweet_len'),
            avg_hashtags=Avg('num_hashtags'),
            avg_mentions=Avg('num_mentions')
        )
        
        data = {
            'brand': {
                'id': brand.id,
                'name': brand.name,
                'display_name': brand.display_name,
                'total_mentions': brand.total_mentions
            },
            'sentiment_breakdown': list(sentiment_breakdown),
            'recent_tweets': [
                {
                    'id': tweet.id,
                    'text': tweet.tweet_text[:100] + '...' if len(tweet.tweet_text) > 100 else tweet.tweet_text,
                    'sentiment': tweet.sentiment_label,
                    'date': tweet.date.isoformat(),
                    'engagement_score': tweet.engagement_score
                }
                for tweet in recent_tweets
            ],
            'engagement_stats': engagement_stats
        }
        
    except Brand.DoesNotExist:
        data = {'error': 'Brand not found'}
    
    return JsonResponse(data)
