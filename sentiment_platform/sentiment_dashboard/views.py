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


def algorithms_page(request):
    """Algorithms and methodologies page"""
    context = {
        'page_title': 'Sentiment Analysis Algorithms & Methodologies',
    }
    return render(request, 'dashboard/algorithms.html', context)


def recommendations_page(request):
    """Brand recommendations page"""
    # Get brands with their sentiment data for recommendations
    brands_data = Brand.objects.annotate(
        mentions_count=Count('tweets'),
        positive_count=Count('tweets', filter=Q(tweets__sentiment_label='Positive')),
        negative_count=Count('tweets', filter=Q(tweets__sentiment_label='Negative')),
        neutral_count=Count('tweets', filter=Q(tweets__sentiment_label='Neutral'))
    ).filter(mentions_count__gt=0).order_by('-mentions_count')

    # Calculate sentiment scores and generate recommendations
    brand_recommendations = []
    for brand_data in brands_data[:20]:  # Top 20 brands
        mentions_count = brand_data.mentions_count
        if mentions_count > 0:
            positive_ratio = brand_data.positive_count / mentions_count
            negative_ratio = brand_data.negative_count / mentions_count
            sentiment_score = (positive_ratio - negative_ratio) * 100

            # Generate recommendations based on sentiment patterns
            recommendations = generate_brand_recommendations(brand_data, sentiment_score, positive_ratio, negative_ratio, mentions_count)

            brand_recommendations.append({
                'brand': brand_data,
                'total_mentions': mentions_count,
                'sentiment_score': round(sentiment_score, 1),
                'positive_ratio': round(positive_ratio * 100, 1),
                'negative_ratio': round(negative_ratio * 100, 1),
                'recommendations': recommendations,
                'priority': get_recommendation_priority(sentiment_score, negative_ratio)
            })

    # Sort by priority (high priority first)
    brand_recommendations.sort(key=lambda x: x['priority'], reverse=True)

    context = {
        'brand_recommendations': brand_recommendations,
        'categories': Brand.objects.values_list('category', flat=True).distinct(),
    }

    return render(request, 'dashboard/recommendations.html', context)


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


def recommendations_data(request):
    """AJAX endpoint for recommendations data"""
    category = request.GET.get('category', '')
    brand_id = request.GET.get('brand_id', '')

    brands_query = Brand.objects.annotate(
        mentions_count=Count('tweets'),
        positive_count=Count('tweets', filter=Q(tweets__sentiment_label='Positive')),
        negative_count=Count('tweets', filter=Q(tweets__sentiment_label='Negative'))
    ).filter(mentions_count__gt=0)

    if category:
        brands_query = brands_query.filter(category=category)
    if brand_id:
        brands_query = brands_query.filter(id=brand_id)

    recommendations_data = []
    for brand in brands_query:
        mentions_count = brand.mentions_count
        if mentions_count > 0:
            positive_ratio = brand.positive_count / mentions_count
            negative_ratio = brand.negative_count / mentions_count
            sentiment_score = (positive_ratio - negative_ratio) * 100

            recommendations = generate_brand_recommendations(brand, sentiment_score, positive_ratio, negative_ratio, mentions_count)

            recommendations_data.append({
                'brand_id': brand.id,
                'brand_name': brand.name,
                'category': brand.category,
                'sentiment_score': round(sentiment_score, 1),
                'recommendations': recommendations
            })

    return JsonResponse({'recommendations': recommendations_data})


def generate_brand_recommendations(brand, sentiment_score, positive_ratio, negative_ratio, total_mentions=None):
    """Generate actionable recommendations for a brand"""
    recommendations = []

    # Get total mentions from parameter or calculate
    if total_mentions is None:
        total_mentions = getattr(brand, 'mentions_count', 0)
        if total_mentions == 0:
            total_mentions = brand.tweets.count()

    # Sentiment-based recommendations
    if sentiment_score < -20:
        recommendations.append({
            'type': 'critical',
            'title': 'Critical Sentiment Alert',
            'description': 'Immediate action required to address negative sentiment',
            'actions': [
                'Conduct crisis communication assessment',
                'Engage with negative feedback directly',
                'Review recent product/service changes',
                'Implement customer service improvements'
            ]
        })
    elif sentiment_score < 0:
        recommendations.append({
            'type': 'warning',
            'title': 'Negative Sentiment Trend',
            'description': 'Monitor and address declining sentiment',
            'actions': [
                'Analyze negative feedback patterns',
                'Improve customer communication',
                'Address common complaints',
                'Enhance product quality'
            ]
        })
    elif sentiment_score > 30:
        recommendations.append({
            'type': 'success',
            'title': 'Strong Positive Sentiment',
            'description': 'Leverage positive momentum for growth',
            'actions': [
                'Amplify positive customer stories',
                'Expand successful initiatives',
                'Engage with brand advocates',
                'Launch referral programs'
            ]
        })

    # Volume-based recommendations
    if total_mentions < 50:
        recommendations.append({
            'type': 'info',
            'title': 'Low Brand Visibility',
            'description': 'Increase brand awareness and engagement',
            'actions': [
                'Boost social media presence',
                'Launch awareness campaigns',
                'Engage with industry influencers',
                'Create shareable content'
            ]
        })

    # Category-specific recommendations
    if brand.category == 'Technology':
        recommendations.append({
            'type': 'info',
            'title': 'Tech Industry Focus',
            'description': 'Technology-specific optimization strategies',
            'actions': [
                'Highlight innovation and features',
                'Engage with tech communities',
                'Provide technical support excellence',
                'Share development updates'
            ]
        })
    elif brand.category == 'Gaming':
        recommendations.append({
            'type': 'info',
            'title': 'Gaming Community Engagement',
            'description': 'Gaming-focused community building',
            'actions': [
                'Engage with gaming communities',
                'Support esports and events',
                'Listen to player feedback',
                'Create gaming content'
            ]
        })

    return recommendations


def get_recommendation_priority(sentiment_score, negative_ratio):
    """Calculate priority score for recommendations"""
    priority = 0

    # Higher priority for negative sentiment
    if sentiment_score < -20:
        priority += 100
    elif sentiment_score < 0:
        priority += 50

    # Higher priority for high negative ratio
    if negative_ratio > 0.4:
        priority += 30
    elif negative_ratio > 0.3:
        priority += 15

    return priority
