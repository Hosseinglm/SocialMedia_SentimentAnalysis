"""
Django REST Framework views for Sentiment Intelligence Platform
"""
from rest_framework import viewsets, status
from rest_framework.decorators import action
from rest_framework.response import Response
from rest_framework.views import APIView
from django.db.models import Count, Q, Avg
from django.utils import timezone
from datetime import datetime, timedelta
import json

from .models import Brand, SentimentTweet, SentimentAnalysis, AlertRule, Alert
from .serializers import (
    BrandSerializer, SentimentTweetSerializer, SentimentTweetListSerializer,
    SentimentAnalysisSerializer, AlertRuleSerializer, AlertSerializer,
    SentimentStatsSerializer, BrandComparisonSerializer, TrendAnalysisSerializer,
    KeywordAnalysisSerializer, CrisisDetectionSerializer
)


class BrandViewSet(viewsets.ModelViewSet):
    """ViewSet for Brand model"""
    queryset = Brand.objects.all()
    serializer_class = BrandSerializer
    
    @action(detail=True, methods=['get'])
    def sentiment_breakdown(self, request, pk=None):
        """Get detailed sentiment breakdown for a brand"""
        brand = self.get_object()
        tweets = brand.tweets.all()
        
        # Date filtering
        date_from = request.query_params.get('date_from')
        date_to = request.query_params.get('date_to')
        
        if date_from:
            tweets = tweets.filter(date__gte=date_from)
        if date_to:
            tweets = tweets.filter(date__lte=date_to)
        
        # Calculate metrics
        total = tweets.count()
        sentiment_counts = tweets.values('sentiment_label').annotate(count=Count('id'))
        
        breakdown = {
            'brand': brand.name,
            'total_mentions': total,
            'date_range': {
                'from': date_from,
                'to': date_to
            },
            'sentiment_distribution': {},
            'engagement_metrics': {
                'avg_length': tweets.aggregate(avg_len=Avg('tweet_len'))['avg_len'] or 0,
                'avg_hashtags': tweets.aggregate(avg_hash=Avg('num_hashtags'))['avg_hash'] or 0,
                'avg_mentions': tweets.aggregate(avg_ment=Avg('num_mentions'))['avg_ment'] or 0,
            }
        }
        
        for item in sentiment_counts:
            sentiment = item['sentiment_label']
            count = item['count']
            breakdown['sentiment_distribution'][sentiment] = {
                'count': count,
                'percentage': round((count / total) * 100, 1) if total > 0 else 0
            }
        
        return Response(breakdown)
    
    @action(detail=False, methods=['get'])
    def comparison(self, request):
        """Compare sentiment across multiple brands"""
        brand_names = request.query_params.getlist('brands')
        if not brand_names:
            return Response({'error': 'Please provide brand names'}, status=400)
        
        brands = Brand.objects.filter(name__in=brand_names)
        comparison_data = []
        
        for brand in brands:
            tweets = brand.tweets.all()
            total = tweets.count()
            
            if total > 0:
                sentiment_counts = tweets.values('sentiment_label').annotate(count=Count('id'))
                sentiment_breakdown = {}
                
                for item in sentiment_counts:
                    sentiment = item['sentiment_label']
                    count = item['count']
                    sentiment_breakdown[sentiment] = {
                        'count': count,
                        'percentage': round((count / total) * 100, 1)
                    }
                
                # Calculate sentiment score (positive - negative)
                pos_pct = sentiment_breakdown.get('Positive', {}).get('percentage', 0)
                neg_pct = sentiment_breakdown.get('Negative', {}).get('percentage', 0)
                sentiment_score = pos_pct - neg_pct
                
                comparison_data.append({
                    'brand_name': brand.name,
                    'total_mentions': total,
                    'sentiment_breakdown': sentiment_breakdown,
                    'sentiment_score': sentiment_score,
                    'engagement_avg': tweets.aggregate(
                        avg_eng=Avg('num_hashtags') + Avg('num_mentions')
                    )['avg_eng'] or 0
                })
        
        return Response({
            'comparison': comparison_data,
            'generated_at': timezone.now()
        })


class SentimentTweetViewSet(viewsets.ReadOnlyModelViewSet):
    """ViewSet for SentimentTweet model (read-only)"""
    queryset = SentimentTweet.objects.all()
    serializer_class = SentimentTweetSerializer
    
    def get_serializer_class(self):
        if self.action == 'list':
            return SentimentTweetListSerializer
        return SentimentTweetSerializer
    
    def get_queryset(self):
        queryset = SentimentTweet.objects.select_related('brand')
        
        # Filtering
        brand = self.request.query_params.get('brand')
        sentiment = self.request.query_params.get('sentiment')
        date_from = self.request.query_params.get('date_from')
        date_to = self.request.query_params.get('date_to')
        source = self.request.query_params.get('source')
        
        if brand:
            queryset = queryset.filter(brand__name__icontains=brand)
        if sentiment:
            queryset = queryset.filter(sentiment_label=sentiment)
        if date_from:
            queryset = queryset.filter(date__gte=date_from)
        if date_to:
            queryset = queryset.filter(date__lte=date_to)
        if source:
            queryset = queryset.filter(source=source)
        
        return queryset


class SentimentStatsAPIView(APIView):
    """API view for overall sentiment statistics"""
    
    def get(self, request):
        """Get comprehensive sentiment statistics"""
        tweets = SentimentTweet.objects.all()
        
        # Date filtering
        date_from = request.query_params.get('date_from')
        date_to = request.query_params.get('date_to')
        
        if date_from:
            tweets = tweets.filter(date__gte=date_from)
        if date_to:
            tweets = tweets.filter(date__lte=date_to)
        
        total_tweets = tweets.count()
        
        # Sentiment distribution
        sentiment_counts = tweets.values('sentiment_label').annotate(count=Count('id'))
        sentiment_distribution = {}
        
        for item in sentiment_counts:
            sentiment = item['sentiment_label']
            count = item['count']
            sentiment_distribution[sentiment] = {
                'count': count,
                'percentage': round((count / total_tweets) * 100, 1) if total_tweets > 0 else 0
            }
        
        # Top brands by mention count
        top_brands = list(tweets.values('brand__name').annotate(
            count=Count('id')
        ).order_by('-count')[:10])
        
        # Date range
        date_range = {
            'earliest': tweets.order_by('date').first().date if tweets.exists() else None,
            'latest': tweets.order_by('-date').first().date if tweets.exists() else None
        }
        
        # Engagement statistics
        engagement_stats = {
            'avg_tweet_length': tweets.aggregate(avg_len=Avg('tweet_len'))['avg_len'] or 0,
            'avg_hashtags': tweets.aggregate(avg_hash=Avg('num_hashtags'))['avg_hash'] or 0,
            'avg_mentions': tweets.aggregate(avg_ment=Avg('num_mentions'))['avg_ment'] or 0,
            'high_engagement_count': tweets.filter(
                Q(num_hashtags__gt=2) | Q(num_mentions__gt=2)
            ).count()
        }
        
        stats_data = {
            'total_tweets': total_tweets,
            'sentiment_distribution': sentiment_distribution,
            'top_brands': top_brands,
            'date_range': date_range,
            'engagement_stats': engagement_stats
        }
        
        serializer = SentimentStatsSerializer(stats_data)
        return Response(serializer.data)


class TrendAnalysisAPIView(APIView):
    """API view for trend analysis"""
    
    def get(self, request):
        """Get sentiment trends over time"""
        period = request.query_params.get('period', 'daily')  # daily, weekly, monthly
        brand_name = request.query_params.get('brand')
        
        tweets = SentimentTweet.objects.all()
        
        if brand_name:
            tweets = tweets.filter(brand__name__icontains=brand_name)
        
        # Group by time period and sentiment
        if period == 'daily':
            date_trunc = 'day'
        elif period == 'weekly':
            date_trunc = 'week'
        else:
            date_trunc = 'month'
        
        # This is a simplified version - in production, you'd use proper date truncation
        trends = tweets.extra(
            select={'period': f"DATE_TRUNC('{date_trunc}', date)"}
        ).values('period', 'sentiment_label').annotate(count=Count('id')).order_by('period')
        
        trend_data = {
            'period': period,
            'brand': brand_name,
            'sentiment_trends': list(trends),
            'key_insights': [
                'Sentiment analysis trends calculated',
                f'Data grouped by {period}',
                f'Total data points: {tweets.count()}'
            ]
        }
        
        serializer = TrendAnalysisSerializer(trend_data)
        return Response(serializer.data)


class CrisisDetectionAPIView(APIView):
    """API view for crisis detection"""
    
    def get(self, request):
        """Detect potential sentiment crises"""
        # Look for negative sentiment spikes in last 24 hours
        last_24h = timezone.now() - timedelta(hours=24)
        recent_tweets = SentimentTweet.objects.filter(date__gte=last_24h)
        
        total_recent = recent_tweets.count()
        negative_recent = recent_tweets.filter(sentiment_label='Negative').count()
        
        if total_recent > 0:
            negative_ratio = negative_recent / total_recent
        else:
            negative_ratio = 0
        
        # Determine risk level
        if negative_ratio > 0.6:
            risk_level = 'CRITICAL'
            risk_score = 0.9
        elif negative_ratio > 0.4:
            risk_level = 'HIGH'
            risk_score = 0.7
        elif negative_ratio > 0.3:
            risk_level = 'MEDIUM'
            risk_score = 0.5
        else:
            risk_level = 'LOW'
            risk_score = 0.2
        
        # Find affected brands
        affected_brands = list(recent_tweets.filter(
            sentiment_label='Negative'
        ).values('brand__name').annotate(
            neg_count=Count('id')
        ).order_by('-neg_count')[:5])
        
        crisis_data = {
            'risk_level': risk_level,
            'risk_score': risk_score,
            'affected_brands': affected_brands,
            'warning_signals': [
                f'{negative_ratio:.1%} negative sentiment in last 24h',
                f'{negative_recent} negative mentions detected'
            ],
            'recommended_actions': [
                'Monitor sentiment closely',
                'Prepare crisis communication plan',
                'Analyze root causes of negative sentiment'
            ],
            'monitoring_suggestions': [
                'Set up real-time alerts',
                'Increase monitoring frequency',
                'Track competitor sentiment'
            ]
        }
        
        serializer = CrisisDetectionSerializer(crisis_data)
        return Response(serializer.data)
