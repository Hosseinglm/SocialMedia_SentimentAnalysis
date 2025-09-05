"""
URL configuration for Sentiment API
"""
from django.urls import path, include
from rest_framework.routers import DefaultRouter
from . import views

# Create router and register viewsets
router = DefaultRouter()
router.register(r'brands', views.BrandViewSet)
router.register(r'tweets', views.SentimentTweetViewSet)

urlpatterns = [
    # Router URLs
    path('', include(router.urls)),

    # Custom API endpoints
    path('stats/', views.SentimentStatsAPIView.as_view(), name='sentiment-stats'),
    path('trends/', views.TrendAnalysisAPIView.as_view(), name='trend-analysis'),
    path('crisis/', views.CrisisDetectionAPIView.as_view(), name='crisis-detection'),
]
