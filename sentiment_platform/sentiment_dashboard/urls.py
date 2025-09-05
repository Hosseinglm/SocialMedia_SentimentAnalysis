"""
URL configuration for Sentiment Dashboard
"""
from django.urls import path
from . import views

app_name = 'dashboard'

urlpatterns = [
    # Dashboard pages
    path('', views.dashboard_home, name='home'),
    path('brands/', views.brand_analysis, name='brand_analysis'),
    path('trends/', views.trend_analysis, name='trend_analysis'),
    path('alerts/', views.alerts_dashboard, name='alerts'),
    path('api-explorer/', views.api_explorer, name='api_explorer'),
    
    # AJAX endpoints
    path('data/', views.dashboard_data, name='dashboard_data'),
    path('brand/<int:brand_id>/data/', views.brand_detail_data, name='brand_detail_data'),
]
