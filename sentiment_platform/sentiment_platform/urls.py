"""
URL configuration for Sentiment Intelligence Platform.
"""
from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static
from django.http import JsonResponse
from django.views.generic import TemplateView

def api_root(request):
    """API root endpoint with available endpoints"""
    return JsonResponse({
        'message': 'Welcome to Augment Sentiment Intelligence Platform',
        'version': '1.0.0',
        'endpoints': {
            'admin': '/admin/',
            'api': '/api/',
            'dashboard': '/dashboard/',
            'sentiment_analysis': '/api/sentiment/',
            'brand_analysis': '/api/brands/',
            'trend_analysis': '/api/trends/',
            'reports': '/api/reports/',
        },
        'documentation': '/api/docs/',
        'status': 'active'
    })

urlpatterns = [
    # Admin interface
    path('admin/', admin.site.urls),
    
    # API root
    path('', api_root, name='api_root'),
    
    # API endpoints
    path('api/', include('sentiment_api.urls')),
    
    # Dashboard interface
    path('dashboard/', include('sentiment_dashboard.urls')),
    
    # Health check endpoint
    path('health/', lambda request: JsonResponse({'status': 'healthy', 'service': 'sentiment_intelligence'})),
]

# Serve static and media files in development
if settings.DEBUG:
    urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
