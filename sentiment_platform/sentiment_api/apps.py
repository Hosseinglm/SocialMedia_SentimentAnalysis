from django.apps import AppConfig


class SentimentApiConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'sentiment_api'
    verbose_name = 'Sentiment Analysis API'
