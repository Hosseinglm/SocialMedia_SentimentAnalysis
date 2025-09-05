#!/bin/bash
cd /var/www/SocialMedia_SentimentAnalysis/sentiment_platform
source venv/bin/activate
python manage.py runserver 0.0.0.0:8006
