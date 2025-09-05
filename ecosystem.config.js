module.exports = {
  apps: [{
    name: 'Sentiment',
    script: 'manage.py',
    args: 'runserver 0.0.0.0:8006',
    cwd: '/var/www/SocialMedia_SentimentAnalysis/sentiment_platform',
    interpreter: '/var/www/SocialMedia_SentimentAnalysis/sentiment_platform/venv/bin/python',
    instances: 1,
    autorestart: true,
    watch: false,
    max_memory_restart: '1G',
    env: {
      DJANGO_SETTINGS_MODULE: 'sentiment_platform.settings'
    }
  }]
};
