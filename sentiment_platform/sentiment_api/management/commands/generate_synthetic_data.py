"""
Django management command to generate synthetic sentiment data
"""
import random
import csv
from datetime import datetime, timedelta
from django.core.management.base import BaseCommand
from django.utils import timezone
from django.db import transaction
from sentiment_api.models import Brand, SentimentTweet


class Command(BaseCommand):
    help = 'Generate synthetic sentiment data for testing and demonstration'
    
    # Extended brand list with categories
    BRANDS_DATA = {
        # Technology
        'Apple': 'Technology',
        'Google': 'Technology', 
        'Microsoft': 'Technology',
        'Amazon': 'Technology',
        'Meta': 'Technology',
        'Netflix': 'Technology',
        'Tesla': 'Technology',
        'Samsung': 'Technology',
        'Intel': 'Technology',
        'NVIDIA': 'Technology',
        
        # Gaming
        'PlayStation': 'Gaming',
        'Xbox': 'Gaming',
        'Nintendo': 'Gaming',
        'Steam': 'Gaming',
        'Epic Games': 'Gaming',
        'Activision': 'Gaming',
        'EA Sports': 'Gaming',
        'Ubisoft': 'Gaming',
        'Riot Games': 'Gaming',
        'Blizzard': 'Gaming',
        
        # Social Media
        'Twitter': 'Social Media',
        'Instagram': 'Social Media',
        'TikTok': 'Social Media',
        'YouTube': 'Social Media',
        'LinkedIn': 'Social Media',
        'Snapchat': 'Social Media',
        'Discord': 'Social Media',
        'Reddit': 'Social Media',
        'WhatsApp': 'Social Media',
        'Telegram': 'Social Media',
        
        # Telecommunications
        'Verizon': 'Telecommunications',
        'AT&T': 'Telecommunications',
        'T-Mobile': 'Telecommunications',
        'Sprint': 'Telecommunications',
        'Comcast': 'Telecommunications',
        'Spectrum': 'Telecommunications',
        'Cox': 'Telecommunications',
        'Xfinity': 'Telecommunications',
        
        # Automotive
        'BMW': 'Automotive',
        'Mercedes': 'Automotive',
        'Toyota': 'Automotive',
        'Ford': 'Automotive',
        'Honda': 'Automotive',
        'Audi': 'Automotive',
        'Volkswagen': 'Automotive',
        'Hyundai': 'Automotive',
        
        # Food & Beverage
        'McDonald\'s': 'Food & Beverage',
        'Starbucks': 'Food & Beverage',
        'Coca-Cola': 'Food & Beverage',
        'Pepsi': 'Food & Beverage',
        'KFC': 'Food & Beverage',
        'Subway': 'Food & Beverage',
        'Pizza Hut': 'Food & Beverage',
        'Domino\'s': 'Food & Beverage',
    }
    
    # Sample tweet templates by sentiment
    TWEET_TEMPLATES = {
        'Positive': [
            "Just got the new {brand} product and I'm absolutely loving it! #amazing",
            "{brand} customer service was fantastic today! Really impressed 👍",
            "Can't believe how good {brand} is getting. Keep up the great work!",
            "Shoutout to {brand} for making my day better! Excellent experience",
            "{brand} just announced something incredible! So excited 🎉",
            "Been using {brand} for years and they never disappoint. Quality stuff!",
            "The new {brand} update is exactly what I needed. Perfect timing!",
            "{brand} team really knows what they're doing. Impressed!",
        ],
        'Negative': [
            "Really disappointed with {brand} lately. Not what it used to be 😞",
            "{brand} customer service was terrible today. Very frustrated!",
            "Why is {brand} making these bad decisions? Come on...",
            "Had a horrible experience with {brand}. Won't be back.",
            "{brand} needs to fix their issues ASAP. This is unacceptable.",
            "Switching from {brand} after this disaster. Done with them.",
            "The new {brand} changes are awful. Who thought this was good?",
            "{brand} used to be great but now it's just disappointing.",
        ],
        'Neutral': [
            "Saw the {brand} announcement today. Interesting developments.",
            "{brand} released their quarterly report. Numbers look standard.",
            "Using {brand} for work. It does what it's supposed to do.",
            "The {brand} event is happening next week. Might check it out.",
            "{brand} stock moved today. Market seems mixed on the news.",
            "Comparing {brand} with competitors. Each has pros and cons.",
            "Reading about {brand}'s new strategy. Time will tell if it works.",
            "{brand} is trending today for some reason. Not sure why.",
        ],
        'Irrelevant': [
            "Going to the store later. Need to pick up some groceries.",
            "Beautiful weather today! Perfect for a walk in the park.",
            "Just finished watching a great movie. Highly recommend it!",
            "Coffee tastes extra good this morning. Great start to the day.",
            "Traffic is crazy today. Glad I left early for work.",
            "Weekend plans are coming together nicely. Should be fun!",
            "Learning something new every day. Knowledge is power!",
            "Time flies when you're having fun. Can't believe it's Friday!",
        ]
    }
    
    def add_arguments(self, parser):
        parser.add_argument(
            '--count',
            type=int,
            default=5000,
            help='Number of synthetic tweets to generate (default: 5000)'
        )
        parser.add_argument(
            '--days-back',
            type=int,
            default=90,
            help='Generate data for the last N days (default: 90)'
        )
        parser.add_argument(
            '--clear-existing',
            action='store_true',
            help='Clear existing data before generating synthetic data'
        )
    
    def handle(self, *args, **options):
        count = options['count']
        days_back = options['days_back']
        clear_existing = options['clear_existing']
        
        self.stdout.write(
            self.style.SUCCESS(f'Generating {count} synthetic sentiment tweets...')
        )
        
        # Clear existing data if requested
        if clear_existing:
            self.stdout.write('Clearing existing data...')
            SentimentTweet.objects.all().delete()
            Brand.objects.all().delete()
            self.stdout.write(self.style.SUCCESS('Existing data cleared'))
        
        # Create brands
        self.stdout.write('Creating brands...')
        brands = self.create_brands()
        
        # Generate synthetic tweets
        self.stdout.write('Generating synthetic tweets...')
        self.generate_synthetic_tweets(brands, count, days_back)
        
        self.stdout.write(
            self.style.SUCCESS(
                f'Successfully generated {count} synthetic tweets for {len(brands)} brands!'
            )
        )
    
    @transaction.atomic
    def create_brands(self):
        """Create brand objects"""
        brands = {}
        
        for brand_name, category in self.BRANDS_DATA.items():
            brand, created = Brand.objects.get_or_create(
                name=brand_name,
                defaults={
                    'display_name': brand_name,
                    'category': category
                }
            )
            brands[brand_name] = brand
            
            if created:
                self.stdout.write(f'  Created brand: {brand_name} ({category})')
        
        return brands
    
    def generate_synthetic_tweets(self, brands, count, days_back):
        """Generate synthetic tweet data"""
        sentiments = ['Positive', 'Negative', 'Neutral', 'Irrelevant']
        
        # Sentiment distribution (more realistic)
        sentiment_weights = {
            'Positive': 0.35,    # 35%
            'Negative': 0.25,    # 25%
            'Neutral': 0.30,     # 30%
            'Irrelevant': 0.10   # 10%
        }
        
        # Brand popularity weights (some brands get more mentions)
        popular_brands = ['Apple', 'Google', 'Microsoft', 'Amazon', 'Tesla', 'PlayStation', 'Xbox']
        
        tweets_to_create = []
        brand_names = list(brands.keys())
        
        for i in range(count):
            # Select brand (popular brands get higher weight)
            if random.random() < 0.4 and any(brand in brand_names for brand in popular_brands):
                available_popular = [b for b in popular_brands if b in brand_names]
                brand_name = random.choice(available_popular)
            else:
                brand_name = random.choice(brand_names)
            
            brand = brands[brand_name]
            
            # Select sentiment based on weights
            sentiment = random.choices(
                sentiments, 
                weights=[sentiment_weights[s] for s in sentiments]
            )[0]
            
            # Generate tweet text
            if sentiment == 'Irrelevant':
                tweet_text = random.choice(self.TWEET_TEMPLATES[sentiment])
            else:
                template = random.choice(self.TWEET_TEMPLATES[sentiment])
                tweet_text = template.format(brand=brand_name)
            
            # Generate random date within the specified range
            end_date = timezone.now()
            start_date = end_date - timedelta(days=days_back)
            random_date = start_date + timedelta(
                seconds=random.randint(0, int((end_date - start_date).total_seconds()))
            )
            
            # Generate other fields
            tweet_len = len(tweet_text)
            num_hashtags = tweet_text.count('#')
            num_mentions = tweet_text.count('@')
            
            # Generate synthetic tweet ID
            tweet_id = f"synthetic_{i+1}_{random.randint(100000, 999999)}"
            
            # Generate author
            authors = [
                'user123', 'social_media_fan', 'tech_enthusiast', 'gamer_pro',
                'daily_user', 'brand_watcher', 'consumer_voice', 'opinion_maker',
                'trend_follower', 'product_reviewer', 'casual_user', 'power_user'
            ]
            author = random.choice(authors) + str(random.randint(1, 9999))
            
            tweet = SentimentTweet(
                tweet_id=tweet_id,
                brand=brand,
                sentiment_label=sentiment,
                tweet_text=tweet_text,
                clean_tweet=tweet_text.lower(),  # Simple cleaning
                author=author,
                date=random_date,
                source='synthetic',
                tweet_len=tweet_len,
                num_hashtags=num_hashtags,
                num_mentions=num_mentions,
            )
            tweets_to_create.append(tweet)
            
            # Bulk create in batches
            if len(tweets_to_create) >= 1000:
                SentimentTweet.objects.bulk_create(tweets_to_create, ignore_conflicts=True)
                tweets_to_create = []
                self.stdout.write(f'  Generated {i+1}/{count} tweets...')
        
        # Create remaining tweets
        if tweets_to_create:
            SentimentTweet.objects.bulk_create(tweets_to_create, ignore_conflicts=True)
        
        self.stdout.write(self.style.SUCCESS(f'Generated {count} synthetic tweets'))
