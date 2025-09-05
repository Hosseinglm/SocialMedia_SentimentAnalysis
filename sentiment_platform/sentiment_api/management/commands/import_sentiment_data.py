"""
Django management command to import sentiment data from CSV files
"""
import csv
import os
from datetime import datetime
from django.core.management.base import BaseCommand, CommandError
from django.utils import timezone
from django.db import transaction
from sentiment_api.models import Brand, SentimentTweet


class Command(BaseCommand):
    help = 'Import sentiment data from CSV files into Django models'
    
    def add_arguments(self, parser):
        parser.add_argument(
            '--csv-file',
            type=str,
            default='../datasets/clean_tweets.csv',
            help='Path to the CSV file to import (default: ../datasets/clean_tweets.csv)'
        )
        parser.add_argument(
            '--batch-size',
            type=int,
            default=1000,
            help='Number of records to process in each batch (default: 1000)'
        )
        parser.add_argument(
            '--clear-existing',
            action='store_true',
            help='Clear existing data before import'
        )
        parser.add_argument(
            '--max-records',
            type=int,
            help='Maximum number of records to import (for testing)'
        )
    
    def handle(self, *args, **options):
        csv_file = options['csv_file']
        batch_size = options['batch_size']
        clear_existing = options['clear_existing']
        max_records = options['max_records']
        
        # Check if file exists
        if not os.path.exists(csv_file):
            raise CommandError(f'CSV file not found: {csv_file}')
        
        self.stdout.write(
            self.style.SUCCESS(f'Starting import from: {csv_file}')
        )
        
        # Clear existing data if requested
        if clear_existing:
            self.stdout.write('Clearing existing data...')
            SentimentTweet.objects.all().delete()
            Brand.objects.all().delete()
            self.stdout.write(self.style.SUCCESS('Existing data cleared'))
        
        # Import data
        try:
            with open(csv_file, 'r', encoding='utf-8') as file:
                reader = csv.DictReader(file)
                
                # Validate CSV headers
                required_fields = [
                    'tweet_id', 'brand', 'sentiment_label', 'tweet_text',
                    'clean_tweet', 'author', 'date', 'source', 'tweet_len',
                    'num_hashtags', 'num_mentions'
                ]
                
                missing_fields = [field for field in required_fields if field not in reader.fieldnames]
                if missing_fields:
                    raise CommandError(f'Missing required fields in CSV: {missing_fields}')
                
                # Process data in batches
                batch = []
                total_processed = 0
                total_created = 0
                total_skipped = 0
                
                for row in reader:
                    if max_records and total_processed >= max_records:
                        break
                    
                    try:
                        # Process the row
                        processed_row = self.process_row(row)
                        if processed_row:
                            batch.append(processed_row)
                        else:
                            total_skipped += 1
                        
                        # Process batch when it reaches batch_size
                        if len(batch) >= batch_size:
                            created_count = self.process_batch(batch)
                            total_created += created_count
                            batch = []
                            
                            self.stdout.write(
                                f'Processed {total_processed + 1} records, '
                                f'created {total_created}, skipped {total_skipped}'
                            )
                        
                        total_processed += 1
                        
                    except Exception as e:
                        self.stdout.write(
                            self.style.WARNING(f'Error processing row {total_processed + 1}: {e}')
                        )
                        total_skipped += 1
                        continue
                
                # Process remaining batch
                if batch:
                    created_count = self.process_batch(batch)
                    total_created += created_count
                
                self.stdout.write(
                    self.style.SUCCESS(
                        f'Import completed! '
                        f'Total processed: {total_processed}, '
                        f'Created: {total_created}, '
                        f'Skipped: {total_skipped}'
                    )
                )
                
        except Exception as e:
            raise CommandError(f'Error importing data: {e}')
    
    def process_row(self, row):
        """Process a single CSV row and return processed data"""
        try:
            # Parse date
            date_str = row['date']
            try:
                # Try different date formats
                for date_format in ['%Y-%m-%d', '%Y-%m-%d %H:%M:%S', '%m/%d/%Y', '%d/%m/%Y']:
                    try:
                        parsed_date = datetime.strptime(date_str, date_format)
                        break
                    except ValueError:
                        continue
                else:
                    # If no format works, use current date
                    parsed_date = datetime.now()
                    self.stdout.write(
                        self.style.WARNING(f'Could not parse date "{date_str}", using current date')
                    )
            except Exception:
                parsed_date = datetime.now()
            
            # Make timezone aware
            if timezone.is_naive(parsed_date):
                parsed_date = timezone.make_aware(parsed_date)
            
            # Validate and clean data
            tweet_id = str(row['tweet_id']).strip()
            brand_name = str(row['brand']).strip()
            sentiment_label = str(row['sentiment_label']).strip()
            tweet_text = str(row['tweet_text']).strip()
            clean_tweet = str(row.get('clean_tweet', '')).strip()
            author = str(row.get('author', '')).strip()
            source = str(row.get('source', 'train')).strip()
            
            # Validate required fields
            if not tweet_id or not brand_name or not sentiment_label or not tweet_text:
                return None
            
            # Parse numeric fields
            try:
                tweet_len = int(row.get('tweet_len', len(tweet_text)))
                num_hashtags = int(row.get('num_hashtags', 0))
                num_mentions = int(row.get('num_mentions', 0))
            except (ValueError, TypeError):
                tweet_len = len(tweet_text)
                num_hashtags = 0
                num_mentions = 0
            
            return {
                'tweet_id': tweet_id,
                'brand_name': brand_name,
                'sentiment_label': sentiment_label,
                'tweet_text': tweet_text,
                'clean_tweet': clean_tweet,
                'author': author,
                'date': parsed_date,
                'source': source,
                'tweet_len': tweet_len,
                'num_hashtags': num_hashtags,
                'num_mentions': num_mentions,
            }
            
        except Exception as e:
            self.stdout.write(
                self.style.WARNING(f'Error processing row: {e}')
            )
            return None
    
    @transaction.atomic
    def process_batch(self, batch):
        """Process a batch of rows and create database objects"""
        created_count = 0
        
        # Get or create brands
        brand_names = set(row['brand_name'] for row in batch)
        brands = {}
        
        for brand_name in brand_names:
            brand, created = Brand.objects.get_or_create(
                name=brand_name,
                defaults={
                    'display_name': brand_name,
                    'category': self.guess_brand_category(brand_name)
                }
            )
            brands[brand_name] = brand
        
        # Create tweets
        tweets_to_create = []
        
        for row in batch:
            # Check if tweet already exists
            if SentimentTweet.objects.filter(tweet_id=row['tweet_id']).exists():
                continue
            
            tweet = SentimentTweet(
                tweet_id=row['tweet_id'],
                brand=brands[row['brand_name']],
                sentiment_label=row['sentiment_label'],
                tweet_text=row['tweet_text'],
                clean_tweet=row['clean_tweet'],
                author=row['author'],
                date=row['date'],
                source=row['source'],
                tweet_len=row['tweet_len'],
                num_hashtags=row['num_hashtags'],
                num_mentions=row['num_mentions'],
            )
            tweets_to_create.append(tweet)
        
        # Bulk create tweets
        if tweets_to_create:
            SentimentTweet.objects.bulk_create(tweets_to_create, ignore_conflicts=True)
            created_count = len(tweets_to_create)
        
        return created_count
    
    def guess_brand_category(self, brand_name):
        """Guess brand category based on brand name"""
        gaming_keywords = [
            'game', 'gaming', 'play', 'xbox', 'playstation', 'nintendo',
            'steam', 'epic', 'battlefield', 'callofduty', 'fortnite',
            'league', 'dota', 'csgo', 'overwatch', 'apex', 'valorant'
        ]
        
        tech_keywords = [
            'microsoft', 'apple', 'google', 'amazon', 'facebook',
            'twitter', 'instagram', 'youtube', 'netflix', 'spotify'
        ]
        
        telecom_keywords = [
            'verizon', 'att', 'tmobile', 'sprint', 'comcast'
        ]
        
        brand_lower = brand_name.lower()
        
        if any(keyword in brand_lower for keyword in gaming_keywords):
            return 'Gaming'
        elif any(keyword in brand_lower for keyword in tech_keywords):
            return 'Technology'
        elif any(keyword in brand_lower for keyword in telecom_keywords):
            return 'Telecommunications'
        else:
            return 'Other'
