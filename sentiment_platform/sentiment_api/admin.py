"""
Django admin configuration for Sentiment Intelligence Platform
"""
from django.contrib import admin
from django.utils.html import format_html
from django.urls import reverse
from django.utils.safestring import mark_safe
from .models import Brand, SentimentTweet, SentimentAnalysis, AlertRule, Alert


@admin.register(Brand)
class BrandAdmin(admin.ModelAdmin):
    list_display = ['name', 'display_name', 'category', 'total_mentions', 'created_at']
    list_filter = ['category', 'created_at']
    search_fields = ['name', 'display_name']
    readonly_fields = ['created_at', 'total_mentions']
    
    def total_mentions(self, obj):
        return obj.total_mentions
    total_mentions.short_description = 'Total Mentions'


@admin.register(SentimentTweet)
class SentimentTweetAdmin(admin.ModelAdmin):
    list_display = ['tweet_id', 'brand', 'sentiment_label', 'date', 'tweet_len', 'engagement_score', 'source']
    list_filter = ['sentiment_label', 'source', 'brand', 'date']
    search_fields = ['tweet_id', 'tweet_text', 'clean_tweet']
    readonly_fields = ['created_at', 'updated_at', 'engagement_score']
    date_hierarchy = 'date'
    
    fieldsets = (
        ('Basic Information', {
            'fields': ('tweet_id', 'brand', 'sentiment_label', 'source')
        }),
        ('Content', {
            'fields': ('tweet_text', 'clean_tweet')
        }),
        ('Metadata', {
            'fields': ('author', 'date', 'tweet_len')
        }),
        ('Engagement', {
            'fields': ('num_hashtags', 'num_mentions', 'engagement_score')
        }),
        ('System', {
            'fields': ('created_at', 'updated_at'),
            'classes': ('collapse',)
        }),
    )
    
    def engagement_score(self, obj):
        score = obj.engagement_score
        if score > 5:
            color = 'green'
        elif score > 2:
            color = 'orange'
        else:
            color = 'red'
        return format_html(
            '<span style="color: {};">{}</span>',
            color, score
        )
    engagement_score.short_description = 'Engagement Score'


@admin.register(SentimentAnalysis)
class SentimentAnalysisAdmin(admin.ModelAdmin):
    list_display = ['title', 'analysis_type', 'created_at', 'created_by', 'brands_count']
    list_filter = ['analysis_type', 'created_at', 'created_by']
    search_fields = ['title', 'description']
    readonly_fields = ['created_at', 'brands_count']
    filter_horizontal = ['brands']
    
    fieldsets = (
        ('Basic Information', {
            'fields': ('analysis_type', 'title', 'description')
        }),
        ('Parameters', {
            'fields': ('brands', 'date_from', 'date_to')
        }),
        ('Results', {
            'fields': ('results', 'insights', 'recommendations'),
            'classes': ('collapse',)
        }),
        ('Metadata', {
            'fields': ('created_at', 'created_by'),
            'classes': ('collapse',)
        }),
    )
    
    def brands_count(self, obj):
        return obj.brands.count()
    brands_count.short_description = 'Brands Count'


@admin.register(AlertRule)
class AlertRuleAdmin(admin.ModelAdmin):
    list_display = ['name', 'brand', 'alert_type', 'threshold_value', 'is_active', 'last_triggered']
    list_filter = ['alert_type', 'is_active', 'brand']
    search_fields = ['name', 'brand__name']
    readonly_fields = ['created_at', 'last_triggered']
    
    fieldsets = (
        ('Basic Information', {
            'fields': ('name', 'alert_type', 'brand')
        }),
        ('Threshold Settings', {
            'fields': ('threshold_value', 'time_window_hours')
        }),
        ('Status', {
            'fields': ('is_active', 'last_triggered')
        }),
        ('Metadata', {
            'fields': ('created_at',),
            'classes': ('collapse',)
        }),
    )


@admin.register(Alert)
class AlertAdmin(admin.ModelAdmin):
    list_display = ['title', 'rule', 'severity', 'status', 'triggered_at', 'status_badge']
    list_filter = ['severity', 'status', 'triggered_at', 'rule__brand']
    search_fields = ['title', 'message', 'rule__name']
    readonly_fields = ['triggered_at']
    
    fieldsets = (
        ('Alert Information', {
            'fields': ('rule', 'title', 'message', 'severity')
        }),
        ('Status', {
            'fields': ('status', 'triggered_at', 'acknowledged_at', 'resolved_at')
        }),
        ('Data', {
            'fields': ('data',),
            'classes': ('collapse',)
        }),
    )
    
    def status_badge(self, obj):
        colors = {
            'open': 'red',
            'acknowledged': 'orange',
            'resolved': 'green'
        }
        return format_html(
            '<span style="color: {}; font-weight: bold;">{}</span>',
            colors.get(obj.status, 'black'),
            obj.get_status_display()
        )
    status_badge.short_description = 'Status'


# Customize admin site
admin.site.site_header = "Augment Sentiment Intelligence Platform"
admin.site.site_title = "Sentiment Intelligence Admin"
admin.site.index_title = "Welcome to Sentiment Intelligence Administration"
