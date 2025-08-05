#!/usr/bin/env python3
"""
Uses DistilBERT to analyze sentiment of Amazon reviews with much better accuracy.

Usage:
    python simple_nlp_analyzer.py reviews.json
"""

import json
import argparse
from typing import List, Dict
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Import transformers for better sentiment analysis
try:
    from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    print("⚠️  Transformers not available. Install with: pip install transformers torch")
    TRANSFORMERS_AVAILABLE = False
    from textblob import TextBlob


class SimpleReviewAnalyzer:
    def __init__(self):
        # Initialize the sentiment analyzer
        if TRANSFORMERS_AVAILABLE:
            print("🧠 Loading DistilBERT sentiment model...")
            self.sentiment_analyzer = pipeline(
                "sentiment-analysis",
                model="distilbert-base-uncased-finetuned-sst-2-english",
                return_all_scores=True
            )
            print("✅ Model loaded successfully!")
        else:
            print("⚠️  Using TextBlob as fallback (less accurate)")
            self.sentiment_analyzer = None
    
    def load_reviews(self, json_file: str) -> List[Dict]:
        with open(json_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def analyze_sentiment(self, reviews: List[Dict]) -> Dict:
        sentiments = []
        
        for i, review in enumerate(reviews):
            text = review.get('text', '').strip()
            if len(text) > 10:
                try:
                    if self.sentiment_analyzer and TRANSFORMERS_AVAILABLE:
                        # Use DistilBERT for much better sentiment analysis
                        results = self.sentiment_analyzer(text[:512])  # Limit length for model
                        
                        # Find the correct scores by label (order might vary)
                        positive_score = 0
                        negative_score = 0
                        for score_info in results[0]:
                            if score_info['label'] == 'POSITIVE':
                                positive_score = score_info['score']
                            elif score_info['label'] == 'NEGATIVE':
                                negative_score = score_info['score']
                        
                        # Calculate polarity (-1 to 1 range)
                        polarity = positive_score - negative_score
                        
                        # Determine sentiment with 3-way classification
                        if abs(polarity) < 0.3:  # More sensitive to NEUTRAL
                            sentiment_label = 'NEUTRAL'
                            confidence = max(positive_score, negative_score)
                        elif positive_score > negative_score:
                            sentiment_label = 'POSITIVE'
                            confidence = positive_score
                        else:
                            sentiment_label = 'NEGATIVE'
                            confidence = negative_score
                        
                        # Debug output for all reviews to see classification
                        if i < 6:  # Show all reviews in our test
                            print(f"🔍 Review {i+1}: '{text[:50]}...'")
                            print(f"   Rating: {review.get('rating', 0)} stars")
                            print(f"   Positive score: {positive_score:.3f}")
                            print(f"   Negative score: {negative_score:.3f}")
                            print(f"   Sentiment: {sentiment_label} (confidence: {confidence:.3f})")
                            print(f"   Polarity: {polarity:.3f} (threshold: ±0.3)")
                            print()
                        
                    else:
                        # Fallback to TextBlob
                        blob = TextBlob(text)
                        polarity = blob.sentiment.polarity
                        
                        # 3-way classification for TextBlob fallback
                        if abs(polarity) < 0.1:  # Close to zero = NEUTRAL
                            sentiment_label = 'NEUTRAL'
                            confidence = 0.5  # Neutral confidence
                        elif polarity >= 0:
                            sentiment_label = 'POSITIVE'
                            confidence = (polarity + 1) / 2
                        else:
                            sentiment_label = 'NEGATIVE'  
                            confidence = (-polarity + 1) / 2
                    
                    sentiments.append({
                        'sentiment': sentiment_label,
                        'confidence': confidence,
                        'polarity': polarity,
                        'rating': review.get('rating', 0)
                    })
                    
                    # Progress indicator
                    if (i + 1) % 10 == 0:
                        print(f"📊 Analyzed {i + 1}/{len(reviews)} reviews...")
                        
                except Exception as e:
                    print(f"⚠️  Error analyzing review {i}: {e}")
                    continue
        
        # Calculate stats
        sentiment_counts = Counter([s['sentiment'] for s in sentiments])
        avg_confidence = np.mean([s['confidence'] for s in sentiments])
        
        positive = [s for s in sentiments if s['sentiment'] == 'POSITIVE']
        negative = [s for s in sentiments if s['sentiment'] == 'NEGATIVE']
        neutral = [s for s in sentiments if s['sentiment'] == 'NEUTRAL']
        
        # Collect additional data for enhanced charts
        polarity_values = [s['polarity'] for s in sentiments]
        confidence_values = [s['confidence'] for s in sentiments]
        
        # Calculate confidence distribution
        confidence_ranges = {'Low (0.5-0.7)': 0, 'Medium (0.7-0.9)': 0, 'High (0.9-1.0)': 0}
        for conf in confidence_values:
            if conf < 0.7:
                confidence_ranges['Low (0.5-0.7)'] += 1
            elif conf < 0.9:
                confidence_ranges['Medium (0.7-0.9)'] += 1
            else:
                confidence_ranges['High (0.9-1.0)'] += 1
        
        return {
            'total_analyzed': len(sentiments),
            'distribution': dict(sentiment_counts),
            'avg_confidence': avg_confidence,
            'avg_rating_positive': np.mean([s['rating'] for s in positive]) if positive else 0,
            'avg_rating_negative': np.mean([s['rating'] for s in negative]) if negative else 0,
            'avg_rating_neutral': np.mean([s['rating'] for s in neutral]) if neutral else 0,
            'avg_polarity': np.mean([s['polarity'] for s in sentiments]),
            'polarity_data': polarity_values,
            'confidence_distribution': confidence_ranges,
            'confidence_values': confidence_values
        }
    
    def create_chart(self, sentiment_data: Dict):
        # Set up professional styling
        plt.style.use('default')
        plt.rcParams['font.size'] = 10
        plt.rcParams['axes.titlesize'] = 12
        plt.rcParams['axes.labelsize'] = 10
        
        # Create figure with better proportions
        fig = plt.figure(figsize=(16, 12))
        fig.suptitle('Amazon Review Sentiment Analysis Dashboard', fontsize=20, fontweight='bold', y=0.98)
        
        # Color scheme
        colors = {
            'POSITIVE': '#2E8B57',  # Sea green
            'NEUTRAL': '#FFD700',   # Gold
            'NEGATIVE': '#DC143C',  # Crimson
            'background': '#F8F9FA',
            'text': '#2C3E50'
        }
        
        # 1. Sentiment Distribution Pie Chart
        plt.subplot(2, 3, 1)
        labels = list(sentiment_data['distribution'].keys())
        sizes = list(sentiment_data['distribution'].values())
        
        # Enhanced color mapping
        chart_colors = [colors.get(label, '#808080') for label in labels]
        
        # Create pie chart with better styling
        wedges, texts, autotexts = plt.pie(sizes, labels=labels, autopct='%1.1f%%', 
                                          colors=chart_colors, startangle=90,
                                          textprops={'fontsize': 11, 'fontweight': 'bold'})
        
        # Enhance pie chart text
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
        
        plt.title('Sentiment Distribution', fontsize=14, fontweight='bold', pad=20)
        
        # 2. Average Rating by Sentiment
        plt.subplot(2, 3, 2)
        sentiments = ['POSITIVE', 'NEUTRAL', 'NEGATIVE']
        ratings = [
            sentiment_data['avg_rating_positive'], 
            sentiment_data['avg_rating_neutral'], 
            sentiment_data['avg_rating_negative']
        ]
        
        # Only show categories that have data
        valid_data = [(sent, rating) for sent, rating in zip(sentiments, ratings) if rating > 0]
        if valid_data:
            valid_sentiments, valid_ratings = zip(*valid_data)
            valid_colors = [colors.get(sent, '#808080') for sent in valid_sentiments]
            
            bars = plt.bar(valid_sentiments, valid_ratings, color=valid_colors, alpha=0.8, edgecolor='black', linewidth=1)
            
            # Add value labels on bars
            for bar, rating in zip(bars, valid_ratings):
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                        f'{rating:.1f}', ha='center', va='bottom', fontweight='bold')
        
        plt.title('Average Rating by Sentiment', fontsize=14, fontweight='bold', pad=20)
        plt.ylabel('Average Rating (Stars)', fontweight='bold')
        plt.ylim(0, 5.5)
        plt.grid(axis='y', alpha=0.3)
        
        # 3. Sentiment Count Comparison
        plt.subplot(2, 3, 3)
        sentiment_counts = list(sentiment_data['distribution'].values())
        sentiment_labels = list(sentiment_data['distribution'].keys())
        count_colors = [colors.get(label, '#808080') for label in sentiment_labels]
        
        bars = plt.bar(sentiment_labels, sentiment_counts, color=count_colors, alpha=0.8, edgecolor='black', linewidth=1)
        
        # Add count labels on bars
        for bar, count in zip(bars, sentiment_counts):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{count}', ha='center', va='bottom', fontweight='bold')
        
        plt.title('Review Count by Sentiment', fontsize=14, fontweight='bold', pad=20)
        plt.ylabel('Number of Reviews', fontweight='bold')
        plt.grid(axis='y', alpha=0.3)
        
        # 4. Polarity Distribution (if we have polarity data)
        plt.subplot(2, 3, 4)
        if 'polarity_data' in sentiment_data:
            polarity_values = sentiment_data['polarity_data']
            plt.hist(polarity_values, bins=20, color='skyblue', alpha=0.7, edgecolor='black')
            plt.axvline(x=0, color='red', linestyle='--', alpha=0.7, label='Neutral Line')
            plt.title('Polarity Distribution', fontsize=14, fontweight='bold', pad=20)
            plt.xlabel('Polarity Score (-1 to +1)', fontweight='bold')
            plt.ylabel('Number of Reviews', fontweight='bold')
            plt.legend()
            plt.grid(alpha=0.3)
        else:
            # Placeholder for polarity info
            plt.text(0.5, 0.5, 'Polarity data not available', ha='center', va='center', 
                    transform=plt.gca().transAxes, fontsize=12)
            plt.title('Polarity Distribution', fontsize=14, fontweight='bold', pad=20)
        
        # 5. Confidence Analysis
        plt.subplot(2, 3, 5)
        if 'confidence_distribution' in sentiment_data:
            confidence_data = sentiment_data['confidence_distribution']
            confidence_levels = list(confidence_data.keys())
            confidence_counts = list(confidence_data.values())
            
            # Color coding for confidence levels
            confidence_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']  # Red, Teal, Blue
            
            bars = plt.bar(confidence_levels, confidence_counts, color=confidence_colors, alpha=0.8, edgecolor='black', linewidth=1)
            
            # Add count labels on bars
            for bar, count in zip(bars, confidence_counts):
                if count > 0:  # Only show labels for non-zero values
                    height = bar.get_height()
                    plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                            f'{count}', ha='center', va='bottom', fontweight='bold')
        else:
            plt.text(0.5, 0.5, 'Confidence data not available', ha='center', va='center', 
                    transform=plt.gca().transAxes, fontsize=12)
        
        plt.title('Confidence Level Distribution', fontsize=14, fontweight='bold', pad=20)
        plt.ylabel('Number of Reviews', fontweight='bold')
        plt.xticks(rotation=45)
        plt.grid(axis='y', alpha=0.3)
        
        # 6. Summary Dashboard
        plt.subplot(2, 3, 6)
        summary_text = f"""
        📊 ANALYSIS SUMMARY
        
        📝 Total Reviews: {sentiment_data['total_analyzed']}
        🎯 Average Confidence: {sentiment_data['avg_confidence']:.1%}
        📈 Average Polarity: {sentiment_data.get('avg_polarity', 0):.2f}
        
        🧠 Model: {'DistilBERT' if TRANSFORMERS_AVAILABLE else 'TextBlob'}
        
        📊 SENTIMENT BREAKDOWN:
        """
        
        for sentiment, count in sentiment_data['distribution'].items():
            percentage = (count / sentiment_data['total_analyzed']) * 100
            summary_text += f"\n   {sentiment}: {count} ({percentage:.1f}%)"
        
        # Determine overall mood
        avg_polarity = sentiment_data.get('avg_polarity', 0)
        if avg_polarity > 0.1:
            mood = "😊 Overall Positive"
        elif avg_polarity < -0.1:
            mood = "😞 Overall Negative"
        else:
            mood = "😐 Mixed/Neutral"
        
        summary_text += f"\n\n🎭 Overall Mood: {mood}"
        
        plt.text(0.05, 0.95, summary_text, transform=plt.gca().transAxes, 
                fontsize=11, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.5", facecolor='lightgray', alpha=0.8))
        
        plt.axis('off')
        plt.title('Summary Dashboard', fontsize=14, fontweight='bold', pad=20)
        
        # Adjust layout and save
        plt.tight_layout()
        plt.subplots_adjust(top=0.94)
        
        # Save with high quality
        plt.savefig('sentiment_analysis.png', dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        plt.close()
        print("📊 Professional sentiment analysis chart saved as 'sentiment_analysis.png'")
    
    def run_analysis(self, json_file: str) -> Dict:
        reviews = self.load_reviews(json_file)
        sentiment_results = self.analyze_sentiment(reviews)
        self.create_chart(sentiment_results)
        
        return {
            'total_reviews': len(reviews),
            'avg_rating': np.mean([r.get('rating', 0) for r in reviews if r.get('rating')]),
            'sentiment': sentiment_results
        }
    
    def print_report(self, results: Dict):
        print("\n" + "="*40)
        print("    SENTIMENT ANALYSIS REPORT")
        print("="*40)
        
        print(f"\nOverview:")
        print(f"   Total Reviews: {results['total_reviews']}")
        print(f"   Average Rating: {results['avg_rating']:.1f}/5.0")
        
        sentiment = results['sentiment']
        print(f"\nSentiment Analysis:")
        print(f"   Reviews Analyzed: {sentiment['total_analyzed']}")
        print(f"   Confidence: {sentiment['avg_confidence']:.1%}")
        
        for sent, count in sentiment['distribution'].items():
            percentage = (count / sentiment['total_analyzed']) * 100
            print(f"    {sent}: {count} ({percentage:.1f}%)")
        
        if sentiment.get('avg_polarity'):
            polarity = sentiment['avg_polarity']
            if polarity > 0.1:
                mood = "Overall Positive"
            elif polarity < -0.1:
                mood = "Overall Negative"
            else:
                mood = "Mixed/Neutral"
            print(f"\nOverall Mood: {mood} (polarity: {polarity:.2f})")
        
        if sentiment['avg_rating_positive'] > 0:
            print(f"\nRating Correlation:")
            print(f"   Positive sentiment: {sentiment['avg_rating_positive']:.1f} stars")
            if sentiment['avg_rating_neutral'] > 0:
                print(f"   Neutral sentiment: {sentiment['avg_rating_neutral']:.1f} stars")
            print(f"   Negative sentiment: {sentiment['avg_rating_negative']:.1f} stars")
        
        print("\n" + "="*40)


def main():
    parser = argparse.ArgumentParser(description='Sentiment analysis for Amazon reviews')
    parser.add_argument('json_file', help='Reviews JSON file')
    parser.add_argument('--output', '-o', default='sentiment_results.json', help='Output file')
    
    args = parser.parse_args()
    
    analyzer = SimpleReviewAnalyzer()
    results = analyzer.run_analysis(args.json_file)
    analyzer.print_report(results)
    
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main() 