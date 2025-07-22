#!/usr/bin/env python3
"""
Uses TextBlob to analyze sentiment of Amazon reviews.

Usage:
    python simple_nlp_analyzer.py reviews.json
"""

import json
import argparse
from typing import List, Dict
import numpy as np
from collections import Counter
from textblob import TextBlob
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


class SimpleReviewAnalyzer:
    def __init__(self):
        pass
    
    def load_reviews(self, json_file: str) -> List[Dict]:
        with open(json_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def analyze_sentiment(self, reviews: List[Dict]) -> Dict:
        sentiments = []
        
        for review in reviews:
            text = review.get('text', '').strip()
            if len(text) > 10:
                try:
                    # Use TextBlob for sentiment analysis
                    blob = TextBlob(text)
                    polarity = blob.sentiment.polarity  # Range: -1 (negative) to 1 (positive)
                    
                    # Convert polarity to POSITIVE/NEGATIVE labels
                    if polarity >= 0:
                        sentiment_label = 'POSITIVE'
                        confidence = (polarity + 1) / 2  # Convert to 0-1 range
                    else:
                        sentiment_label = 'NEGATIVE'  
                        confidence = (-polarity + 1) / 2  # Convert to 0-1 range
                    
                    sentiments.append({
                        'sentiment': sentiment_label,
                        'confidence': confidence,
                        'polarity': polarity,  # Keep raw polarity for analysis
                        'rating': review.get('rating', 0)
                    })
                except:
                    continue
        
        # Calculate stats
        sentiment_counts = Counter([s['sentiment'] for s in sentiments])
        avg_confidence = np.mean([s['confidence'] for s in sentiments])
        
        positive = [s for s in sentiments if s['sentiment'] == 'POSITIVE']
        negative = [s for s in sentiments if s['sentiment'] == 'NEGATIVE']
        
        return {
            'total_analyzed': len(sentiments),
            'distribution': dict(sentiment_counts),
            'avg_confidence': avg_confidence,
            'avg_rating_positive': np.mean([s['rating'] for s in positive]) if positive else 0,
            'avg_rating_negative': np.mean([s['rating'] for s in negative]) if negative else 0,
            'avg_polarity': np.mean([s['polarity'] for s in sentiments])
        }
    
    def create_chart(self, sentiment_data: Dict):
        plt.figure(figsize=(10, 6))
        
        # Pie chart
        plt.subplot(2, 2, 1)
        labels = list(sentiment_data['distribution'].keys())
        sizes = list(sentiment_data['distribution'].values())
        colors = ['lightgreen' if l == 'POSITIVE' else 'lightcoral' for l in labels]
        plt.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors)
        plt.title('Sentiment Distribution')
        
        # Bar chart with ratings
        plt.subplot(2, 2, 2)
        sentiments = ['POSITIVE', 'NEGATIVE']
        ratings = [sentiment_data['avg_rating_positive'], sentiment_data['avg_rating_negative']]
        colors = ['green', 'red']
        plt.bar(sentiments, ratings, color=colors)
        plt.title('Average Rating by Sentiment')
        plt.ylabel('Average Rating')
        plt.ylim(0, 5)
        
        # Polarity histogram
        plt.subplot(2, 1, 2)
        plt.text(0.1, 0.8, f"Overall Sentiment Analysis", fontsize=14, weight='bold')
        plt.text(0.1, 0.6, f"• Average Confidence: {sentiment_data['avg_confidence']:.1%}", fontsize=12)
        plt.text(0.1, 0.4, f"• Average Polarity: {sentiment_data.get('avg_polarity', 0):.2f}", fontsize=12)
        plt.text(0.1, 0.2, f"• Total Reviews Analyzed: {sentiment_data['total_analyzed']}", fontsize=12)
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.axis('off')
        plt.title('Analysis Summary')
        
        plt.tight_layout()
        plt.savefig('sentiment_analysis.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("Chart saved as 'sentiment_analysis.png'")
    
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