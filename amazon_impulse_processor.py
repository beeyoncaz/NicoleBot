#!/usr/bin/env python3
"""
Enhanced Amazon Impulse Buying Detector

Uses improved chi-square analysis with stopword filtering and master keyword database.
"""

import json
import gzip
import requests
from pathlib import Path
import argparse
import re
import csv
import os
from collections import Counter

# Import from impulse_tracker_updated
from impulse_tracker_updated import impulse_label, chi2_keyterms, tokens, IMPULSE_RE

# Optional pandas import
try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

# Improved stopword filtering (with fallback if sklearn not available)
try:
    from sklearn.feature_extraction import text
    EN_STOP = set(text.ENGLISH_STOP_WORDS)
except ImportError:
    EN_STOP = {"a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "he", "in", "is", "it", "its", "of", "on", "that", "the", "to", "was", "were", "will", "with", "the", "this", "but", "they", "have", "had", "what", "said", "each", "which", "do", "how", "their", "if", "up", "out", "many", "then", "them", "these", "so", "some", "her", "would", "make", "like", "into", "him", "time", "has", "two", "more", "go", "no", "way", "could", "my", "than", "first", "been", "call", "who", "oil", "sit", "now", "find", "down", "day", "did", "get", "come", "made", "may", "part"}

EXTRA_STOP = {"not", "just", "even", "really", "don", "didn", "don't", "good", "great", "nice"}
STOP = EN_STOP | EXTRA_STOP

def load_master_keywords(filepath="master_impulse_keywords.csv"):
    """
    Load pre-computed impulse keywords from master CSV
    """
    if not os.path.exists(filepath):
        return []
    
    try:
        if HAS_PANDAS:
            df = pd.read_csv(filepath)
            # Handle different CSV formats
            if 'word' in df.columns:
                return df['word'].tolist()
            else:
                # Handle the current format with unnamed columns
                return df.iloc[:, 1].tolist()  # Second column contains words
        else:
            # Manual CSV reading fallback
            keywords = []
            with open(filepath, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                for row in reader:
                    if len(row) >= 2 and row[1] and row[1] != 'word':  # Skip header
                        keywords.append(row[1])
            return keywords
    except Exception as e:
        print(f"Error loading master keywords: {e}")
        return []

class ImpulseAnalyzer:    
    CATEGORIES = [
        'All_Beauty', 'Amazon_Fashion', 'Appliances', 'Arts_Crafts_and_Sewing',
        'Automotive', 'Baby_Products', 'Beauty_and_Personal_Care', 'Books',
        'CDs_and_Vinyl', 'Cell_Phones_and_Accessories', 'Clothing_Shoes_and_Jewelry',
        'Digital_Music', 'Electronics', 'Gift_Cards', 'Grocery_and_Gourmet_Food',
        'Handmade_Products', 'Health_and_Household', 'Home_and_Kitchen',
        'Industrial_and_Scientific', 'Kindle_Store', 'Musical_Instruments',
        'Office_Products', 'Patio_Lawn_and_Garden', 'Pet_Supplies',
        'Software', 'Sports_and_Outdoors', 'Tools_and_Home_Improvement',
        'Toys_and_Games', 'Video_Games'
    ]
    
    BASE_URL = "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/"
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
    def download_reviews(self, category: str):
        """
        Download review file for a category
        """
        review_url = f"{self.BASE_URL}review_categories/{category}.jsonl.gz"
        review_file = self.data_dir / f"{category}_reviews.jsonl.gz"
        
        if not review_file.exists():
            print(f"Downloading {category} reviews...")
            self._download_file(review_url, review_file)
        else:
            print(f"Using existing file: {review_file}")
                
        return review_file
    
    def _download_file(self, url: str, filepath: Path):
        """Download a file with progress"""
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        with open(filepath, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    
        print(f"Downloaded: {filepath}")
    
    def load_reviews(self, filepath: Path, max_reviews: int = 1000):
        """Load and process reviews"""
        print(f"Loading {max_reviews} reviews...")
        reviews = []
        count = 0
        
        with gzip.open(filepath, 'rt', encoding='utf-8') as f:
            for line in f:
                if count >= max_reviews:
                    break
                    
                review = json.loads(line.strip())
                
                processed_review = {
                    'title': review.get('title', ''),
                    'text': review.get('text', ''),
                    'rating': float(review.get('rating', 0)),
                    'timestamp': review.get('timestamp', 0),
                    'price': None  # Simplified for now
                }
                
                reviews.append(processed_review)
                count += 1
                
                if count % 500 == 0:
                    print(f"Loaded {count} reviews...")
                    
        print(f"Loaded {len(reviews)} reviews total")
        return reviews
    
    def analyze_impulse_enhanced(self, reviews, category="Unknown"):
        print("Running enhanced impulse analysis...")
        
        master_keywords = load_master_keywords()
        
        # Get keyword scores if available (from CSV with chi-square values)
        keyword_scores = {}
        if os.path.exists("master_impulse_keywords.csv"):
            try:
                with open("master_impulse_keywords.csv", 'r', encoding='utf-8') as f:
                    reader = csv.reader(f)
                    next(reader, None)  # Skip header
                    for row in reader:
                        if len(row) >= 3:
                            keyword = row[1]
                            try:
                                score = float(row[2])
                                keyword_scores[keyword] = score
                            except ValueError:
                                keyword_scores[keyword] = 1.0
            except Exception as e:
                print(f"Warning: Could not load keyword scores: {e}")
        
        for keyword in master_keywords:
            if keyword not in keyword_scores:
                keyword_scores[keyword] = 1.0
        
        results = {'impulse': 0, 'non_impulse': 0, 'reviews': []}
        
        for review in reviews:
            text = (review.get('title', '') + ' ' + review.get('text', '')).lower()
            
            impulse_score = self.calculate_impulse_score(text, keyword_scores)
            
            if impulse_score >= 10.0: # Classification threshold
                results['impulse'] += 1
                results['reviews'].append({
                    **review,
                    'impulse_score': impulse_score
                })
            else:
                results['non_impulse'] += 1
        
        return {
            'total_reviews': len(reviews),
            'results': results,
            'master_keywords': master_keywords[:20]
        }
    
    def calculate_impulse_score(self, text, keyword_scores):
        """
        Calculate impulse score based on multiple factors:
        - Number of keywords found
        - Keyword weights (chi-square scores)
        - Keyword density
        - Strong impulse patterns (regex)
        """
        score = 0.0
        words = text.split()
        word_count = len(words)
        
        if IMPULSE_RE.search(text):
            score += 3.0 
        
        keywords_found = 0
        for keyword in keyword_scores:
            if keyword in text:
                occurrences = text.count(keyword)
                weight = keyword_scores[keyword] if keyword_scores[keyword] > 0 else 1.0
                
                # Normalize weight (chi-square values can be very high)
                normalized_weight = min(weight / 100.0, 2.0)
                
                score += occurrences * normalized_weight
                keywords_found += occurrences
                
        if keywords_found >= 2:
            score += 0.5

        if word_count > 0:
            density = (keywords_found / word_count) * 100
            if density > 2.0: 
                score += 0.5
        
        return score
    
    def print_enhanced_results(self, results):
        print("\n" + "="*60)
        print("SMART IMPULSE BUYING ANALYSIS")
        print("="*60)
        
        print(f"Total Reviews: {results['total_reviews']:,}")
        total = results['results']['impulse'] + results['results']['non_impulse']
        rate = results['results']['impulse'] / total if total > 0 else 0
        print(f"Total impulse: {results['results']['impulse']:,} impulse ({rate:.1%})")
        print(f"Classification threshold: 2.0 (scores >= 2.0 = impulse)")
        
        print(f"\nMASTER KEYWORDS (Top 10):")
        for i, word in enumerate(results['master_keywords'][:10], 1):
            print(f"  {i:2d}. {word}")
        
        print(f"\nSAMPLE IMPULSE REVIEWS (with scores):")
        for i, review in enumerate(results['results']['reviews'][:3], 1):
            score = review.get('impulse_score', 0)
            print(f"\n{i}. [Score: {score:.1f}] Title: {review['title']}")
            print(f"   Rating: {review['rating']}")
            print(f"   Text: {review['text'][:150]}...")
            
        # Show score distribution
        if results['results']['reviews']:
            scores = [r.get('impulse_score', 0) for r in results['results']['reviews']]
            avg_score = sum(scores) / len(scores)
            max_score = max(scores)
            print(f"\nSCORE STATISTICS:")
            print(f"  Average impulse score: {avg_score:.1f}")
            print(f"  Highest score: {max_score:.1f}")
            print(f"  Score breakdown:")
            score_ranges = [(2.0, 3.0), (3.0, 5.0), (5.0, 10.0), (10.0, float('inf'))]
            for low, high in score_ranges:
                count = sum(1 for s in scores if low <= s < high)
                if count > 0:
                    high_str = f"{high:.1f}" if high != float('inf') else "∞"
                    print(f"    {low:.1f} - {high_str}: {count} reviews")

def main():
    parser = argparse.ArgumentParser(description='Enhanced Amazon impulse buying analyzer')
    parser.add_argument('--category', required=True, choices=ImpulseAnalyzer.CATEGORIES,
                       help='Amazon category to analyze')
    parser.add_argument('--max-reviews', type=int, default=1000,
                       help='Maximum reviews to analyze (default: 1000)')
    
    args = parser.parse_args()
    
    analyzer = ImpulseAnalyzer()
    
    # Download and load reviews
    print(f"Analyzing {args.category} category...")
    review_file = analyzer.download_reviews(args.category)
    reviews = analyzer.load_reviews(review_file, args.max_reviews)
    
    # Enhanced analysis
    results = analyzer.analyze_impulse_enhanced(reviews, args.category)
    
    # Show results
    analyzer.print_enhanced_results(results)
    
    print(f"\nDone! Analyzed {results['total_reviews']} reviews")

if __name__ == "__main__":
    main() 