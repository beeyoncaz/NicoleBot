#!/usr/bin/env python3

import json
import gzip
import requests
from pathlib import Path
import argparse

from impulse_tracker import impulse_label

class SimpleImpulseAnalyzer:
    """
    Simple analyzer for impulse buying patterns
    """
    
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
        """
        Download a file
        """
        
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        with open(filepath, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    
        print(f"Downloaded: {filepath}")
    
    def load_reviews(self, filepath: Path, max_reviews: int = 1000):
        """
        Load and process reviews
        """
        
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
                    'price': None 
                }
                
                reviews.append(processed_review)
                count += 1
                
                if count % 250 == 0:
                    print(f"Loaded {count} reviews...")
                    
        print(f"Loaded {len(reviews)} reviews total")
        return reviews
    
    def analyze_impulse(self, reviews):
        """
        Classify reviews as impulse or non-impulse
        """
        
        print("Analyzing impulse patterns...")
        
        impulse_count = 0
        non_impulse_count = 0
        impulse_reviews = []
        
        for review in reviews:
            if impulse_label(review):
                impulse_count += 1
                impulse_reviews.append(review)
            else:
                non_impulse_count += 1
        
        total = len(reviews)
        impulse_rate = impulse_count / total if total > 0 else 0
        
        return {
            'total_reviews': total,
            'impulse_count': impulse_count,
            'non_impulse_count': non_impulse_count,
            'impulse_rate': impulse_rate,
            'sample_impulse_reviews': impulse_reviews[:5] 
        }
    
    def print_results(self, results):
        print("\n" + "="*40)
        print("IMPULSE BUYING RESULTS")
        print("="*40)
        
        print(f"Total Reviews: {results['total_reviews']:,}")
        print(f"Impulse Purchases: {results['impulse_count']:,}")
        print(f"Non-Impulse Purchases: {results['non_impulse_count']:,}")
        print(f"Impulse Rate: {results['impulse_rate']:.1%}")
        
        print(f"\nSample Impulse Reviews:")
        for i, review in enumerate(results['sample_impulse_reviews'], 1):
            print(f"\n{i}. Title: {review['title']}")
            print(f"   Rating: {review['rating']}")
            print(f"   Text: {review['text'][:150]}...")

def main():
    parser = argparse.ArgumentParser(description='Simple Amazon impulse buying analyzer')
    parser.add_argument('--category', required=True, choices=SimpleImpulseAnalyzer.CATEGORIES,
                       help='Amazon category to analyze')
    parser.add_argument('--max-reviews', type=int, default=1000,
                       help='Maximum reviews to analyze (default: 1000)')
    
    args = parser.parse_args()
    
    analyzer = SimpleImpulseAnalyzer()
    
    print(f"Analyzing {args.category} category...")
    review_file = analyzer.download_reviews(args.category)
    reviews = analyzer.load_reviews(review_file, args.max_reviews)
    
    results = analyzer.analyze_impulse(reviews)
    
    analyzer.print_results(results)
    
    print(f"\nDone! Analyzed {results['total_reviews']} reviews")

if __name__ == "__main__":
    main() 