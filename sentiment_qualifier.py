# Documentation
# python sentiment_qualifier.py \
  --master_csv keywords/master_impulse_keywords.csv \
  --out_csv    keywords/master_impulse_with_sentiment.csv
# Argument 1: Your csv file (master_csv, my_csv, wtvr)
# Argument 2: Your output csv file (name)
# Please ensure you are in proper directory with csv file

import argparse
import pandas as pd
from transformers import pipeline

def map_label_to_score(label: str, score: float) -> float:
    if "star" in label:
        try:
            n = int(label.split()[0])
            return (n - 3) / 2.0
        except:
            pass
    # fallback for POSITIVE/NEGATIVE
    return score if "POSITIVE" in label.upper() else -score

def main():
    p = argparse.ArgumentParser(
        description="Take your master χ²‐lexicon and append a sentiment score"
    )
    p.add_argument(
        "--master_csv", required=True,
        help="path to existing master_impulse_keywords.csv"
    )
    p.add_argument(
        "--out_csv", required=True,
        help="where to write the sentiment‐augmented CSV"
    )
    args = p.parse_args()

    # 1. load your existing lexicon
    df = pd.read_csv(args.master_csv)

    # 2. set up the HF sentiment pipeline
    sent = pipeline(
        "sentiment-analysis",
        model="nlptown/bert-base-multilingual-uncased-sentiment"
    )

    # 3. score each unique word exactly once
    words = df["word"].unique().tolist()
    scores = {}
    for w in words:
        out = sent(w)[0]
        scores[w] = map_label_to_score(out["label"], out["score"])

    # 4. join back to your DataFrame
    df["sentiment_score"] = df["word"].map(scores)

    # 5. write out the new CSV
    df.to_csv(args.out_csv, index=False)
    print(f"Wrote {len(df)} rows with sentiment to {args.out_csv}")

if __name__=="__main__":
    main()
