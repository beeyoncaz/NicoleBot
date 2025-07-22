import argparse, json
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer # Needed for our code
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from impulse_keywords import impulse_label  # your weak labeller

def build_features(reviews, master_csv):
    # load χ²‑lexicon + sentiment
    master = pd.read_csv(master_csv)
    # keep only the words in your final vocab
    vocab = master["word"].tolist()
    # build a per‑word weight = χ2 × sentiment
    weights = master["chi2"].values * master["sentiment_score"].values

    # binary bag-of-words on that vocab
    cv = CountVectorizer(vocabulary=vocab, binary=True)
    X_bow = cv.fit_transform(
        [r["title"] + " " + r["text"] for r in reviews]
    ).astype(float)

    # 3) one single “weighted χ²×sentiment” feature
    #    by dotting the binary matrix with your weight vector
    X_weighted = X_bow.dot(weights.reshape(-1,1))

    # 4) (optional) stack on any other numeric features like price/hour-of-day:
    #    X_extra = ...
    #    X = np.hstack([X_weighted, X_extra])
    return X_weighted

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--master_csv", required=True)
    p.add_argument("--in_reviews", required=True,
                   help="JSONL file of reviews to train/test on")
    args = p.parse_args()

    # 1. load data + labels
    with open(args.in_reviews) as f:
        reviews = [json.loads(line) for line in f]
    y = np.array([impulse_label(r) for r in reviews])

    # 2. featurise
    X = build_features(reviews, args.master_csv)
  
    # 3. train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 4. fit logistic regression
    clf = LogisticRegression(class_weight="balanced", max_iter=1000)
    clf.fit(X_train, y_train)

    # 5. evaluate & predict
    proba = clf.predict_proba(X_test)[:,1]
    print("Test AUC:", roc_auc_score(y_test, proba))
    # optionally save proba with IDs to a CSV for downstream use

if __name__=="__main__":
    main()
