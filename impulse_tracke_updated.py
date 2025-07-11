# Use Pearson-Chi Square Statistic (Associated Word: How Much More Frequent Does the Word Show Up)
# Next Step: Feature Selection/Lexicon Building; Provide Ranking and Prioritization of Tokens
# Update: Use a stopword imported library to remove irrelevant/insignificant words that don't add anything to the library
# NOTE WHEN RUNNING: YOU NEED ONE OF THE CATEGORY DATA (preferrably json file) in order to run the code. PLEASE FORMAT YOUR INPUT IN THE TERMINAL LIKE THIS:
# Further Update: Modified code so that keywords are always written to a master csv file (alr pushed to Github). 
# IF YOU ARE GOING TO EXTRACT FROM THE CSV FILE, PLEASE DO NOT MIX THE KEYWORDS; you may modify the argparse section to include your NEW CSV FILE PATH
# PLEASE FORMAT YOUR INPUT IN THE TERMINAL LIKE THIS:
# python impulse_tracker.py < 'file_name'. If you adding a csv file link, you need to add the argument in between
# Please contact Gavin Gong for any issues.

import re, sys, json, datetime as dt, csv, os
from collections import Counter
import argparse

from sklearn.feature_extraction import text
EN_STOP = set(text.ENGLISH_STOP_WORDS)
EXTRA_STOP = {"not", "just", "even", "really", "don", "didn", "don't"}
STOP = EN_STOP | EXTRA_STOP

IMPULSE_RE = re.compile(
    r"(impulse[ -]?buy|bought on a whim|couldn.?t resist|spur of the moment|"
    r"late[- ]night purchase|didn.?t even need)", flags=re.I)

def impulse_label(r):
    txt = (r["title"] + " " + r["text"])
    if IMPULSE_RE.search(txt):
        return 1
    score = 0
    if r.get("price") is not None and r["price"] < 20:
        score += 1
    hour = dt.datetime.utcfromtimestamp(r["timestamp"]/1000).hour
    if 0 <= hour < 4:
        score += 1
    if r["rating"] in (1.0, 2.0):
        score += 1
    return 1 if score >= 2 else 0

TOKENISE = re.compile(r"[A-Za-z']{3,}")
def tokens(text):
    return TOKENISE.findall(text.lower())

def chi2_keyterms(reviews, *, min_df=5, top_k=30):
    N_imp = N_non = 0
    df_imp = Counter()
    df_non = Counter()

    for r in reviews:
        y = impulse_label(r)
        seen = set(tokens(r["title"] + " " + r["text"]))
        seen = {w for w in seen if w not in STOP}

        if y:
            N_imp += 1
            df_imp.update(seen)
        else:
            N_non += 1
            df_non.update(seen)

    keyterms = []
    for w in (df_imp.keys() | df_non.keys()):
        if (df_imp[w] + df_non[w]) < min_df:
            continue
        A, B = df_imp[w], df_non[w]
        C, D = N_imp - A, N_non - B
        num = abs(A*D - B*C) - 0.5*(N_imp+N_non)
        chi2 = (num*num)*(N_imp+N_non)/((A+B)*(C+D)*(A+C)*(B+D))
        keyterms.append((w, chi2))

    keyterms.sort(key=lambda t: t[1], reverse=True)
    return keyterms[:top_k]

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--min_df",     type=int,   default=5,
                   help="min doc-freq to keep a token")
    p.add_argument("--top_k",      type=int,   default=30,
                   help="number of keyterms to print")
    p.add_argument("--source",     type=str,   default="",
                   help="label for these keyterms (e.g. filename or category)")
    p.add_argument("--master_csv", type=str,   default="master_impulse_keywords.csv",
                   help="path to aggregate CSV to append to")
    args = p.parse_args()

    # load and score
    reviews = [json.loads(line) for line in sys.stdin]
    terms   = chi2_keyterms(reviews, min_df=args.min_df, top_k=args.top_k)

    # print to stdout
    for w, score in terms:
        print(f"{w:20s}  χ² = {score:6.1f}")

    # append to master CSV
    write_header = not os.path.exists(args.master_csv)
    with open(args.master_csv, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["source", "word", "chi2"])
        for w, score in terms:
            writer.writerow([args.source, w, f"{score:.3f}"])
