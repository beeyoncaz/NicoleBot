import re, sys, json, datetime as dt
from collections import Counter
import argparse

from sklearn.feature_extraction import text
EN_STOP = set(text.ENGLISH_STOP_WORDS)

# Add any extra tokens you want to ban; in several cases you can manually update the list if need be below
# other high-frequency words that aren’t helpful shoudl be added here too
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
        seen = set(tokens(r["title"] + " " + r["text"])) # Change
        seen = {w for w in seen if w not in STOP} # Change

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
    ap = argparse.ArgumentParser()
    ap.add_argument("--min_df", type=int, default=5,
                    help="minimum doc-freq to keep a token")
    ap.add_argument("--top_k", type=int, default=30,
                    help="number of keyterms to print")
    args = ap.parse_args()

    reviews = [json.loads(line) for line in sys.stdin]
    for word, score in chi2_keyterms(reviews,
                                     min_df=args.min_df,
                                     top_k=args.top_k):
        print(f"{word:20s}  χ² = {score:6.1f}")


# Use Pearson-Chi Square Statistic (Associated Word: How Much More Frequent Does the Word Show Up)
# Next Step: Feature Selection/Lexicon Building; Provide Ranking and Prioritization of Tokens
# Update: Use a stopword imported library to remove irrelevant/insignificant words that don't add anything to the library
# NOTE WHEN RUNNING: YOU NEED ONE OF THE CATEGORY DATA (preferrably json file) in order to run the code. PLEASE FORMAT YOUR INPUT IN THE TERMINAL LIKE THIS:
# python impulse_tracker.py < 'file_name'. Please contact Gavin Gong for any issues.
