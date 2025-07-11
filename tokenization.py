# File will provide the tokenization code for the amazon_impulse_processor.py code
IMPULSE_TERMS = { "impulse", "whim", "spur", "caught", "splurge", ... }

import re

# e.g. split on words (alphanumeric + underscore)
TOKENISE = re.compile(r"\w+")

def detect_impulse(text, threshold=1):
    tokens = set(t.lower() for t in TOKENISE.findall(text))
    hits   = tokens & IMPULSE_TERMS
    if len(hits) >= threshold:
        return True, hits
    return False, hits

# Example:
msg = "I love the spur of finding impulse on a whim, until I am caught doing stupid shit"
flag, terms = detect_impulse(msg)
if flag:
    print("Fatty McFatass Eating at a McDonalds. FATFUCK", terms)
