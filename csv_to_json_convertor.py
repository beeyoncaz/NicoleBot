import json

# Your existing dictionary
reviews = {
    "Reviewer Names": [],
    "Review Titles": [],
    "Review Texts": [],
    "Review Dates":[],
    "Review Star Ratings": []
}

# Convert to a list of review dictionaries (one per review)
reviews_json_list = []
num_reviews = len(reviews["Reviewer Names"])

for i in range(num_reviews):
    review = {
        "Reviewer Name": reviews["Reviewer Names"][i],
        "Review Title": reviews["Review Titles"][i] if i < len(reviews["Review Titles"]) else "N/A",
        "Review Text": reviews["Review Texts"][i] if i < len(reviews["Review Texts"]) else "N/A",
        "Review Date": reviews["Review Dates"][i] if i < len(reviews["Review Dates"]) else "N/A",
        "Star Rating": reviews["Review Star Ratings"][i] if i < len(reviews["Review Star Ratings"]) else "N/A",
    }
    reviews_json_list.append(review)

# ✅ Option 1: Print JSON to terminal (nicely formatted)
print(json.dumps(reviews_json_list, indent=2))

# ✅ Option 2: Save to JSON file
with open("amazon_reviews1.json", "w", encoding="utf-8") as f:
    json.dump(reviews_json_list, f, indent=2, ensure_ascii=False)

print("✅ Data successfully saved to amazon_reviews1.json")
