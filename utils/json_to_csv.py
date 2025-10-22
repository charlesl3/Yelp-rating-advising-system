"""
json_to_csv.py
-----------------
Converts the original Yelp Academic Dataset (JSON lines format)
into a smaller CSV file containing only 'text' and 'stars' columns.

Usage:
    python json_to_csv.py
"""

import pandas as pd

def convert_json_to_csv(input_path: str, output_path: str, min_star: int = 1, max_star: int = 5):
    """
    Convert Yelp JSON review data to a simplified CSV.

    Args:
        input_path (str): Path to Yelp JSON file (e.g., yelp_academic_dataset_review.json)
        output_path (str): Path to save the resulting CSV
        min_star (int): Minimum star rating to keep
        max_star (int): Maximum star rating to keep
    """
    print(f"📂 Loading dataset from {input_path} ...")
    df = pd.read_json(input_path, lines=True)

    df = df[['text', 'stars']]
    df = df[df['stars'].between(min_star, max_star)]

    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    df.to_csv(output_path, index=False)

    print(f"✅ Saved cleaned dataset to {output_path} ({len(df):,} rows)")

if __name__ == "__main__":
    convert_json_to_csv("yelp_academic_dataset_review.json", "yelp_reviews_cleaned.csv")
