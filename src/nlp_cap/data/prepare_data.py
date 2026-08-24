"""
prepare_data.py
Downloads the Amazon Reviews Polarity dataset from HuggingFace,
samples a balanced subset, and saves it as reviews_dataset.csv.
"""

import os
import pandas as pd
from datasets import load_dataset

# Number of samples per class (positive/negative)
SAMPLES_PER_CLASS = 2500
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "reviews_dataset.csv")


def prepare_dataset():
    """Download Amazon Reviews Polarity and create a balanced sample."""
    print("[*] Downloading Amazon Reviews Polarity dataset from HuggingFace...")
    print("    (This may take a few minutes on first run)\n")

    # Load the dataset — only the train split
    dataset = load_dataset("fancyzhx/amazon_polarity", split="train")

    # Convert to DataFrame
    df = pd.DataFrame(dataset)

    # The dataset has:
    #   label: 0 = Negative, 1 = Positive
    #   content: review text
    #   title: review title
    print(f"[+] Full dataset loaded: {len(df):,} reviews")
    print(f"    Label distribution:\n{df['label'].value_counts().to_string()}\n")

    # Sample balanced subset
    df_negative = df[df["label"] == 0].sample(n=SAMPLES_PER_CLASS, random_state=42)
    df_positive = df[df["label"] == 1].sample(n=SAMPLES_PER_CLASS, random_state=42)
    df_sampled = pd.concat([df_negative, df_positive]).sample(frac=1, random_state=42)

    # Map labels to readable strings
    label_map = {0: "Negative", 1: "Positive"}
    df_sampled["sentiment"] = df_sampled["label"].map(label_map)

    # Combine title + content for richer text
    df_sampled["review_text"] = df_sampled["title"] + ". " + df_sampled["content"]

    # Keep only what we need
    df_final = df_sampled[["review_text", "sentiment"]].reset_index(drop=True)

    # Save
    df_final.to_csv(OUTPUT_FILE, index=False)
    print(f"[+] Saved {len(df_final):,} reviews to: {OUTPUT_FILE}")
    print(f"    Sentiment distribution:\n{df_final['sentiment'].value_counts().to_string()}")

    return df_final


if __name__ == "__main__":
    prepare_dataset()
