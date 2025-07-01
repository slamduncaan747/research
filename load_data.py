import pandas as pd
from datasets import load_dataset
import os


def save_to_csv(texts, output_file):
    """Save texts to a CSV file with 'text' column."""
    df = pd.DataFrame({"text": texts})
    df.to_csv(output_file, index=False, encoding="utf-8")


def main():
    # Output directory
    output_dir = "tinystories_100mb_splits"
    os.makedirs(output_dir, exist_ok=True)

    # Load TinyStories dataset
    dataset = load_dataset("roneneldan/TinyStories")

    # Total target samples for ~100MB (~107,000 stories)
    total_samples = 107000
    train_ratio = 0.8  # ~85,600 for train
    val_size = len(dataset["validation"]["text"])  # Use all ~22,000 validation stories
    test_size = (
        total_samples - train_ratio * total_samples - val_size
    )  # ~22,000 for test

    # Subsample train and create test from train data
    train_texts = dataset["train"]["text"][: int(train_ratio * total_samples)]
    val_texts = dataset["validation"]["text"]

    # Save to CSV
    save_to_csv(train_texts, os.path.join(output_dir, "train.csv"))
    save_to_csv(val_texts, os.path.join(output_dir, "val.csv"))


if __name__ == "__main__":
    main()
