import pandas as pd
import tiktoken
import random
import difficulty_metrics as dm
import matplotlib.pyplot as plt
import numpy as np


def create_text_segments(df, text_column="text", enc=None):
    """Create segments of different lengths from the dataset"""

    # Define pools by token length ranges
    pools = {
        1: {"range": (2, 20), "segments": []},
        2: {"range": (21, 50), "segments": []},
        3: {"range": (51, 100), "segments": []},
        4: {"range": (101, 150), "segments": []},
        5: {"range": (151, 200), "segments": []},
    }

    print("Creating text segments...")
    for idx, row in df.iterrows():
        if idx % 1000 == 0:
            print(f"Processed {idx} rows")

        text = row[text_column]
        if pd.isna(text) or len(text.strip()) == 0:
            continue

        tokens = enc.encode_ordinary(text)
        if len(tokens) < 2:
            continue

        # Create segments for each pool using sliding window
        for pool_id, pool_info in pools.items():
            min_size, max_size = pool_info["range"]

            # Skip if text is too short for this pool
            if len(tokens) < min_size:
                continue

            # Use sliding window with step size based on pool size
            step_size = max(1, min_size // 4)  # Smaller step for more coverage

            for start in range(0, len(tokens) - min_size + 1, step_size):
                # Random window size within range
                window_size = random.randint(
                    min_size, min(max_size, len(tokens) - start)
                )
                chunk = tokens[start : start + window_size]

                if min_size <= len(chunk) <= max_size:
                    decoded_text = enc.decode(chunk)
                    pools[pool_id]["segments"].append(decoded_text)

    # Sample from each pool to balance dataset
    target_size = 8000  # Reduced for faster processing
    combined_segments = []

    for pool_id, pool_info in pools.items():
        segments = pool_info["segments"]
        if len(segments) == 0:
            continue

        # Sample segments
        sample_size = min(target_size, len(segments))
        sampled = random.sample(segments, sample_size)
        combined_segments.extend(sampled)
        print(
            f"Pool {pool_id} ({pool_info['range']}): {len(segments)} -> {sample_size} segments"
        )

    return combined_segments


def calculate_metrics(segments):
    """Calculate MTLD and length metrics for segments"""

    results = []
    print(f"\nCalculating metrics for {len(segments)} segments...")

    for idx, text in enumerate(segments):
        if idx % 1000 == 0:
            print(f"Processed {idx} segments")

        try:
            # Skip very short segments
            if len(text.split()) < 2:
                continue

            # Calculate MTLD
            mtld = dm.MTLD({"text": text})

            # Get token info
            token_info = dm.TokenzeText({"text": text})

            results.append({"text": text, "mtld": mtld, "length": token_info["len"]})

        except Exception as e:
            print(f"Error processing segment {idx}: {e}")
            continue

    return pd.DataFrame(results)


def create_clusters(df):
    """Create clusters based on length ranges and MTLD complexity"""

    # Define clustering strategy: (min_len, max_len, n_complexity_levels)
    cluster_config = [
        (2, 4, 1),  # Very short: single cluster
        (5, 15, 2),  # Short: 2 complexity levels
        (15, 40, 3),  # Medium: 3 complexity levels
        (40, 100, 3),  # Medium-long: 3 complexity levels
        (100, 200, 2),  # Long: 2 complexity levels
    ]

    df = df.copy()
    df["cluster"] = -1
    current_cluster = 0

    for min_len, max_len, n_levels in cluster_config:
        # Get segments in this length range
        mask = (df["length"] >= min_len) & (df["length"] <= max_len)
        group_data = df[mask].copy()

        if len(group_data) == 0:
            continue

        # Sort by MTLD and split into complexity levels
        group_data = group_data.sort_values("mtld")
        splits = np.array_split(group_data, n_levels)

        for split in splits:
            df.loc[split.index, "cluster"] = current_cluster
            current_cluster += 1

    # Remove unclustered data
    df = df[df["cluster"] >= 0]

    return df, current_cluster


def plot_clusters(df, n_clusters):
    """Create visualization of clusters"""

    plt.figure(figsize=(15, 10))

    # Generate distinct colors
    colors = plt.cm.tab20(np.linspace(0, 1, n_clusters))

    for cluster_id in range(n_clusters):
        cluster_data = df[df["cluster"] == cluster_id]
        if len(cluster_data) == 0:
            continue

        plt.scatter(
            cluster_data["length"],
            cluster_data["mtld"],
            label=f"Cluster {cluster_id}",
            color=colors[cluster_id % len(colors)],
            alpha=0.6,
            s=30,
        )

    plt.xlabel("Length (tokens)")
    plt.ylabel("MTLD Score")
    plt.title("Text Complexity Clusters for Curriculum Learning")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def save_cluster_examples(df, n_clusters, filename="cluster_examples.txt"):
    """Save example texts from each cluster"""

    with open(filename, "w", encoding="utf-8") as f:
        f.write("CURRICULUM LEARNING CLUSTERS\n")
        f.write("=" * 80 + "\n\n")

        for cluster_id in range(n_clusters):
            cluster_data = df[df["cluster"] == cluster_id]
            if len(cluster_data) == 0:
                continue

            # Sort by MTLD for consistent sampling
            sorted_data = cluster_data.sort_values("mtld")

            # Get representative examples
            low_idx = 0
            mid_idx = len(sorted_data) // 2
            high_idx = -1

            examples = [
                ("LOW COMPLEXITY", sorted_data.iloc[low_idx]),
                ("MEDIUM COMPLEXITY", sorted_data.iloc[mid_idx]),
                ("HIGH COMPLEXITY", sorted_data.iloc[high_idx]),
            ]

            f.write(f"CLUSTER {cluster_id}\n")
            f.write(
                f"Length range: {cluster_data['length'].min():.0f}-{cluster_data['length'].max():.0f} tokens\n"
            )
            f.write(
                f"MTLD range: {cluster_data['mtld'].min():.2f}-{cluster_data['mtld'].max():.2f}\n"
            )
            f.write(f"Sample size: {len(cluster_data)}\n")
            f.write("-" * 40 + "\n")

            for label, example in examples:
                f.write(f"\n{label}:\n")
                f.write(f"Length: {example['length']}, MTLD: {example['mtld']:.2f}\n")
                f.write(f"Text: {example['text']}\n")

            f.write("\n" + "=" * 80 + "\n")


def print_cluster_stats(df):
    """Print statistics for each cluster"""

    print("\nCLUSTER STATISTICS:")
    print("=" * 60)

    stats = (
        df.groupby("cluster")
        .agg(
            {
                "length": ["count", "mean", "std", "min", "max"],
                "mtld": ["mean", "std", "min", "max"],
            }
        )
        .round(2)
    )

    stats.columns = [
        "Count",
        "Len_Mean",
        "Len_Std",
        "Len_Min",
        "Len_Max",
        "MTLD_Mean",
        "MTLD_Std",
        "MTLD_Min",
        "MTLD_Max",
    ]

    print(stats)


def main():
    """Main execution function"""

    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)

    # Initialize tokenizer
    enc = tiktoken.get_encoding("gpt2")

    # Load data
    print("Loading dataset...")
    df = pd.read_csv("tinystories_100mb_splits/train.csv", encoding="utf-8")
    df = df.dropna(subset=["text"])
    print(f"Loaded {len(df)} rows")

    # Create segments
    segments = create_text_segments(df, enc=enc)

    # Calculate metrics
    processed_df = calculate_metrics(segments)
    print(f"\nFinal dataset: {len(processed_df)} segments")

    # Create clusters
    clustered_df, n_clusters = create_clusters(processed_df)
    print(f"\nCreated {n_clusters} clusters")

    # Print statistics
    print_cluster_stats(clustered_df)

    # Create visualization
    plot_clusters(clustered_df, n_clusters)

    # Save examples
    save_cluster_examples(clustered_df, n_clusters)
    print(f"\nExamples saved to cluster_examples.txt")

    # Save the clustered dataset
    clustered_df.to_csv("clustered_curriculum_data.csv", index=False)
    print("Clustered dataset saved to clustered_curriculum_data.csv")

    return clustered_df


if __name__ == "__main__":
    clustered_data = main()
