import pandas as pd
import numpy as np
import tiktoken
import json
from pathlib import Path


def csv_to_bins(
    csv_file="clustered_curriculum_data.csv",
    output_dir="curriculum_bins",
    val_ratio=0.1,
):
    """Convert clustered CSV data to binary files for curriculum learning"""

    # Load the clustered data
    print(f"Loading {csv_file}...")
    df = pd.read_csv(csv_file)
    print(f"Loaded {len(df)} samples with {df['cluster'].nunique()} clusters")

    # Initialize tokenizer
    enc = tiktoken.get_encoding("gpt2")

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # Split into train/val (stratified by cluster to maintain distribution)
    train_df, val_df = train_val_split(df, val_ratio=val_ratio)

    # Create binary files
    metadata = create_binary_files(train_df, val_df, enc, output_dir)

    print(f"\nConversion complete! Files saved to {output_dir}/")
    print(f"Created {len(metadata['clusters'])} cluster files + 1 validation file")

    return metadata


def train_val_split(df, val_ratio=0.1, random_state=42):
    """Split data into train/val randomly (no regard for cluster distribution)"""

    # Shuffle the entire dataset randomly
    df_shuffled = df.sample(frac=1, random_state=random_state).reset_index(drop=True)

    # Calculate split point
    split_idx = int(len(df_shuffled) * (1 - val_ratio))

    # Split into train and validation
    train_df = df_shuffled[:split_idx].copy()
    val_df = df_shuffled[split_idx:].copy()

    print(f"Train set: {len(train_df)} samples")
    print(f"Validation set: {len(val_df)} samples")
    print(f"Validation clusters: {sorted(val_df['cluster'].unique())}")

    return train_df, val_df


def create_binary_files(train_df, val_df, enc, output_dir):
    """Create separate .bin files for each cluster and one validation file"""

    output_path = Path(output_dir)

    # Get all clusters that exist in the training data
    train_clusters = sorted(train_df["cluster"].unique())

    # Create metadata dictionary
    metadata = {
        "encoding": "gpt2",
        "clusters": {},
        "total_train_samples": len(train_df),
        "total_val_samples": len(val_df),
        "val_file": "val.bin",
        "available_clusters": [
            int(c) for c in train_clusters
        ],  # Convert to regular Python ints
    }

    print(f"\nCreating binary files...")

    # Create validation file (random samples, mixed clusters)
    print("Creating validation file...")
    val_tokens = []
    for _, row in val_df.iterrows():
        tokens = enc.encode_ordinary(row["text"])
        val_tokens.extend(tokens)
        val_tokens.append(enc.encode_ordinary("<|endoftext|>")[0])  # Add separator

    val_array = np.array(val_tokens, dtype=np.uint16)
    val_file = output_path / "val.bin"
    val_array.tofile(val_file)

    print(f"✓ Created {val_file.name} with {len(val_tokens):,} tokens")

    # Create separate bin file for each cluster (only from training data)
    for cluster_id in train_clusters:
        cluster_data = train_df[train_df["cluster"] == cluster_id]

        print(f"Creating cluster {cluster_id} file...")

        # Tokenize all texts in this cluster
        cluster_tokens = []
        for _, row in cluster_data.iterrows():
            tokens = enc.encode_ordinary(row["text"])
            cluster_tokens.extend(tokens)
            cluster_tokens.append(
                enc.encode_ordinary("<|endoftext|>")[0]
            )  # Add separator

        # Convert to numpy array and save
        cluster_array = np.array(cluster_tokens, dtype=np.uint16)
        cluster_file = output_path / f"cluster_{cluster_id}.bin"
        cluster_array.tofile(cluster_file)

        # Store metadata (convert cluster_id to string for JSON)
        metadata["clusters"][str(cluster_id)] = {
            "file": f"cluster_{cluster_id}.bin",
            "samples": len(cluster_data),
            "tokens": len(cluster_tokens),
            "avg_length": float(cluster_data["length"].mean()),
            "avg_mtld": float(cluster_data["mtld"].mean()),
            "length_range": [
                int(cluster_data["length"].min()),
                int(cluster_data["length"].max()),
            ],
            "mtld_range": [
                float(cluster_data["mtld"].min()),
                float(cluster_data["mtld"].max()),
            ],
        }

        print(
            f"✓ Created {cluster_file.name} with {len(cluster_tokens):,} tokens ({len(cluster_data)} samples)"
        )

    # Save metadata
    metadata_file = output_path / "metadata.json"
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"✓ Saved metadata to {metadata_file.name}")

    return metadata


def print_summary(metadata):
    """Print a summary of the created files"""

    print("\n" + "=" * 60)
    print("CURRICULUM DATA SUMMARY")
    print("=" * 60)

    print(f"Total training samples: {metadata['total_train_samples']:,}")
    print(f"Total validation samples: {metadata['total_val_samples']:,}")
    print(f"Number of clusters: {len(metadata['clusters'])}")

    print(f"\nCluster breakdown:")
    for cluster_id, info in metadata["clusters"].items():
        print(f"  Cluster {cluster_id}:")
        print(f"    Samples: {info['samples']:,}")
        print(f"    Tokens: {info['tokens']:,}")
        print(f"    Avg length: {info['avg_length']:.1f}")
        print(f"    Avg MTLD: {info['avg_mtld']:.2f}")
        print(f"    Length range: {info['length_range']}")


if __name__ == "__main__":
    # Convert the CSV to binary files
    metadata = csv_to_bins(
        csv_file="clustered_curriculum_data.csv",
        output_dir="curriculum_bins",
        val_ratio=0.1,
    )

    # Print summary
    print_summary(metadata)
