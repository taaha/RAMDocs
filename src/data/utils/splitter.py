import json
import random
from pathlib import Path
from typing import List, Dict, Any


def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """Load data from a JSONL file."""
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():  # Skip empty lines
                data.append(json.loads(line))
    return data


def save_jsonl(data: List[Dict[str, Any]], file_path: str) -> None:
    """Save data to a JSONL file."""
    with open(file_path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def split_data(
    data: List[Dict[str, Any]], train_ratio: float = 0.8, seed: int = 42
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Split data into train and test sets with a fixed seed for reproducibility.

    Args:
        data: List of data items to split
        train_ratio: Ratio of data to use for training (default: 0.8)
        seed: Random seed for reproducibility (default: 42)

    Returns:
        tuple: (train_data, test_data)
    """
    # Set the random seed for reproducibility
    random.seed(seed)

    # Shuffle the data
    shuffled_data = data.copy()
    random.shuffle(shuffled_data)

    # Calculate split point
    split_idx = int(len(shuffled_data) * train_ratio)

    # Split the data
    train_data = shuffled_data[:split_idx]
    test_data = shuffled_data[split_idx:]

    return train_data, test_data


def split_jsonl_file(
    input_file: str, train_ratio: float = 0.8, seed: int = 42, output_dir: str = "."
) -> tuple[str, str]:
    """
    Split a JSONL file into train and test sets.

    Args:
        input_file: Path to input JSONL file
        train_ratio: Ratio of data to use for training (default: 0.8)
        seed: Random seed for reproducibility (default: 42)
        output_dir: Output directory for train and test files (default: current directory)

    Returns:
        tuple: (train_file_path, test_file_path)
    """
    # Load the data
    print(f"Loading data from {input_file}...")
    data = load_jsonl(input_file)
    print(f"Loaded {len(data)} items")

    # Split the data
    print(f"Splitting data with train_ratio={train_ratio}, seed={seed}...")
    train_data, test_data = split_data(data, train_ratio, seed)

    print(f"Train set: {len(train_data)} items")
    print(f"Test set: {len(test_data)} items")

    # Create output directory if it doesn't exist
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Generate output filenames
    input_path = Path(input_file)
    base_name = input_path.stem

    train_file = output_path / f"{base_name}_train.jsonl"
    test_file = output_path / f"{base_name}_test.jsonl"

    # Save the split data
    print(f"Saving train data to {train_file}...")
    save_jsonl(train_data, str(train_file))

    print(f"Saving test data to {test_file}...")
    save_jsonl(test_data, str(test_file))

    print("Data splitting completed successfully!")
    print(f"Train file: {train_file}")
    print(f"Test file: {test_file}")

    return str(train_file), str(test_file)


if __name__ == "__main__":
    # Example usage
    input_file = "../RAMDocs_test.jsonl"
    train_file, test_file = split_jsonl_file(
        input_file=input_file, train_ratio=0.5, seed=42, output_dir="../split_data"
    )
