"""
Convert the user's existing dataset (JSON array with prompt + label)
into the SFT chat-messages JSONL format for training.
"""

import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import SYSTEM_PROMPT, TRAIN_SPLIT, VAL_SPLIT, DATA_DIR


def convert_dataset(input_file: str, output_dir: str = None):
    """Convert prompt/label JSON to chat-messages JSONL."""
    output_dir = Path(output_dir or DATA_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load source data
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"Loaded {len(data)} examples from {input_file}")

    # Convert to SFT chat format
    examples = []
    tier_counts = {1: 0, 2: 0, 3: 0, 4: 0}

    for item in data:
        prompt = item["prompt"]
        label = int(item["label"])
        confidence = round(random.uniform(0.88, 0.99), 2)

        example = {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
                {
                    "role": "assistant",
                    "content": json.dumps({
                        "model_tier": label,
                        "confidence": confidence,
                    }),
                },
            ]
        }
        examples.append(example)
        tier_counts[label] += 1

    # Print distribution
    print(f"\nTier distribution:")
    for tier, count in sorted(tier_counts.items()):
        pct = count / len(examples) * 100
        print(f"  Tier {tier}: {count} ({pct:.1f}%)")

    # Shuffle and split
    random.seed(42)
    random.shuffle(examples)

    n = len(examples)
    train_end = int(n * TRAIN_SPLIT)
    val_end = train_end + int(n * VAL_SPLIT)

    splits = {
        "train": examples[:train_end],
        "val": examples[train_end:val_end],
        "test": examples[val_end:],
    }

    # Save
    for split_name, split_data in splits.items():
        filepath = output_dir / f"{split_name}.jsonl"
        with open(filepath, "w", encoding="utf-8") as f:
            for ex in split_data:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
        print(f"Saved {len(split_data)} examples to {filepath}")

    print(f"\nDone! Files are in {output_dir}")


if __name__ == "__main__":
    input_file = sys.argv[1] if len(sys.argv) > 1 else str(
        Path(__file__).parent.parent.parent / "combined_prompts_1_to_4.json"
    )
    convert_dataset(input_file)
