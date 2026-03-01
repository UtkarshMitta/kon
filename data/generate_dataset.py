"""
Synthetic dataset generation for the Mistral Query Router.
Uses a strong LLM (Mistral Large) as a teacher to generate diverse,
balanced training examples across all 4 tiers.
"""

import json
import random
import argparse
import sys
from pathlib import Path
from tqdm import tqdm

# Add parent dir to path for config import
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    SYSTEM_PROMPT,
    DATASET_SIZE,
    EXAMPLES_PER_TIER,
    TRAIN_SPLIT,
    VAL_SPLIT,
    TEST_SPLIT,
    GENERATION_MODEL,
    MISTRAL_API_KEY,
    DATA_DIR,
    TIER_LABELS,
    TIER_MAP,
)
from data.tier_rubric import TIER_RUBRIC, QUERY_CATEGORIES, get_seed_examples_flat


# ──────────────────────────────────────────────
# Generation prompt template
# ──────────────────────────────────────────────
GENERATION_PROMPT = """You are generating training data for a query router model. 
The router classifies user queries into complexity tiers.

{tier_description}

Category: {category}

Generate {count} diverse, realistic user queries that belong to Tier {tier_num} ({tier_label}).
The queries should:
- Be in the "{category}" category
- Match the complexity level described above
- Be diverse in phrasing and specific topics
- Be realistic queries a user might actually ask an AI assistant
- NOT include the tier label or any metadata — just the raw query text

Respond with a JSON array of strings, nothing else:
["query 1", "query 2", ...]"""


def generate_with_mistral(prompt: str, api_key: str) -> str:
    """Generate text using Mistral API."""
    from mistralai import Mistral

    client = Mistral(api_key=api_key)
    response = client.chat.complete(
        model=GENERATION_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.8,
        max_tokens=4096,
    )
    return response.choices[0].message.content


def parse_queries_from_response(response: str) -> list[str]:
    """Parse a JSON array of queries from the LLM response."""
    # Try direct JSON parse
    try:
        # Find the JSON array in the response
        start = response.index("[")
        end = response.rindex("]") + 1
        return json.loads(response[start:end])
    except (ValueError, json.JSONDecodeError):
        # Fallback: extract quoted strings
        import re
        matches = re.findall(r'"([^"]+)"', response)
        return matches


def format_as_chat_messages(query: str, tier_label: str) -> dict:
    """Format a single example as chat messages for SFT training."""
    tier_num = TIER_MAP[tier_label]
    # Confidence is high for training data (teacher-labeled)
    confidence = round(random.uniform(0.85, 0.99), 2)

    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": query},
            {
                "role": "assistant",
                "content": json.dumps(
                    {"model_tier": tier_num, "confidence": confidence}
                ),
            },
        ]
    }


def generate_synthetic_data(
    api_key: str,
    count_per_tier: int = EXAMPLES_PER_TIER,
    dry_run: bool = False,
) -> list[dict]:
    """Generate synthetic training data using teacher model."""
    all_examples = []

    # Step 1: Include seed examples
    print("📋 Adding seed examples...")
    for query, tier_label in get_seed_examples_flat():
        example = format_as_chat_messages(query, tier_label)
        all_examples.append(example)
    print(f"   Added {len(all_examples)} seed examples")

    if dry_run:
        print("🏃 Dry run — skipping LLM generation")
        return all_examples

    # Step 2: Generate synthetic examples via LLM
    remaining_per_tier = {
        label: count_per_tier - len(TIER_RUBRIC[label]["seed_examples"])
        for label in TIER_LABELS
    }

    for tier_label in TIER_LABELS:
        tier = TIER_RUBRIC[tier_label]
        remaining = remaining_per_tier[tier_label]

        if remaining <= 0:
            continue

        # Distribute across categories
        queries_per_category = max(1, remaining // len(QUERY_CATEGORIES))

        print(f"\n🔄 Generating Tier {tier['tier']} ({tier_label}) examples...")

        for category in tqdm(QUERY_CATEGORIES, desc=f"  Tier {tier['tier']}"):
            criteria_str = "\n".join(f"  - {c}" for c in tier["criteria"])
            tier_desc = (
                f"Tier {tier['tier']} ({tier_label}): {tier['description']}\n"
                f"Criteria:\n{criteria_str}"
            )

            prompt = GENERATION_PROMPT.format(
                tier_description=tier_desc,
                category=category,
                count=queries_per_category,
                tier_num=tier["tier"],
                tier_label=tier_label,
            )

            try:
                response = generate_with_mistral(prompt, api_key)
                queries = parse_queries_from_response(response)

                for q in queries[:queries_per_category]:
                    example = format_as_chat_messages(q.strip(), tier_label)
                    all_examples.append(example)

            except Exception as e:
                print(f"    ⚠️  Error generating for {category}: {e}")
                continue

    return all_examples


def split_and_save(examples: list[dict], output_dir: Path):
    """Shuffle, split into train/val/test, and save as JSONL files."""
    output_dir.mkdir(parents=True, exist_ok=True)

    random.shuffle(examples)
    n = len(examples)
    train_end = int(n * TRAIN_SPLIT)
    val_end = train_end + int(n * VAL_SPLIT)

    splits = {
        "train": examples[:train_end],
        "val": examples[train_end:val_end],
        "test": examples[val_end:],
    }

    for split_name, split_data in splits.items():
        filepath = output_dir / f"{split_name}.jsonl"
        with open(filepath, "w", encoding="utf-8") as f:
            for example in split_data:
                f.write(json.dumps(example, ensure_ascii=False) + "\n")
        print(f"💾 Saved {len(split_data)} examples to {filepath}")

    # Also save a combined file for convenience
    all_path = output_dir / "all.jsonl"
    with open(all_path, "w", encoding="utf-8") as f:
        for example in examples:
            f.write(json.dumps(example, ensure_ascii=False) + "\n")
    print(f"💾 Saved {len(examples)} total examples to {all_path}")


def validate_dataset(filepath: str | Path):
    """Validate the format of a generated dataset file."""
    filepath = Path(filepath)
    if not filepath.exists():
        print(f"❌ File not found: {filepath}")
        return False

    errors = 0
    tier_counts = {1: 0, 2: 0, 3: 0, 4: 0}
    total = 0

    with open(filepath, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            total += 1
            try:
                data = json.loads(line.strip())
            except json.JSONDecodeError:
                print(f"  ❌ Line {i}: Invalid JSON")
                errors += 1
                continue

            # Check structure
            if "messages" not in data:
                print(f"  ❌ Line {i}: Missing 'messages' key")
                errors += 1
                continue

            messages = data["messages"]
            if len(messages) < 2:
                print(f"  ❌ Line {i}: Need at least user + assistant messages")
                errors += 1
                continue

            # Check assistant response is valid JSON
            assistant_msg = messages[-1]
            if assistant_msg.get("role") != "assistant":
                print(f"  ❌ Line {i}: Last message is not assistant")
                errors += 1
                continue

            try:
                output = json.loads(assistant_msg["content"])
                tier = output.get("model_tier")
                conf = output.get("confidence")

                if tier not in [1, 2, 3, 4]:
                    print(f"  ❌ Line {i}: Invalid model_tier: {tier}")
                    errors += 1
                elif not (0.0 <= conf <= 1.0):
                    print(f"  ❌ Line {i}: Invalid confidence: {conf}")
                    errors += 1
                else:
                    tier_counts[tier] += 1
            except (json.JSONDecodeError, TypeError):
                print(f"  ❌ Line {i}: Assistant content is not valid JSON")
                errors += 1

    # Summary
    print(f"\n📊 Validation Summary for {filepath.name}:")
    print(f"   Total examples: {total}")
    print(f"   Errors: {errors}")
    print(f"   Tier distribution:")
    for tier, count in sorted(tier_counts.items()):
        pct = (count / total * 100) if total > 0 else 0
        print(f"     Tier {tier}: {count} ({pct:.1f}%)")

    if errors == 0:
        print("   ✅ All examples valid!")
    else:
        print(f"   ⚠️  {errors} errors found")

    return errors == 0


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic router training data")
    parser.add_argument("--count", type=int, default=DATASET_SIZE, help="Total examples to generate")
    parser.add_argument("--output", type=str, default=str(DATA_DIR), help="Output directory")
    parser.add_argument("--dry-run", action="store_true", help="Only use seed examples, no LLM calls")
    parser.add_argument("--validate", type=str, help="Validate an existing dataset file")
    parser.add_argument("--api-key", type=str, default=MISTRAL_API_KEY, help="Mistral API key")
    args = parser.parse_args()

    if args.validate:
        validate_dataset(args.validate)
        return

    if not args.dry_run and not args.api_key:
        print("❌ MISTRAL_API_KEY required for generation. Use --dry-run for seed-only mode.")
        print("   Set via environment variable or --api-key flag.")
        sys.exit(1)

    count_per_tier = args.count // len(TIER_LABELS)
    print(f"🚀 Generating {args.count} examples ({count_per_tier} per tier)")
    print(f"   Output: {args.output}")
    print(f"   Teacher model: {GENERATION_MODEL}")

    examples = generate_synthetic_data(
        api_key=args.api_key,
        count_per_tier=count_per_tier,
        dry_run=args.dry_run,
    )

    output_dir = Path(args.output)
    split_and_save(examples, output_dir)

    # Validate the training split
    print("\n🔍 Validating training split...")
    validate_dataset(output_dir / "train.jsonl")


if __name__ == "__main__":
    main()
