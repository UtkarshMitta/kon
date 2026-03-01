"""
Retrain trigger for the Mistral Query Router.
Checks feedback count and triggers retraining when threshold is met.
Merges feedback into training dataset and launches AutoTrain/transformers training.
"""

import json
import sys
import argparse
import shutil
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    FEEDBACK_THRESHOLD,
    DATA_DIR,
    HF_DATASET_ID,
    HF_TOKEN,
    BASE_MODEL,
    HF_REPO_ID,
    WANDB_PROJECT,
)
from feedback.collect import FeedbackCollector


def merge_feedback_into_dataset(
    feedback_file: str | Path,
    train_file: str | Path,
    output_file: str | Path = None,
) -> int:
    """
    Merge feedback entries into the training dataset.
    
    Args:
        feedback_file: Path to feedback JSONL
        train_file: Path to existing training JSONL
        output_file: Path for merged output (defaults to overwriting train_file)
    
    Returns:
        Number of new examples added
    """
    collector = FeedbackCollector(feedback_file)
    new_examples = collector.to_training_format()

    if not new_examples:
        print("⚠️  No feedback entries to merge")
        return 0

    output_file = output_file or train_file
    train_file = Path(train_file)

    # Load existing training data
    existing = []
    if train_file.exists():
        with open(train_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    existing.append(json.loads(line))

    print(f"📂 Existing training examples: {len(existing)}")
    print(f"📝 New feedback examples: {len(new_examples)}")

    # Merge
    merged = existing + new_examples

    # Save
    with open(output_file, "w", encoding="utf-8") as f:
        for example in merged:
            f.write(json.dumps(example, ensure_ascii=False) + "\n")

    print(f"💾 Merged dataset saved: {len(merged)} total examples → {output_file}")
    return len(new_examples)


def push_dataset_to_hub(data_dir: str | Path, dataset_id: str = HF_DATASET_ID):
    """Push updated dataset to HuggingFace Hub."""
    from huggingface_hub import HfApi

    api = HfApi(token=HF_TOKEN)
    data_dir = Path(data_dir)

    print(f"📤 Pushing dataset to HF Hub: {dataset_id}")

    # Create repo if it doesn't exist
    try:
        api.create_repo(dataset_id, repo_type="dataset", exist_ok=True)
    except Exception as e:
        print(f"⚠️  Could not create repo: {e}")

    # Upload all JSONL files
    for jsonl_file in data_dir.glob("*.jsonl"):
        api.upload_file(
            path_or_fileobj=str(jsonl_file),
            path_in_repo=jsonl_file.name,
            repo_id=dataset_id,
            repo_type="dataset",
        )
        print(f"   Uploaded {jsonl_file.name}")

    print(f"✅ Dataset pushed to https://huggingface.co/datasets/{dataset_id}")


def trigger_retrain(
    method: str = "transformers",
    data_path: str = None,
    model: str = BASE_MODEL,
    repo_id: str = HF_REPO_ID,
):
    """Trigger a retraining run."""
    import subprocess

    data_path = data_path or str(DATA_DIR)
    train_script = Path(__file__).parent.parent / "train" / "train_autotrain.py"

    run_name = f"retrain-{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir = f"./output/{run_name}"

    cmd = [
        sys.executable, str(train_script),
        "--data-path", data_path,
        "--output-dir", output_dir,
        "--model", model,
        "--repo-id", repo_id,
        "--method", method,
    ]

    print(f"\n🚀 Triggering retrain: {run_name}")
    print(f"   Method: {method}")
    print(f"   Data: {data_path}")
    print(f"   Model: {model}")
    print(f"   Output: {output_dir}")
    print(f"   Command: {' '.join(cmd)}\n")

    subprocess.run(cmd, check=True)


def run_retrain_pipeline(
    method: str = "transformers",
    mode: str = "sft",
    push_dataset: bool = True,
    push_model: bool = True,
    force: bool = False,
):
    """
    Full retrain pipeline:
    1. Check feedback threshold
    2. Merge feedback into training data (SFT or DPO)
    3. Push updated dataset to HF Hub
    4. Trigger retraining (SFT or RL/DPO)
    5. Reset feedback counter
    """
    collector = FeedbackCollector()

    # Step 1: Check threshold
    count = collector.count
    print(f"\n{'=' * 50}")
    print(f"RETRAIN PIPELINE ({mode.upper()} Mode)")
    print(f"{'=' * 50}")
    print(f"  Feedback count: {count}/{FEEDBACK_THRESHOLD}")

    if not force and count < FEEDBACK_THRESHOLD:
        print(f"  ⏳ Not enough feedback yet. Need {FEEDBACK_THRESHOLD - count} more.")
        return False
    
    if force:
        print("  ⚡ Force mode active: skipping threshold check.")

    # Step 2: Merge/Export feedback
    print(f"\n📋 Step 1: Exporting feedback for {mode.upper()}...")
    if mode == "dpo":
        dpo_file = DATA_DIR / "dpo_feedback.jsonl"
        examples = collector.to_dpo_format()
        with open(dpo_file, "w", encoding="utf-8") as f:
            for ex in examples:
                f.write(json.dumps(ex) + "\n")
        print(f"  💾 Exported {len(examples)} DPO examples to {dpo_file}")
    else:
        train_file = DATA_DIR / "train.jsonl"
        added = merge_feedback_into_dataset(
            feedback_file=collector.feedback_file,
            train_file=train_file,
        )
        if added == 0:
            print("  ⚠️  No new examples to add. Skipping retrain.")
            return False

    # Step 3: Push dataset to HF Hub
    if push_dataset:
        print(f"\n📋 Step 2: Pushing dataset to HF Hub...")
        try:
            push_dataset_to_hub(DATA_DIR)
        except Exception as e:
            print(f"  ⚠️  Could not push dataset: {e}")

    # Step 4: Trigger retraining
    print(f"\n📋 Step 3: Triggering {mode.upper()} training...")
    try:
        import subprocess
        train_script = "train_dpo.py" if mode == "dpo" else "train_autotrain.py"
        script_path = Path(__file__).parent.parent / "train" / train_script
        
        cmd = [sys.executable, str(script_path)]
        if mode == "sft":
            cmd.extend(["--method", method])
            
        print(f"🚀 Executing: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
    except Exception as e:
        print(f"  ❌ Training failed: {e}")
        return False

    # Step 5: Reset feedback
    print(f"\n📋 Step 4: Resetting feedback counter...")
    collector.reset(archive=True)

    print(f"\n✅ Retrain pipeline complete!")
    return True


def main():
    parser = argparse.ArgumentParser(description="Retrain trigger for query router")
    parser.add_argument("--check", action="store_true", help="Check if retrain is needed")
    parser.add_argument("--force", action="store_true", help="Force retrain regardless of threshold")
    parser.add_argument("--mode", choices=["sft", "dpo"], default="sft", help="Retrain mode: SFT or RL/DPO")
    parser.add_argument("--method", choices=["autotrain", "transformers"], default="transformers")
    parser.add_argument("--no-push", action="store_true", help="Don't push to HF Hub")
    args = parser.parse_args()

    if args.check:
        collector = FeedbackCollector()
        count = collector.count
        stats = collector.get_stats()
        print(f"📊 Feedback: {count}/{FEEDBACK_THRESHOLD}")
        print(f"   Ready for retrain: {'✅ Yes' if stats['ready_for_retrain'] else '❌ No'}")
        return

    # Run pipeline
    run_retrain_pipeline(
        method=args.method,
        mode=args.mode,
        push_dataset=not args.no_push,
        force=args.force,
    )


if __name__ == "__main__":
    main()
