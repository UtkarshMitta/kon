"""
Feedback collection for the Mistral Query Router.
Collects human corrections and stores them for retraining.
"""

import json
import sys
import argparse
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    FEEDBACK_FILE,
    FEEDBACK_DIR,
    FEEDBACK_THRESHOLD,
    SYSTEM_PROMPT,
    TIER_MAP,
    TIER_LABELS,
)


class FeedbackCollector:
    """
    Collects and stores human corrections for the query router.

    Usage:
        collector = FeedbackCollector()
        collector.add(
            query="Explain quantum entanglement in detail",
            predicted_tier=2,
            corrected_tier=3,
        )
        print(collector.count)  # Number of pending feedbacks
    """

    def __init__(self, feedback_file: str | Path = FEEDBACK_FILE):
        self.feedback_file = Path(feedback_file)
        self.feedback_file.parent.mkdir(parents=True, exist_ok=True)

    @property
    def count(self) -> int:
        """Number of feedback entries in the current batch."""
        if not self.feedback_file.exists():
            return 0
        return sum(1 for _ in open(self.feedback_file, "r", encoding="utf-8"))

    @property
    def is_threshold_reached(self) -> bool:
        """Whether the feedback threshold for retraining is met."""
        return self.count >= FEEDBACK_THRESHOLD

    def add(
        self,
        query: str,
        predicted_tier: int,
        corrected_tier: int,
        confidence: float = None,
        notes: str = None,
    ) -> dict:
        """
        Add a human correction.

        Args:
            query: The original user query
            predicted_tier: What the model predicted (1-4)
            corrected_tier: What the human says it should be (1-4)
            confidence: The model's confidence for this prediction
            notes: Optional human notes explaining the correction

        Returns:
            The feedback entry dict
        """
        # Validate tiers
        if predicted_tier not in [1, 2, 3, 4]:
            raise ValueError(f"predicted_tier must be 1-4, got {predicted_tier}")
        if corrected_tier not in [1, 2, 3, 4]:
            raise ValueError(f"corrected_tier must be 1-4, got {corrected_tier}")

        entry = {
            "query": query,
            "predicted_tier": predicted_tier,
            "corrected_tier": corrected_tier,
            "confidence": confidence,
            "notes": notes,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "is_correct": predicted_tier == corrected_tier,
        }

        # Append to feedback file
        with open(self.feedback_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        current_count = self.count
        print(f"📝 Feedback #{current_count} recorded: "
              f"Tier {predicted_tier} → {corrected_tier} "
              f"({'✓ correct' if entry['is_correct'] else '✗ corrected'})")

        if self.is_threshold_reached:
            print(f"🔔 Threshold reached! {current_count}/{FEEDBACK_THRESHOLD} feedbacks collected.")
            print(f"   Run `python feedback/retrain_trigger.py` to trigger retraining.")

        return entry

    def get_all(self) -> list[dict]:
        """Load all feedback entries."""
        if not self.feedback_file.exists():
            return []

        entries = []
        with open(self.feedback_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    entries.append(json.loads(line))
        return entries

    def to_sft_format(self) -> list[dict]:
        """
        Convert feedback entries to SFT training format.
        Uses the corrected tier (human label) as the ground truth.
        """
        training_examples = []
        for entry in self.get_all():
            example = {
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": entry["query"]},
                    {
                        "role": "assistant",
                        "content": json.dumps({
                            "model_tier": entry["corrected_tier"],
                            "confidence": 0.95,
                        }),
                    },
                ]
            }
            training_examples.append(example)
        return training_examples

    def to_dpo_format(self) -> list[dict]:
        """
        Convert feedback entries to DPO (Direct Preference Optimization) format.
        Creates (prompt, chosen, rejected) triads.
        """
        dpo_examples = []
        for entry in self.get_all():
            if entry["is_correct"]:
                continue  # DPO needs a preference (Chosen > Rejected)
            
            # For corrections, the model's prediction is 'rejected' 
            # and the human correction is 'chosen'
            example = {
                "prompt": f"{SYSTEM_PROMPT}\n\nUser Query: {entry['query']}",
                "chosen": json.dumps({"model_tier": entry["corrected_tier"], "confidence": 1.0}),
                "rejected": json.dumps({"model_tier": entry["predicted_tier"], "confidence": entry.get("confidence", 0.5)})
            }
            dpo_examples.append(example)
        return dpo_examples

    def get_stats(self) -> dict:
        """Get feedback statistics."""
        entries = self.get_all()
        if not entries:
            return {"total": 0, "correct": 0, "accuracy": 0.0, "by_tier": {}}

        correct = sum(1 for e in entries if e["is_correct"])
        total = len(entries)

        # Per-tier breakdown
        tier_stats = {}
        for tier in [1, 2, 3, 4]:
            tier_entries = [e for e in entries if e["corrected_tier"] == tier]
            tier_stats[tier] = {
                "total": len(tier_entries),
                "corrected_from": {
                    t: sum(1 for e in tier_entries if e["predicted_tier"] == t)
                    for t in [1, 2, 3, 4]
                },
            }

        return {
            "total": total,
            "correct": correct,
            "incorrect": total - correct,
            "accuracy": correct / total if total > 0 else 0.0,
            "by_tier": tier_stats,
            "threshold": FEEDBACK_THRESHOLD,
            "ready_for_retrain": total >= FEEDBACK_THRESHOLD,
        }

    def reset(self, archive: bool = True):
        """
        Reset feedback after retraining.
        Optionally archive old feedback.
        """
        if not self.feedback_file.exists():
            return

        if archive:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            archive_path = self.feedback_file.parent / f"feedback_archive_{timestamp}.jsonl"
            self.feedback_file.rename(archive_path)
            print(f"📦 Archived feedback to {archive_path}")
        else:
            self.feedback_file.unlink()
            print("🗑️  Feedback file deleted")


def main():
    parser = argparse.ArgumentParser(description="Feedback collection for query router")
    parser.add_argument("--add", nargs=3, metavar=("QUERY", "PREDICTED", "CORRECTED"),
                        help="Add feedback: query predicted_tier corrected_tier")
    parser.add_argument("--stats", action="store_true", help="Show feedback statistics")
    parser.add_argument("--export", type=str, help="Export feedback as training JSONL (SFT)")
    parser.add_argument("--export-dpo", type=str, help="Export feedback as DPO JSONL (RL)")
    parser.add_argument("--reset", action="store_true", help="Reset feedback (with archive)")
    parser.add_argument("--test", action="store_true", help="Add test feedback entries")
    args = parser.parse_args()

    collector = FeedbackCollector()

    if args.add:
        query, predicted, corrected = args.add
        collector.add(query, int(predicted), int(corrected))

    elif args.stats:
        stats = collector.get_stats()
        print(f"\n📊 Feedback Statistics:")
        print(f"   Total: {stats['total']}")
        print(f"   Correct: {stats['correct']} ({stats['accuracy']:.1%})")
        print(f"   Incorrect: {stats['incorrect']}")
        print(f"   Threshold: {stats['threshold']}")
        print(f"   Ready for retrain: {'✅ Yes' if stats['ready_for_retrain'] else '❌ No'}")

    elif args.export:
        examples = collector.to_training_format()
        with open(args.export, "w", encoding="utf-8") as f:
            for ex in examples:
                f.write(json.dumps(ex) + "\n")
        print(f"💾 Exported {len(examples)} SFT training examples to {args.export}")

    elif args.export_dpo:
        examples = collector.to_dpo_format()
        with open(args.export_dpo, "w", encoding="utf-8") as f:
            for ex in examples:
                f.write(json.dumps(ex) + "\n")
        print(f"💾 Exported {len(examples)} DPO (RL) training examples to {args.export_dpo}")

    elif args.reset:
        collector.reset(archive=True)

    elif args.test:
        print("🧪 Adding test feedback entries...")
        test_feedbacks = [
            ("What is 2+2?", 1, 1),
            ("Explain quantum computing", 2, 3),
            ("Hello!", 2, 1),
            ("Design a distributed system", 3, 4),
            ("Translate hello to Spanish", 1, 2),
        ]
        for query, pred, correct in test_feedbacks:
            collector.add(query, pred, correct)
        print(f"\n   Total feedback: {collector.count}")
        print(f"   Threshold reached: {collector.is_threshold_reached}")

    else:
        print(f"📝 Feedback count: {collector.count}/{FEEDBACK_THRESHOLD}")
        print("   Use --add, --stats, --export, --reset, or --test")


if __name__ == "__main__":
    main()
