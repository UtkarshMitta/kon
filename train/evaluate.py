"""
Evaluation script for the Mistral Query Router.
Computes accuracy, per-tier F1, confusion matrix, and confidence calibration.
Logs all metrics and visualizations to W&B.
"""

import json
import sys
import argparse
import numpy as np
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import WANDB_PROJECT, DATA_DIR, TIER_LABELS, TIER_MAP


def load_predictions(pred_file: str) -> tuple[list, list, list]:
    """
    Load predictions from a JSONL file.
    Each line: {"query": "...", "true_tier": 1, "predicted_tier": 2, "confidence": 0.85}
    Returns: (true_labels, predicted_labels, confidences)
    """
    true_labels = []
    pred_labels = []
    confidences = []

    with open(pred_file, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line.strip())
            true_labels.append(data["true_tier"])
            pred_labels.append(data["predicted_tier"])
            confidences.append(data.get("confidence", 0.5))

    return true_labels, pred_labels, confidences


def compute_metrics(true_labels: list, pred_labels: list, confidences: list) -> dict:
    """Compute classification metrics."""
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        precision_score,
        recall_score,
        classification_report,
        confusion_matrix,
    )

    accuracy = accuracy_score(true_labels, pred_labels)
    f1_macro = f1_score(true_labels, pred_labels, average="macro", zero_division=0)
    f1_weighted = f1_score(true_labels, pred_labels, average="weighted", zero_division=0)
    precision = precision_score(true_labels, pred_labels, average="macro", zero_division=0)
    recall = recall_score(true_labels, pred_labels, average="macro", zero_division=0)

    # Per-tier F1
    tier_f1 = f1_score(true_labels, pred_labels, average=None, labels=[1, 2, 3, 4], zero_division=0)

    # Confusion matrix
    cm = confusion_matrix(true_labels, pred_labels, labels=[1, 2, 3, 4])

    # Classification report
    report = classification_report(
        true_labels, pred_labels,
        labels=[1, 2, 3, 4],
        target_names=["small", "medium", "large", "xlarge"],
        zero_division=0,
    )

    # Confidence calibration (Expected Calibration Error)
    ece = compute_ece(true_labels, pred_labels, confidences)

    metrics = {
        "accuracy": accuracy,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "precision_macro": precision,
        "recall_macro": recall,
        "ece": ece,
        "per_tier_f1": {
            "small": float(tier_f1[0]),
            "medium": float(tier_f1[1]),
            "large": float(tier_f1[2]),
            "xlarge": float(tier_f1[3]),
        },
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
    }

    return metrics


def compute_ece(true_labels: list, pred_labels: list, confidences: list, n_bins: int = 10) -> float:
    """Compute Expected Calibration Error."""
    confidences_arr = np.array(confidences)
    correct = np.array([t == p for t, p in zip(true_labels, pred_labels)])

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    total = len(confidences)

    for i in range(n_bins):
        mask = (confidences_arr > bin_boundaries[i]) & (confidences_arr <= bin_boundaries[i + 1])
        if mask.sum() == 0:
            continue
        bin_conf = confidences_arr[mask].mean()
        bin_acc = correct[mask].mean()
        ece += mask.sum() / total * abs(bin_acc - bin_conf)

    return float(ece)


def plot_confusion_matrix(cm, labels, save_path: str = None):
    """Create and optionally save a confusion matrix heatmap."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=labels, yticklabels=labels, ax=ax,
    )
    ax.set_xlabel("Predicted Tier")
    ax.set_ylabel("True Tier")
    ax.set_title("Query Router Confusion Matrix")

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"💾 Confusion matrix saved to {save_path}")

    return fig


def log_to_wandb(metrics: dict, cm_fig=None, run_name: str = None):
    """Log all metrics and visualizations to W&B."""
    import wandb

    wandb.init(
        project=WANDB_PROJECT,
        name=run_name or "evaluation",
        job_type="evaluation",
    )

    # Log scalar metrics
    wandb.log({
        "eval/accuracy": metrics["accuracy"],
        "eval/f1_macro": metrics["f1_macro"],
        "eval/f1_weighted": metrics["f1_weighted"],
        "eval/precision_macro": metrics["precision_macro"],
        "eval/recall_macro": metrics["recall_macro"],
        "eval/ece": metrics["ece"],
    })

    # Log per-tier F1
    for tier, f1 in metrics["per_tier_f1"].items():
        wandb.log({f"eval/f1_{tier}": f1})

    # Log confusion matrix figure
    if cm_fig:
        wandb.log({"eval/confusion_matrix": wandb.Image(cm_fig)})

    # Log classification report as text
    wandb.log({"eval/classification_report": wandb.Html(
        f"<pre>{metrics['classification_report']}</pre>"
    )})

    wandb.finish()
    print("📊 Metrics logged to W&B")


def evaluate_from_test_set(
    model_path: str,
    test_file: str,
    output_file: str = None,
):
    """Run inference on test set and save predictions."""
    from inference.router import MistralRouter

    router = MistralRouter(model_path)

    predictions = []
    with open(test_file, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line.strip())
            messages = data["messages"]

            # Extract query and true tier
            query = next(m["content"] for m in messages if m["role"] == "user")
            true_output = json.loads(
                next(m["content"] for m in messages if m["role"] == "assistant")
            )

            # Get prediction
            result = router.route(query)

            predictions.append({
                "query": query,
                "true_tier": true_output["model_tier"],
                "predicted_tier": result["model_tier"],
                "confidence": result["confidence"],
            })

    # Save predictions
    if output_file:
        with open(output_file, "w", encoding="utf-8") as f:
            for pred in predictions:
                f.write(json.dumps(pred) + "\n")
        print(f"💾 Predictions saved to {output_file}")

    return predictions


def main():
    parser = argparse.ArgumentParser(description="Evaluate the Mistral Query Router")
    parser.add_argument("--predictions", type=str, help="Path to predictions JSONL file")
    parser.add_argument("--model-path", type=str, help="Path to fine-tuned model (runs inference on test set)")
    parser.add_argument("--test-file", type=str, default=str(DATA_DIR / "test.jsonl"), help="Test set file")
    parser.add_argument("--output", type=str, default="predictions.jsonl", help="Output predictions file")
    parser.add_argument("--wandb", action="store_true", help="Log metrics to W&B")
    parser.add_argument("--save-cm", type=str, default="confusion_matrix.png", help="Save confusion matrix")
    args = parser.parse_args()

    # Either load existing predictions or generate them
    if args.predictions:
        print(f"📂 Loading predictions from {args.predictions}")
        true_labels, pred_labels, confidences = load_predictions(args.predictions)
    elif args.model_path:
        print(f"🔄 Running inference on test set with model: {args.model_path}")
        predictions = evaluate_from_test_set(args.model_path, args.test_file, args.output)
        true_labels = [p["true_tier"] for p in predictions]
        pred_labels = [p["predicted_tier"] for p in predictions]
        confidences = [p["confidence"] for p in predictions]
    else:
        print("❌ Provide either --predictions or --model-path")
        sys.exit(1)

    # Compute metrics
    print("\n📊 Computing metrics...")
    metrics = compute_metrics(true_labels, pred_labels, confidences)

    # Print results
    print(f"\n{'=' * 50}")
    print("EVALUATION RESULTS")
    print(f"{'=' * 50}")
    print(f"  Accuracy:       {metrics['accuracy']:.4f}")
    print(f"  F1 (macro):     {metrics['f1_macro']:.4f}")
    print(f"  F1 (weighted):  {metrics['f1_weighted']:.4f}")
    print(f"  Precision:      {metrics['precision_macro']:.4f}")
    print(f"  Recall:         {metrics['recall_macro']:.4f}")
    print(f"  ECE:            {metrics['ece']:.4f}")
    print(f"\n  Per-Tier F1:")
    for tier, f1 in metrics["per_tier_f1"].items():
        print(f"    {tier:8s}: {f1:.4f}")
    print(f"\n{metrics['classification_report']}")

    # Plot confusion matrix
    import numpy as np
    cm = np.array(metrics["confusion_matrix"])
    cm_fig = plot_confusion_matrix(cm, ["small", "medium", "large", "xlarge"], args.save_cm)

    # Log to W&B
    if args.wandb:
        log_to_wandb(metrics, cm_fig)

    print("✅ Evaluation complete!")


if __name__ == "__main__":
    main()
