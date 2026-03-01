"""
Training script for the Mistral Query Router using HF AutoTrain SFT + W&B.
Fine-tunes Ministral 3B with QLoRA to classify queries into model tiers.
"""

import os
import sys
import json
import subprocess
import argparse
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    BASE_MODEL,
    HF_REPO_ID,
    WANDB_PROJECT,
    DATA_DIR,
    LEARNING_RATE,
    TRAINING_EPOCHS,
    BATCH_SIZE,
    GRADIENT_ACCUMULATION_STEPS,
    MAX_SEQ_LENGTH,
    LORA_RANK,
    LORA_ALPHA,
    LORA_DROPOUT,
    QUANTIZATION,
    HF_TOKEN,
    WANDB_API_KEY,
)


def check_prerequisites():
    """Verify all required tools and credentials are available."""
    checks = []

    # Check autotrain
    try:
        result = subprocess.run(
            ["autotrain", "--version"],
            capture_output=True, text=True, timeout=10
        )
        checks.append(("autotrain", True, result.stdout.strip()))
    except (FileNotFoundError, subprocess.TimeoutExpired):
        checks.append(("autotrain", False, "Not installed. Run: pip install autotrain-advanced"))

    # Check W&B
    if WANDB_API_KEY or os.getenv("WANDB_API_KEY"):
        checks.append(("W&B API Key", True, "Set"))
    else:
        checks.append(("W&B API Key", False, "Missing. Set WANDB_API_KEY env var"))

    # Check HF token
    if HF_TOKEN or os.getenv("HF_TOKEN"):
        checks.append(("HF Token", True, "Set"))
    else:
        checks.append(("HF Token", False, "Missing. Set HF_TOKEN env var"))

    # Check training data
    train_file = DATA_DIR / "train.jsonl"
    if train_file.exists():
        line_count = sum(1 for _ in open(train_file))
        checks.append(("Training data", True, f"{line_count} examples"))
    else:
        checks.append(("Training data", False, f"Not found at {train_file}"))

    # Print results
    print("=" * 50)
    print("PREREQUISITE CHECK")
    print("=" * 50)
    all_ok = True
    for name, ok, detail in checks:
        status = "✅" if ok else "❌"
        print(f"  {status} {name}: {detail}")
        if not ok:
            all_ok = False

    return all_ok


def build_autotrain_command(
    data_path: str,
    model: str = BASE_MODEL,
    project_name: str = "mistral-router",
    push_to_hub: bool = True,
    repo_id: str = HF_REPO_ID,
    dry_run: bool = False,
) -> list[str]:
    """Build the AutoTrain CLI command."""
    cmd = [
        "autotrain", "llm",
        "--train",
        "--model", model,
        "--data-path", data_path,
        "--project-name", project_name,
        "--text-column", "text",
        "--trainer", "sft",
        "--peft",
        "--quantization", QUANTIZATION,
        "--lr", str(LEARNING_RATE),
        "--epochs", str(TRAINING_EPOCHS),
        "--batch-size", str(BATCH_SIZE),
        "--gradient-accumulation", str(GRADIENT_ACCUMULATION_STEPS),
        "--block-size", str(MAX_SEQ_LENGTH),
        "--lora-r", str(LORA_RANK),
        "--lora-alpha", str(LORA_ALPHA),
        "--lora-dropout", str(LORA_DROPOUT),
        "--log", "wandb",
    ]

    if push_to_hub and repo_id:
        cmd.extend(["--push-to-hub", "--repo-id", repo_id])

    return cmd


def train_with_transformers(
    data_path: str,
    output_dir: str = "./output",
    model: str = BASE_MODEL,
    push_to_hub: bool = True,
    repo_id: str = HF_REPO_ID,
):
    """
    Alternative: Train directly with transformers + trl (if AutoTrain has issues).
    This gives more control over the training loop.
    """
    import torch
    import wandb
    from datasets import load_dataset
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        TrainingArguments,
    )
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from trl import SFTTrainer, SFTConfig

    # Initialize W&B
    wandb.init(
        project=WANDB_PROJECT,
        name=f"router-{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        config={
            "base_model": model,
            "lora_rank": LORA_RANK,
            "lora_alpha": LORA_ALPHA,
            "learning_rate": LEARNING_RATE,
            "epochs": TRAINING_EPOCHS,
            "batch_size": BATCH_SIZE,
            "quantization": QUANTIZATION,
        },
    )

    print(f"🔄 Loading model: {model}")

    # QLoRA config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    # Load model + tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model_obj = AutoModelForCausalLM.from_pretrained(
        model,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model_obj = prepare_model_for_kbit_training(model_obj)

    # LoRA config
    lora_config = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )

    # Load dataset
    print(f"📂 Loading dataset from: {data_path}")
    dataset = load_dataset("json", data_files={
        "train": str(Path(data_path) / "train.jsonl"),
        "validation": str(Path(data_path) / "val.jsonl"),
    })

    def format_chat(example):
        """Format chat messages into a single string using the tokenizer's chat template."""
        text = tokenizer.apply_chat_template(
            example["messages"], tokenize=False, add_generation_prompt=False
        )
        return {"text": text}

    dataset = dataset.map(format_chat)

    # Training config
    training_args = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=TRAINING_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        bf16=True,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=3,
        report_to="wandb",
        max_seq_length=MAX_SEQ_LENGTH,
        packing=False,
        dataset_text_field="text",
        push_to_hub=push_to_hub,
        hub_model_id=repo_id if push_to_hub else None,
    )

    # Trainer
    trainer = SFTTrainer(
        model=model_obj,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        peft_config=lora_config,
        tokenizer=tokenizer,
    )

    print("🚀 Starting training...")
    trainer.train()

    # Save final model
    print("💾 Saving model...")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    if push_to_hub:
        print(f"📤 Pushing to HF Hub: {repo_id}")
        trainer.push_to_hub()

    wandb.finish()
    print("✅ Training complete!")


def main():
    parser = argparse.ArgumentParser(description="Train the Mistral Query Router")
    parser.add_argument("--data-path", type=str, default=str(DATA_DIR), help="Path to training data")
    parser.add_argument("--output-dir", type=str, default="./output", help="Output directory")
    parser.add_argument("--model", type=str, default=BASE_MODEL, help="Base model to fine-tune")
    parser.add_argument("--repo-id", type=str, default=HF_REPO_ID, help="HF Hub repo ID")
    parser.add_argument("--no-push", action="store_true", help="Don't push to HF Hub")
    parser.add_argument("--check", action="store_true", help="Only check prerequisites")
    parser.add_argument("--method", choices=["autotrain", "transformers"], default="transformers",
                        help="Training method: autotrain CLI or direct transformers")
    parser.add_argument("--dry-run", action="store_true", help="Build command but don't execute")
    args = parser.parse_args()

    if args.check:
        check_prerequisites()
        return

    if not check_prerequisites():
        print("\n❌ Prerequisites not met. Fix the issues above and try again.")
        sys.exit(1)

    if args.method == "autotrain":
        cmd = build_autotrain_command(
            data_path=args.data_path,
            model=args.model,
            push_to_hub=not args.no_push,
            repo_id=args.repo_id,
            dry_run=args.dry_run,
        )
        print(f"\n🏃 AutoTrain command:\n  {' '.join(cmd)}\n")

        if args.dry_run:
            print("🏃 Dry run — not executing")
            return

        subprocess.run(cmd, check=True)

    else:  # transformers
        train_with_transformers(
            data_path=args.data_path,
            output_dir=args.output_dir,
            model=args.model,
            push_to_hub=not args.no_push,
            repo_id=args.repo_id,
        )


if __name__ == "__main__":
    main()
