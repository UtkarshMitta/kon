"""
Launch training on HuggingFace's compute (AutoTrain remote).
Uses your hackathon credits — no local GPU needed.
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv(Path(__file__).parent.parent / ".env")

from huggingface_hub import HfApi

HF_TOKEN = os.getenv("HF_TOKEN")
WANDB_API_KEY = os.getenv("WANDB_API_KEY")
HF_REPO_ID = os.getenv("HF_REPO_ID", "mistral-hackaton-2026/mistral-query-router")
HF_DATASET_ID = os.getenv("HF_DATASET_ID", "mistral-hackaton-2026/router-training-data")
WANDB_PROJECT = os.getenv("WANDB_PROJECT", "mistral-router")

BASE_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"


def launch_remote_training():
    """Launch AutoTrain training job on HuggingFace's infrastructure."""
    from autotrain.app_utils import run_training

    # AutoTrain remote training params
    params = {
        "model": BASE_MODEL,
        "data_path": HF_DATASET_ID,
        "trainer": "sft",
        "text_column": "text",
        "chat_template": "mistral",
        "learning_rate": 2e-4,
        "num_epochs": 3,
        "batch_size": 4,
        "gradient_accumulation": 4,
        "block_size": 512,
        "peft": True,
        "quantization": "int4",
        "lora_r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "target_modules": "q_proj,k_proj,v_proj,o_proj",
        "push_to_hub": True,
        "repo_id": HF_REPO_ID,
        "token": HF_TOKEN,
        "log": "wandb",
    }

    print("=" * 50)
    print("LAUNCHING REMOTE TRAINING ON HF")
    print("=" * 50)
    print(f"  Base model:  {BASE_MODEL}")
    print(f"  Dataset:     {HF_DATASET_ID}")
    print(f"  Output:      {HF_REPO_ID}")
    print(f"  W&B project: {WANDB_PROJECT}")
    print(f"  Method:      SFT + QLoRA (int4)")
    print("=" * 50)


def launch_via_cli():
    """Launch training using the AutoTrain CLI (simpler approach)."""
    import subprocess

    cmd = [
        "autotrain", "llm",
        "--train",
        "--model", BASE_MODEL,
        "--data-path", HF_DATASET_ID,
        "--project-name", "mistral-query-router",
        "--text-column", "text",
        "--trainer", "sft",
        "--peft",
        "--quantization", "int4",
        "--lr", "2e-4",
        "--epochs", "3",
        "--batch-size", "4",
        "--gradient-accumulation", "4",
        "--block-size", "512",
        "--lora-r", "16",
        "--lora-alpha", "32",
        "--lora-dropout", "0.05",
        "--log", "wandb",
        "--push-to-hub",
        "--username", "mistral-hackaton-2026",
        "--token", HF_TOKEN,
    ]

    print("=" * 50)
    print("LAUNCHING TRAINING VIA AUTOTRAIN CLI")
    print("=" * 50)
    print(f"  Base model:  {BASE_MODEL}")
    print(f"  Dataset:     {HF_DATASET_ID}")
    print(f"  Output:      {HF_REPO_ID}")
    print(f"  W&B project: {WANDB_PROJECT}")
    print()
    print("Command:")
    print(f"  {' '.join(cmd)}")
    print("=" * 50)

    # Set W&B env var so AutoTrain picks it up
    env = os.environ.copy()
    env["WANDB_API_KEY"] = WANDB_API_KEY
    env["WANDB_PROJECT"] = WANDB_PROJECT

    subprocess.run(cmd, env=env, check=True)
    print("\nTraining complete! Model pushed to:", HF_REPO_ID)


if __name__ == "__main__":
    if not HF_TOKEN:
        print("Error: HF_TOKEN not set in .env")
        sys.exit(1)

    launch_via_cli()
