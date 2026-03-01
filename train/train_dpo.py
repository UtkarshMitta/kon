"""
Training script for the Mistral Query Router using RL (DPO - Direct Preference Optimization).
Aligns the SFT-tuned model using collected human feedback.
"""

import os
import sys
import json
import torch
import wandb
import argparse
from pathlib import Path
from datetime import datetime
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, prepare_model_for_kbit_training
from trl import DPOTrainer, DPOConfig

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    BASE_MODEL,
    HF_REPO_ID,
    WANDB_PROJECT,
    DATA_DIR,
    LEARNING_RATE,
    TRAINING_EPOCHS,
    BATCH_SIZE,
    MAX_SEQ_LENGTH,
    LORA_RANK,
    LORA_ALPHA,
    LORA_DROPOUT,
    HF_TOKEN,
)

def train_dpo(
    data_path: str,
    output_dir: str = "./output/dpo",
    model_id: str = BASE_MODEL,
    push_to_hub: bool = True,
    repo_id: str = HF_REPO_ID,
):
    # Initialize W&B
    wandb.init(
        project=WANDB_PROJECT,
        name=f"dpo-router-{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        config={
            "method": "DPO",
            "base_model": model_id,
            "learning_rate": LEARNING_RATE / 2, # DPO usually needs lower LR
            "epochs": TRAINING_EPOCHS,
        },
    )

    print(f"🔄 Loading model for RL/DPO: {model_id}")

    # QLoRA config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    # Load model + tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model = prepare_model_for_kbit_training(model)

    # DPO also needs a reference model (often the same model, but frozen)
    # The DPOTrainer handles creating the reference model internally if ref_model is None
    
    # LoRA config
    lora_config = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )

    # Load DPO dataset
    print(f"📂 Loading DPO dataset from: {data_path}")
    dataset = load_dataset("json", data_files={"train": str(Path(data_path) / "dpo_feedback.jsonl")})

    # DPO Config
    dpo_config = DPOConfig(
        output_dir=output_dir,
        num_train_epochs=TRAINING_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE / 2,
        report_to="wandb",
        max_prompt_length=MAX_SEQ_LENGTH // 2,
        max_length=MAX_SEQ_LENGTH,
        push_to_hub=push_to_hub,
        hub_model_id=repo_id if push_to_hub else None,
    )

    # Trainer
    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=dpo_config,
        train_dataset=dataset["train"],
        tokenizer=tokenizer,
        peft_config=lora_config,
        beta=0.1, # DPO temperature
    )

    print("🚀 Starting RL/DPO alignment...")
    trainer.train()

    print("💾 Saving RL-aligned model...")
    trainer.save_model(output_dir)
    
    if push_to_hub:
        print(f"📤 Pushing RL-aligned model to HF Hub: {repo_id}")
        trainer.push_to_hub()

    wandb.finish()
    print("✅ RL Alignment complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RL Alignment for Mistral Router")
    parser.add_argument("--data-path", type=str, default=str(DATA_DIR), help="Path to dpo_feedback.jsonl")
    parser.add_argument("--model", type=str, default=BASE_MODEL, help="Base/SFT model")
    parser.add_argument("--repo-id", type=str, default=HF_REPO_ID, help="Target HF repo")
    args = parser.parse_args()
    
    train_dpo(data_path=args.data_path, model_id=args.model, repo_id=args.repo_id)
