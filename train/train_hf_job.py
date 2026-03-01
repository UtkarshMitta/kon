"""
Standalone training script for HF Jobs.
Run with: hf jobs uv run --flavor a100-large --with "trl datasets peft bitsandbytes wandb" --secrets HF_TOKEN --secrets WANDB_API_KEY --timeout 2h train_hf_job.py
"""

import os
import torch
import wandb
from datetime import datetime
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

# ──────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────
BASE_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"
DATASET_ID = "mistral-hackaton-2026/router-training-data"
OUTPUT_REPO = "mistral-hackaton-2026/mistral-query-router"
WANDB_PROJECT = "mistral-router"

HF_TOKEN = os.getenv("HF_TOKEN")
WANDB_API_KEY = os.getenv("WANDB_API_KEY")

# Hyperparameters
LEARNING_RATE = 2e-4
EPOCHS = 3
BATCH_SIZE = 4
GRADIENT_ACCUMULATION = 4
MAX_SEQ_LENGTH = 512
LORA_RANK = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05

# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
def main():
    print("=" * 50)
    print("MISTRAL QUERY ROUTER - SFT TRAINING")
    print("=" * 50)
    print(f"  Model:   {BASE_MODEL}")
    print(f"  Dataset: {DATASET_ID}")
    print(f"  Output:  {OUTPUT_REPO}")
    print(f"  GPU:     {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print("=" * 50)

    # Init W&B
    if WANDB_API_KEY:
        wandb.login(key=WANDB_API_KEY)
        wandb.init(
            project=WANDB_PROJECT,
            name=f"router-{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            config={
                "base_model": BASE_MODEL,
                "lora_rank": LORA_RANK,
                "learning_rate": LEARNING_RATE,
                "epochs": EPOCHS,
            },
        )

    # QLoRA config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, token=HF_TOKEN)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Load model
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        token=HF_TOKEN,
        trust_remote_code=True,
    )
    model = prepare_model_for_kbit_training(model)

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
    print("Loading dataset...")
    dataset = load_dataset(DATASET_ID, token=HF_TOKEN)

    # Format chat messages
    def format_chat(example):
        text = tokenizer.apply_chat_template(
            example["messages"], tokenize=False, add_generation_prompt=False
        )
        return {"text": text}

    dataset = dataset.map(format_chat)

    # Training config
    training_args = SFTConfig(
        output_dir="./output",
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION,
        learning_rate=LEARNING_RATE,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        bf16=True,
        logging_steps=5,
        eval_strategy="epoch",
        save_strategy="steps",
        save_steps=50,
        save_total_limit=3,
        report_to="wandb" if WANDB_API_KEY else "none",
        max_seq_length=MAX_SEQ_LENGTH,
        packing=False,
        dataset_text_field="text",
        push_to_hub=True,
        hub_model_id=OUTPUT_REPO,
        hub_token=HF_TOKEN,
    )

    # Train
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset.get("validation", dataset.get("val")),
        peft_config=lora_config,
        processing_class=tokenizer,
    )

    print("Starting training...")
    trainer.train()

    # Save & push
    print("Saving model...")
    trainer.save_model("./output")
    tokenizer.save_pretrained("./output")
    trainer.push_to_hub()

    if WANDB_API_KEY:
        wandb.finish()

    print(f"Done! Model pushed to: https://huggingface.co/{OUTPUT_REPO}")


if __name__ == "__main__":
    main()
