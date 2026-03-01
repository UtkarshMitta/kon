"""
Central configuration for the Mistral Query Router.
All hyperparameters, paths, and settings in one place.
"""

import os
from pathlib import Path

# ──────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data" / "raw"
FEEDBACK_DIR = PROJECT_ROOT / "feedback" / "store"

# ──────────────────────────────────────────────
# Model
# ──────────────────────────────────────────────
BASE_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"
HF_REPO_ID = os.getenv("HF_REPO_ID", "your-username/mistral-query-router")
HF_DATASET_ID = os.getenv("HF_DATASET_ID", "your-username/router-training-data")

# ──────────────────────────────────────────────
# Tier Definitions
# ──────────────────────────────────────────────
TIER_LABELS = ["small", "medium", "large", "xlarge"]
TIER_MAP = {
    "small": 1,
    "medium": 2,
    "large": 3,
    "xlarge": 4,
}
TIER_MODEL_MAP = {
    1: "ministral-3b-latest",       # Simple queries
    2: "mistral-small-latest",      # Moderate queries
    3: "mistral-medium-latest",     # Complex queries
    4: "mistral-large-latest",      # Expert queries
}

# ──────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────
WANDB_PROJECT = os.getenv("WANDB_PROJECT", "mistral-router")
LEARNING_RATE = 2e-4
TRAINING_EPOCHS = 3
BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 4
MAX_SEQ_LENGTH = 512
LORA_RANK = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
QUANTIZATION = "int4"  # QLoRA 4-bit

# ──────────────────────────────────────────────
# Dataset Generation
# ──────────────────────────────────────────────
DATASET_SIZE = 5000  # Total examples to generate
EXAMPLES_PER_TIER = DATASET_SIZE // len(TIER_LABELS)
TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.1
TEST_SPLIT = 0.1
GENERATION_MODEL = "mistral-large-latest"  # Teacher model for synthetic data

# ──────────────────────────────────────────────
# Feedback Loop
# ──────────────────────────────────────────────
FEEDBACK_THRESHOLD = 20  # Retrain after N feedback entries
FEEDBACK_FILE = FEEDBACK_DIR / "feedback.jsonl"

# ──────────────────────────────────────────────
# System Prompt (used in training data + inference)
# ──────────────────────────────────────────────
SYSTEM_PROMPT = """You are a query complexity router for Mistral AI models. Given a user query, classify it into the appropriate model tier and provide a confidence score.

Output ONLY valid JSON in this exact format:
{"model_tier": <1|2|3|4>, "confidence": <0.0-1.0>}

Tier definitions:
- Tier 1 (small): Simple factual lookups, greetings, trivial math, yes/no questions
- Tier 2 (medium): Summarization, translation, moderate reasoning, simple coding
- Tier 3 (large): Complex analysis, multi-step reasoning, debugging, comparisons
- Tier 4 (xlarge): Expert-level research, novel problem-solving, proofs, system design"""

# ──────────────────────────────────────────────
# API Keys (from environment)
# ──────────────────────────────────────────────
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY", "")
HF_TOKEN = os.getenv("HF_TOKEN", "")
WANDB_API_KEY = os.getenv("WANDB_API_KEY", "")
