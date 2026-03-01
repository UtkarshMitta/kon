# Mistral Query Router

A fine-tuned **Ministral 3B** model that classifies user queries into optimal Mistral model tiers (1–4), enabling cost-efficient LLM routing. Built with **HuggingFace AutoTrain**, monitored via **Weights & Biases**, with a **continuous feedback loop** for auto-retraining.

## Model Tiers

| Tier | Label | Target Model | Use Case |
|------|-------|-------------|----------|
| 1 | small | Ministral 3B/8B | Simple facts, greetings, trivial math |
| 2 | medium | Mistral Small 3.2 | Summarization, translation, moderate reasoning |
| 3 | large | Mistral Medium 3.1 | Complex analysis, multi-step reasoning, coding |
| 4 | xlarge | Mistral Large 3 | Expert research, proofs, system design |

## Quick Start

### 1. Setup

```bash
cd router-model
pip install -r requirements.txt

# Copy and fill in your API keys
cp .env.example .env
```

### 2. Generate Training Data

```bash
# Dry run with seed examples only (no API needed)
python data/generate_dataset.py --dry-run

# Full generation with Mistral API (5000 examples)
python data/generate_dataset.py --api-key YOUR_MISTRAL_KEY

# Validate the generated dataset
python data/generate_dataset.py --validate data/raw/train.jsonl
```

### 3. Train the Model

```bash
# Check prerequisites
python train/train_autotrain.py --check

# Train with transformers + QLoRA + W&B logging
python train/train_autotrain.py --method transformers

# Or use AutoTrain CLI
python train/train_autotrain.py --method autotrain
```

### 4. Run Inference

```bash
# Single query
python inference/router.py --model-path ./output --query "What is 2+2?"

# Interactive mode
python inference/router.py --model-path ./output --interactive

# Test queries
python inference/router.py --model-path ./output --test
```

### 5. Evaluate

```bash
# Run evaluation on test set, log to W&B
python train/evaluate.py --model-path ./output --wandb
```

## Feedback Loop

Collect human corrections and auto-retrain after 20 feedbacks:

```bash
# Add feedback (query, predicted_tier, corrected_tier)
python feedback/collect.py --add "Explain quantum computing" 2 3

# Check feedback stats
python feedback/collect.py --stats

# Trigger retraining (automatically checks threshold)
python feedback/retrain_trigger.py

# Force retrain regardless of threshold
python feedback/retrain_trigger.py --force

# Optional: Start webhook listener for auto-retrain
python feedback/webhook_listener.py
```

## Architecture

```
User Query → Router Model (Ministral 3B + LoRA)
                ↓
          {model_tier: 1-4, confidence: 0.0-1.0}
                ↓
         Route to: Ministral 3B | Mistral Small | Mistral Medium | Mistral Large

Human Feedback → Feedback Store (JSONL)
                      ↓ (≥20 entries)
                 Auto-Retrain → W&B Dashboard
                      ↓
                Updated Router Model
```

## Project Structure

```
router-model/
├── config.py                  # All settings
├── requirements.txt           # Dependencies
├── .env.example               # API key template
├── data/
│   ├── tier_rubric.py         # Tier definitions + seed examples
│   ├── generate_dataset.py    # Synthetic data generation
│   └── raw/                   # Generated datasets
├── train/
│   ├── train_autotrain.py     # Training (AutoTrain/transformers + W&B)
│   └── evaluate.py            # Evaluation + W&B logging
├── inference/
│   └── router.py              # Inference wrapper
└── feedback/
    ├── collect.py             # Feedback collection
    ├── retrain_trigger.py     # Auto-retrain pipeline
    └── webhook_listener.py    # HF Webhook listener (optional)
```

## W&B Integration

All training runs are logged to W&B with:
- Loss curves, learning rate schedules
- GPU memory & utilization
- Per-tier F1 scores and confusion matrix (via evaluate.py)
- Model checkpoints as artifacts

Set your W&B API key in `.env` and metrics appear in your [W&B dashboard](https://wandb.ai).
