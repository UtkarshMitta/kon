# Mistral Query Router — Elastic Intelligence Platform

A brilliant, self-improving **Mistral Router** built for enterprise cost-efficiency. It uses a fine-tuned **Ministral 3B** model to intelligently classify and route user queries to the optimal Mistral model tier (1–4), driving up to **90%+ cost savings** without sacrificing response quality.

Built with **HuggingFace**, **FastAPI**, **React**, and **Weights & Biases**, featuring a continuous **DPO Reinforcement Learning** feedback loop.

---

## 🚀 Key Features

*   **Intelligent Routing:** SFT-tuned classifier dynamically routes to Ministral 3B, Mistral Small, Medium, or Large based on query complexity.
*   **Live ROI Dashboard:** A stunning React/Vite interface showing real-time cost savings vs. a static baseline (Mistral Medium).
*   **Elastic Context Compression:** A background summarization agent (Ministral 3B) compresses chat history into a persistent <200 token briefing, maintaining high-fidelity session memory for pennies.
*   **Reinforcement Learning (DPO):** Granular human feedback is logged directly to **W&B Tables**, triggering Direct Preference Optimization (DPO) to continually align and improve the router.
*   **W&B Integration:** Real-time logging of routing decisions, costs, feedback arrays, and RL training curves.

---

## 🧠 Model Tiers

| Tier | Complexity | Assigned Model | Target Use Case & Cost |
| :--- | :--- | :--- | :--- |
| **Tier 1** | Simple | `ministral-3b-latest` | Greetings, facts, trivial math ($0.10 / 1M) |
| **Tier 2** | Moderate | `mistral-small-latest` | Summarization, code snippets ($0.30 / 1M) |
| **Tier 3** | Complex | `mistral-medium-latest` | Multi-step reasoning ($2.00 / 1M) |
| **Tier 4** | Expert | `mistral-large-latest` | Synthesis, system design ($1.50 / 1M) |

---

## 🛠️ Tech Stack

*   **Models:** Mistral API (Ministral 3B up to Mistral Large), HuggingFace Endpoints
*   **Training:** QLoRA, TRL (DPOTrainer), HuggingFace Datasets
*   **Backend:** FastAPI, Python, Uvicorn
*   **Frontend:** React, Vite, CSS Animations
*   **MLOps & Observability:** Weights & Biases (W&B)

---

## 🚦 Quick Start

### 1. Installation 
```bash
git clone https://github.com/UtkarshMitta/kon.git
cd kon
pip install -r requirements.txt

# Configure your environment
cp .env.example .env
# Fill in your MISTRAL_API_KEY, WANDB_API_KEY, and HF_TOKEN in .env
```

### 2. Run the Production Dashboard
Our production setup uses a FastAPI backend and a React/Vite frontend.

**Start the Backend (Port 8000):**
```bash
python web/backend/app.py
```

**Start the Frontend (Port 5173):**
```bash
cd web/frontend
npm i
npm run dev
```
Navigate to `http://localhost:5173` to experience the live routing and ROI dashboard!

### 3. Terminal Agent (Granular Feedback Mode)
If you prefer the command-line, run the standalone agent loop. This mode features granular tier-correction prompts to generate high-quality RL datasets.

```bash
python main.py
```

---

## 🔄 DPO Reinforcement Learning Loop

The platform gets smarter with every user interaction:
1.  **Feedback Collection:** The `handler.py` logs user satisfaction (chosen vs. rejected tiers) into local JSONL and syncs heavily with **W&B Tables**.
2.  **DPO Training:** Run the DPO training script to align the model against user preferences.
    ```bash
    python train/train_dpo.py --data-path ./data
    ```
3.  **Deployment:** The newly aligned LoRA is pushed to the HuggingFace Hub and dynamically loaded into the Inference Endpoint.

---

## 📁 Repository Structure

```
kon/
├── config.py                 # Global constants (models, thresholds, limits)
├── main.py                   # Terminal interactive agent loop
├── .env.example              # API key template
│
├── web/
│   ├── backend/app.py        # FastAPI server endpoints
│   └── frontend/             # React/Vite live ROI dashboard
│
├── inference/
│   ├── router.py             # Inference class for the HF classifier endpoint
│   ├── context_manager.py    # Background context compression agent
│   ├── cost_calculator.py    # Per-token ROI math engine
│   └── wandb_logger.py       # Weights & Biases sync
│
├── train/
│   └── train_dpo.py          # RL alignment script (TRL DPOTrainer)
│
└── data/
    └── tier_rubric.py        # Prompt engineering guidelines for SFT data
```

Built for the **Mistral Hackathon** 🔥
