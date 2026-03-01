# Mistral Query Router (kon demo)

This repo contains a **stand‑alone Mistral query router**, implemented in `model-router.py`.

It takes plain text prompts, chooses the appropriate Mistral model tier (1–4), maintains a
JSONL conversation history with periodic summarization, and logs per‑query token usage and
relative cost vs a virtual tier‑3 baseline.

---

## 1. What `model-router.py` does

`model-router.py` defines a single public class:

- `MistralRouter` – a stateful router around the Mistral Conversations API:
  - **Routing**:
    - Classifies each incoming prompt into a **tier 1–4**.
    - Maps tiers to concrete models:
      - Tier 1 → `ministral-3b-latest`
      - Tier 2 → `mistral-small-latest`
      - Tier 3 → `mistral-medium-latest`
      - Tier 4 → `mistral-large-latest`
  - **State & history**:
    - Appends every user/assistant exchange to `conversation_history.jsonl`.
    - Persists lightweight router state (current model, conversation id) in `.router_state.json`.
  - **Summarization & compaction**:
    - After a fixed number of turns (`SUMMARY_INTERVAL`), calls a summarizer model
      (`magistral-small-2509`) to summarize the full history.
    - Rewrites `conversation_history.jsonl` to a single `summary` entry, then continues
      appending new turns after that summary.
  - **Model switching with context preservation**:
    - On a model change, reads the compacted history and starts a new conversation on the
      new model, prepending the summary + recent turns so context is preserved.
  - **Cost logging**:
    - Uses the **real token usage** returned by the Mistral API (no estimates).
    - For each routed query and each summarization call, writes one JSON object to
      `router_cost_log.jsonl` with:
      - `prompt_tokens`, `completion_tokens`, `total_tokens`
      - The **actual model tier** and cost (based on current model)
      - A **virtual tier‑3 “baseline” cost** for the same tokens
      - `savings_pct` – percentage savings vs that tier‑3 baseline

The router itself is self‑contained; Kon’s TUI and the rest of the agent are not required
to use it.

---

## 2. Basic usage

You interact with the router directly from Python:

```python
from model_router import MistralRouter

router = MistralRouter()  # uses MISTRAL_API_KEY from your environment

# 1) Start a conversation (router chooses tier for you)
result = router.route({"prompt": "Explain what a Python list comprehension is."})
print(result["model"], result["output"])

# 2) Continue the same conversation
followup = router.route({"prompt": "Now show me a short code example."})
print(followup["model"], followup["output"])

# 3) Force a specific model tier / id if you want (optional)
hard_routed = router.route(
    {"prompt": "Design a small REST API for a todo app.", "model": "mistral-large-latest"}
)
print(hard_routed["model"], hard_routed["output"])

# 4) When you’re done
router.reset()
```

Environment:

- Set `MISTRAL_API_KEY` in your shell or `.env.local` so the router can talk to Mistral.
- The router will create / update:
  - `conversation_history.jsonl`
  - `.router_state.json`
  - `router_cost_log.jsonl`

---

## 3. Inspecting routing and cost

Because each call writes exactly one JSON object to `router_cost_log.jsonl`, you can do
all your analysis offline, for example:

```bash
cat router_cost_log.jsonl | jq .
```

Each line looks roughly like:

```json
{
  "timestamp": "...",
  "kind": "route",
  "is_summarisation": false,
  "model": "mistral-small-latest",
  "decision_tier": 2,
  "actual_tier": 2,
  "baseline_tier": 3,
  "prompt_tokens": 210,
  "completion_tokens": 96,
  "total_tokens": 306,
  "actual_cost": 0.00002,
  "baseline_cost": 0.00025,
  "savings_pct": 92.0
}
```

This gives you:

- Per‑query **actual cost** (based on the chosen model).
- A virtual **tier‑3 baseline cost** for comparison.
- A simple **percentage savings** metric you can aggregate however you like.

---

## 4. Demo script

For a short demo (e.g. a 2–3 minute video) that exercises:

- Tier switching (1 → 2 → 3 → 4),
- Summarization and history compaction,
- And context preservation,

