# 🚀 Sharing Your Mistral Router Agent

Your friend can easily plug this agent into their workflow. Here is the step-by-step guide for them:

## 1. The Core Files
The easiest way for them to get started is by using the standalone script:
- **`mistral_router_standalone.py`**: A clean, self-contained version of the router logic.

If they want the full context-managed system:
- `main.py`: The entry point/orchestrator.
- `inference/router.py`: The "Brain" that decides the complexity.
- `inference/context_manager.py`: The "Memory" (Elastic Context).
- `config.py`: The configuration.

## 2. Prerequisites
They must install:
```bash
pip install mistralai python-dotenv requests
```

## 3. Configuration (.env)
They need to create a `.env` file with these keys:
```text
# Mistral Platform
MISTRAL_API_KEY=your_key_here

# Hugging Face (Custom LoRA Handler)
HF_TOKEN=your_token_here
HF_ENDPOINT_URL=https://x0ymdma0fdt3ttkz.us-east4.gcp.endpoints.huggingface.cloud
```

> [!IMPORTANT]
> This endpoint uses a **Custom Inference Handler**. This means the URL provided above is the root address (no `/v1/chat/completions` suffix). The standalone script handles this automatically.

## 4. How to Use in a Custom Workflow
If they want to use this *inside* their own Python script (instead of running `main.py`), they can import it like this:

```python
from inference.router import MistralRouter
from inference.context_manager import ContextManager
import os

# 1. Setup
mistral_key = os.getenv("MISTRAL_API_KEY")
router = MistralRouter("mistral-hackaton-2026/mistral-query-router", mistral_api_key=mistral_key)
context = ContextManager(api_key=mistral_key)

# 2. Process a Query
user_query = "Write a complex Java backend."

# Decision: Which model?
result = router.route(user_query)
tier = result["model_tier"]
model_name = result["model_name"]

# Memory: Should we summarize?
if context.should_pivot(tier):
    context.generate_pivot_briefing()

messages = context.get_messages_for_api(user_query)

# Execution: Get answer
response = router.get_model_response(model_name, messages)
print(response)

# Save result back to memory
context.add_turn(user_query, response, tier)
```

## 5. Cost Savings
Remind them that this system **saves money** by:
1. Using a tiny 7B model for intelligence-based routing.
2. Only summarizing context when the model changes (saving thousands of context tokens).
