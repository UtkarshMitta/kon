"""
Adaptive Query Router — Backend Module

Accepts JSON input with `prompt` and `model`, routes to the correct Mistral
model via the Conversations API.

Summarization logic:
  - Every exchange is immediately appended to conversation_history.jsonl.
  - Every SUMMARY_INTERVAL (20) calls, a summary is generated via
    magistral-small-2509.  The JSONL file is then COMPACTED: all prior
    entries are replaced by a single summary entry.  New exchanges are
    appended after this summary entry going forward.
  - On model switch, NO summary API call is made.  The current file
    contents (summary + recent entries) are used as context for the
    new model.

Usage from another Python file:
    from model_router import MistralRouter

    router = MistralRouter()
    result = router.route({"prompt": "Hello", "model": "mistral-medium-latest"})
    result = router.route({"prompt": "Follow up"})
    result = router.route({"prompt": "Code this", "model": "codestral-latest"})
    router.reset()
"""

import datetime
import json
import os
import re
from pathlib import Path

import requests
from dotenv import load_dotenv
from mistralai import Mistral

HISTORY_FILE = Path(__file__).parent / "conversation_history.jsonl"
STATE_FILE = Path(__file__).parent / ".router_state.json"
# Per-query cost log (one JSON object per line).
COST_FILE = Path(__file__).parent / "router_cost_log.jsonl"


# ---------------------------------------------------------------------------
#  MistralRouter
# ---------------------------------------------------------------------------
class MistralRouter:
    """
    Stateful router that directs prompts to Mistral agents / models.

    Public API
    ----------
    route(payload)   — main entry  {"prompt": str, "model": str?}
    restart(payload) — restart from entry  {"from_entry_id": str, "prompt": str}
    reset()          — clear all state
    """

    SUMMARISER_MODEL = "magistral-small-2509"
    DEFAULT_MODEL = "ministral-3b-latest"
    SUMMARY_INTERVAL = 5

    BASE_ROUTER_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"

    ROUTER_SYSTEM_PROMPT = (
        "You are a query complexity router for Mistral AI models. "
        "Given a user query, classify it into the appropriate model tier and provide "
        "a confidence score. If the user query requires tools (like 'bash' or 'write'), "
        "always output them in the exact <tool_call> format requested.\n\n"
        "Output ONLY valid JSON for the classification in this exact format:\n"
        '{"model_tier": <1|2|3|4>, "confidence": <0.0-1.0>}\n\n'
        "Tier definitions:\n"
        "- Tier 1 (small): Simple factual lookups, greetings, trivial math, yes/no questions\n"
        "- Tier 2 (medium): Summarization, translation, moderate reasoning, simple coding\n"
        "- Tier 3 (large): Complex analysis, multi-step reasoning, debugging, comparisons\n"
        "- Tier 4 (xlarge): Expert-level research, novel problem-solving, proofs, system design"
    )

    def __init__(self, api_key: str | None = None):
        # agent cache:  model_name -> agent_id
        self._agents: dict[str, str] = {}

        self.TIER_MODEL_MAP = {
            1: "ministral-3b-latest",  # Simple queries
            2: "mistral-small-latest",  # Moderate queries
            3: "mistral-medium-latest",  # Complex queries
            4: "mistral-large-latest",  # Expert queries
        }
        # Invert mapping so we can go from model_id -> tier.
        self.MODEL_TIER_MAP = {v: k for k, v in self.TIER_MODEL_MAP.items()}
        # Virtual baseline tier: Mistral Medium (tier 3).
        self.BASELINE_TIER = 3

        env_path = Path(__file__).parent / ".env.local"
        if env_path.exists():
            load_dotenv(dotenv_path=env_path)
        else:
            load_dotenv()

        self._api_key = api_key or os.environ.get("MISTRAL_API_KEY")
        if not self._api_key:
            raise RuntimeError(
                "MISTRAL_API_KEY not found.  Pass it explicitly or set it in "
                ".env.local / environment."
            )

        self._client = Mistral(api_key=self._api_key)

        # HF Router config
        self._hf_endpoint = os.environ.get("HF_ENDPOINT_URL")
        self._hf_token = os.environ.get("HF_TOKEN")

        # Log endpoint for debugging
        if os.environ.get("KON_DEBUG"):
            with open("/tmp/mistral_router_debug.log", "a") as f:
                f.write(f"\n[DEBUG] Loaded HF_ENDPOINT_URL: {self._hf_endpoint}\n")

        # conversation state
        self._current_model: str | None = None
        self._conversation_id: str | None = None

        # Load persisted state if exists
        self._load_state()

    def _load_state(self) -> None:
        """Load internal state from disk."""
        if STATE_FILE.exists():
            try:
                data = json.loads(STATE_FILE.read_text())
                self._agents = data.get("agents", {})
                self._current_model = data.get("current_model")
                self._conversation_id = data.get("conversation_id")
                print(f"  📂 Loaded persistent state from {STATE_FILE.name}")
            except Exception as e:
                print(f"  ⚠️ Error loading state: {e}")

    def _save_state(self) -> None:
        """Persist internal state to disk."""
        try:
            data = {
                "agents": self._agents,
                "current_model": self._current_model,
                "conversation_id": self._conversation_id,
                "last_update": datetime.datetime.now(datetime.UTC).isoformat(),
            }
            STATE_FILE.write_text(json.dumps(data, indent=2))
        except Exception as e:
            print(f"  ⚠️ Error saving state: {e}")

    # ------------------------------------------------------------------
    #  Public methods
    # ------------------------------------------------------------------

    def route(self, payload: dict) -> dict:
        prompt = payload.get("prompt")
        if not prompt:
            raise ValueError("payload must contain a non-empty 'prompt' key")

        requested_model = payload.get("model")
        decision_tier: int | None = None

        # Check if conversation history is large enough to trigger summary BEFORE the request
        if self._should_summarise():
            print(
                f"  📋 Conversation history >= {self.SUMMARY_INTERVAL} entries. "
                "Triggering summary BEFORE request."
            )
            self._run_periodic_summary()

        # Dynamic routing logic
        if not requested_model:
            decision = self._classify_query(prompt)
            decision_tier = decision.get("model_tier")
            target_model = decision.get("model_name", self.DEFAULT_MODEL)

            log_msg = (
                f"[{datetime.datetime.now()}] 🤖 HF Router Decision: "
                f"Tier {decision.get('model_tier')} ({decision.get('confidence')} confidence) "
                f"-> {target_model}\n"
            )
            print(log_msg.strip())
            with open("/tmp/mistral_router_debug.log", "a") as f:
                f.write(log_msg)

            if self._current_model is None:
                result = self._start_conversation(target_model, prompt)
            elif target_model == self._current_model:
                result = self._continue_conversation(prompt)
            else:
                print(f"  🔄 Model Switch DETECTED: {self._current_model} -> {target_model}")
                result = self._switch_model(target_model, prompt)
        else:
            # Explicit model request (e.g. via --model flag or config)
            if self._current_model is None:
                result = self._start_conversation(requested_model, prompt)
            elif requested_model == self._current_model:
                result = self._continue_conversation(prompt)
            else:
                print(f"  🔄 Manual Model Switch: {self._current_model} -> {requested_model}")
                result = self._switch_model(requested_model, prompt)

        # Log per-query cost information based on returned usage.
        usage = result.get("usage") or {}
        self._log_cost_record(
            kind="route",
            model=result.get("model"),
            usage=usage,
            is_summarisation=False,
            decision_tier=decision_tier,
        )

        return result

    def restart(self, payload: dict) -> dict:
        """Restart the current conversation from a specific entry."""
        if not self._conversation_id:
            raise RuntimeError("No active conversation to restart.")

        from_entry_id = payload.get("from_entry_id")
        prompt = payload.get("prompt")
        if not from_entry_id or not prompt:
            raise ValueError("payload must contain 'from_entry_id' and 'prompt'")

        response = self._client.beta.conversations.restart(
            conversation_id=self._conversation_id, from_entry_id=from_entry_id, inputs=prompt
        )
        self._conversation_id = response.conversation_id
        result = self._format_response(response)
        self._log_to_file("user", prompt)
        self._log_to_file("assistant", result.get("output") or "")
        return result

    def reset(self) -> None:
        """Clear all state."""
        self._current_model = None
        self._conversation_id = None
        # Wipe state file
        if STATE_FILE.exists():
            STATE_FILE.unlink()
        # Wipe the history file
        if HISTORY_FILE.exists():
            HISTORY_FILE.write_text("")

    def _classify_query(self, query: str) -> dict:
        """Predicts the optimal tier for a query using the Dedicated HF Endpoint."""
        if not self._hf_endpoint or not self._hf_token:
            raise RuntimeError(
                "Hugging Face Router credentials missing (HF_ENDPOINT_URL or HF_TOKEN). "
                "Routing cannot continue without these."
            )

        headers = {"Authorization": f"Bearer {self._hf_token}", "Content-Type": "application/json"}

        # We MUST use the Mistral Chat Template format (<s>[INST] ... [/INST])
        templated_prompt = (
            f"<s>[INST] {self.ROUTER_SYSTEM_PROMPT}\n\nUser Query: {query}\n\n"
            "Remember: Output ONLY the JSON object. "
            'Example: {"model_tier": 2, "confidence": 0.8} [/INST]'
        )

        payload = {
            "inputs": templated_prompt,
            "parameters": {"max_new_tokens": 128, "temperature": 0.01, "do_sample": False},
        }

        try:
            endpoint = self._hf_endpoint.rstrip("/")
            res = requests.post(endpoint, headers=headers, json=payload, timeout=20)
            res.raise_for_status()
            res_json = res.json()

            # Custom handlers return a list of dictionaries: [{"generated_text": "..."}]
            if isinstance(res_json, list) and len(res_json) > 0:
                raw_content = res_json[0].get("generated_text", "")

                with open("/tmp/mistral_router_debug.log", "a") as f:
                    f.write(
                        f"[{datetime.datetime.now()}] 📥 Raw HF Response Content: {raw_content}\n"
                    )

                result = self._parse_json(raw_content)

                # Map back to model name
                tier = result.get("model_tier", 1)
                result["model_name"] = self.TIER_MODEL_MAP.get(tier, self.DEFAULT_MODEL)
                return result
            else:
                raise ValueError(f"Unexpected API response: {res_json}")

        except Exception as e:
            print(f"  ❌ HF Router Critical Error: {e}")
            raise RuntimeError(f"Hugging Face Router failed: {e}") from e

    def _parse_json(self, text: str) -> dict:
        """Robustly extract JSON from model output."""
        try:
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if match:
                return json.loads(match.group(0))
            return json.loads(text)
        except Exception as e:
            raise RuntimeError(
                f"Failed to extract or parse JSON from router output: {text}. Error: {e}"
            ) from e

    # ------------------------------------------------------------------
    #  Internal — conversation lifecycle
    # ------------------------------------------------------------------

    def _start_conversation(self, model: str, prompt: str) -> dict:
        agent_id = self._get_or_create_agent(model)
        response = self._client.beta.conversations.start(agent_id=agent_id, inputs=prompt)
        self._current_model = model
        self._conversation_id = response.conversation_id

        result = self._format_response(response)
        self._log_to_file("user", prompt)
        self._log_to_file("assistant", result.get("output") or "")
        return result

    def _continue_conversation(self, prompt: str) -> dict:
        if not self._conversation_id:
            raise RuntimeError("Cannot continue conversation: no active conversation ID")

        response = self._client.beta.conversations.append(
            conversation_id=self._conversation_id, inputs=prompt
        )
        self._conversation_id = response.conversation_id

        result = self._format_response(response)
        self._log_to_file("user", prompt)
        self._log_to_file("assistant", result.get("output") or "")
        return result

    def _switch_model(self, new_model: str, prompt: str) -> dict:
        """
        Model-switch workflow (NO summary API call):
        1. Read current JSONL file contents (summary + recent entries).
        2. Start a new conversation on new_model with that context + prompt.
        """
        context = self._read_context_from_file()

        combined_prompt = (
            f"Here is the context of our prior conversation:\n\n"
            f"{context}\n\n---\n\n"
            f"Now, continuing with your expertise, please respond to:\n\n"
            f"{prompt}"
        )
        result = self._start_conversation(new_model, combined_prompt)

        # Rewrite the last "user" log entry to show the original prompt
        # (the combined_prompt is an implementation detail)
        self._rewrite_last_user_entry(prompt)

        return result

    def _should_summarise(self) -> bool:
        """Check if the JSONL file has SUMMARY_INTERVAL or more entries."""
        if not HISTORY_FILE.exists():
            return False

        with open(HISTORY_FILE, encoding="utf-8") as f:
            lines = [line for line in f if line.strip()]

        return len(lines) >= self.SUMMARY_INTERVAL

    def _run_periodic_summary(self) -> None:
        """
        Summarise everything in the JSONL file, then COMPACT the file:
        replace all entries with a single summary entry.
        """
        context = self._read_context_from_file()
        if not context:
            return

        summariser_id = self._get_or_create_agent(self.SUMMARISER_MODEL)
        response = self._client.beta.conversations.start(
            agent_id=summariser_id,
            inputs=(
                "Summarise the following conversation concisely, preserving "
                "all key facts, decisions, and context so that a different "
                "model can continue the conversation without loss:\n\n"
                f"{context}"
            ),
        )
        formatted = self._format_response(response)
        summary_text = formatted.get("output") or ""

        # Log summarisation cost as its own record.
        self._log_cost_record(
            kind="summary",
            model=self.SUMMARISER_MODEL,
            usage=formatted.get("usage") or {},
            is_summarisation=True,
            decision_tier=None,
        )

        # COMPACT: rewrite the JSONL file with just the summary entry
        summary_entry = {
            "role": "summary",
            "content": summary_text,
            "timestamp": datetime.datetime.now(datetime.UTC).isoformat(),
        }
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            f.write(json.dumps(summary_entry, ensure_ascii=False) + "\n")

        print(f"  ✅ JSONL compacted — summary ({len(summary_text or '')} chars)")

    # ------------------------------------------------------------------
    #  Internal — cost logging
    # ------------------------------------------------------------------

    def _log_cost_record(
        self,
        *,
        kind: str,
        model: str | None,
        usage: dict | None,
        is_summarisation: bool,
        decision_tier: int | None,
    ) -> None:
        """
        Append a single JSON object with cost information to COST_FILE.

        We only use token counts reported by the API (`usage`) and never
        hallucinate them. Baseline cost is computed by treating the same
        tokens as if they had run on the tier-3 (Mistral Medium) model.
        """
        try:
            usage = usage or {}
            prompt_tokens = int(usage.get("prompt_tokens") or 0)
            completion_tokens = int(usage.get("completion_tokens") or 0)
            total_tokens = int(usage.get("total_tokens") or 0)

            # If the API didn't report usage, don't log anything.
            if prompt_tokens == 0 and completion_tokens == 0 and total_tokens == 0:
                return

            # Pricing: USD per 1M tokens (input, output).
            pricing_by_tier = {
                1: {"input": 0.10, "output": 0.10},
                2: {"input": 0.05, "output": 0.08},
                3: {"input": 0.40, "output": 2.00},
                4: {"input": 0.50, "output": 1.50},
            }

            def _cost_for(tier: int | None) -> float | None:
                if tier is None or tier not in pricing_by_tier:
                    return None
                rates = pricing_by_tier[tier]
                input_cost = (prompt_tokens / 1_000_000) * rates["input"]
                output_cost = (completion_tokens / 1_000_000) * rates["output"]
                return input_cost + output_cost

            actual_tier = self.MODEL_TIER_MAP.get(model or "", None)
            baseline_tier = self.BASELINE_TIER

            actual_cost = _cost_for(actual_tier)
            baseline_cost = _cost_for(baseline_tier)

            savings_pct = None
            if actual_cost is not None and baseline_cost is not None and baseline_cost > 0.0:
                savings_pct = (baseline_cost - actual_cost) / baseline_cost * 100.0

            record = {
                "timestamp": datetime.datetime.now(datetime.UTC).isoformat(),
                "kind": kind,  # "route" or "summary"
                "is_summarisation": is_summarisation,
                "model": model,
                "decision_tier": decision_tier,
                "actual_tier": actual_tier,
                "baseline_tier": baseline_tier,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
                "actual_cost": actual_cost,
                "baseline_cost": baseline_cost,
                "savings_pct": savings_pct,
            }

            COST_FILE.parent.mkdir(parents=True, exist_ok=True)
            with open(COST_FILE, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
        except Exception:
            # Cost logging must never break routing; fail silently.
            return

    # ------------------------------------------------------------------
    #  Internal — file I/O
    # ------------------------------------------------------------------

    def _log_to_file(self, role: str, content: str) -> None:
        """Immediately append a single entry to the JSONL file."""
        entry = {
            "role": role,
            "content": content,
            "model": self._current_model,
            "conversation_id": self._conversation_id,
            "timestamp": datetime.datetime.now(datetime.UTC).isoformat(),
        }
        with open(HISTORY_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        self._save_state()

    def _read_context_from_file(self) -> str:
        """
        Read the JSONL file and build a context string.
        - If a 'summary' entry exists, it becomes the first block.
        - All user/assistant entries are listed after it.
        """
        if not HISTORY_FILE.exists() or HISTORY_FILE.stat().st_size == 0:
            return ""

        parts = []
        with open(HISTORY_FILE, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                role = entry.get("role", "")
                content = entry.get("content", "")
                if role == "summary":
                    parts.append(f"## Summary of earlier conversation\n\n{content}")
                else:
                    parts.append(f"[{role}]: {content}")

        return "\n\n".join(parts)

    def _rewrite_last_user_entry(self, original_prompt: str) -> None:
        """
        After a model switch, the last 'user' entry in the JSONL contains
        the combined prompt (context + real prompt).  Rewrite it to show
        only the original user prompt.
        """
        if not HISTORY_FILE.exists():
            return

        lines = []
        with open(HISTORY_FILE, encoding="utf-8") as f:
            lines = f.readlines()

        # Find the last user entry (second-to-last line, since assistant follows)
        for i in range(len(lines) - 1, -1, -1):
            entry = json.loads(lines[i])
            if entry.get("role") == "user":
                entry["content"] = original_prompt
                lines[i] = json.dumps(entry, ensure_ascii=False) + "\n"
                break

        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            f.writelines(lines)

    # ------------------------------------------------------------------
    #  Internal — agent management
    # ------------------------------------------------------------------

    def _get_or_create_agent(self, model: str) -> str:
        if model in self._agents:
            return self._agents[model]

        agent = self._client.beta.agents.create(
            model=model, description=f"Adaptive router agent for {model}", name=f"router-{model}"
        )
        self._agents[model] = agent.id
        self._save_state()
        return agent.id

    # ------------------------------------------------------------------
    #  Internal — response helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_text(response) -> str:
        for output in response.outputs:
            if hasattr(output, "content") and output.content:
                if isinstance(output.content, str):
                    return output.content
                elif isinstance(output.content, list):
                    text_parts = []
                    for chunk in output.content:
                        if hasattr(chunk, "text"):
                            text_parts.append(chunk.text)
                        elif hasattr(chunk, "content"):
                            text_parts.append(str(chunk.content))
                        else:
                            text_parts.append(str(chunk))
                    return "".join(text_parts)
                return str(output.content)
        return ""

    def _format_response(self, response) -> dict:
        output_text = self._extract_text(response)
        usage = {}
        if getattr(response, "usage", None):
            u = response.usage
            if isinstance(u, dict):
                p_tok = u.get("prompt_tokens", 0)
                c_tok = u.get("completion_tokens", 0)
                t_tok = u.get("total_tokens", 0)
            else:
                p_tok = getattr(u, "prompt_tokens", 0)
                c_tok = getattr(u, "completion_tokens", 0)
                t_tok = getattr(u, "total_tokens", 0)

            usage = {"prompt_tokens": p_tok, "completion_tokens": c_tok, "total_tokens": t_tok}
            print(
                f"  [Debug] Tokens extracted from API response: "
                f"prompt={p_tok}, completion={c_tok}, total={t_tok}"
            )

        return {
            "conversation_id": response.conversation_id,
            "output": output_text,
            "model": self._current_model,
            "usage": usage,
        }
