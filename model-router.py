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

import json
import os
import datetime
from pathlib import Path

from dotenv import load_dotenv
from mistralai import Mistral


# ---------------------------------------------------------------------------
#  Constants
# ---------------------------------------------------------------------------
SUMMARISER_MODEL = "magistral-small-2509"
DEFAULT_MODEL = "mistral-medium-latest"
SUMMARY_INTERVAL = 5  # compact & summarise every N calls

HISTORY_FILE = Path(__file__).parent / "conversation_history.jsonl"


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

    def __init__(self, api_key: str | None = None):
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

        # agent cache:  model_name -> agent_id
        self._agents: dict[str, str] = {}

        # conversation state
        self._current_model: str | None = None
        self._conversation_id: str | None = None

        self._conversation_id: str | None = None

    # ------------------------------------------------------------------
    #  Public methods
    # ------------------------------------------------------------------

    def route(self, payload: dict) -> dict:
        prompt = payload.get("prompt")
        if not prompt:
            raise ValueError("payload must contain a non-empty 'prompt' key")

        requested_model = payload.get("model")

        # Check if conversation history is large enough to trigger summary BEFORE the request
        if self._should_summarise():
            print(f"  📋 Conversation history >= {SUMMARY_INTERVAL} entries. Triggering summary BEFORE request.")
            self._run_periodic_summary()

        # First call ever
        if self._current_model is None:
            model = requested_model or DEFAULT_MODEL
            result = self._start_conversation(model, prompt)
        # Same model (or no model specified) → continue
        elif requested_model is None or requested_model == self._current_model:
            result = self._continue_conversation(prompt)
        # Different model → switch
        else:
            result = self._switch_model(requested_model, prompt)

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
            conversation_id=self._conversation_id,
            from_entry_id=from_entry_id,
            inputs=prompt,
        )
        self._conversation_id = response.conversation_id
        result = self._format_response(response)
        self._log_to_file("user", prompt)
        self._log_to_file("assistant", result["output"])
        return result

    def reset(self) -> None:
        """Clear all state."""
        self._current_model = None
        self._conversation_id = None
        # Wipe the history file
        if HISTORY_FILE.exists():
            HISTORY_FILE.write_text("")

    # ------------------------------------------------------------------
    #  Internal — conversation lifecycle
    # ------------------------------------------------------------------

    def _start_conversation(self, model: str, prompt: str) -> dict:
        agent_id = self._get_or_create_agent(model)
        response = self._client.beta.conversations.start(
            agent_id=agent_id,
            inputs=prompt,
        )
        self._current_model = model
        self._conversation_id = response.conversation_id

        result = self._format_response(response)
        self._log_to_file("user", prompt)
        self._log_to_file("assistant", result["output"])
        return result

    def _continue_conversation(self, prompt: str) -> dict:
        response = self._client.beta.conversations.append(
            conversation_id=self._conversation_id,
            inputs=prompt,
        )
        self._conversation_id = response.conversation_id

        result = self._format_response(response)
        self._log_to_file("user", prompt)
        self._log_to_file("assistant", result["output"])
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
            
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            lines = [line for line in f if line.strip()]
        
        return len(lines) >= SUMMARY_INTERVAL

    def _run_periodic_summary(self) -> None:
        """
        Summarise everything in the JSONL file, then COMPACT the file:
        replace all entries with a single summary entry.
        """
        context = self._read_context_from_file()
        if not context:
            return

        summariser_id = self._get_or_create_agent(SUMMARISER_MODEL)
        response = self._client.beta.conversations.start(
            agent_id=summariser_id,
            inputs=(
                "Summarise the following conversation concisely, preserving "
                "all key facts, decisions, and context so that a different "
                "model can continue the conversation without loss:\n\n"
                f"{context}"
            ),
        )
        summary_text = self._extract_text(response)

        # COMPACT: rewrite the JSONL file with just the summary entry
        summary_entry = {
            "role": "summary",
            "content": summary_text,
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        }
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            f.write(json.dumps(summary_entry, ensure_ascii=False) + "\n")

        print(f"  ✅ JSONL compacted — replaced all entries with summary ({len(summary_text)} chars)")

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
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        }
        with open(HISTORY_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def _read_context_from_file(self) -> str:
        """
        Read the JSONL file and build a context string.
        - If a 'summary' entry exists, it becomes the first block.
        - All user/assistant entries are listed after it.
        """
        if not HISTORY_FILE.exists() or HISTORY_FILE.stat().st_size == 0:
            return ""

        parts = []
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
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
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
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
            model=model,
            description=f"Adaptive router agent for {model}",
            name=f"router-{model}",
        )
        self._agents[model] = agent.id
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

            usage = {
                "prompt_tokens": p_tok,
                "completion_tokens": c_tok,
                "total_tokens": t_tok,
            }
            print(f"  [Debug] Tokens extracted from API response: prompt={p_tok}, completion={c_tok}, total={t_tok}")

        return {
            "conversation_id": response.conversation_id,
            "output": output_text,
            "model": self._current_model,
            "usage": usage,
        }
