import asyncio
import datetime
import json
import os
import re
import sys
from collections.abc import AsyncIterator
from pathlib import Path

from ...core.types import (
    ImageContent,
    Message,
    MetadataPart,
    StopReason,
    StreamDone,
    StreamPart,
    TextContent,
    TextPart,
    ToolCallDelta,
    ToolCallStart,
    ToolDefinition,
    ToolResultMessage,
    UserMessage,
)
from ..base import BaseProvider, LLMStream, ProviderConfig


class ModelRouterProvider(BaseProvider):
    name = "model-router"

    def __init__(self, config: ProviderConfig):
        super().__init__(config)

        # 1. Detect project root more robustly
        def find_root(start_path: Path) -> Path:
            curr = start_path.resolve()
            for _ in range(10):
                if (curr / "model-router.py").exists() or (curr / "pyproject.toml").exists():
                    return curr
                if curr.parent == curr:
                    break
                curr = curr.parent
            return start_path.resolve().parent.parent.parent.parent.parent

        project_root = find_root(Path(__file__))

        # Diagnostic logging to file
        self.log_file = Path("/tmp/kon_router_debug.log")
        with open(self.log_file, "a") as f:
            f.write(f"\n--- ModelRouter Init at {datetime.datetime.now()} ---\n")
            f.write(f"  Detected project root: {project_root}\n")
            f.write(f"  CWD: {os.getcwd()}\n")
            f.write(f"  __file__: {__file__}\n")

        if str(project_root) not in sys.path:
            sys.path.append(str(project_root))

        try:
            import importlib.util

            router_file = project_root / "model-router.py"
            spec = importlib.util.spec_from_file_location("model_router", str(router_file))
            if spec and spec.loader:
                mr_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mr_module)
                self.router_class = mr_module.MistralRouter
            else:
                raise ImportError(f"Could not load {router_file}")
        except Exception as e:
            with open(self.log_file, "a") as f:
                f.write(f"  ❌ Failed to load model-router.py: {e}\n")
            raise RuntimeError(f"Failed to load model-router.py: {e}") from e

        from dotenv import load_dotenv

        # 2. Load environment variables from detected root
        env_files = [".env.local", ".env"]
        loaded = False
        for env_file in env_files:
            env_path = project_root / env_file
            if env_path.exists():
                with open(self.log_file, "a") as f:
                    f.write(f"  📂 Loading environment from: {env_path}\n")
                load_dotenv(dotenv_path=env_path, override=True)
                loaded = True
                break

        if not loaded:
            with open(self.log_file, "a") as f:
                f.write(f"  ⚠️ No .env or .env.local found in {project_root}\n")

        # Use the API key provided in config or env
        api_key = config.api_key or os.environ.get("MISTRAL_API_KEY")
        if not api_key:
            # Check CWD as a last resort
            cwd_env = Path.cwd() / ".env.local"
            if cwd_env.exists():
                with open(self.log_file, "a") as f:
                    f.write(f"  📂 Loading from CWD .env.local: {cwd_env}\n")
                load_dotenv(dotenv_path=cwd_env, override=True)
            api_key = os.environ.get("MISTRAL_API_KEY")

        with open(self.log_file, "a") as f:
            status = "FOUND" if api_key else "NOT FOUND"
            f.write(f"  Result: MISTRAL_API_KEY is {status}\n")
            if api_key:
                f.write(f"  Key starts with: {api_key[:4]}...\n")
            else:
                f.write(f"  Environ keys: {list(os.environ.keys())}\n")

        if not api_key:
            raise ValueError("MISTRAL_API_KEY not found in config or environment.")

        self._router = self.router_class(api_key=api_key)

    async def _stream_impl(
        self,
        messages: list[Message],
        *,
        system_prompt: str | None = None,
        tools: list[ToolDefinition] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMStream:
        # 1. Extract the latest user message
        # Since model-router preserves state, we only send the newest input.
        latest_input = ""
        for msg in reversed(messages):
            if isinstance(msg, UserMessage):
                if isinstance(msg.content, str):
                    latest_input = msg.content
                else:
                    parts = []
                    for c in msg.content:
                        if isinstance(c, TextContent):
                            if c.text:
                                parts.append(c.text)
                        elif isinstance(c, ImageContent) and c.data:
                            parts.append(f"Image: {c.data}")
                    latest_input = "\n".join(parts)
                break
            elif isinstance(msg, ToolResultMessage):
                # Handle tool results as well
                latest_input = f"Tool result for {msg.tool_name}:\n"
                parts = []
                for c in msg.content:
                    if isinstance(c, TextContent):
                        if c.text:
                            parts.append(c.text)
                    elif isinstance(c, ImageContent) and c.data:
                        parts.append(f"Image: {c.data}")
                latest_input += "\n".join(parts)
                break

        # 2. Add System Prompt and Tool definitions if it's the first message or if context changed
        # For simplicity and to follow user's "two-way street" request:
        # We will embed the system instructions and tool definitions into the prompt if they exist.
        full_prompt = ""
        if system_prompt:
            full_prompt += f"System Instructions:\n{system_prompt}\n\n"

        if tools:
            full_prompt += "## Tool Usage Instructions\n"
            full_prompt += (
                "You possess various tools. To use one, you MUST output a tool call "
                "using the following format EXACTLY:\n"
            )
            full_prompt += '<tool_call name="TOOL_NAME">{"PARAM_NAME": "VALUE"}</tool_call>\n\n'
            full_prompt += "### CRITICAL RULES:\n"
            full_prompt += "1. The JSON inside the tags must be a SINGLE FLAT DICTIONARY.\n"
            full_prompt += (
                "2. You MUST provide ALL required arguments in the JSON. "
                "NEVER output an empty {} if the tool requires parameters.\n"
            )
            full_prompt += "3. For the 'bash' tool, provide the 'command' parameter.\n"
            full_prompt += (
                "4. For the 'write' tool, provide both 'path' and 'content' parameters.\n\n"
            )
            full_prompt += "### EXAMPLE OF CORRECT TOOL CALL:\n"
            full_prompt += "User: Create a hello world script.\n"
            full_prompt += (
                "Assistant: "
                '<tool_call name="write">{"path": "hello.py", "content": "print(\'hello\')"}'
                "</tool_call>\n\n"
            )
            full_prompt += "### Available Tools:\n"
            for t in tools:
                full_prompt += (
                    f"- {t.name}: {t.description}. Parameters: {json.dumps(t.parameters)}\n"
                )
            full_prompt += "\n"

        full_prompt += f"User Input: {latest_input}\n"

        # 3. Call the router (synchronous call, so we wrap in thread)
        payload = {"prompt": full_prompt}

        # Determine if we should pass a model override.
        # If the model is 'model-router' or the baseline 'ministral-3b-latest'
        # we omit it to allow the dynamic router to classify.
        model_override = self.config.model
        if model_override and model_override not in ["model-router", "ministral-3b-latest"]:
            payload["model"] = model_override
            with open(self.log_file, "a") as f:
                f.write(f"  🎯 Using explicit model override: {model_override}\n")
        else:
            with open(self.log_file, "a") as f:
                f.write("  🧠 No explicit override; triggering dynamic classification.\n")

        loop = asyncio.get_running_loop()
        # The route() method in MistralRouter is synchronous
        # Errors will now correctly bubble up to the caller
        result = await loop.run_in_executor(None, self._router.route, payload)
        output_text = result.get("output", "")

        # DEBUG: Print raw output to see what the model actually returned
        print(f"\n[DEBUG] Raw model output: {output_text}\n")

        llm_stream = LLMStream()

        # Extract model name from router result.
        # If no model is found, we let it fail or default at the Agent level.
        actual_model = result.get("model") or self.config.model

        async def _yield_metadata_then_process():
            yield MetadataPart(model=actual_model, provider="model-router")
            async for part in self._process_output(output_text):
                yield part

        llm_stream.set_iterator(_yield_metadata_then_process())
        return llm_stream

    def _clean_json(self, text: str) -> str:
        """Clean markdown blocks and other common noise from JSON strings."""
        text = text.strip()
        # Remove triple backticks if present
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text)
            text = re.sub(r"\s*```$", "", text)
        return text.strip()

    def _repair_json(self, text: str) -> str:
        """Attempt to repair common JSON failures from smaller models."""
        text = self._clean_json(text)
        if not text:
            return "{}"

        # 1. Try direct parse
        try:
            json.loads(text)
            return text
        except json.JSONDecodeError:
            pass

        # 2. Extract between { and }
        match = re.search(r"(\{.*\})", text, re.DOTALL)
        if match:
            candidate = match.group(1)
            try:
                json.loads(candidate)
                return candidate
            except json.JSONDecodeError:
                # 3. Handle unescaped newlines in values
                # This is a bit aggressive: replace literal newlines in values
                # but ONLY if followed by a comma or closing brace later.
                # Or just escape all newlines not preceded by backslashes.
                # Better: try to find the path and content keys via regex.
                pass

        return text  # Fallback to original

    async def _process_output(self, text: str) -> AsyncIterator[StreamPart]:
        # Simple parser for <tool_call name="xxx">json</tool_call>
        # Note: MistralRouter returns the full text at once, so we simulate streaming
        # or just yield the whole thing if it's small.

        # Pattern to find tool calls
        pattern = r'<tool_call name="(?P<name>[^"]+)">(?P<args>.*?)</tool_call>'

        last_pos = 0
        for tool_index, match in enumerate(re.finditer(pattern, text, re.DOTALL)):
            # Yield text before tool call
            pre_text = text[last_pos : match.start()].strip()
            if pre_text:
                yield TextPart(text=pre_text)

            tool_name = match.group("name")
            raw_args = match.group("args")
            tool_args_str = self._repair_json(raw_args)

            # Robust parsing: handle raw strings or nested objects
            try:
                # Try to parse the arguably "repaired" string
                args_json = json.loads(tool_args_str)

                # If it's a dict with only ONE key named after the tool, unwrap it.
                # This handles cases like {"write": {"path": "...", "content": "..."}}
                if (
                    isinstance(args_json, dict)
                    and len(args_json) == 1
                    and next(iter(args_json.keys())) == tool_name
                ):
                    val = next(iter(args_json.values()))
                    if isinstance(val, dict):
                        args_json = val

                # Special case unwrapping for generic nesting
                while (
                    isinstance(args_json, dict)
                    and len(args_json) == 1
                    and isinstance(next(iter(args_json.values())), dict)
                    and next(iter(args_json.keys())) not in ["path", "content", "command"]
                ):
                    args_json = next(iter(args_json.values()))

                # If tool is 'bash' and 'command' is missing
                if (
                    tool_name == "bash"
                    and isinstance(args_json, dict)
                    and "command" not in args_json
                ):
                    if len(args_json) == 1:
                        args_json = {"command": next(iter(args_json.values()))}
                    elif not args_json:
                        args_json = {"command": raw_args.strip()}

                # Final check for 'write' tool
                if tool_name == "write":
                    if not isinstance(args_json, dict):
                        args_json = {}
                    if "path" not in args_json or "content" not in args_json:
                        # Try regex fallback if keys are missing
                        path_m = re.search(r'"path":\s*"([^"]+)"', raw_args)
                        # Content is harder because it can have quotes.
                        content_m = re.search(r'"content":\s*"(.*)"\s*\}', raw_args, re.DOTALL)
                        if path_m and "path" not in args_json:
                            args_json["path"] = path_m.group(1)
                        if content_m and "content" not in args_json:
                            # Try to strip trailing " and }
                            c = content_m.group(1).strip()
                            args_json["content"] = c

                tool_args_str = json.dumps(args_json)
            except Exception:
                # If everything failed, last effort check
                if tool_name == "bash" and tool_args_str:
                    tool_args_str = json.dumps({"command": tool_args_str})
                elif tool_name == "write":
                    # Try to find path and content in raw text
                    pm = re.search(r'path["\s:]+([^"\n]+)', tool_args_str)
                    cm = re.search(r'content["\s:]+(.*)', tool_args_str, re.DOTALL)
                    if pm and cm:
                        tool_args_str = json.dumps(
                            {
                                "path": pm.group(1).strip(" \"'"),
                                "content": cm.group(1).strip(" \"'\n}"),
                            }
                        )
                    else:
                        # If we can't find anything, don't yield an empty dict!
                        # Instead, try to use the raw args if they look like a path
                        if len(tool_args_str.splitlines()) == 1 and "/" in tool_args_str:
                            tool_args_str = json.dumps(
                                {"path": tool_args_str.strip(), "content": ""}
                            )
                        else:
                            # SKIP broken tool calls that we can't repair
                            continue

            # Start tool call
            if (not tool_args_str or tool_args_str == "{}") and tool_name == "write":
                continue

            yield ToolCallStart(id=f"mr-{tool_index}", name=tool_name, index=tool_index)
            # Send arguments
            yield ToolCallDelta(index=tool_index, arguments_delta=tool_args_str)

            last_pos = match.end()

        # Yield remaining text
        post_text = text[last_pos:].strip()
        if post_text:
            yield TextPart(text=post_text)
        yield StreamDone(stop_reason=StopReason.STOP)

    def should_retry_for_error(self, error: Exception) -> bool:
        return False
