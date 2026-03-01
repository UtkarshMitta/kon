import asyncio
import json
import os
import re
import sys
from collections.abc import AsyncIterator
from pathlib import Path

from ...core.types import (
    Message,
    StopReason,
    StreamDone,
    StreamPart,
    TextPart,
    ToolCallStart,
    ToolCallDelta,
    ToolDefinition,
    UserMessage,
    ToolResultMessage,
)
from ..base import BaseProvider, LLMStream, ProviderConfig


class ModelRouterProvider(BaseProvider):
    name = "model-router"

    def __init__(self, config: ProviderConfig):
        super().__init__(config)
        
        # Add project root to sys.path to import model-router.py
        project_root = str(Path(__file__).parent.parent.parent.parent.parent)
        if project_root not in sys.path:
            sys.path.append(project_root)
                    
        try:
            # The file is model-router.py, so the module name in python needs to be handled
            # Since it has a hyphen, we use importlib or rename hint.
            # Actually, I'll just load it dynamically.
            import importlib.util
            spec = importlib.util.spec_from_file_location("model_router", os.path.join(project_root, "model-router.py"))
            if spec and spec.loader:
                mr_module = importlib.util.module_from_spec(spec)
                if mr_module and spec.loader:
                    spec.loader.exec_module(mr_module)
                    self.router_class = getattr(mr_module, "MistralRouter")
                else:
                    raise ImportError("Could not initialize mr_module")
            else:
                raise ImportError("Could not load spec or loader for model-router.py")
        except Exception as e:
            raise RuntimeError(f"Failed to load model-router.py: {e}")

        # Use the API key provided in config or env
        api_key = config.api_key or os.environ.get("MISTRAL_API_KEY")
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
                        if hasattr(c, "text"):
                            parts.append(getattr(c, "text"))
                    latest_input = "\n".join(parts)
                break
            elif isinstance(msg, ToolResultMessage):
                # Handle tool results as well
                latest_input = f"Tool result for {msg.tool_name}:\n"
                parts = []
                for c in msg.content:
                    if hasattr(c, "text"):
                        parts.append(c.text)
                latest_input += "\n".join(parts)
                break

        # 2. Add System Prompt and Tool definitions if it's the first message or if context changed
        # For simplicity and to follow user's "two-way street" request:
        # We will embed the system instructions and tool definitions into the prompt if they exist.
        full_prompt = ""
        if system_prompt:
            full_prompt += f"System Instructions:\n{system_prompt}\n\n"
        
        if tools:
            full_prompt += "Available Tools:\n"
            for t in tools:
                full_prompt += f"- {t.name}: {t.description}. Parameters: {json.dumps(t.parameters)}\n"
            full_prompt += "\nTo use a tool, output: <tool_call name=\"tool_name\">{\"arg\": \"value\"}</tool_call>\n\n"

        full_prompt += f"User Input: {latest_input}\n"

        # 3. Call the router (synchronous call, so we wrap in thread)
        payload = {
            "prompt": full_prompt,
            "model": self.config.model or "ministral-3b-latest"
        }
        
        loop = asyncio.get_running_loop()
        try:
            # The route() method in MistralRouter is synchronous
            result = await loop.run_in_executor(None, self._router.route, payload)
            output_text = result.get("output", "")
        except Exception as e:
            output_text = f"Error from model-router: {str(e)}"

        llm_stream = LLMStream()
        llm_stream.set_iterator(self._process_output(output_text))
        return llm_stream

    async def _process_output(self, text: str) -> AsyncIterator[StreamPart]:
        # Simple parser for <tool_call name="xxx">json</tool_call>
        # Note: MistralRouter returns the full text at once, so we simulate streaming 
        # or just yield the whole thing if it's small.
        
        # Pattern to find tool calls
        pattern = r'<tool_call name="(?P<name>[^"]+)">(?P<args>.*?)</tool_call>'
        
        last_pos = 0
        tool_index = 0
        
        for match in re.finditer(pattern, text, re.DOTALL):
            # Yield text before tool call
            pre_text = text[last_pos:match.start()].strip()
            if pre_text:
                yield TextPart(text=pre_text)
            
            tool_name = match.group("name")
            tool_args_str = match.group("args").strip()
            
            # Start tool call
            yield ToolCallStart(id=f"mr-{tool_index}", name=tool_name, index=tool_index)
            # Send arguments
            yield ToolCallDelta(index=tool_index, arguments_delta=tool_args_str)
            
            last_pos = match.end()
            tool_index += 1
            
        # Yield remaining text
        post_text = text[last_pos:].strip()
        if post_text:
            yield TextPart(text=post_text)
        yield StreamDone(stop_reason=StopReason.STOP)            
        yield StreamDone(stop_reason=StopReason.STOP)

    def should_retry_for_error(self, error: Exception) -> bool:
        return False
