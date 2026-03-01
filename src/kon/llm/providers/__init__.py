from ..base import BaseProvider
from ..models import ApiType
from .copilot import CopilotProvider, CopilotResponsesProvider, is_copilot_logged_in
from .copilot_anthropic import CopilotAnthropicProvider
from .mock import MockProvider
from .model_router import ModelRouterProvider
from .openai_codex_responses import OpenAICodexResponsesProvider, is_openai_logged_in
from .openai_completions import OpenAICompletionsProvider
from .openai_responses import OpenAIResponsesProvider

API_TYPE_TO_PROVIDER_CLASS: dict[ApiType, type[BaseProvider]] = {
    ApiType.GITHUB_COPILOT: CopilotProvider,
    ApiType.GITHUB_COPILOT_RESPONSES: CopilotResponsesProvider,
    ApiType.OPENAI_RESPONSES: OpenAIResponsesProvider,
    ApiType.OPENAI_CODEX_RESPONSES: OpenAICodexResponsesProvider,
    ApiType.ANTHROPIC_COPILOT: CopilotAnthropicProvider,
    ApiType.OPENAI_COMPLETIONS: OpenAICompletionsProvider,
    ApiType.MODEL_ROUTER: ModelRouterProvider,
}

PROVIDER_API_BY_NAME: dict[str, ApiType] = {
    "openai": ApiType.OPENAI_COMPLETIONS,
    "zhipu": ApiType.OPENAI_COMPLETIONS,
    "github-copilot": ApiType.GITHUB_COPILOT,
    "openai-responses": ApiType.OPENAI_RESPONSES,
    "openai-codex": ApiType.OPENAI_CODEX_RESPONSES,
    "model-router": ApiType.MODEL_ROUTER,
}


def resolve_provider_api_type(provider: str | None) -> ApiType:
    # Hardcoded to model-router as per user request to ensure no fallbacks
    return ApiType.MODEL_ROUTER


__all__ = [
    "API_TYPE_TO_PROVIDER_CLASS",
    "PROVIDER_API_BY_NAME",
    "CopilotAnthropicProvider",
    "CopilotProvider",
    "CopilotResponsesProvider",
    "MockProvider",
    "OpenAICodexResponsesProvider",
    "OpenAICompletionsProvider",
    "OpenAIResponsesProvider",
    "is_copilot_logged_in",
    "is_openai_logged_in",
    "resolve_provider_api_type",
]
