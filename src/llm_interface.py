import os
from typing import Optional

from dotenv import load_dotenv
try:
    from together import Together
except ModuleNotFoundError:  # pragma: no cover - optional dependency for tests
    Together = None  # type: ignore[assignment]

try:
    import openai
except ModuleNotFoundError:  # pragma: no cover - optional dependency for tests
    openai = None  # type: ignore[assignment]

load_dotenv()

DEFAULT_OPENROUTER_MODEL = "openai/gpt-4o-mini"
DEFAULT_TOGETHER_MODEL = "openai/gpt-oss-20b"
DEFAULT_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


def _require_env(key: str) -> str:
    value = os.getenv(key)
    if not value:
        raise RuntimeError(f"{key} not set in environment")
    return value


def _extract_content(response) -> str:
    message = response.choices[0].message  # type: ignore[attr-defined]
    content = getattr(message, "content", None)
    if not content:
        raise RuntimeError("LLM response missing content")
    return content  # type: ignore[return-value]


def _generate_together(prompt: str, model: Optional[str] = None) -> str:
    if Together is None:
        raise RuntimeError("together package is not installed; cannot use Together provider.")

    api_key = _require_env("TOGETHER_API_KEY")
    model_name = model or os.getenv("TOGETHER_MODEL", DEFAULT_TOGETHER_MODEL)

    client = Together(api_key=api_key)
    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
    )
    return _extract_content(response)


def _generate_openrouter(prompt: str, model: Optional[str] = None) -> str:
    if openai is None:
        raise RuntimeError("openai package is not installed; cannot use OpenRouter provider.")

    api_key = _require_env("OPENROUTER_API_KEY")
    model_name = model or os.getenv("OPENROUTER_MODEL", DEFAULT_OPENROUTER_MODEL)
    base_url = os.getenv("OPENROUTER_BASE_URL", DEFAULT_OPENROUTER_BASE_URL)

    client = openai.OpenAI(api_key=api_key, base_url=base_url)
    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=2000,
    )
    return _extract_content(response)


def generate(prompt: str, *, provider: str = "openrouter", model: Optional[str] = None) -> str:
    if provider == "together":
        return _generate_together(prompt, model=model)
    if provider == "openrouter":
        return _generate_openrouter(prompt, model=model)
    raise ValueError(f"Unsupported provider '{provider}'")


def call_llm(
    prompt: str,
    model: str = DEFAULT_OPENROUTER_MODEL,
    *,
    provider: str = "openrouter",
) -> str:
    """Call the configured LLM provider with sensible defaults."""
    provider_model = None if model == DEFAULT_OPENROUTER_MODEL else model
    response = generate(prompt, provider=provider, model=provider_model)
    if not response:
        raise RuntimeError("Empty response from LLM provider")
    return response
