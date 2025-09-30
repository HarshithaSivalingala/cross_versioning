import re
from typing import Optional
import openai
from together import Together


def _generate_together(prompt: str, model: Optional[str] = None) -> str:
    if Together is None:
        raise ImportError("together package not installed. Run: pip install together")

    try:
        client = Together()
        response = client.chat.completions.create(
            model=model or "openai/gpt-oss-20b",
            messages=[{"role": "user", "content": prompt}],
        )
        content = response.choices[0].message.content # type: ignore
        print(content)
        return content # type: ignore
    except Exception as e:
        raise RuntimeError(f"Together API error: {str(e)}") from e


def _generate_openrouter(prompt: str, model: Optional[str] = None) -> str:
    if openai is None:
        raise ImportError("openai package not installed. Run: pip install openai")

    try:
        client = openai.OpenAI(
            api_key="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",
            base_url="https://openrouter.ai/api/v1",
        )

        response = client.chat.completions.create(
            model=model or "mistralai/mistral-7b-instruct",  # Any OpenRouter-supported model
            messages=[{"role": "user", "content": prompt}],
            max_tokens=2000,
        )
        print(len(response.choices))
        print(response.choices[0].message)
        return response.choices[0].message.content # type: ignore
    except Exception as e:
        raise RuntimeError(f"OpenRouter API error: {str(e)}") from e


def generate(prompt: str, *, provider: str = "together", model: Optional[str] = None) -> str:
    if provider == "together":
        return _generate_together(prompt, model=model)
    if provider == "openrouter":
        return _generate_openrouter(prompt, model=model)
    raise ValueError(f"Unsupported provider '{provider}'")


def call_llm(
    prompt: str,
    model: str = "gemini-1.5-flash",
    *,
    provider: str = "together",
) -> str:
    """Call the configured LLM provider (Together default, OpenRouter optional)."""
    try:
        provider_model = None if model == "gemini-1.5-flash" else model
        response = generate(prompt, provider=provider, model=provider_model)

        if not response:
            raise Exception("Empty response from LLM provider")

        return response

    except ImportError as exc:
        raise Exception(str(exc)) from exc
    except Exception as e:
        raise Exception(f" API call failed: {str(e)}")
