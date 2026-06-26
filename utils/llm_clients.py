"""
LLM Client Utilities
====================

Provides unified interface for OpenAI and Anthropic API clients.
Used throughout the evaluation examples for model-based grading.

Book Reference: Chapter 3 (LLM-as-Judge requires API access)

Features:
- Automatic retry with exponential backoff for transient errors
- Unified interface for OpenAI and Anthropic
- Token usage tracking
"""

import os
from typing import Optional, Literal
from dataclasses import dataclass

try:
    from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
    TENACITY_AVAILABLE = True
except ImportError:
    TENACITY_AVAILABLE = False


@dataclass
class LLMResponse:
    """Standardized response from LLM APIs."""
    content: str
    model: str
    input_tokens: int
    output_tokens: int

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


def get_openai_client():
    """
    Get an OpenAI client instance.

    Requires OPENAI_API_KEY environment variable.

    Returns:
        OpenAI client instance

    Example:
        >>> client = get_openai_client()
        >>> response = client.chat.completions.create(
        ...     model="gpt-4o",
        ...     messages=[{"role": "user", "content": "Hello"}]
        ... )
    """
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("openai package required. Install with: pip install openai")

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY environment variable not set. "
            "Set it with: export OPENAI_API_KEY='your-key'"
        )

    return OpenAI(api_key=api_key)


def get_anthropic_client():
    """
    Get an Anthropic client instance.

    Requires ANTHROPIC_API_KEY environment variable.

    Returns:
        Anthropic client instance

    Example:
        >>> client = get_anthropic_client()
        >>> response = client.messages.create(
        ...     model="claude-sonnet-4-20250514",
        ...     max_tokens=1024,
        ...     messages=[{"role": "user", "content": "Hello"}]
        ... )
    """
    try:
        from anthropic import Anthropic
    except ImportError:
        raise ImportError("anthropic package required. Install with: pip install anthropic")

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError(
            "ANTHROPIC_API_KEY environment variable not set. "
            "Set it with: export ANTHROPIC_API_KEY='your-key'"
        )

    return Anthropic(api_key=api_key)


def call_llm(
    prompt: str,
    system: Optional[str] = None,
    model: str = "gpt-4o-mini",
    provider: Literal["openai", "anthropic"] = "openai",
    temperature: float = 0.0,
    max_tokens: int = 1024,
    timeout: float = 60.0,
) -> LLMResponse:
    """
    Unified LLM calling interface.

    Abstracts away provider-specific API differences for simpler code.

    Args:
        prompt: User message to send
        system: Optional system prompt
        model: Model identifier (provider-specific)
        provider: "openai" or "anthropic"
        temperature: Sampling temperature (0.0 = deterministic)
        max_tokens: Maximum response tokens
        timeout: Per-request timeout in seconds. Without this a stalled
            connection can hang the call indefinitely (and, combined with
            retries, multiply that hang).

    Returns:
        LLMResponse with content and token usage

    Example:
        >>> response = call_llm(
        ...     prompt="What is 2+2?",
        ...     system="You are a helpful math tutor.",
        ...     model="gpt-4o-mini",
        ...     provider="openai"
        ... )
        >>> print(response.content)
        "2 + 2 equals 4."
    """
    if provider == "openai":
        return _call_openai(prompt, system, model, temperature, max_tokens, timeout)
    elif provider == "anthropic":
        return _call_anthropic(prompt, system, model, temperature, max_tokens, timeout)
    else:
        raise ValueError(f"Unknown provider: {provider}. Use 'openai' or 'anthropic'.")


def _transient_exceptions(module_name: str) -> tuple:
    """Transient API error types worth retrying for a provider SDK.

    Returns a tuple of exception classes (timeout / connection / rate-limit /
    5xx). Returning an empty tuple disables retries rather than retrying on
    every exception (which would mask non-transient bugs like auth errors or
    malformed requests).
    """
    try:
        mod = __import__(module_name)
    except ImportError:
        return ()
    names = ("APITimeoutError", "APIConnectionError", "RateLimitError",
             "InternalServerError")
    return tuple(getattr(mod, n) for n in names if hasattr(mod, n))


def _make_openai_call(client, messages, model, temperature, max_tokens, timeout):
    """Make the actual OpenAI API call (separated for retry decorator)."""
    return client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
    )


def _call_openai(
    prompt: str,
    system: Optional[str],
    model: str,
    temperature: float,
    max_tokens: int,
    timeout: float = 60.0,
) -> LLMResponse:
    """Call OpenAI API with automatic retry on transient errors."""
    client = get_openai_client()

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    transient = _transient_exceptions("openai")
    if TENACITY_AVAILABLE and transient:
        @retry(
            retry=retry_if_exception_type(transient),
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=1, min=1, max=10),
            reraise=True
        )
        def call_with_retry():
            return _make_openai_call(client, messages, model, temperature, max_tokens, timeout)
        response = call_with_retry()
    else:
        response = _make_openai_call(client, messages, model, temperature, max_tokens, timeout)

    return LLMResponse(
        content=response.choices[0].message.content,
        model=response.model,
        input_tokens=response.usage.prompt_tokens,
        output_tokens=response.usage.completion_tokens,
    )


def _make_anthropic_call(client, kwargs):
    """Make the actual Anthropic API call (separated for retry decorator)."""
    return client.messages.create(**kwargs)


def _call_anthropic(
    prompt: str,
    system: Optional[str],
    model: str,
    temperature: float,
    max_tokens: int,
    timeout: float = 60.0,
) -> LLMResponse:
    """Call Anthropic API with automatic retry on transient errors."""
    client = get_anthropic_client()

    kwargs = {
        "model": model,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "messages": [{"role": "user", "content": prompt}],
        "timeout": timeout,
    }

    if system:
        kwargs["system"] = system

    transient = _transient_exceptions("anthropic")
    if TENACITY_AVAILABLE and transient:
        @retry(
            retry=retry_if_exception_type(transient),
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=1, min=1, max=10),
            reraise=True
        )
        def call_with_retry():
            return _make_anthropic_call(client, kwargs)
        response = call_with_retry()
    else:
        response = _make_anthropic_call(client, kwargs)

    return LLMResponse(
        content=response.content[0].text,
        model=response.model,
        input_tokens=response.usage.input_tokens,
        output_tokens=response.usage.output_tokens,
    )


# Demo usage
if __name__ == "__main__":
    print("LLM Client Utilities")
    print("=" * 50)
    print("\nThis module provides:")
    print("  - get_openai_client(): Returns OpenAI client")
    print("  - get_anthropic_client(): Returns Anthropic client")
    print("  - call_llm(): Unified interface for both providers")
    print("\nSet environment variables before use:")
    print("  export OPENAI_API_KEY='your-key'")
    print("  export ANTHROPIC_API_KEY='your-key'")
