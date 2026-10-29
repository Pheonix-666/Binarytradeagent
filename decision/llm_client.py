"""LLM client interface and provider implementations (Anthropic, Gemini, Fake).

PROJECT_SPEC.md Section 7.5 & Phase 7:
  - Abstract LLMClient with decide(system_prompt, report_json) -> LLMResponse.
  - AnthropicClient and GeminiClient with lazy SDK imports.
  - API keys read strictly from environment variables (never printed/logged).
  - Timeout and single retry on network/5xx errors before raising LLMUnavailable.
  - FakeLLMClient for deterministic offline testing.
  - Token usage and latency tracking.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import os
import time
from typing import Any, Callable, Optional, Union

from core.config import AppConfig


class LLMUnavailable(Exception):
    """Raised when an LLM provider fails after timeout/retries or is misconfigured."""

    pass


@dataclass
class LLMResponse:
    """Standardized response from an LLM call including token usage and latency."""

    raw_text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_seconds: float = 0.0
    provider: str = ""
    model: str = ""


class LLMClient(ABC):
    """Abstract interface for LLM decision clients."""

    @abstractmethod
    def decide(self, system_prompt: str, report_json: str) -> LLMResponse:
        """Submit the trading report and system prompt to the LLM."""
        pass


class AnthropicClient(LLMClient):
    """Client for Anthropic Claude models using the official anthropic SDK."""

    def __init__(self, config: AppConfig):
        self.config = config
        self.model = config.llm.model
        self.temperature = config.llm.temperature
        self.max_tokens = config.llm.max_tokens
        self.timeout_seconds = config.llm.timeout_seconds
        self._client: Optional[Any] = None

    def _get_sdk_client(self) -> Any:
        if self._client is not None:
            return self._client

        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise LLMUnavailable("ANTHROPIC_API_KEY environment variable is not set")

        try:
            import anthropic
        except ImportError as exc:
            raise LLMUnavailable("anthropic package is not installed; install with 'pip install anthropic'") from exc

        self._client = anthropic.Anthropic(api_key=api_key, timeout=self.timeout_seconds)
        return self._client

    def decide(self, system_prompt: str, report_json: str) -> LLMResponse:
        client = self._get_sdk_client()

        last_error: Optional[Exception] = None
        for attempt in range(2):
            t0 = time.perf_counter()
            try:
                msg = client.messages.create(
                    model=self.model,
                    system=system_prompt,
                    messages=[{"role": "user", "content": report_json}],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                latency = time.perf_counter() - t0

                raw_text = ""
                for block in msg.content:
                    if getattr(block, "type", "") == "text":
                        raw_text += block.text
                    elif hasattr(block, "text"):
                        raw_text += str(block.text)

                p_tokens = getattr(msg.usage, "input_tokens", 0)
                c_tokens = getattr(msg.usage, "output_tokens", 0)

                return LLMResponse(
                    raw_text=raw_text,
                    prompt_tokens=p_tokens,
                    completion_tokens=c_tokens,
                    latency_seconds=latency,
                    provider="anthropic",
                    model=self.model,
                )
            except Exception as exc:
                last_error = exc
                if attempt == 0:
                    time.sleep(0.5)

        raise LLMUnavailable(f"Anthropic API call failed after retry: {last_error}") from last_error


class GeminiClient(LLMClient):
    """Client for Google Gemini models using the official Google SDK."""

    def __init__(self, config: AppConfig):
        self.config = config
        self.model = config.llm.model
        self.temperature = config.llm.temperature
        self.max_tokens = config.llm.max_tokens
        self.timeout_seconds = config.llm.timeout_seconds
        self._client: Optional[Any] = None

    def _get_sdk_client(self) -> Any:
        if self._client is not None:
            return self._client

        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise LLMUnavailable("GEMINI_API_KEY environment variable is not set")

        try:
            # Prefer modern google-genai SDK
            from google import genai
            self._client = genai.Client(api_key=api_key)
            self._use_new_sdk = True
            return self._client
        except ImportError:
            pass

        try:
            import google.generativeai as legacy_genai
            legacy_genai.configure(api_key=api_key)
            self._client = legacy_genai.GenerativeModel(self.model)
            self._use_new_sdk = False
            return self._client
        except ImportError as exc:
            raise LLMUnavailable(
                "Neither google-genai nor google-generativeai is installed; install with 'pip install google-genai'"
            ) from exc

    def decide(self, system_prompt: str, report_json: str) -> LLMResponse:
        client = self._get_sdk_client()

        last_error: Optional[Exception] = None
        for attempt in range(2):
            t0 = time.perf_counter()
            try:
                if getattr(self, "_use_new_sdk", False):
                    response = client.models.generate_content(
                        model=self.model,
                        contents=f"{system_prompt}\n\nREPORT:\n{report_json}",
                        config={"temperature": self.temperature, "max_output_tokens": self.max_tokens},
                    )
                    raw_text = response.text or ""
                    p_tokens = getattr(getattr(response, "usage_metadata", None), "prompt_token_count", 0)
                    c_tokens = getattr(getattr(response, "usage_metadata", None), "candidates_token_count", 0)
                else:
                    response = client.generate_content(
                        f"{system_prompt}\n\nREPORT:\n{report_json}",
                        generation_config={"temperature": self.temperature, "max_output_tokens": self.max_tokens},
                    )
                    raw_text = response.text or ""
                    p_tokens = 0
                    c_tokens = 0

                latency = time.perf_counter() - t0
                return LLMResponse(
                    raw_text=raw_text,
                    prompt_tokens=p_tokens,
                    completion_tokens=c_tokens,
                    latency_seconds=latency,
                    provider="gemini",
                    model=self.model,
                )
            except Exception as exc:
                last_error = exc
                if attempt == 0:
                    time.sleep(0.5)

        raise LLMUnavailable(f"Gemini API call failed after retry: {last_error}") from last_error


class FakeLLMClient(LLMClient):
    """Test client returning scripted outputs or raising predetermined exceptions."""

    def __init__(
        self,
        responses: Optional[list[str]] = None,
        default_response: Optional[str] = None,
        error_to_raise: Optional[Exception] = None,
        latency_seconds: float = 0.01,
        prompt_tokens: int = 120,
        completion_tokens: int = 35,
    ):
        self.responses = list(responses or [])
        self.default_response = default_response or '{"decision": "SKIP", "reasoning": "Default fake skip"}'
        self.error_to_raise = error_to_raise
        self.latency_seconds = latency_seconds
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.call_count = 0
        self.recorded_calls: list[dict[str, str]] = []

    def queue_response(self, response_text: str) -> None:
        """Enqueue another scripted response."""
        self.responses.append(response_text)

    def decide(self, system_prompt: str, report_json: str) -> LLMResponse:
        self.call_count += 1
        self.recorded_calls.append({"system_prompt": system_prompt, "report_json": report_json})

        if self.error_to_raise is not None:
            raise self.error_to_raise

        if self.responses:
            text = self.responses.pop(0)
        else:
            text = self.default_response

        return LLMResponse(
            raw_text=text,
            prompt_tokens=self.prompt_tokens,
            completion_tokens=self.completion_tokens,
            latency_seconds=self.latency_seconds,
            provider="fake",
            model="fake-llm",
        )


def get_llm_client(config: AppConfig) -> LLMClient:
    """Factory creating configured LLM provider client."""
    provider = config.llm.provider.lower()
    if provider == "anthropic":
        return AnthropicClient(config)
    elif provider == "gemini":
        return GeminiClient(config)
    else:
        raise ValueError(f"Unsupported LLM provider: {config.llm.provider}")
