"""Robust LLM client with exponential backoff, JSON extraction, and error handling."""

import json
import logging
import re
import time
from typing import Any, Dict, List, Optional

import requests
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from config import (
    BACKOFF_MULTIPLIER,
    BASE_DELAY,
    DEEPSEEK_API_KEY,
    DEEPSEEK_ENDPOINT,
    GEMINI_API_KEY,
    GEMINI_ENDPOINT,
    MAX_DELAY,
    MAX_RETRIES,
    SILICONFLOW_API_KEY,
    SILICONFLOW_ENDPOINT,
)

logger = logging.getLogger(__name__)


# ── JSON extraction ───────────────────────────────────────────────────

def extract_json(text: str) -> Dict[str, Any]:
    """Extract a JSON object from text that may contain markdown fences or extra content.

    Tries multiple strategies in order; raises ValueError if all fail.
    """
    if not text or not text.strip():
        raise ValueError("Empty response — cannot extract JSON")

    original = text.strip()

    strategies: List[str] = []

    # Strategy 1: strip ```json ... ``` fences
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", original, re.DOTALL)
    if m:
        strategies.append(m.group(1).strip())

    # Strategy 2: find first { ... } pair (greedy — last closing brace)
    m = re.search(r"\{[\s\S]*\}", original)
    if m:
        strategies.append(m.group(0).strip())

    # Strategy 3: raw text as-is
    strategies.append(original)

    errors: List[str] = []
    for i, candidate in enumerate(strategies):
        try:
            return json.loads(candidate)
        except json.JSONDecodeError as e:
            errors.append(f"Strategy {i + 1}: {e}")

    raise ValueError(f"JSON extraction failed:\n" + "\n".join(errors))


# ── Retry predicates ──────────────────────────────────────────────────

def _is_retryable(exception: Exception) -> bool:
    """Return True if the exception warrants a retry (429, 5xx, network)."""
    if isinstance(exception, (requests.Timeout, requests.ConnectionError)):
        return True
    if isinstance(exception, requests.HTTPError):
        status = exception.response.status_code if hasattr(exception, "response") else 0
        return status in (429, 500, 502, 503, 504)
    # ValueError from JSON extraction — don't retry (it's a data problem)
    return False


# ── Core client ───────────────────────────────────────────────────────

class LLMClient:
    """Unified client for Gemini, SiliconFlow (Qwen), and DeepSeek."""

    def __init__(self) -> None:
        self.session = requests.Session()
        self.session.headers.setdefault("Content-Type", "application/json")
        self._call_count: Dict[str, int] = {}

    # ── Gemini (REST API) ─────────────────────────────────────────────

    @retry(
        retry=retry_if_exception(_is_retryable),
        wait=wait_exponential(multiplier=BACKOFF_MULTIPLIER, min=BASE_DELAY, max=MAX_DELAY),
        stop=stop_after_attempt(MAX_RETRIES),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
    def call_gemini(
        self, system_prompt: str, user_message: str, temperature: float = 0.7
    ) -> Dict[str, Any]:
        """Call Gemini 2.5 Flash via REST API, return parsed JSON."""
        url = f"{GEMINI_ENDPOINT}?key={GEMINI_API_KEY}"
        # Combine system prompt into the user message (Gemini REST pattern)
        full_prompt = f"[SYSTEM]\n{system_prompt}\n\n[USER]\n{user_message}"

        payload = {
            "contents": [{"parts": [{"text": full_prompt}]}],
            "generationConfig": {"temperature": temperature},
        }

        resp = self.session.post(url, json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()

        # Extract text from Gemini response
        try:
            text = data["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError):
            raise ValueError(f"Unexpected Gemini response structure: {json.dumps(data)[:500]}")

        if not text:
            raise ValueError("Gemini returned empty text")

        self._bump("gemini")
        return extract_json(text)

    # ── OpenAI-compatible (SiliconFlow / DeepSeek) ────────────────────

    @retry(
        retry=retry_if_exception(_is_retryable),
        wait=wait_exponential(multiplier=BACKOFF_MULTIPLIER, min=BASE_DELAY, max=MAX_DELAY),
        stop=stop_after_attempt(MAX_RETRIES),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
    def _call_openai_compatible(
        self,
        endpoint: str,
        api_key: str,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
    ) -> str:
        """Call an OpenAI-compatible chat completions endpoint, return raw text."""
        headers = {"Authorization": f"Bearer {api_key}"}
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
        }

        resp = self.session.post(endpoint, json=payload, headers=headers, timeout=120)
        resp.raise_for_status()
        data = resp.json()

        try:
            text = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError):
            raise ValueError(f"Unexpected API response structure: {json.dumps(data)[:500]}")

        if not text:
            raise ValueError("API returned empty content")

        return text.strip()

    def call_qwen(
        self,
        system_prompt: str,
        user_message: str,
        model: str = "Qwen/Qwen2.5-7B-Instruct",
        temperature: float = 0.7,
    ) -> Dict[str, Any]:
        """Call Qwen via SiliconFlow, return parsed JSON."""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]
        text = self._call_openai_compatible(
            SILICONFLOW_ENDPOINT, SILICONFLOW_API_KEY, model, messages, temperature
        )
        self._bump("qwen")
        return extract_json(text)

    def call_deepseek(
        self,
        system_prompt: str,
        user_message: str,
        model: str = "deepseek-chat",
        temperature: float = 0.3,
    ) -> str:
        """Call DeepSeek — returns raw text (Judge output is not JSON)."""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]
        text = self._call_openai_compatible(
            DEEPSEEK_ENDPOINT, DEEPSEEK_API_KEY, model, messages, temperature
        )
        self._bump("deepseek")
        # Strip any markdown fences the Judge might wrap around its output
        text = re.sub(r"^```(?:[\w]*)\s*", "", text.strip())
        text = re.sub(r"\s*```$", "", text)
        return text.strip()

    # ── Helpers ───────────────────────────────────────────────────────

    def _bump(self, provider: str) -> None:
        self._call_count[provider] = self._call_count.get(provider, 0) + 1

    @property
    def stats(self) -> Dict[str, int]:
        return dict(self._call_count)
