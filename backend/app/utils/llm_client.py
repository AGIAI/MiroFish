"""
LLM Client Wrapper
Unified OpenAI-format API calls with timeout, retry, and provider fallback.
"""

import json
import logging
import re
import time
from typing import Optional, Dict, Any, List
from openai import OpenAI

from ..config import Config

# Default timeout for LLM API calls (connect, read) in seconds
_DEFAULT_TIMEOUT = 120.0
# Retry configuration
_MAX_RETRIES = 3
_RETRY_BASE_DELAY = 2.0


class LLMClient:
    """LLM Client with timeout and retry support."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout: float = _DEFAULT_TIMEOUT,
    ):
        self.api_key = api_key or Config.LLM_API_KEY
        self.base_url = base_url or Config.LLM_BASE_URL
        self.model = model or Config.LLM_MODEL_NAME
        self.timeout = timeout

        if not self.api_key:
            raise ValueError("LLM_API_KEY is not configured")

        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout,
            max_retries=0,  # We handle retries ourselves for better control
        )

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 4096,
        response_format: Optional[Dict] = None
    ) -> str:
        """
        Send a chat request with automatic retry on transient failures.

        Args:
            messages: List of messages
            temperature: Temperature parameter
            max_tokens: Maximum number of tokens
            response_format: Response format (e.g., JSON mode)

        Returns:
            Model response text
        """
        logger = logging.getLogger('mirofish.llm')
        kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        if response_format:
            kwargs["response_format"] = response_format

        last_exception = None
        for attempt in range(_MAX_RETRIES):
            try:
                response = self.client.chat.completions.create(**kwargs)
                content = response.choices[0].message.content
                # Some models may include <think> reasoning content, which needs to be removed
                content = re.sub(r'<think>[\s\S]*?</think>', '', content).strip()
                return content
            except Exception as e:
                last_exception = e
                error_str = str(e).lower()
                # Only retry on transient errors (rate limit, server error, timeout, connection)
                is_transient = any(kw in error_str for kw in [
                    'rate limit', '429', '500', '502', '503', '529',
                    'timeout', 'connection', 'overloaded',
                ])
                if is_transient and attempt < _MAX_RETRIES - 1:
                    delay = _RETRY_BASE_DELAY * (2 ** attempt)
                    logger.warning(
                        f"LLM call attempt {attempt + 1} failed ({type(e).__name__}), "
                        f"retrying in {delay:.0f}s..."
                    )
                    time.sleep(delay)
                else:
                    raise

        raise last_exception  # Should not reach here, but safety net

    def _clean_json_response(self, response: str) -> str:
        """Strip markdown fences and whitespace from an LLM response."""
        cleaned = response.strip()
        cleaned = re.sub(r'^```(?:json)?\s*\n?', '', cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r'\n?```\s*$', '', cleaned)
        return cleaned.strip()

    def chat_json(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.3,
        max_tokens: int = 4096
    ) -> Dict[str, Any]:
        """
        Send a chat request and return JSON.
        Falls back to prompting without response_format if the provider
        rejects the parameter (e.g. Groq, DeepSeek, local models).

        Args:
            messages: List of messages
            temperature: Temperature parameter
            max_tokens: Maximum number of tokens

        Returns:
            Parsed JSON object
        """
        logger = logging.getLogger('mirofish.llm')

        # First attempt: use response_format for providers that support it
        try:
            response = self.chat(
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format={"type": "json_object"}
            )
        except Exception as e:
            error_str = str(e).lower()
            if any(kw in error_str for kw in ['response_format', 'json_object', 'not supported', 'invalid parameter']):
                logger.warning("Provider rejected response_format, retrying without it")
                response = self.chat(
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
            else:
                raise

        cleaned_response = self._clean_json_response(response)

        try:
            return json.loads(cleaned_response)
        except json.JSONDecodeError:
            raise ValueError(f"Invalid JSON format returned by LLM: {cleaned_response}")
