import logging
import os
from enum import Enum
from typing import Dict, Optional
from dotenv import load_dotenv

from openai import OpenAI


load_dotenv()

class Provider(str, Enum):
    OPENAI = "openai"
    GEMINI = "gemini"


DEFAULT_MODELS: Dict[Provider, str] = {
    Provider.OPENAI: "gpt-4o-mini",
    Provider.GEMINI: "gemini-2.0-flash",
}

GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai"


class LLMWrapper:
    def __init__(
        self,
        provider: Provider,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: int = 60,
        logger: Optional[logging.Logger] = None,
    ):
        self.provider = provider
        self.model = model or DEFAULT_MODELS[provider]
        self._logger = logger or logging.getLogger(__name__)

        api_key = api_key or self._default_api_key(provider)
        if not api_key:
            raise ValueError(
                f"Missing API key for {provider}. Set {self._env_key(provider)} or pass api_key=."
            )

        if base_url is None and provider == Provider.GEMINI:
            base_url = GEMINI_BASE_URL

        self._client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)

    @staticmethod
    def _env_key(provider: Provider) -> str:
        return "OPENAI_API_KEY" if provider == Provider.OPENAI else "GEMINI_API_KEY"

    def _default_api_key(self, provider: Provider) -> Optional[str]:
        key = os.environ.get(self._env_key(provider))
        if key:
            return key
        if provider == Provider.GEMINI:
            return os.environ.get("GOOGLE_API_KEY")
        return None

    def call(
        self,
        prompt: str,
        text_chunk: str,
        temperature: float = 0.0,
        max_tokens: int = 800,
    ) -> str:
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": text_chunk},
        ]

        self._logger.debug("LLM call: model=%s, chunk_len=%d", self.model, len(text_chunk))

        response = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        if not response.choices:
            self._logger.error("Model returned no choices.")
            raise ValueError("Model returned no choices.")

        content = response.choices[0].message.content
        if not content:
            self._logger.error("Model returned empty response.")
            raise ValueError("Model returned empty response.")

        return content.strip()