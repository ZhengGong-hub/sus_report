import json
import logging
import os
from enum import Enum
from typing import Any, Dict, Optional

from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()


class Provider(str, Enum):
    OPENAI = "openai"
    GEMINI = "gemini"


DEFAULT_MODELS: Dict[Provider, str] = {
    Provider.OPENAI: "gpt-5-mini",
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

    def call_structured(
        self,
        prompt: str,
        text_chunk: str,
        json_schema: Dict[str, Any],
        schema_name: str = "schema",
    ) -> Dict[str, Any]:
        """
        Chat completion using OpenAI response_format=json_schema.

        Returns a parsed dict matching the provided JSON schema.
        """
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": text_chunk},
        ]

        self._logger.debug(
            "LLM structured call: model=%s, chunk_len=%d, schema=%s",
            self.model,
            len(text_chunk),
            schema_name,
        )

        response = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": schema_name,
                    "schema": json_schema,
                },
            },
        )

        if not response.choices:
            self._logger.error("Model returned no choices (structured call).")
            raise ValueError("Model returned no choices.")

        content = response.choices[0].message.content
        if not content:
            self._logger.error("Model returned empty response (structured call).")
            raise ValueError("Model returned empty response.")

        # The OpenAI API returns JSON text; parse it.
        if isinstance(content, list):
            raw = "".join(str(part) for part in content)
        else:
            raw = str(content)

        try:
            data: Dict[str, Any] = json.loads(raw)
        except json.JSONDecodeError as exc:
            self._logger.error("Failed to parse structured JSON response: %s", raw)
            raise ValueError("Failed to parse structured JSON response.") from exc

        return data