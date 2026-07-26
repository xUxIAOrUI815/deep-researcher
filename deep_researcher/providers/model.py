from __future__ import annotations

from dataclasses import dataclass
import json
import os
import time
from typing import Any

import httpx

from deep_researcher.contracts import BudgetUsage
from deep_researcher.kernel import (
    ModelInvocationError,
    ModelRequest,
    ModelResponse,
)


@dataclass(frozen=True)
class OpenAICompatibleModelConfig:
    """Explicit configuration for an OpenAI-compatible chat endpoint."""

    api_key: str
    base_url: str = "https://api.deepseek.com"
    model: str = "deepseek-chat"
    timeout_seconds: float = 90.0
    temperature: float = 0.1
    input_cost_per_million: float = 0.0
    output_cost_per_million: float = 0.0
    require_json_object: bool = True

    def __post_init__(self) -> None:
        if not self.api_key.strip():
            raise ValueError("A non-empty model API key is required")
        if not self.base_url.startswith(("https://", "http://")):
            raise ValueError("Model base_url must be HTTP(S)")
        if self.timeout_seconds <= 0:
            raise ValueError("Model timeout must be positive")
        if self.input_cost_per_million < 0 or self.output_cost_per_million < 0:
            raise ValueError("Model prices cannot be negative")

    @classmethod
    def from_environment(cls) -> "OpenAICompatibleModelConfig":
        raw_key = os.getenv("DEEPSEEK_API_KEY", "")
        api_key = raw_key.strip().strip('"').strip("'")
        if api_key.startswith("DEEPSEEK_API_KEY="):
            api_key = api_key.split("=", 1)[1].strip().strip('"').strip("'")
        return cls(
            api_key=api_key,
            base_url=os.getenv(
                "DEEPSEEK_API_BASE",
                "https://api.deepseek.com",
            ).rstrip("/"),
            model=(
                os.getenv("DEEPSEEK_MODEL_NAME")
                or os.getenv("DEEPSEEK_MODEL")
                or "deepseek-chat"
            ),
            timeout_seconds=float(os.getenv("MODEL_TIMEOUT_SECONDS", "90")),
            temperature=float(os.getenv("MODEL_TEMPERATURE", "0.1")),
            input_cost_per_million=float(
                os.getenv("MODEL_INPUT_COST_PER_MILLION", "0")
            ),
            output_cost_per_million=float(
                os.getenv("MODEL_OUTPUT_COST_PER_MILLION", "0")
            ),
            require_json_object=os.getenv(
                "MODEL_REQUIRE_JSON_OBJECT",
                "1",
            ).strip().casefold()
            not in {"0", "false", "no"},
        )


class OpenAICompatibleModelAdapter:
    """Bounded, structured ModelAdapter for production Background001 roles."""

    def __init__(
        self,
        config: OpenAICompatibleModelConfig,
        *,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self.config = config
        self._http_client = http_client

    async def complete(self, request: ModelRequest) -> ModelResponse:
        return await self._invoke(request, repair=None)

    async def repair(
        self,
        request: ModelRequest,
        invalid_response: ModelResponse,
        errors: tuple[str, ...],
    ) -> ModelResponse:
        return await self._invoke(
            request,
            repair={
                "validation_errors": list(errors),
                "invalid_response": (
                    invalid_response.structured
                    if invalid_response.structured is not None
                    else invalid_response.content
                ),
                "instruction": (
                    "Return a corrected JSON object that conforms exactly to "
                    "the supplied output schema. Do not add prose."
                ),
            },
        )

    async def _invoke(
        self,
        request: ModelRequest,
        *,
        repair: dict[str, Any] | None,
    ) -> ModelResponse:
        messages = [
            {"role": "system", "content": request.system},
            *[
                {
                    "role": str(item.get("role", "user")),
                    "content": self._content(item.get("content")),
                }
                for item in request.messages
            ],
            {
                "role": "user",
                "content": self._content(
                    {
                        "output_schema": request.command_schema,
                        "response_rule": (
                            "Return exactly one JSON object matching this schema."
                        ),
                    }
                ),
            },
        ]
        if repair is not None:
            messages.append(
                {"role": "user", "content": self._content(repair)}
            )
        body: dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature,
            "max_tokens": request.max_output_tokens,
        }
        if self.config.require_json_object:
            body["response_format"] = {"type": "json_object"}
        started = time.monotonic()
        try:
            if self._http_client is None:
                async with httpx.AsyncClient(
                    timeout=self.config.timeout_seconds
                ) as client:
                    response = await client.post(
                        f"{self.config.base_url}/chat/completions",
                        headers={
                            "Authorization": f"Bearer {self.config.api_key}",
                            "Content-Type": "application/json",
                        },
                        json=body,
                    )
            else:
                response = await self._http_client.post(
                    f"{self.config.base_url}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.config.api_key}",
                        "Content-Type": "application/json",
                    },
                    json=body,
                    timeout=self.config.timeout_seconds,
                )
            response.raise_for_status()
            payload = response.json()
            choice = payload["choices"][0]
            content = str(choice["message"].get("content") or "")
            structured = self._structured(content)
            usage = payload.get("usage") or {}
            input_tokens = int(
                usage.get("prompt_tokens", usage.get("input_tokens", 0)) or 0
            )
            output_tokens = int(
                usage.get(
                    "completion_tokens",
                    usage.get("output_tokens", 0),
                )
                or 0
            )
            cost = (
                input_tokens * self.config.input_cost_per_million
                + output_tokens * self.config.output_cost_per_million
            ) / 1_000_000
            return ModelResponse(
                content=content,
                structured=structured,
                usage=BudgetUsage(
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    cost_usd=cost,
                    model_calls=1,
                ),
                latency_ms=(time.monotonic() - started) * 1000,
                finish_reason=str(choice.get("finish_reason") or "stop"),
                response_id=str(payload.get("id") or ""),
            )
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code
            retryable = status in {408, 409, 425, 429, 500, 502, 503, 504}
            raise ModelInvocationError(
                f"Model provider returned HTTP {status}",
                retryable=retryable,
            ) from exc
        except (httpx.TimeoutException, httpx.NetworkError) as exc:
            raise ModelInvocationError(
                f"Model provider transport failed: {type(exc).__name__}",
                retryable=True,
            ) from exc
        except (KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ModelInvocationError(
                f"Model provider returned an invalid structured response: {exc}",
                retryable=False,
            ) from exc

    @staticmethod
    def _content(value: Any) -> str:
        if isinstance(value, str):
            return value
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )

    @staticmethod
    def _structured(content: str) -> Any:
        candidate = content.strip()
        if candidate.startswith("```"):
            lines = candidate.splitlines()
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            candidate = "\n".join(lines)
        return json.loads(candidate)


class EnvironmentModelAdapter:
    """Lazily resolves live provider configuration without hiding omissions."""

    def __init__(self) -> None:
        self._adapter: OpenAICompatibleModelAdapter | None = None

    def _resolved(self) -> OpenAICompatibleModelAdapter:
        if self._adapter is not None:
            return self._adapter
        try:
            config = OpenAICompatibleModelConfig.from_environment()
        except ValueError as exc:
            raise ModelInvocationError(
                "Live model configuration is unavailable: DEEPSEEK_API_KEY "
                "must be configured before starting a research run.",
                retryable=False,
            ) from exc
        self._adapter = OpenAICompatibleModelAdapter(config)
        return self._adapter

    async def complete(self, request: ModelRequest) -> ModelResponse:
        return await self._resolved().complete(request)

    async def repair(
        self,
        request: ModelRequest,
        invalid_response: ModelResponse,
        errors: tuple[str, ...],
    ) -> ModelResponse:
        return await self._resolved().repair(request, invalid_response, errors)
