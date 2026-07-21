from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Mapping, Protocol

import httpx

from deep_researcher.contracts import RunEvent


class EventExporter(Protocol):
    @property
    def name(self) -> str:
        ...

    def export(self, event: RunEvent) -> None:
        ...


class EventExportError(RuntimeError):
    """External telemetry export failed after local event persistence."""


@dataclass(frozen=True)
class OTLPHTTPConfig:
    endpoint: str
    service_name: str = "deep-researcher"
    headers: Mapping[str, str] | None = None
    timeout_seconds: float = 10.0

    def __post_init__(self) -> None:
        if not self.endpoint.startswith(("http://", "https://")):
            raise ValueError("OTLP endpoint must use HTTP or HTTPS")
        if self.timeout_seconds <= 0:
            raise ValueError("OTLP timeout must be positive")


class OTLPHTTPEventExporter:
    """Exports RunEvent records through the standard OTLP/HTTP JSON logs API."""

    name = "otlp_http"

    def __init__(self, config: OTLPHTTPConfig, *, client: httpx.Client | None = None) -> None:
        self.config = config
        self._client = client

    @staticmethod
    def _hex_id(value: str, byte_length: int) -> str:
        return hashlib.sha256(value.encode("utf-8")).hexdigest()[: byte_length * 2]

    @staticmethod
    def _attributes(event: RunEvent) -> list[dict[str, Any]]:
        values: dict[str, Any] = {
            "event.id": event.event_id,
            "event.type": event.event_type.value,
            "event.sequence": event.sequence_no,
            "run.id": event.run_id,
            "thread.id": event.thread_id,
            "actor.id": event.actor_id,
            "producer.id": event.producer_id,
            "span.kind": event.span_kind.value,
            "run.status": event.status.value,
            "latency.ms": event.latency_ms,
            "usage.input_tokens": event.usage.input_tokens,
            "usage.output_tokens": event.usage.output_tokens,
            "usage.cost_usd": event.usage.cost_usd,
            "usage.tool_calls": event.usage.tool_calls,
            "usage.search_calls": event.usage.search_calls,
            "usage.retries": event.usage.retries,
            "usage.errors": event.usage.errors,
        }
        if event.task_id:
            values["task.id"] = event.task_id
        if event.causation_event_id:
            values["event.causation_id"] = event.causation_event_id
        attributes: list[dict[str, Any]] = []
        for key, value in values.items():
            if isinstance(value, bool):
                encoded = {"boolValue": value}
            elif isinstance(value, int):
                encoded = {"intValue": str(value)}
            elif isinstance(value, float):
                encoded = {"doubleValue": value}
            else:
                encoded = {"stringValue": str(value)}
            attributes.append({"key": key, "value": encoded})
        return attributes

    def _payload(self, event: RunEvent) -> dict[str, Any]:
        timestamp_ns = int(event.occurred_at.timestamp() * 1_000_000_000)
        observed_ns = int(event.recorded_at.timestamp() * 1_000_000_000)
        return {
            "resourceLogs": [
                {
                    "resource": {
                        "attributes": [
                            {"key": "service.name", "value": {"stringValue": self.config.service_name}},
                            {"key": "service.namespace", "value": {"stringValue": "background001"}},
                        ]
                    },
                    "scopeLogs": [
                        {
                            "scope": {"name": "deep_researcher.events", "version": event.schema_version},
                            "logRecords": [
                                {
                                    "timeUnixNano": str(timestamp_ns),
                                    "observedTimeUnixNano": str(observed_ns),
                                    "severityText": event.level.value.upper(),
                                    "body": {"stringValue": json.dumps(event.payload, ensure_ascii=False, sort_keys=True)},
                                    "attributes": self._attributes(event),
                                    "traceId": self._hex_id(event.trace_id, 16),
                                    "spanId": self._hex_id(event.span_id, 8),
                                }
                            ],
                        }
                    ],
                }
            ]
        }

    def export(self, event: RunEvent) -> None:
        endpoint = self.config.endpoint.rstrip("/")
        if not endpoint.endswith("/v1/logs"):
            endpoint += "/v1/logs"
        headers = {"Content-Type": "application/json", **dict(self.config.headers or {})}
        try:
            if self._client is not None:
                response = self._client.post(endpoint, headers=headers, json=self._payload(event))
            else:
                response = httpx.post(
                    endpoint,
                    headers=headers,
                    json=self._payload(event),
                    timeout=self.config.timeout_seconds,
                )
            response.raise_for_status()
        except Exception as exc:
            raise EventExportError(f"OTLP export failed: {type(exc).__name__}: {exc}") from exc
