from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path

from .exporters import OTLPHTTPConfig, OTLPHTTPEventExporter
from .observer import PersistentEventObserver
from .recorder import EventRecorder
from .sqlite_store import SQLiteEventStore


@dataclass
class EventRuntime:
    store: SQLiteEventStore
    recorder: EventRecorder
    observer: PersistentEventObserver

    def close(self) -> None:
        self.store.close()

    def __enter__(self) -> "EventRuntime":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()


def build_event_runtime(
    database_path: str | Path,
    *,
    otlp_endpoint: str | None = None,
    otlp_headers: dict[str, str] | None = None,
) -> EventRuntime:
    store = SQLiteEventStore(database_path)
    endpoint = otlp_endpoint if otlp_endpoint is not None else os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "").strip()
    exporters = ()
    if endpoint:
        exporters = (
            OTLPHTTPEventExporter(
                OTLPHTTPConfig(
                    endpoint=endpoint,
                    service_name=os.getenv("OTEL_SERVICE_NAME", "deep-researcher"),
                    headers=otlp_headers,
                )
            ),
        )
    recorder = EventRecorder(store, exporters)
    observer = PersistentEventObserver(recorder)
    return EventRuntime(store=store, recorder=recorder, observer=observer)
