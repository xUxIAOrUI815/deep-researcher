from __future__ import annotations

from dataclasses import dataclass
import threading

from deep_researcher.contracts import RunEvent

from .exporters import EventExporter
from .store import AppendResult, EventStore, PendingExport, SequenceConflict


@dataclass(frozen=True)
class RecordOutcome:
    append: AppendResult
    exported_to: tuple[str, ...]
    failed_exports: dict[str, str]


class EventRecorder:
    """Commits locally before attempting any external telemetry export."""

    def __init__(self, store: EventStore, exporters: tuple[EventExporter, ...] = ()) -> None:
        names = [exporter.name for exporter in exporters]
        if len(names) != len(set(names)):
            raise ValueError("event exporter names must be unique")
        self.store = store
        self.exporters = {exporter.name: exporter for exporter in exporters}
        self._lock = threading.RLock()

    def record(self, event: RunEvent) -> RecordOutcome:
        with self._lock:
            appended = self.store.append(event, export_targets=tuple(self.exporters))
        exported: list[str] = []
        failed: dict[str, str] = {}
        for name, exporter in self.exporters.items():
            try:
                exporter.export(appended.event)
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                self.store.mark_export_failed(appended.event.event_id, name, error)
                failed[name] = error
            else:
                self.store.mark_exported(appended.event.event_id, name)
                exported.append(name)
        return RecordOutcome(append=appended, exported_to=tuple(exported), failed_exports=failed)

    def retry_pending(self, *, exporter_name: str | None = None, limit: int = 100) -> tuple[RecordOutcome, ...]:
        outcomes: list[RecordOutcome] = []
        pending = self.store.pending_exports(exporter_name=exporter_name, limit=limit)
        for item in pending:
            exporter = self.exporters.get(item.exporter_name)
            if exporter is None:
                continue
            try:
                exporter.export(item.event)
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                self.store.mark_export_failed(item.event.event_id, item.exporter_name, error)
                outcomes.append(
                    RecordOutcome(
                        append=AppendResult(event=item.event, inserted=False),
                        exported_to=(),
                        failed_exports={item.exporter_name: error},
                    )
                )
            else:
                self.store.mark_exported(item.event.event_id, item.exporter_name)
                outcomes.append(
                    RecordOutcome(
                        append=AppendResult(event=item.event, inserted=False),
                        exported_to=(item.exporter_name,),
                        failed_exports={},
                    )
                )
        return tuple(outcomes)
