from __future__ import annotations

import threading

from .models import ToolAdapter, ToolDefinition


class ToolRegistry:
    """Thread-safe registry with immutable versions and explicit activation."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._entries: dict[tuple[str, str], tuple[ToolDefinition, ToolAdapter]] = {}
        self._active: dict[str, str] = {}

    def register(self, definition: ToolDefinition, adapter: ToolAdapter, *, activate: bool = False) -> ToolDefinition:
        key = (definition.name, definition.version)
        with self._lock:
            existing = self._entries.get(key)
            if existing is not None:
                if existing[0] != definition or existing[1] is not adapter:
                    raise ValueError(f"tool version is immutable: {definition.identity}")
                return definition
            self._entries[key] = (definition, adapter)
            if definition.name not in self._active or activate:
                self._active[definition.name] = definition.version
        return definition

    def activate(self, name: str, version: str) -> ToolDefinition:
        with self._lock:
            entry = self._entries.get((name, version))
            if entry is None:
                raise KeyError(f"unknown tool version: {name}@{version}")
            self._active[name] = version
            return entry[0]

    def resolve(self, name: str, version: str | None = None) -> tuple[ToolDefinition, ToolAdapter]:
        with self._lock:
            selected = version or self._active.get(name)
            if selected is None:
                raise KeyError(f"unknown tool: {name}")
            entry = self._entries.get((name, selected))
            if entry is None:
                raise KeyError(f"unknown tool version: {name}@{selected}")
            return entry

    def definitions(self, *, active_only: bool = True) -> tuple[ToolDefinition, ...]:
        with self._lock:
            if active_only:
                values = [self._entries[(name, version)][0] for name, version in self._active.items()]
            else:
                values = [entry[0] for entry in self._entries.values()]
        return tuple(sorted(values, key=lambda item: (item.name.casefold(), item.version)))

    def active_version(self, name: str) -> str | None:
        with self._lock:
            return self._active.get(name)
