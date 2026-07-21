from __future__ import annotations

import threading

from deep_researcher.contracts import AgentSpec


class AgentSpecRegistry:
    """Thread-safe immutable-version registry used by every logical role."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._by_identity: dict[tuple[str, str], AgentSpec] = {}
        self._by_id: dict[str, AgentSpec] = {}

    def register(self, spec: AgentSpec) -> AgentSpec:
        key = (spec.name.casefold(), spec.version)
        with self._lock:
            existing = self._by_identity.get(key) or self._by_id.get(spec.agent_spec_id)
            if existing is not None and existing != spec:
                raise ValueError("AgentSpec identity/version is immutable")
            self._by_identity[key] = spec
            self._by_id[spec.agent_spec_id] = spec
        return spec

    def get(self, agent_spec_id: str) -> AgentSpec | None:
        with self._lock:
            return self._by_id.get(agent_spec_id)

    def resolve(self, name: str, version: str) -> AgentSpec | None:
        with self._lock:
            return self._by_identity.get((name.casefold(), version))

    def require(self, agent_spec_id: str) -> AgentSpec:
        spec = self.get(agent_spec_id)
        if spec is None:
            raise KeyError(agent_spec_id)
        if not spec.enabled:
            raise ValueError(f"AgentSpec is disabled: {agent_spec_id}")
        return spec

    def list(self, *, enabled_only: bool = True) -> tuple[AgentSpec, ...]:
        with self._lock:
            specs = tuple(self._by_id.values())
        return tuple(sorted(
            (spec for spec in specs if spec.enabled or not enabled_only),
            key=lambda item: (item.role.value, item.name.casefold(), item.version),
        ))
