from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from deep_researcher.knowledge import (
    KnowledgeRuntime,
    VectorRetrievalAdapter,
    build_knowledge_runtime,
    restore_knowledge_runtime,
)

from .engine import EvidenceVerificationEngine
from .events import EvidenceEventSink
from .graph import CandidateKnowledgeView, VerifiedKnowledgeView
from .models import SemanticVerificationAdapter, VerificationPolicy


@dataclass
class EvidenceRuntime:
    knowledge: KnowledgeRuntime
    engine: EvidenceVerificationEngine
    candidates: CandidateKnowledgeView
    verified: VerifiedKnowledgeView

    def integrity_check(self) -> None:
        self.knowledge.integrity_check()

    def close(self) -> None:
        self.knowledge.close()

    def __enter__(self) -> "EvidenceRuntime":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()


def build_evidence_runtime(
    root: str | Path,
    *,
    semantic_adapter: SemanticVerificationAdapter,
    event_sink: EvidenceEventSink,
    policy: VerificationPolicy,
    vector_adapter: VectorRetrievalAdapter | None = None,
    verifier_id: str = "agent_spec_evidence_verifier_1_0_0",
) -> EvidenceRuntime:
    knowledge = build_knowledge_runtime(root, vector_adapter=vector_adapter)
    runtime = EvidenceRuntime(
        knowledge=knowledge,
        engine=EvidenceVerificationEngine(
            repository=knowledge.repository,
            artifact_store=knowledge.artifacts,
            semantic_adapter=semantic_adapter,
            event_sink=event_sink,
            policy=policy,
            verifier_id=verifier_id,
        ),
        candidates=CandidateKnowledgeView(knowledge.repository),
        verified=VerifiedKnowledgeView(knowledge.repository),
    )
    runtime.integrity_check()
    return runtime


def restore_evidence_runtime(
    backup: str | Path,
    destination: str | Path,
    *,
    semantic_adapter: SemanticVerificationAdapter,
    event_sink: EvidenceEventSink,
    policy: VerificationPolicy,
    verifier_id: str = "agent_spec_evidence_verifier_1_0_0",
) -> EvidenceRuntime:
    knowledge = restore_knowledge_runtime(backup, destination)
    runtime = EvidenceRuntime(
        knowledge=knowledge,
        engine=EvidenceVerificationEngine(
            repository=knowledge.repository,
            artifact_store=knowledge.artifacts,
            semantic_adapter=semantic_adapter,
            event_sink=event_sink,
            policy=policy,
            verifier_id=verifier_id,
        ),
        candidates=CandidateKnowledgeView(knowledge.repository),
        verified=VerifiedKnowledgeView(knowledge.repository),
    )
    runtime.integrity_check()
    return runtime
