from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
import re
from typing import Any, Mapping
from urllib.parse import urlsplit

from deep_researcher.artifacts.store import ArtifactStore
from deep_researcher.contracts import (
    ArtifactKind,
    AtomicFact,
    Citation,
    CitationStatus,
    Claim,
    ClaimStatus,
    Conflict,
    EntityProvenance,
    Evidence,
    EvidenceQuote,
    EvidenceRelation,
    EvidenceStatus,
    FactStatus,
    Passage,
    PassageStatus,
    Report,
    ReportStatus,
    Section,
    SectionStatus,
    SnapshotStatus,
    Source,
    SourceLevel,
    SourceSnapshot,
    SourceStatus,
    SourceType,
    utc_now,
)

from .normalization import canonicalize_url, normalize_text
from .repository import KnowledgeRepository
from .sqlite_storage import entity_id
from .storage import KnowledgeEntity


_VALID_ID = re.compile(r"^[a-z][a-z0-9_]*_[A-Za-z0-9][A-Za-z0-9_.:-]*$")


def stable_id(prefix: str, value: str) -> str:
    normalized = str(value or "").strip()
    if _VALID_ID.fullmatch(normalized) and normalized.startswith(f"{prefix}_"):
        return normalized
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:24]
    return f"{prefix}_{digest}"


def _entity_id(prefix: str, run_id: str, value: str) -> str:
    return stable_id(prefix, f"{run_id}:{value}")


def _stable_payload_key(prefix: str, value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, default=str, separators=(",", ":")
    )
    return f"{prefix}:{hashlib.sha256(encoded.encode('utf-8')).hexdigest()}"


def _source_classification(raw: Mapping[str, Any]) -> tuple[SourceType, SourceLevel]:
    type_value = str(raw.get("source_type") or raw.get("type") or "").strip().casefold()
    level_value = (
        str(raw.get("source_level") or raw.get("level") or "").strip().casefold()
    )
    try:
        source_type = SourceType(type_value)
    except ValueError:
        source_type = SourceType.OTHER
    try:
        source_level = SourceLevel(level_value)
    except ValueError:
        source_level = {
            SourceType.PRIMARY: SourceLevel.PRIMARY,
            SourceType.SECONDARY: SourceLevel.SECONDARY,
            SourceType.TERTIARY: SourceLevel.TERTIARY,
            SourceType.OFFICIAL_DOCUMENTATION: SourceLevel.PRIMARY,
            SourceType.DATASET: SourceLevel.PRIMARY,
            SourceType.ACADEMIC: SourceLevel.SECONDARY,
        }.get(source_type, SourceLevel.UNKNOWN)
    return source_type, source_level


_SCHOLARLY_PRIMARY_HOSTS = (
    "aclanthology.org",
    "arxiv.org",
    "dl.acm.org",
    "doi.org",
    "ieeexplore.ieee.org",
    "link.springer.com",
    "nature.com",
    "onlinelibrary.wiley.com",
    "openreview.net",
    "proceedings.neurips.cc",
    "sciencedirect.com",
)
_COMMUNITY_HOSTS = (
    "facebook.com",
    "linkedin.com",
    "medium.com",
    "reddit.com",
    "youtube.com",
)


def _host_matches(host: str, candidates: tuple[str, ...]) -> bool:
    return any(host == item or host.endswith(f".{item}") for item in candidates)


def _deterministic_source_profile(
    url: str,
) -> tuple[SourceType, SourceLevel, float] | None:
    """Classify authority from stable source identity, never search rank.

    Provider ``score`` fields are relevance scores. Treating them as source
    authority lets a highly ranked blog outrank a paper original, and a later
    read (which has no relevance score) can accidentally downgrade an already
    discovered primary source to zero. The profile below is deliberately
    conservative and records only source classes that can be inferred from the
    canonical host itself.
    """

    host = (urlsplit(url).hostname or "").casefold().removeprefix("www.")
    if not host:
        return None
    if _host_matches(host, _SCHOLARLY_PRIMARY_HOSTS):
        return SourceType.PRIMARY, SourceLevel.PRIMARY, 0.9
    if (
        host.endswith(".gov")
        or ".gov." in host
        or host.endswith(".mil")
    ):
        return SourceType.OFFICIAL_DOCUMENTATION, SourceLevel.PRIMARY, 0.9
    if (
        host.endswith(".edu")
        or ".edu." in host
        or host.endswith(".ac.uk")
    ):
        return SourceType.PRIMARY, SourceLevel.PRIMARY, 0.82
    if _host_matches(host, _COMMUNITY_HOSTS):
        return SourceType.COMMUNITY, SourceLevel.TERTIARY, 0.2
    if host.startswith("docs.") or host.startswith("developer."):
        return SourceType.OFFICIAL_DOCUMENTATION, SourceLevel.PRIMARY, 0.72
    return None


def _default_authority(
    source_type: SourceType,
    source_level: SourceLevel,
) -> float:
    if source_type in {
        SourceType.PRIMARY,
        SourceType.OFFICIAL_DOCUMENTATION,
        SourceType.DATASET,
    }:
        return 0.8
    if source_type == SourceType.ACADEMIC:
        return 0.72
    if source_type in {SourceType.SECONDARY, SourceType.NEWS}:
        return 0.5
    if source_type in {SourceType.TERTIARY, SourceType.COMMUNITY}:
        return 0.25
    return {
        SourceLevel.PRIMARY: 0.8,
        SourceLevel.SECONDARY: 0.5,
        SourceLevel.TERTIARY: 0.25,
        SourceLevel.UNKNOWN: 0.0,
    }[source_level]


def _source_profile(
    raw: Mapping[str, Any],
    url: str,
    existing: Source | None,
) -> tuple[SourceType, SourceLevel, float]:
    source_type, source_level = _source_classification(raw)
    deterministic = _deterministic_source_profile(url)
    if deterministic is not None:
        source_type, source_level, authority = deterministic
    else:
        explicit_authority = raw.get("authority_score")
        try:
            authority = (
                max(0.0, min(1.0, float(explicit_authority)))
                if explicit_authority is not None
                else _default_authority(source_type, source_level)
            )
        except (TypeError, ValueError):
            authority = _default_authority(source_type, source_level)

    if existing is not None:
        authority = max(authority, existing.authority_score)
        level_rank = {
            SourceLevel.UNKNOWN: 0,
            SourceLevel.TERTIARY: 1,
            SourceLevel.SECONDARY: 2,
            SourceLevel.PRIMARY: 3,
        }
        if level_rank[existing.source_level] > level_rank[source_level]:
            source_type = existing.source_type
            source_level = existing.source_level
    return source_type, source_level, authority


def _aware_datetime(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    return (
        parsed if parsed.tzinfo is not None and parsed.utcoffset() is not None else None
    )


@dataclass(frozen=True)
class IngestionResult:
    artifact_ids: tuple[str, ...]
    entity_ids: tuple[str, ...]
    issues: tuple[str, ...]


class KnowledgeIngestionService:
    """Normalizes draft outputs into artifact-backed evidence-domain entities."""

    def __init__(
        self, artifact_store: ArtifactStore, repository: KnowledgeRepository
    ) -> None:
        self.artifact_store = artifact_store
        self.repository = repository

    def _stable_entity(self, candidate: KnowledgeEntity) -> KnowledgeEntity:
        """Preserve persisted timestamps so replayed ingestion is revision-idempotent."""
        existing = self.repository.get(entity_id(candidate))
        if existing is None or type(existing) is not type(candidate):
            return candidate
        if self._is_exact_candidate_replay(existing, candidate):
            return existing
        ignored = {
            "created_at",
            "updated_at",
            "discovered_at",
            "fetched_at",
            "extracted_at",
            "verified_at",
        }
        left = {
            key: value
            for key, value in existing.model_dump(mode="python").items()
            if key not in ignored
        }
        right = {
            key: value
            for key, value in candidate.model_dump(mode="python").items()
            if key not in ignored
        }
        if left == right:
            return existing
        values = {}
        for name in (
            "created_at",
            "discovered_at",
            "fetched_at",
            "extracted_at",
            "verified_at",
        ):
            if hasattr(existing, name):
                values[name] = getattr(existing, name)
        return candidate.model_copy(update=values)

    @staticmethod
    def _is_exact_candidate_replay(
        existing: KnowledgeEntity,
        candidate: KnowledgeEntity,
    ) -> bool:
        ignored_by_type: dict[type[KnowledgeEntity], set[str]] = {
            Evidence: {"status", "verification_id", "verified_at"},
            AtomicFact: {"status", "verification_id", "verified_at"},
            Claim: {
                "status",
                "high_impact",
                "support_score",
                "verification_id",
                "verified_at",
            },
            Conflict: {
                "status",
                "severity",
                "high_impact",
                "resolution",
                "resolution_kind",
                "resolution_evidence_ids",
            },
            Section: {
                "status",
                "coverage_score",
                "citation_score",
                "coverage_status",
                "unsupported_claim_ids",
                "conflicted_claim_ids",
            },
        }
        ignored = ignored_by_type.get(type(candidate))
        if ignored is None:
            return False
        existing_provenance = getattr(existing, "provenance")
        candidate_provenance = getattr(candidate, "provenance")
        if not set(candidate_provenance.source_artifact_ids).issubset(
            existing_provenance.source_artifact_ids
        ):
            return False
        if (
            existing_provenance.run_id != candidate_provenance.run_id
            or existing_provenance.task_id != candidate_provenance.task_id
        ):
            return False
        ignored = {
            *ignored,
            "created_at",
            "updated_at",
            "discovered_at",
            "fetched_at",
            "extracted_at",
            "provenance",
        }
        left = {
            key: value
            for key, value in existing.model_dump(mode="python").items()
            if key not in ignored
        }
        right = {
            key: value
            for key, value in candidate.model_dump(mode="python").items()
            if key not in ignored
        }
        return left == right

    @staticmethod
    def _provenance(
        *,
        producer_id: str,
        run_id: str,
        task_id: str | None,
        source_artifact_ids: tuple[str, ...] = (),
    ) -> EntityProvenance:
        return EntityProvenance(
            producer_id=producer_id,
            run_id=run_id,
            task_id=task_id,
            source_artifact_ids=source_artifact_ids,
        )

    def ingest_research_observation(
        self,
        outputs: Mapping[str, Any],
        *,
        run_id: str,
        task_id: str | None,
    ) -> IngestionResult:
        task_id = stable_id("task", task_id) if task_id else None
        issues: list[str] = []
        artifact_ids: list[str] = []
        entities: list[KnowledgeEntity] = []
        output_artifact = self.artifact_store.put_json(
            dict(outputs),
            kind=ArtifactKind.SEARCH_RESPONSE,
            producer_id="agent_researcher",
            run_id=run_id,
            task_id=task_id,
            content_schema="ResearchObservation@1",
            idempotency_key=_stable_payload_key("research-observation", outputs),
        )
        artifact_ids.append(output_artifact.artifact_id)
        sources = list(outputs.get("sources", []) or [])
        passages = list(outputs.get("passages", []) or [])
        scraped = list(outputs.get("scraped_data_cache", []) or [])
        scraped_by_url = {
            canonicalize_url(str(item.get("url", ""))): item
            for item in scraped
            if isinstance(item, Mapping) and item.get("url")
        }

        for raw_source in sources:
            if not isinstance(raw_source, Mapping) or not raw_source.get("url"):
                issues.append("researcher source without URL was skipped")
                continue
            try:
                url = canonicalize_url(str(raw_source["url"]))
            except ValueError as exc:
                issues.append(
                    f"invalid source URL skipped: {raw_source.get('url')}: {exc}"
                )
                continue
            source_id = _entity_id("source", run_id, url)
            existing_source = self.repository.sources.get(source_id)
            discovered_at = (
                existing_source.discovered_at if existing_source else utc_now()
            )
            source_type, source_level, authority_score = _source_profile(
                raw_source,
                url,
                existing_source,
            )
            source_artifact_ids = (output_artifact.artifact_id,)
            if existing_source is not None:
                source_artifact_ids = tuple(
                    dict.fromkeys(
                        (
                            *existing_source.provenance.source_artifact_ids,
                            output_artifact.artifact_id,
                        )
                    )
                )
            source = Source(
                source_id=source_id,
                canonical_url=url,
                source_type=source_type,
                source_level=source_level,
                status=SourceStatus.ACCESSIBLE,
                title=(
                    normalize_text(str(raw_source.get("title", "")))
                    or (existing_source.title if existing_source else None)
                ),
                publisher=(
                    normalize_text(str(raw_source.get("publisher", "")))
                    or (existing_source.publisher if existing_source else None)
                ),
                authority_score=authority_score,
                published_at=(
                    _aware_datetime(raw_source.get("published_at"))
                    or (
                        existing_source.published_at
                        if existing_source is not None
                        else None
                    )
                ),
                discovered_at=discovered_at,
                updated_at=utc_now(),
                provenance=self._provenance(
                    producer_id="agent_researcher",
                    run_id=run_id,
                    task_id=task_id,
                    source_artifact_ids=source_artifact_ids,
                ),
                metadata={
                    "legacy_source_id": raw_source.get("source_id"),
                    "extraction_method": raw_source.get("extraction_method"),
                    "search_query": raw_source.get("query"),
                    "search_relevance_score": raw_source.get("score"),
                    "authority_basis": (
                        "canonical_host"
                        if _deterministic_source_profile(url) is not None
                        else "declared_source_class"
                    ),
                },
            )
            source = self._stable_entity(source)
            entities.append(source)

            source_passages = []
            for item in passages:
                if not isinstance(item, Mapping):
                    continue
                matches = str(item.get("source_id", "")) == str(
                    raw_source.get("source_id", "")
                )
                if not matches and item.get("url"):
                    try:
                        matches = canonicalize_url(str(item["url"])) == url
                    except ValueError:
                        matches = False
                if matches:
                    source_passages.append(item)
            scraped_item = scraped_by_url.get(url, {})
            body = normalize_text(str(scraped_item.get("markdown", "")))
            if not body:
                body = "\n\n".join(
                    normalize_text(str(item.get("text", "")))
                    for item in source_passages
                ).strip()
            if not body:
                issues.append(f"source has no snapshot body: {url}")
                continue
            snapshot_artifact = self.artifact_store.put_text(
                body,
                kind=ArtifactKind.SOURCE_SNAPSHOT,
                producer_id="agent_researcher",
                run_id=run_id,
                task_id=task_id,
                content_schema="WebSnapshot@1",
                source_artifact_ids=(output_artifact.artifact_id,),
                metadata={
                    "canonical_url": url,
                    "fetch_method": scraped_item.get("fetch_method")
                    or raw_source.get("extraction_method"),
                },
                idempotency_key=f"source-snapshot:{source_id}:{hashlib.sha256(body.encode('utf-8')).hexdigest()}:{output_artifact.artifact_id}",
            )
            artifact_ids.append(snapshot_artifact.artifact_id)
            previous_snapshots = self.repository.source_snapshots(source_id)
            matching_snapshot = next(
                (
                    item
                    for item in previous_snapshots
                    if item.content_hash == snapshot_artifact.content_hash
                ),
                None,
            )
            if matching_snapshot is not None:
                snapshot = matching_snapshot
            else:
                source_version = (
                    max((item.source_version for item in previous_snapshots), default=0)
                    + 1
                )
                snapshot = SourceSnapshot(
                    snapshot_id=_entity_id(
                        "snapshot",
                        run_id,
                        f"{source_id}:{snapshot_artifact.content_hash}",
                    ),
                    source_id=source_id,
                    artifact_id=snapshot_artifact.artifact_id,
                    status=SnapshotStatus.NORMALIZED,
                    source_level=source_level,
                    source_version=source_version,
                    content_hash=snapshot_artifact.content_hash,
                    final_url=url,
                    media_type=snapshot_artifact.media_type,
                    http_status=int(scraped_item.get("http_status", 200) or 200),
                    capture_method=str(
                        scraped_item.get("fetch_method")
                        or raw_source.get("extraction_method")
                        or "unknown"
                    )[:120],
                    provenance=self._provenance(
                        producer_id="agent_researcher",
                        run_id=run_id,
                        task_id=task_id,
                        source_artifact_ids=(output_artifact.artifact_id,),
                    ),
                    metadata={
                        "fetch_method": scraped_item.get("fetch_method")
                        or raw_source.get("extraction_method")
                    },
                )
                entities.append(snapshot)

            for ordinal, raw_passage in enumerate(source_passages):
                text = normalize_text(str(raw_passage.get("text", "")))
                if not text:
                    continue
                passage_artifact = self.artifact_store.put_text(
                    text,
                    kind=ArtifactKind.CLEANED_CONTENT,
                    producer_id="agent_researcher",
                    run_id=run_id,
                    task_id=task_id,
                    content_schema="PassageText@1",
                    source_artifact_ids=(snapshot_artifact.artifact_id,),
                    metadata={"source_id": source_id, "canonical_url": url},
                    idempotency_key=f"passage:{snapshot.snapshot_id}:{ordinal}:{hashlib.sha256(text.encode('utf-8')).hexdigest()}:{snapshot_artifact.artifact_id}",
                )
                artifact_ids.append(passage_artifact.artifact_id)
                passage_start = body.find(text)
                extraction_method = str(
                    raw_passage.get("extraction_method")
                    or scraped_item.get("fetch_method")
                    or raw_source.get("extraction_method")
                    or "unknown"
                )[:120]
                entities.append(
                    self._stable_entity(
                        Passage(
                            passage_id=_entity_id(
                                "passage",
                                run_id,
                                f"{snapshot.snapshot_id}:{ordinal}:{passage_artifact.content_hash}",
                            ),
                            snapshot_id=snapshot.snapshot_id,
                            text_artifact_id=passage_artifact.artifact_id,
                            ordinal=ordinal,
                            locator=f"passage:{ordinal}",
                            content_hash=passage_artifact.content_hash,
                            extraction_method=extraction_method,
                            char_start=passage_start if passage_start >= 0 else None,
                            char_end=(
                                passage_start + len(text)
                                if passage_start >= 0
                                else None
                            ),
                            status=PassageStatus.ACCEPTED,
                            provenance=self._provenance(
                                producer_id="agent_researcher",
                                run_id=run_id,
                                task_id=task_id,
                                source_artifact_ids=(
                                    snapshot_artifact.artifact_id,
                                    passage_artifact.artifact_id,
                                ),
                            ),
                            metadata={
                                "legacy_passage_id": raw_passage.get("passage_id"),
                                "legacy_source_id": raw_source.get("source_id"),
                                "canonical_url": url,
                                "title": raw_passage.get("title"),
                                "query": raw_passage.get("query"),
                                "extraction_method": raw_passage.get(
                                    "extraction_method"
                                ),
                            },
                        )
                    )
                )

        if entities:
            self.repository.save_graph(*entities)
        return IngestionResult(
            artifact_ids=tuple(dict.fromkeys(artifact_ids)),
            entity_ids=tuple(dict.fromkeys(entity_id(entity) for entity in entities)),
            issues=tuple(issues),
        )

    def ingest_candidate_knowledge(
        self,
        outputs: Mapping[str, Any],
        *,
        run_id: str,
        task_id: str | None,
    ) -> IngestionResult:
        task_id = stable_id("task", task_id) if task_id else None
        issues: list[str] = []
        artifact_ids: list[str] = []
        output_artifact = self.artifact_store.put_json(
            dict(outputs),
            kind=ArtifactKind.MODEL_OUTPUT,
            producer_id="agent_distiller",
            run_id=run_id,
            task_id=task_id,
            content_schema="CandidateKnowledgeBatch@1",
            idempotency_key=_stable_payload_key("candidate-knowledge", outputs),
        )
        artifact_ids.append(output_artifact.artifact_id)
        passages = self.repository.passages.list(run_id)
        passages_by_legacy_source: dict[str, list[Passage]] = {}
        passages_by_url: dict[str, list[Passage]] = {}
        for passage in passages:
            legacy_source = str(passage.metadata.get("legacy_source_id") or "")
            url = str(passage.metadata.get("canonical_url") or "")
            if legacy_source:
                passages_by_legacy_source.setdefault(legacy_source, []).append(passage)
            if url:
                passages_by_url.setdefault(url, []).append(passage)

        entities: list[KnowledgeEntity] = []
        evidence_map: dict[str, Evidence] = {}
        evidence_by_source: dict[str, list[Evidence]] = {}
        for raw in list(outputs.get("evidence", []) or []):
            if not isinstance(raw, Mapping):
                continue
            legacy_id = str(
                raw.get("id")
                or raw.get("evidence_id")
                or json.dumps(raw, sort_keys=True, default=str)
            )
            legacy_source = str(raw.get("source_id") or "")
            source_url = str(raw.get("source_url") or "")
            candidates: list[Passage] = []
            if source_url:
                try:
                    candidates.extend(
                        passages_by_url.get(canonicalize_url(source_url), [])
                    )
                except ValueError:
                    pass
            candidates.extend(
                passages_by_legacy_source.get(legacy_source, [])
            )
            candidates = list(
                {
                    item.passage_id: item
                    for item in candidates
                }.values()
            )
            if not candidates:
                issues.append(
                    f"evidence has no persisted passage and was skipped: {legacy_id}"
                )
                continue
            grounded = next(
                (
                    (candidate, quotes)
                    for candidate in sorted(
                        candidates,
                        key=lambda item: (
                            item.extraction_method == "search_provider",
                            -item.extracted_at.timestamp(),
                        ),
                    )
                    if (quotes := self._candidate_quotes(raw, candidate))
                ),
                None,
            )
            if grounded is None:
                issues.append(
                    "evidence quote is not an exact substring of its persisted "
                    f"passage and was skipped: {legacy_id}"
                )
                continue
            passage, quotes = grounded
            evidence = Evidence(
                evidence_id=_entity_id("evidence", run_id, legacy_id),
                passage_ids=(passage.passage_id,),
                relation=EvidenceRelation.SUPPORTS,
                status=EvidenceStatus.PROPOSED,
                summary=normalize_text(
                    str(raw.get("summary") or raw.get("quote") or "Candidate evidence")
                )[:4000],
                confidence=max(
                    0.0,
                    min(
                        1.0,
                        float(
                            raw.get("confidence", raw.get("quality_score", 0.5)) or 0.5
                        ),
                    ),
                ),
                relevance=max(
                    0.0, min(1.0, float(raw.get("quality_score", 0.5) or 0.5))
                ),
                source_quality=max(
                    0.0, min(1.0, float(raw.get("quality_score", 0.5) or 0.5))
                ),
                quotes=quotes,
                provenance=self._provenance(
                    producer_id="agent_distiller",
                    run_id=run_id,
                    task_id=task_id,
                    source_artifact_ids=(output_artifact.artifact_id,),
                ),
                metadata={
                    "legacy_evidence_id": legacy_id,
                    "legacy_source_id": legacy_source,
                    "quote": raw.get("quote"),
                },
            )
            evidence = self._stable_entity(evidence)
            entities.append(evidence)
            evidence_map[legacy_id] = evidence
            if legacy_source:
                evidence_by_source.setdefault(legacy_source, []).append(evidence)
            if source_url:
                evidence_by_source.setdefault(source_url, []).append(evidence)

        fact_map: dict[str, AtomicFact] = {}
        for raw in list(outputs.get("atomic_facts", []) or []):
            if not isinstance(raw, Mapping):
                continue
            legacy_id = str(
                raw.get("id")
                or raw.get("fact_id")
                or json.dumps(raw, sort_keys=True, default=str)
            )
            source_key = str(raw.get("source_id") or raw.get("source_url") or "")
            supporting = evidence_by_source.get(source_key, [])
            if not supporting:
                issues.append(
                    f"fact has no persisted evidence and was skipped: {legacy_id}"
                )
                continue
            statement = normalize_text(
                str(raw.get("text") or raw.get("statement") or "")
            )[:4000]
            if not statement:
                issues.append(f"empty fact skipped: {legacy_id}")
                continue
            fact = AtomicFact(
                fact_id=_entity_id("fact", run_id, legacy_id),
                statement=statement,
                evidence_ids=tuple(item.evidence_id for item in supporting[:3]),
                status=FactStatus.PROPOSED,
                confidence=max(0.0, min(1.0, float(raw.get("confidence", 0.5) or 0.5))),
                qualifiers={"section_id": str(raw.get("section_id") or "")},
                provenance=self._provenance(
                    producer_id="agent_distiller",
                    run_id=run_id,
                    task_id=task_id,
                    source_artifact_ids=(output_artifact.artifact_id,),
                ),
            )
            fact = self._stable_entity(fact)
            entities.append(fact)
            fact_map[legacy_id] = fact

        claim_map: dict[str, Claim] = {}
        citations: list[Citation] = []
        for raw in list(outputs.get("claims", []) or []):
            if not isinstance(raw, Mapping):
                continue
            legacy_id = str(
                raw.get("id")
                or raw.get("claim_id")
                or json.dumps(raw, sort_keys=True, default=str)
            )
            facts = tuple(
                fact_map[str(ref)].fact_id
                for ref in raw.get("fact_ids", [])
                if str(ref) in fact_map
            )
            direct_evidence_ids = tuple(
                evidence_map[str(ref)].evidence_id
                for ref in raw.get("evidence_ids", [])
                if str(ref) in evidence_map
            )
            evidence_ids = tuple(
                dict.fromkeys(
                    (
                        *direct_evidence_ids,
                        *(
                            evidence_id
                            for fact_id in facts
                            for evidence_id in next(
                                (
                                    item.evidence_ids
                                    for item in fact_map.values()
                                    if item.fact_id == fact_id
                                ),
                                (),
                            )
                        ),
                    )
                )
            )
            statement = normalize_text(
                str(raw.get("text") or raw.get("statement") or "")
            )[:8000]
            if not statement:
                issues.append(f"empty claim skipped: {legacy_id}")
                continue
            if not evidence_ids:
                issues.append(
                    f"claim has no persisted grounded evidence and was skipped: {legacy_id}"
                )
                continue
            claim = Claim(
                claim_id=_entity_id("claim", run_id, legacy_id),
                statement=statement,
                fact_ids=facts,
                evidence_ids=evidence_ids,
                status=ClaimStatus.DRAFT,
                confidence=max(0.0, min(1.0, float(raw.get("confidence", 0.5) or 0.5))),
                provenance=self._provenance(
                    producer_id="agent_distiller",
                    run_id=run_id,
                    task_id=task_id,
                    source_artifact_ids=(output_artifact.artifact_id,),
                ),
            )
            claim = self._stable_entity(claim)
            entities.append(claim)
            claim_map[legacy_id] = claim
            for evidence_id in evidence_ids:
                evidence_item = next(
                    (
                        item
                        for item in evidence_map.values()
                        if item.evidence_id == evidence_id
                    ),
                    None,
                )
                if evidence_item is None or not evidence_item.quotes:
                    continue
                quote = evidence_item.quotes[0]
                passage = self.repository.passages.get(quote.passage_id)
                if passage is None:
                    continue
                snapshot = self.repository.snapshots.get(passage.snapshot_id)
                if snapshot is None:
                    continue
                citation = Citation(
                    citation_id=_entity_id(
                        "citation",
                        run_id,
                        f"{claim.claim_id}:{evidence_id}:{passage.passage_id}",
                    ),
                    claim_id=claim.claim_id,
                    evidence_id=evidence_id,
                    passage_id=passage.passage_id,
                    snapshot_id=snapshot.snapshot_id,
                    source_id=snapshot.source_id,
                    locator=passage.locator,
                    quote=quote.quote,
                    quote_start=quote.char_start,
                    quote_end=quote.char_end,
                    extraction_method=quote.extraction_method,
                    status=CitationStatus.PROPOSED,
                    provenance=self._provenance(
                        producer_id="agent_research_worker",
                        run_id=run_id,
                        task_id=task_id,
                        source_artifact_ids=(
                            output_artifact.artifact_id,
                            passage.text_artifact_id,
                        ),
                    ),
                )
                citation = self._stable_entity(citation)
                entities.append(citation)
                citations.append(citation)

        for raw in list(outputs.get("conflicts", []) or []):
            if not isinstance(raw, Mapping):
                continue
            legacy_id = str(
                raw.get("id")
                or raw.get("conflict_id")
                or json.dumps(raw, sort_keys=True, default=str)
            )
            claim_ids = tuple(
                claim_map[str(ref)].claim_id
                for ref in raw.get("claim_ids", [])
                if str(ref) in claim_map
            )
            if len(claim_ids) < 2:
                issues.append(
                    f"conflict with fewer than two persisted claims skipped: {legacy_id}"
                )
                continue
            fact_ids = tuple(
                fact_map[str(ref)].fact_id
                for ref in raw.get("fact_ids", [])
                if str(ref) in fact_map
            )
            entities.append(
                self._stable_entity(
                    Conflict(
                        conflict_id=_entity_id("conflict", run_id, legacy_id),
                        claim_ids=claim_ids,
                        fact_ids=fact_ids,
                        summary=normalize_text(
                            str(
                                raw.get("description") or "Conflicting candidate claims"
                            )
                        )[:5000],
                        provenance=self._provenance(
                            producer_id="agent_distiller",
                            run_id=run_id,
                            task_id=task_id,
                            source_artifact_ids=(output_artifact.artifact_id,),
                        ),
                    )
                )
            )

        for pack in list(outputs.get("section_evidence_packs", []) or []):
            if not isinstance(pack, Mapping):
                continue
            artifact = self.artifact_store.put_json(
                dict(pack),
                kind=ArtifactKind.EVIDENCE_PACK,
                producer_id="agent_distiller",
                run_id=run_id,
                task_id=task_id,
                content_schema="SectionEvidencePack@1",
                source_artifact_ids=(output_artifact.artifact_id,),
                idempotency_key=_stable_payload_key("evidence-pack", pack),
            )
            artifact_ids.append(artifact.artifact_id)
        if entities:
            self.repository.save_graph(*entities)
        return IngestionResult(
            artifact_ids=tuple(dict.fromkeys(artifact_ids)),
            entity_ids=tuple(dict.fromkeys(entity_id(entity) for entity in entities)),
            issues=tuple(issues),
        )

    def _candidate_quotes(
        self, raw: Mapping[str, Any], passage: Passage
    ) -> tuple[EvidenceQuote, ...]:
        quote = normalize_text(str(raw.get("quote") or ""))
        if not quote or len(quote) > 3000:
            return ()
        passage_text = self.artifact_store.read_bytes(passage.text_artifact_id).decode(
            "utf-8"
        )
        start = passage_text.find(quote)
        if start < 0:
            return ()
        return (
            EvidenceQuote(
                passage_id=passage.passage_id,
                quote=quote,
                char_start=start,
                char_end=start + len(quote),
                passage_content_hash=passage.content_hash,
                extraction_method=str(
                    raw.get("extraction_method")
                    or passage.extraction_method
                    or "distiller"
                )[:120],
                extracted_at=passage.extracted_at,
            ),
        )

    def create_report_scaffold(
        self,
        report_payload: Mapping[str, Any],
        *,
        report_outline: Mapping[str, Any],
        run_id: str,
        thread_id: str,
        research_question: str,
    ) -> IngestionResult:
        markdown = str(report_payload.get("markdown") or "")
        artifact = self.artifact_store.put_text(
            markdown,
            kind=ArtifactKind.REPORT,
            producer_id="agent_writer",
            run_id=run_id,
            content_schema="ReportScaffold@1",
            idempotency_key=_stable_payload_key("report", report_payload),
        )
        raw_sections = list(report_outline.get("sections", []) or [])
        if not raw_sections:
            raw_sections = [
                {
                    "section_id": section_id,
                    "title": section_id,
                    "goal": "Draft report section",
                    "order": index,
                }
                for index, section_id in enumerate(
                    report_payload.get("section_ids", []) or [], start=1
                )
            ]
        report_id = _entity_id(
            "report",
            run_id,
            str(report_payload.get("report_id") or artifact.content_hash),
        )
        section_ids = tuple(
            _entity_id("section", run_id, str(item.get("section_id") or index))
            for index, item in enumerate(raw_sections, start=1)
        )
        provenance = self._provenance(
            producer_id="agent_writer",
            run_id=run_id,
            task_id=None,
            source_artifact_ids=(artifact.artifact_id,),
        )
        report = Report(
            report_id=report_id,
            thread_id=stable_id("thread", thread_id),
            run_id=run_id,
            title=normalize_text(str(report_outline.get("title") or "Research Report")),
            research_question=normalize_text(research_question),
            section_ids=section_ids,
            status=ReportStatus.DRAFT,
            content_artifact_id=artifact.artifact_id,
            provenance=provenance,
        )
        citation_map = dict(report_payload.get("citation_map", {}) or {})
        existing_claim_ids = {
            claim.claim_id for claim in self.repository.claims.list(run_id)
        }
        sections: list[Section] = []
        for index, raw in enumerate(raw_sections, start=1):
            legacy_section = str(raw.get("section_id") or index)
            claim_ids = tuple(
                _entity_id("claim", run_id, str(reference))
                for reference in citation_map.get(legacy_section, [])
                if _entity_id("claim", run_id, str(reference)) in existing_claim_ids
            )
            sections.append(
                Section(
                    section_id=section_ids[index - 1],
                    report_id=report_id,
                    title=normalize_text(str(raw.get("title") or f"Section {index}")),
                    goal=normalize_text(str(raw.get("goal") or "Draft report section")),
                    order=int(raw.get("order", index) or index),
                    claim_ids=claim_ids,
                    content_artifact_id=artifact.artifact_id,
                    status=SectionStatus.DRAFTING,
                    provenance=provenance,
                )
            )
        report = self._stable_entity(report)
        sections = [self._stable_entity(section) for section in sections]
        self.repository.save_graph(report, *sections)
        return IngestionResult(
            artifact_ids=(artifact.artifact_id,),
            entity_ids=(
                report.report_id,
                *(section.section_id for section in sections),
            ),
            issues=(),
        )
