from __future__ import annotations

import json
from typing import Any, Dict

import pytest

from core.session_knowledge import SessionKnowledgeStore


RESEARCH_ID = "research-session-store"
SESSION_ID = "session-research-session-store"
ROOT_QUERY = "How should cumulative session knowledge behave across rounds?"


@pytest.fixture
def session_store():
    store = SessionKnowledgeStore(":memory:")
    try:
        yield store
    finally:
        store.close()


@pytest.fixture
def initialized_store(session_store: SessionKnowledgeStore):
    session_store.create_or_get_session(
        research_id=RESEARCH_ID,
        session_id=SESSION_ID,
        root_query=ROOT_QUERY,
        metadata_json={"owner": "test"},
    )
    return session_store


def _json(payload: Dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False)


def _source_row(*, round_no: int, source_id: str = "src-1", url: str = "https://example.com/a") -> Dict[str, Any]:
    return {
        "source_id": source_id,
        "research_id": RESEARCH_ID,
        "url": url,
        "title": f"Source {source_id} round {round_no}",
        "domain": "example.com",
        "source_type": "web",
        "authority_score": 0.6 + (round_no * 0.1),
        "freshness_score": 0.4 + (round_no * 0.1),
        "task_id": f"task-round-{round_no}",
        "first_seen_round": round_no,
        "last_seen_round": round_no,
        "first_seen_at": f"2026-04-{round_no:02d}T10:00:00",
        "last_seen_at": f"2026-04-{round_no:02d}T10:30:00",
        "is_active": 1,
        "metadata_json": _json({"round": round_no, "kind": "source"}),
    }


def _claim_row(*, round_no: int, claim_id: str = "claim-1", dedup_key: str = "claim-dedup-1") -> Dict[str, Any]:
    return {
        "claim_id": claim_id,
        "research_id": RESEARCH_ID,
        "task_id": f"task-round-{round_no}",
        "section_id": "sec-1",
        "canonical_text": "Demand for accelerator hardware increased materially.",
        "raw_text": f"Round {round_no} raw claim text",
        "confidence": 0.5 + (round_no * 0.1),
        "status": "active",
        "source_count": round_no,
        "evidence_count": round_no,
        "fact_count": round_no,
        "first_seen_round": round_no,
        "last_seen_round": round_no,
        "created_at": f"2026-04-{round_no:02d}T11:00:00",
        "updated_at": f"2026-04-{round_no:02d}T11:30:00",
        "dedup_key": dedup_key,
        "metadata_json": _json({"round": round_no, "fact_ids": [f"fact-{round_no}"]}),
    }


def _fact_row(*, round_no: int, fact_id: str = "fact-1", dedup_key: str = "fact-dedup-1") -> Dict[str, Any]:
    return {
        "fact_id": fact_id,
        "research_id": RESEARCH_ID,
        "task_id": f"task-round-{round_no}",
        "section_id": "sec-1",
        "canonical_text": "Accelerator demand increased.",
        "raw_text": f"Round {round_no} raw fact text",
        "snippet": f"Round {round_no} snippet",
        "confidence": 0.45 + (round_no * 0.1),
        "verified_count": round_no,
        "source_count": round_no,
        "status": "active",
        "dedup_key": dedup_key,
        "first_seen_round": round_no,
        "last_seen_round": round_no,
        "created_at": f"2026-04-{round_no:02d}T12:00:00",
        "updated_at": f"2026-04-{round_no:02d}T12:30:00",
        "metadata_json": _json({"round": round_no, "source_url": "https://example.com/a"}),
    }


def _evidence_row(
    *,
    round_no: int,
    evidence_id: str = "evidence-1",
    dedup_key: str = "evidence-dedup-1",
    source_id: str = "src-1",
) -> Dict[str, Any]:
    return {
        "evidence_id": evidence_id,
        "research_id": RESEARCH_ID,
        "task_id": f"task-round-{round_no}",
        "section_id": "sec-1",
        "source_id": source_id,
        "quote_text": "GPU spending increased year over year.",
        "summary_text": f"Round {round_no} evidence summary",
        "quality_score": 0.55 + (round_no * 0.1),
        "confidence": 0.5 + (round_no * 0.1),
        "status": "active",
        "dedup_key": dedup_key,
        "first_seen_round": round_no,
        "last_seen_round": round_no,
        "created_at": f"2026-04-{round_no:02d}T13:00:00",
        "updated_at": f"2026-04-{round_no:02d}T13:30:00",
        "metadata_json": _json({"round": round_no, "claim_ids": ["claim-1"]}),
    }


def _conflict_row(*, round_no: int, conflict_id: str = "conflict-1", dedup_key: str = "conflict-dedup-1") -> Dict[str, Any]:
    return {
        "conflict_id": conflict_id,
        "research_id": RESEARCH_ID,
        "task_id": f"task-round-{round_no}",
        "section_id": "sec-1",
        "conflict_type": "numerical_conflict",
        "description": "Different reports disagree on total accelerator market size.",
        "severity": "medium" if round_no == 1 else "high",
        "status": "active",
        "claim_count": round_no,
        "evidence_count": round_no,
        "first_seen_round": round_no,
        "last_seen_round": round_no,
        "created_at": f"2026-04-{round_no:02d}T14:00:00",
        "updated_at": f"2026-04-{round_no:02d}T14:30:00",
        "dedup_key": dedup_key,
        "metadata_json": _json({"round": round_no, "claim_ids": ["claim-1"]}),
    }


def _section_pack_row(
    *,
    round_no: int,
    pack_id: str = "pack-1",
    section_id: str = "sec-1",
    coverage_score: float | None = None,
) -> Dict[str, Any]:
    score = coverage_score if coverage_score is not None else 0.3 + (round_no * 0.2)
    return {
        "pack_id": pack_id,
        "research_id": RESEARCH_ID,
        "section_id": section_id,
        "section_title": "Market Outlook",
        "goal": "Summarize cumulative market evidence.",
        "coverage_score": score,
        "status": "active",
        "claim_count": round_no,
        "fact_count": round_no,
        "evidence_count": round_no,
        "conflict_count": 0 if round_no == 1 else 1,
        "notes": f"Round {round_no} notes",
        "first_seen_round": round_no,
        "last_updated_round": round_no,
        "created_at": f"2026-04-{round_no:02d}T15:00:00",
        "updated_at": f"2026-04-{round_no:02d}T15:30:00",
        "metadata_json": _json({"round": round_no, "section_id": section_id}),
    }


def _coverage_snapshot_row(*, round_no: int, snapshot_id: str) -> Dict[str, Any]:
    return {
        "snapshot_id": snapshot_id,
        "research_id": RESEARCH_ID,
        "round_no": round_no,
        "avg_section_coverage": 0.25 + (round_no * 0.2),
        "evidence_density": 0.4 + (round_no * 0.1),
        "conflict_pressure": 0.1 * round_no,
        "sufficiency_level": "partial" if round_no == 1 else "sufficient_for_writing",
        "completed_section_count": max(0, round_no - 1),
        "partial_section_count": 1,
        "uncovered_section_count": 2 - min(round_no, 2),
        "created_at": f"2026-04-{round_no:02d}T16:00:00",
        "raw_summary_json": _json({"round": round_no, "covered_sections": ["sec-1"] if round_no > 1 else []}),
    }


def _gap_row(*, round_no: int, gap_id: str, gap_text: str, section_id: str = "sec-1", severity: str = "medium") -> Dict[str, Any]:
    return {
        "gap_id": gap_id,
        "research_id": RESEARCH_ID,
        "round_no": round_no,
        "task_id": f"task-round-{round_no}",
        "section_id": section_id,
        "gap_text": gap_text,
        "gap_type": "coverage_gap",
        "severity": severity,
        "status": "open",
        "created_at": f"2026-04-{round_no:02d}T17:00:00",
        "updated_at": f"2026-04-{round_no:02d}T17:30:00",
        "metadata_json": _json({"round": round_no}),
    }


def _fetchone(store: SessionKnowledgeStore, query: str, params: tuple[Any, ...]) -> Dict[str, Any]:
    row = store.conn.execute(query, params).fetchone()
    assert row is not None
    return dict(row)


def _fetchall(store: SessionKnowledgeStore, query: str, params: tuple[Any, ...] = ()) -> list[Dict[str, Any]]:
    return [dict(row) for row in store.conn.execute(query, params).fetchall()]


def test_session_initialization_and_round_increment(initialized_store: SessionKnowledgeStore):
    session = initialized_store.get_session(RESEARCH_ID)

    assert session is not None
    assert session.research_id == RESEARCH_ID
    assert session.session_id == SESSION_ID
    assert session.root_query == ROOT_QUERY
    assert session.current_round == 0
    assert session.metadata_json == {"owner": "test"}

    initialized_store.update_session_round(
        RESEARCH_ID,
        round_no=2,
        current_active_task_id="task-round-2",
        status="active",
    )
    updated = initialized_store.get_session(RESEARCH_ID)

    assert updated is not None
    assert updated.current_round == 2
    assert updated.current_active_task_id == "task-round-2"
    assert updated.status == "active"
    assert updated.last_active_at >= session.last_active_at


def test_session_root_query_is_not_overwritten_by_later_updates(initialized_store: SessionKnowledgeStore):
    initialized_store.create_or_get_session(
        research_id=RESEARCH_ID,
        session_id=SESSION_ID,
        root_query="Distilled 5 clean passages into 2 atomic facts.",
        current_active_task_id="task-1",
    )

    session = initialized_store.get_session(RESEARCH_ID)

    assert session is not None
    assert session.root_query == ROOT_QUERY
    assert session.current_active_task_id == "task-1"


def test_first_ingest_persists_all_core_entities(initialized_store: SessionKnowledgeStore):
    initialized_store.upsert_sources([_source_row(round_no=1)])
    initialized_store.upsert_claims([_claim_row(round_no=1)])
    initialized_store.upsert_facts([_fact_row(round_no=1)])
    initialized_store.upsert_evidence([_evidence_row(round_no=1)])
    initialized_store.upsert_conflicts([_conflict_row(round_no=1)])
    initialized_store.upsert_section_packs([_section_pack_row(round_no=1)])
    initialized_store.upsert_claim_fact_links([(RESEARCH_ID, "claim-1", "fact-1")])
    initialized_store.upsert_claim_evidence_links([(RESEARCH_ID, "claim-1", "evidence-1")])
    initialized_store.upsert_pack_links(
        fact_rows=[(RESEARCH_ID, "pack-1", "fact-1")],
        claim_rows=[(RESEARCH_ID, "pack-1", "claim-1")],
        evidence_rows=[(RESEARCH_ID, "pack-1", "evidence-1")],
        conflict_rows=[(RESEARCH_ID, "pack-1", "conflict-1")],
    )

    source = _fetchone(initialized_store, "SELECT * FROM session_sources WHERE research_id = ? AND source_id = ?", (RESEARCH_ID, "src-1"))
    claim = _fetchone(initialized_store, "SELECT * FROM session_claims WHERE research_id = ? AND claim_id = ?", (RESEARCH_ID, "claim-1"))
    fact = _fetchone(initialized_store, "SELECT * FROM session_facts WHERE research_id = ? AND fact_id = ?", (RESEARCH_ID, "fact-1"))
    evidence = _fetchone(initialized_store, "SELECT * FROM session_evidence WHERE research_id = ? AND evidence_id = ?", (RESEARCH_ID, "evidence-1"))
    conflict = _fetchone(initialized_store, "SELECT * FROM session_conflicts WHERE research_id = ? AND conflict_id = ?", (RESEARCH_ID, "conflict-1"))
    pack = _fetchone(initialized_store, "SELECT * FROM session_section_packs WHERE research_id = ? AND pack_id = ?", (RESEARCH_ID, "pack-1"))

    assert source["url"] == "https://example.com/a"
    assert claim["canonical_text"] == "Demand for accelerator hardware increased materially."
    assert fact["canonical_text"] == "Accelerator demand increased."
    assert evidence["source_id"] == "src-1"
    assert conflict["conflict_type"] == "numerical_conflict"
    assert pack["section_id"] == "sec-1"

    assert _fetchall(initialized_store, "SELECT * FROM session_claim_fact_links WHERE research_id = ?", (RESEARCH_ID,)) == [
        {
            "research_id": RESEARCH_ID,
            "claim_id": "claim-1",
            "fact_id": "fact-1",
            "created_at": _fetchone(
                initialized_store,
                "SELECT * FROM session_claim_fact_links WHERE research_id = ? AND claim_id = ? AND fact_id = ?",
                (RESEARCH_ID, "claim-1", "fact-1"),
            )["created_at"],
        }
    ]
    assert len(_fetchall(initialized_store, "SELECT * FROM session_pack_fact_links WHERE research_id = ?", (RESEARCH_ID,))) == 1
    assert len(_fetchall(initialized_store, "SELECT * FROM session_pack_claim_links WHERE research_id = ?", (RESEARCH_ID,))) == 1
    assert len(_fetchall(initialized_store, "SELECT * FROM session_pack_evidence_links WHERE research_id = ?", (RESEARCH_ID,))) == 1
    assert len(_fetchall(initialized_store, "SELECT * FROM session_pack_conflict_links WHERE research_id = ?", (RESEARCH_ID,))) == 1


def test_cross_round_upsert_and_merge_preserves_prior_knowledge(initialized_store: SessionKnowledgeStore):
    initialized_store.upsert_sources([_source_row(round_no=1, source_id="src-1", url="https://example.com/a")])
    initialized_store.upsert_claims([_claim_row(round_no=1)])
    initialized_store.upsert_facts([_fact_row(round_no=1)])
    initialized_store.upsert_evidence([_evidence_row(round_no=1)])
    initialized_store.upsert_conflicts([_conflict_row(round_no=1)])

    initialized_store.update_session_round(RESEARCH_ID, round_no=2, current_active_task_id="task-round-2")
    initialized_store.upsert_sources([_source_row(round_no=2, source_id="src-2", url="https://example.com/a")])
    initialized_store.upsert_claims([_claim_row(round_no=2, claim_id="claim-2")])
    initialized_store.upsert_facts([_fact_row(round_no=2, fact_id="fact-2")])
    initialized_store.upsert_evidence([_evidence_row(round_no=2, evidence_id="evidence-2")])
    initialized_store.upsert_conflicts([_conflict_row(round_no=2, conflict_id="conflict-2")])

    source_rows = _fetchall(initialized_store, "SELECT * FROM session_sources WHERE research_id = ?", (RESEARCH_ID,))
    claim = _fetchone(initialized_store, "SELECT * FROM session_claims WHERE research_id = ? AND claim_id = ?", (RESEARCH_ID, "claim-1"))
    fact = _fetchone(initialized_store, "SELECT * FROM session_facts WHERE research_id = ? AND fact_id = ?", (RESEARCH_ID, "fact-1"))
    evidence = _fetchone(initialized_store, "SELECT * FROM session_evidence WHERE research_id = ? AND evidence_id = ?", (RESEARCH_ID, "evidence-1"))
    conflict = _fetchone(initialized_store, "SELECT * FROM session_conflicts WHERE research_id = ? AND conflict_id = ?", (RESEARCH_ID, "conflict-1"))

    assert len(source_rows) == 1
    assert source_rows[0]["source_id"] == "src-1"
    assert source_rows[0]["last_seen_round"] == 2
    assert source_rows[0]["first_seen_round"] == 1
    assert source_rows[0]["authority_score"] == pytest.approx(0.8)

    assert claim["claim_id"] == "claim-1"
    assert claim["last_seen_round"] == 2
    assert claim["first_seen_round"] == 1
    assert claim["confidence"] == pytest.approx(0.7)
    assert claim["source_count"] == 2

    assert fact["fact_id"] == "fact-1"
    assert fact["last_seen_round"] == 2
    assert fact["verified_count"] == 2
    assert fact["snippet"] == "Round 2 snippet"

    assert evidence["evidence_id"] == "evidence-1"
    assert evidence["last_seen_round"] == 2
    assert evidence["quality_score"] == pytest.approx(0.75)

    assert conflict["conflict_id"] == "conflict-1"
    assert conflict["last_seen_round"] == 2
    assert conflict["severity"] == "high"
    assert conflict["claim_count"] == 2


def test_source_id_is_unique_per_research_not_global(initialized_store: SessionKnowledgeStore):
    other_research_id = "research-session-store-other"
    initialized_store.create_or_get_session(
        research_id=other_research_id,
        session_id=f"session-{other_research_id}",
        root_query="A second research can reuse the same source id.",
        metadata_json={"owner": "test"},
    )

    first_source = _source_row(round_no=1, source_id="src-shared", url="https://example.com/shared")
    second_source = {
        **_source_row(round_no=1, source_id="src-shared", url="https://example.com/shared"),
        "research_id": other_research_id,
        "title": "Shared source in second research",
    }

    initialized_store.upsert_sources([first_source])
    initialized_store.upsert_sources([second_source])

    rows = _fetchall(
        initialized_store,
        "SELECT research_id, source_id, url FROM session_sources WHERE source_id = ? ORDER BY research_id",
        ("src-shared",),
    )

    assert rows == [
        {"research_id": RESEARCH_ID, "source_id": "src-shared", "url": "https://example.com/shared"},
        {"research_id": other_research_id, "source_id": "src-shared", "url": "https://example.com/shared"},
    ]


def test_legacy_global_source_primary_key_is_migrated(session_store: SessionKnowledgeStore):
    session_store.conn.executescript(
        """
        DROP TABLE session_sources;

        CREATE TABLE session_sources (
            source_id TEXT PRIMARY KEY,
            research_id TEXT NOT NULL,
            url TEXT NOT NULL,
            title TEXT NOT NULL,
            domain TEXT NOT NULL,
            source_type TEXT NOT NULL,
            authority_score REAL NOT NULL DEFAULT 0.0,
            freshness_score REAL NOT NULL DEFAULT 0.0,
            task_id TEXT,
            first_seen_round INTEGER NOT NULL DEFAULT 0,
            last_seen_round INTEGER NOT NULL DEFAULT 0,
            first_seen_at TEXT NOT NULL,
            last_seen_at TEXT NOT NULL,
            is_active INTEGER NOT NULL DEFAULT 1,
            metadata_json TEXT NOT NULL,
            UNIQUE(research_id, url)
        );

        INSERT INTO session_sources (
            source_id,
            research_id,
            url,
            title,
            domain,
            source_type,
            authority_score,
            freshness_score,
            task_id,
            first_seen_round,
            last_seen_round,
            first_seen_at,
            last_seen_at,
            is_active,
            metadata_json
        )
        VALUES (
            'src-shared',
            'research-legacy',
            'https://example.com/shared',
            'Legacy source',
            'example.com',
            'web',
            0.5,
            0.4,
            'task-legacy',
            1,
            1,
            '2026-04-01T10:00:00',
            '2026-04-01T10:30:00',
            1,
            '{}'
        );
        """
    )

    session_store._migrate_session_sources_to_research_scoped_primary_key()
    primary_key_columns = [
        row["name"]
        for row in sorted(session_store.conn.execute("PRAGMA table_info(session_sources)").fetchall(), key=lambda item: item["pk"])
        if row["pk"]
    ]
    session_store.create_or_get_session(
        research_id="research-new",
        session_id="session-research-new",
        root_query="Reuse the legacy source id.",
        metadata_json={"owner": "test"},
    )
    session_store.upsert_sources(
        [
            {
                **_source_row(round_no=1, source_id="src-shared", url="https://example.com/shared"),
                "research_id": "research-new",
                "title": "Shared source after migration",
            }
        ]
    )
    rows = _fetchall(
        session_store,
        "SELECT research_id, source_id FROM session_sources WHERE source_id = ? ORDER BY research_id",
        ("src-shared",),
    )

    assert primary_key_columns == ["research_id", "source_id"]
    assert rows == [
        {"research_id": "research-legacy", "source_id": "src-shared"},
        {"research_id": "research-new", "source_id": "src-shared"},
    ]


def test_same_section_id_pack_is_upserted_not_duplicated(initialized_store: SessionKnowledgeStore):
    initialized_store.upsert_section_packs([_section_pack_row(round_no=1, pack_id="pack-1", section_id="sec-1", coverage_score=0.35)])
    initialized_store.upsert_pack_links(
        fact_rows=[(RESEARCH_ID, "pack-1", "fact-1")],
        claim_rows=[(RESEARCH_ID, "pack-1", "claim-1")],
        evidence_rows=[(RESEARCH_ID, "pack-1", "evidence-1")],
        conflict_rows=[],
    )

    initialized_store.upsert_section_packs([_section_pack_row(round_no=2, pack_id="pack-2", section_id="sec-1", coverage_score=0.9)])
    initialized_store.upsert_pack_links(
        fact_rows=[(RESEARCH_ID, "pack-1", "fact-2")],
        claim_rows=[(RESEARCH_ID, "pack-1", "claim-2")],
        evidence_rows=[(RESEARCH_ID, "pack-1", "evidence-2")],
        conflict_rows=[(RESEARCH_ID, "pack-1", "conflict-1")],
    )

    packs = _fetchall(initialized_store, "SELECT * FROM session_section_packs WHERE research_id = ?", (RESEARCH_ID,))

    assert len(packs) == 1
    assert packs[0]["pack_id"] == "pack-1"
    assert packs[0]["section_id"] == "sec-1"
    assert packs[0]["coverage_score"] == pytest.approx(0.9)
    assert packs[0]["claim_count"] == 2
    assert packs[0]["last_updated_round"] == 2
    assert len(_fetchall(initialized_store, "SELECT * FROM session_pack_fact_links WHERE research_id = ?", (RESEARCH_ID,))) == 2


def test_coverage_snapshots_append_in_order(initialized_store: SessionKnowledgeStore):
    initialized_store.append_coverage_snapshot(_coverage_snapshot_row(round_no=1, snapshot_id="coverage-1"))
    initialized_store.append_coverage_snapshot(_coverage_snapshot_row(round_no=2, snapshot_id="coverage-2"))

    rows = _fetchall(
        initialized_store,
        "SELECT snapshot_id, round_no, sufficiency_level FROM session_coverage_snapshots WHERE research_id = ? ORDER BY round_no ASC",
        (RESEARCH_ID,),
    )
    snapshot = initialized_store.build_session_snapshot(RESEARCH_ID)

    assert rows == [
        {"snapshot_id": "coverage-1", "round_no": 1, "sufficiency_level": "partial"},
        {"snapshot_id": "coverage-2", "round_no": 2, "sufficiency_level": "sufficient_for_writing"},
    ]
    assert snapshot["latest_coverage_snapshot"]["snapshot_id"] == "coverage-2"
    assert snapshot["latest_coverage_snapshot"]["round_no"] == 2
    assert snapshot["latest_coverage_snapshot"]["raw_summary_json"]["round"] == 2


def test_unresolved_gaps_lifecycle_open_update_resolved(initialized_store: SessionKnowledgeStore):
    gap_a_round_1 = _gap_row(round_no=1, gap_id="gap-1", gap_text="Need more vendor revenue evidence")
    gap_b_round_1 = _gap_row(round_no=1, gap_id="gap-2", gap_text="Need better section coverage", section_id="sec-2")

    result_round_1 = initialized_store.upsert_unresolved_gaps([gap_a_round_1, gap_b_round_1])
    open_rows_round_1 = _fetchall(
        initialized_store,
        "SELECT gap_id, status, round_no FROM session_unresolved_gaps WHERE research_id = ? ORDER BY gap_id",
        (RESEARCH_ID,),
    )

    gap_a_round_2 = _gap_row(
        round_no=2,
        gap_id="gap-3",
        gap_text="Need more vendor revenue evidence",
        severity="high",
    )
    result_round_2 = initialized_store.upsert_unresolved_gaps([gap_a_round_2], close_missing=True)
    rows_after_round_2 = _fetchall(
        initialized_store,
        "SELECT gap_id, gap_text, status, round_no, severity FROM session_unresolved_gaps WHERE research_id = ? ORDER BY gap_text",
        (RESEARCH_ID,),
    )
    snapshot = initialized_store.build_session_snapshot(RESEARCH_ID)

    assert result_round_1 == {"inserted": 2, "updated": 0}
    assert open_rows_round_1 == [
        {"gap_id": "gap-1", "status": "open", "round_no": 1},
        {"gap_id": "gap-2", "status": "open", "round_no": 1},
    ]

    assert result_round_2 == {"inserted": 0, "updated": 1}
    assert rows_after_round_2 == [
        {
            "gap_id": "gap-2",
            "gap_text": "Need better section coverage",
            "status": "resolved",
            "round_no": 1,
            "severity": "medium",
        },
        {
            "gap_id": "gap-1",
            "gap_text": "Need more vendor revenue evidence",
            "status": "open",
            "round_no": 2,
            "severity": "high",
        },
    ]
    assert [gap["gap_id"] for gap in snapshot["open_gaps"]] == ["gap-1"]
    assert snapshot["open_gaps"][0]["severity"] == "high"
    assert snapshot["open_gaps"][0]["metadata_json"]["round"] == 2


def test_build_session_snapshot_returns_expected_aggregate(initialized_store: SessionKnowledgeStore):
    initialized_store.update_session_round(RESEARCH_ID, round_no=2, current_active_task_id="task-round-2")

    initialized_store.upsert_sources(
        [
            _source_row(round_no=1, source_id="src-1", url="https://example.com/a"),
            _source_row(round_no=2, source_id="src-2", url="https://example.com/b"),
        ]
    )
    initialized_store.upsert_claims(
        [
            _claim_row(round_no=1, claim_id="claim-1", dedup_key="claim-dedup-1"),
            _claim_row(round_no=2, claim_id="claim-2", dedup_key="claim-dedup-2"),
        ]
    )
    initialized_store.upsert_facts(
        [
            _fact_row(round_no=1, fact_id="fact-1", dedup_key="fact-dedup-1"),
            _fact_row(round_no=2, fact_id="fact-2", dedup_key="fact-dedup-2"),
        ]
    )
    initialized_store.upsert_evidence(
        [
            _evidence_row(round_no=1, evidence_id="evidence-1", dedup_key="evidence-dedup-1", source_id="src-1"),
            _evidence_row(round_no=2, evidence_id="evidence-2", dedup_key="evidence-dedup-2", source_id="src-2"),
        ]
    )
    initialized_store.upsert_conflicts(
        [
            _conflict_row(round_no=1, conflict_id="conflict-1", dedup_key="conflict-dedup-1"),
            _conflict_row(round_no=2, conflict_id="conflict-2", dedup_key="conflict-dedup-2"),
        ]
    )
    initialized_store.upsert_section_packs(
        [
            _section_pack_row(round_no=1, pack_id="pack-1", section_id="sec-1"),
            _section_pack_row(round_no=2, pack_id="pack-2", section_id="sec-2"),
        ]
    )
    initialized_store.upsert_claim_fact_links(
        [
            (RESEARCH_ID, "claim-1", "fact-1"),
            (RESEARCH_ID, "claim-2", "fact-2"),
        ]
    )
    initialized_store.upsert_claim_evidence_links(
        [
            (RESEARCH_ID, "claim-1", "evidence-1"),
            (RESEARCH_ID, "claim-2", "evidence-2"),
        ]
    )
    initialized_store.upsert_pack_links(
        fact_rows=[
            (RESEARCH_ID, "pack-1", "fact-1"),
            (RESEARCH_ID, "pack-2", "fact-2"),
        ],
        claim_rows=[
            (RESEARCH_ID, "pack-1", "claim-1"),
            (RESEARCH_ID, "pack-2", "claim-2"),
        ],
        evidence_rows=[
            (RESEARCH_ID, "pack-1", "evidence-1"),
            (RESEARCH_ID, "pack-2", "evidence-2"),
        ],
        conflict_rows=[
            (RESEARCH_ID, "pack-1", "conflict-1"),
            (RESEARCH_ID, "pack-2", "conflict-2"),
        ],
    )
    initialized_store.append_coverage_snapshot(_coverage_snapshot_row(round_no=2, snapshot_id="coverage-2"))
    initialized_store.append_novelty_snapshot(
        {
            "snapshot_id": "novelty-2",
            "research_id": RESEARCH_ID,
            "round_no": 2,
            "new_fact_count": 1,
            "merged_fact_count": 1,
            "new_source_count": 1,
            "new_claim_count": 1,
            "new_evidence_count": 1,
            "novelty_ratio": 0.5,
            "novelty_level": "medium",
            "created_at": "2026-04-02T18:00:00",
            "metadata_json": _json({"round": 2}),
        }
    )
    initialized_store.upsert_unresolved_gaps([_gap_row(round_no=2, gap_id="gap-1", gap_text="Need one more primary source")])

    snapshot = initialized_store.build_session_snapshot(RESEARCH_ID)

    assert snapshot["session"]["research_id"] == RESEARCH_ID
    assert snapshot["session"]["current_round"] == 2
    assert snapshot["stats"] == {
        "source_count": 2,
        "fact_count": 2,
        "claim_count": 2,
        "evidence_count": 2,
        "conflict_count": 2,
        "section_pack_count": 2,
    }
    assert snapshot["knowledge_refs"]["collection_name"] == f"research_{RESEARCH_ID}"
    assert snapshot["knowledge_refs"]["source_ids"] == ["src-1", "src-2"]
    assert snapshot["knowledge_refs"]["claim_ids"] == ["claim-1", "claim-2"]
    assert snapshot["knowledge_refs"]["fact_ids"] == ["fact-1", "fact-2"]
    assert snapshot["knowledge_refs"]["evidence_ids"] == ["evidence-1", "evidence-2"]
    assert snapshot["knowledge_refs"]["conflict_ids"] == ["conflict-1", "conflict-2"]
    assert snapshot["knowledge_refs"]["section_pack_ids"] == ["pack-1", "pack-2"]
    assert snapshot["source_registry"]["src-1"]["url"] == "https://example.com/a"
    assert snapshot["source_registry"]["src-2"]["url"] == "https://example.com/b"
    assert [item["claim_id"] for item in snapshot["claims"]] == ["claim-1", "claim-2"]
    assert [item["fact_id"] for item in snapshot["facts"]] == ["fact-1", "fact-2"]
    assert [item["evidence_id"] for item in snapshot["evidence"]] == ["evidence-1", "evidence-2"]
    assert [item["conflict_id"] for item in snapshot["conflicts"]] == ["conflict-1", "conflict-2"]
    assert [item["pack_id"] for item in snapshot["section_evidence_packs"]] == ["pack-1", "pack-2"]
    assert snapshot["latest_coverage_snapshot"]["snapshot_id"] == "coverage-2"
    assert snapshot["latest_novelty_snapshot"]["snapshot_id"] == "novelty-2"
    assert [gap["gap_id"] for gap in snapshot["open_gaps"]] == ["gap-1"]
