from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from console_app.app import create_app
from deep_researcher.artifacts import SQLiteArtifactStore
from deep_researcher.contracts import (
    ArtifactKind,
    AtomicFact,
    Budget,
    BudgetUsage,
    Citation,
    Claim,
    ComponentKind,
    Conflict,
    ConflictSeverity,
    EntityProvenance,
    ErrorCategory,
    ErrorRecord,
    EventType,
    Evidence,
    EvidenceRelation,
    Passage,
    Report,
    RunStatus,
    Section,
    SnapshotStatus,
    Source,
    SourceLevel,
    SourceSnapshot,
    SourceStatus,
    SourceType,
    SpanKind,
    TaskEnvelope,
    TaskKind,
    TaskStatus,
    VersionRef,
    utc_now,
)
from deep_researcher.events import SQLiteEventStore
from deep_researcher.knowledge import KnowledgeRepository, SQLiteKnowledgeStorage
from deep_researcher.orchestration import (
    NativeEventSourcedScheduler,
    RunControl,
    SchedulerEvent,
    SchedulerEventType,
    SQLiteSchedulerStore,
    TaskRecord,
)
from deep_researcher.studio import StudioV2Service
from deep_researcher.version_registry import (
    SQLiteVersionRegistryStore,
    VersionRegistry,
)
from tests.test_bg001_event_store import _event


def _budget() -> Budget:
    return Budget(
        max_tokens=1000,
        max_cost_usd=2,
        max_wall_time_seconds=60,
        max_model_calls=10,
        max_tool_calls=10,
        max_search_calls=10,
        max_retries=3,
        max_errors=3,
    )


def _task(
    suffix: str,
    *,
    run_id: str,
    parent: str | None = None,
    dependencies: tuple[str, ...] = (),
    priority: float = 0.5,
) -> TaskEnvelope:
    return TaskEnvelope(
        task_id=f"task_{suffix}",
        run_id=run_id,
        parent_task_id=parent,
        dependency_task_ids=dependencies,
        kind=TaskKind.RESEARCH,
        title=f"Research {suffix}",
        goal=f"Resolve the bounded question for {suffix}.",
        expected_output_schema="TaskResult@1",
        budget=_budget(),
        priority=priority,
        max_attempts=3,
        created_by="agent_supervisor",
    )


def _service(tmp_path, *, run_id: str):
    artifacts = SQLiteArtifactStore(tmp_path / "artifacts.sqlite3")
    knowledge = SQLiteKnowledgeStorage(
        tmp_path / "knowledge.sqlite3",
        artifact_store=artifacts,
    )
    events = SQLiteEventStore(tmp_path / "events.sqlite3")
    scheduler_store = SQLiteSchedulerStore(tmp_path / "scheduler.sqlite3")
    versions = SQLiteVersionRegistryStore(tmp_path / "versions.sqlite3")
    started = _event(
        1,
        EventType.RUN_STARTED,
        run_id=run_id,
        event_id=f"event_{run_id}_started",
    )
    events.append(started)
    service = StudioV2Service(
        event_store=events,
        scheduler_store=scheduler_store,
        knowledge_repository=KnowledgeRepository(knowledge),
        artifact_store=artifacts,
        version_store=versions,
    )
    return service, artifacts, knowledge, events, scheduler_store, versions, started


def _close(artifacts, knowledge, events, scheduler, versions) -> None:
    versions.close()
    scheduler.close()
    knowledge.close()
    artifacts.close()
    events.close()


@pytest.mark.asyncio
async def test_task_dag_covers_split_merge_skip_fail_retry_dependencies_and_rebuildable_diffs(
    tmp_path,
):
    run_id = "run_studio_v2_tasks"
    (
        service,
        artifacts,
        knowledge,
        events,
        scheduler_store,
        versions,
        _,
    ) = _service(tmp_path, run_id=run_id)
    scheduler = NativeEventSourcedScheduler(scheduler_store)
    try:
        await scheduler.create_run(
            run_id,
            max_concurrency=1,
            actor_id="agent_supervisor",
            mutation_id="mutation_studio_run",
        )
        parent = await scheduler.submit(
            _task("studio_parent", run_id=run_id),
            actor_id="agent_supervisor",
            mutation_id="mutation_studio_parent",
        )
        child_a = _task(
            "studio_child_a",
            run_id=run_id,
            parent=parent.task_id,
        )
        child_b = _task(
            "studio_child_b",
            run_id=run_id,
            parent=parent.task_id,
            dependencies=(child_a.task_id,),
        )
        await scheduler.split(
            parent.task_id,
            (child_a, child_b),
            actor_id="agent_supervisor",
            mutation_id="mutation_studio_split",
        )
        await scheduler.prune(
            child_b.task_id,
            reason="duplicate branch",
            actor_id="agent_supervisor",
            mutation_id="mutation_studio_prune",
        )
        merge_source = await scheduler.submit(
            _task("studio_merge_source", run_id=run_id),
            actor_id="agent_supervisor",
            mutation_id="mutation_studio_merge_source",
        )
        merge_target = await scheduler.submit(
            _task("studio_merge_target", run_id=run_id),
            actor_id="agent_supervisor",
            mutation_id="mutation_studio_merge_target",
        )
        await scheduler.merge(
            merge_source.task_id,
            merge_target.task_id,
            actor_id="agent_supervisor",
            mutation_id="mutation_studio_merge",
        )
        retry_task = await scheduler.submit(
            _task(
                "studio_retry",
                run_id=run_id,
                priority=1.0,
            ),
            actor_id="agent_supervisor",
            mutation_id="mutation_studio_retry_submit",
        )
        lease = (
            await scheduler.claim(
                run_id,
                worker_id="agent_worker",
                limit=1,
                lease_seconds=30,
                mutation_id="mutation_studio_claim",
            )
        )[0]
        assert lease.task.task_id == retry_task.task_id
        await scheduler.fail(
            retry_task.task_id,
            error_ref="error_studio_transient",
            worker_id="agent_worker",
            mutation_id="mutation_studio_fail",
            usage=BudgetUsage(
                input_tokens=12,
                output_tokens=4,
                cost_usd=0.03,
                model_calls=1,
            ),
        )
        await scheduler.retry(
            retry_task.task_id,
            actor_id="agent_supervisor",
            mutation_id="mutation_studio_retry",
        )

        nodes = []
        edges = []
        cursor = None
        while True:
            page = service.task_graph(
                run_id,
                cursor=cursor,
                limit=2,
            )
            nodes.extend(page.nodes)
            edges.extend(page.edges)
            if page.next_cursor is None:
                break
            cursor = page.next_cursor
        assert len({item.node_id for item in nodes}) == len(nodes)
        assert {"split", "merge", "dependency"} <= {
            item.edge_type for item in edges
        }
        pruned = next(item for item in nodes if item.node_id == child_b.task_id)
        retried = next(item for item in nodes if item.node_id == retry_task.task_id)
        assert pruned.status == TaskStatus.PRUNED.value
        assert {
            item["event_type"] for item in retried.data["operations"]
        } >= {"task_failed", "task_retried"}
        assert all(item.links for item in (*nodes, *edges))

        before_rebuild = service.scheduler_state_diffs(
            run_id,
            limit=1000,
        ).model_dump(mode="json")
        first_diff_page = service.scheduler_state_diffs(
            run_id,
            limit=2,
        )
        assert first_diff_page.next_after_sequence is not None
        second_diff_page = service.scheduler_state_diffs(
            run_id,
            after_sequence=first_diff_page.next_after_sequence,
            limit=2,
        )
        assert {
            item.event_id for item in first_diff_page.items
        }.isdisjoint(
            item.event_id for item in second_diff_page.items
        )
        assert {
            change.entity_kind
            for item in service.scheduler_state_diffs(
                run_id,
                limit=1000,
            ).items
            for change in item.changes
        } >= {"task", "budget"}
        scheduler_store.rebuild_projection(run_id)
        after_rebuild = service.scheduler_state_diffs(
            run_id,
            limit=1000,
        ).model_dump(mode="json")
        assert after_rebuild == before_rebuild
        assert service.metrics(run_id).budget_health
        scheduler_chain = service.scheduler_error_retry_chain(run_id)
        assert [
            item.event_type for item in scheduler_chain.items
        ] == ["task_failed", "task_retried"]
        assert (
            scheduler_chain.items[1].retry_of_event_id
            == scheduler_chain.items[0].event_id
        )
    finally:
        _close(artifacts, knowledge, events, scheduler_store, versions)


def _seed_evidence_graph(
    artifacts: SQLiteArtifactStore,
    repository: KnowledgeRepository,
    *,
    run_id: str,
    causation_event_id: str,
):
    now = utc_now()
    snapshot_artifact = artifacts.put_text(
        "A primary source states the measured result.",
        kind=ArtifactKind.SOURCE_SNAPSHOT,
        producer_id="tool_scraper",
        run_id=run_id,
        task_id="task_evidence",
        content_schema="SourceSnapshotText@1",
    )
    passage_artifact = artifacts.put_text(
        "A primary source states the measured result.",
        kind=ArtifactKind.CLEANED_CONTENT,
        producer_id="agent_researcher",
        run_id=run_id,
        task_id="task_evidence",
        content_schema="CleanedContent@1",
        source_artifact_ids=(snapshot_artifact.artifact_id,),
    )
    provenance = EntityProvenance(
        producer_id="agent_distiller",
        run_id=run_id,
        task_id="task_evidence",
        causation_event_id=causation_event_id,
        source_artifact_ids=(passage_artifact.artifact_id,),
    )
    source = Source(
        source_id="source_studio_v2",
        canonical_url="https://example.test/primary",
        source_type=SourceType.PRIMARY,
        source_level=SourceLevel.PRIMARY,
        status=SourceStatus.ACCESSIBLE,
        title="Primary source",
        publisher="Example Lab",
        authority_score=0.95,
        metadata={"internal_debug": "never-project-this-value"},
        provenance=provenance,
    )
    snapshot = SourceSnapshot(
        snapshot_id="snapshot_studio_v2",
        source_id=source.source_id,
        artifact_id=snapshot_artifact.artifact_id,
        status=SnapshotStatus.NORMALIZED,
        source_level=SourceLevel.PRIMARY,
        source_version=1,
        content_hash=snapshot_artifact.content_hash,
        final_url=source.canonical_url,
        media_type="text/plain",
        http_status=200,
        fetched_at=now,
        capture_method="governed_scraper",
        provenance=provenance,
    )
    passage = Passage(
        passage_id="passage_studio_v2",
        snapshot_id=snapshot.snapshot_id,
        text_artifact_id=passage_artifact.artifact_id,
        ordinal=0,
        locator="chars:0-44",
        content_hash=passage_artifact.content_hash,
        extraction_method="dom_text_v1",
        extracted_at=now,
        char_start=0,
        char_end=44,
        language="en",
        provenance=provenance,
    )
    evidence = Evidence(
        evidence_id="evidence_studio_v2",
        passage_ids=(passage.passage_id,),
        relation=EvidenceRelation.SUPPORTS,
        summary="The source supports the measured result.",
        confidence=0.92,
        relevance=0.96,
        source_quality=0.95,
        provenance=provenance,
    )
    fact = AtomicFact(
        fact_id="fact_studio_v2",
        statement="The measured result is reproducible.",
        evidence_ids=(evidence.evidence_id,),
        confidence=0.9,
        provenance=provenance,
    )
    claim_a = Claim(
        claim_id="claim_studio_v2_a",
        statement="The result is reproducible.",
        fact_ids=(fact.fact_id,),
        evidence_ids=(evidence.evidence_id,),
        confidence=0.9,
        provenance=provenance,
    )
    claim_b = Claim(
        claim_id="claim_studio_v2_b",
        statement="The result may not reproduce.",
        fact_ids=(fact.fact_id,),
        evidence_ids=(evidence.evidence_id,),
        confidence=0.55,
        provenance=provenance,
    )
    citation = Citation(
        citation_id="citation_studio_v2",
        claim_id=claim_a.claim_id,
        evidence_id=evidence.evidence_id,
        passage_id=passage.passage_id,
        snapshot_id=snapshot.snapshot_id,
        source_id=source.source_id,
        locator=passage.locator,
        quote="A primary source states the measured result.",
        extraction_method="dom_text_v1",
        provenance=provenance,
    )
    conflict = Conflict(
        conflict_id="conflict_studio_v2",
        claim_ids=(claim_a.claim_id, claim_b.claim_id),
        fact_ids=(fact.fact_id,),
        summary="The two claims disagree about reproducibility.",
        severity=ConflictSeverity.HIGH,
        high_impact=True,
        provenance=provenance,
    )
    section = Section(
        section_id="section_studio_v2",
        report_id="report_studio_v2",
        title="Findings",
        goal="Present verified findings.",
        order=0,
        claim_ids=(claim_a.claim_id, claim_b.claim_id),
        required_claim_ids=(claim_a.claim_id,),
        citation_ids=(citation.citation_id,),
        conflicted_claim_ids=(claim_a.claim_id, claim_b.claim_id),
        provenance=provenance,
    )
    report = Report(
        report_id="report_studio_v2",
        thread_id="thread_studio_v2",
        run_id=run_id,
        title="Studio evidence report",
        research_question="Is the result reproducible?",
        section_ids=(section.section_id,),
        provenance=provenance,
    )
    repository.save_graph(
        source,
        snapshot,
        passage,
        evidence,
        fact,
        claim_a,
        claim_b,
        citation,
        conflict,
        section,
        report,
    )
    return snapshot, conflict


def test_evidence_graph_snapshot_conflicts_event_diffs_errors_metrics_and_versions(
    tmp_path,
):
    run_id = "run_studio_v2_evidence"
    (
        service,
        artifacts,
        knowledge,
        events,
        scheduler_store,
        versions,
        started,
    ) = _service(tmp_path, run_id=run_id)
    try:
        snapshot, conflict = _seed_evidence_graph(
            artifacts,
            service.knowledge_repository,
            run_id=run_id,
            causation_event_id=started.event_id,
        )
        prompt_artifact = artifacts.put_text(
            "Use evidence-backed synthesis.",
            kind=ArtifactKind.PROMPT,
            producer_id="agent_release_manager",
            run_id=run_id,
            content_schema="Prompt@1",
        )
        prompt_ref = VersionRef(
            version_id="version_studio_v2_prompt",
            kind=ComponentKind.PROMPT,
            name="research-writer",
            version="2.0.0",
            artifact_id=prompt_artifact.artifact_id,
            content_hash=prompt_artifact.content_hash,
        )
        registry = VersionRegistry(
            store=versions,
            artifact_store=artifacts,
        )
        registry.register(prompt_ref)
        gate_artifact = artifacts.put_json(
            {"decision_id": "decision_studio_v2_release", "passed": True},
            redact=False,
            kind=ArtifactKind.RELEASE_GATE_DECISION,
            producer_id="evaluation_release_gate",
            run_id=run_id,
            content_schema="ReleaseGateDecision@1",
        )
        registry.promote(
            prompt_ref.version_id,
            gate_decision_id="decision_studio_v2_release",
            gate_decision_artifact_id=gate_artifact.artifact_id,
            reason="Semantic and non-regression gates passed.",
            actor_id="agent_release_manager",
            occurred_at=utc_now(),
        )
        error = ErrorRecord(
            error_id="error_studio_v2_provider",
            category=ErrorCategory.TRANSIENT_PROVIDER,
            code="provider_timeout",
            message="Provider timed out after the governed deadline.",
            retryable=True,
            attempt=1,
            actor_id="agent_researcher",
            task_id="task_evidence",
        )
        failed = _event(
            2,
            EventType.TASK_STATE_CHANGED,
            run_id=run_id,
            event_id="event_studio_v2_failed",
            error=error,
        ).model_copy(
            update={
                "task_id": "task_evidence",
                "attempt": 1,
            }
        )
        events.append(failed)
        retried = _event(
            3,
            EventType.RETRY_SCHEDULED,
            run_id=run_id,
            event_id="event_studio_v2_retry",
            payload={"reason": "retryable provider error"},
        ).model_copy(
            update={
                "task_id": "task_evidence",
                "attempt": 2,
                "causation_event_id": failed.event_id,
            }
        )
        events.append(retried)
        evidence_changed = _event(
            4,
            EventType.EVIDENCE_CHANGED,
            run_id=run_id,
            event_id="event_studio_v2_evidence_changed",
            payload={
                "subject_id": "claim_studio_v2_a",
                "change": "claim_evidence_verified",
                "state_changes": [
                    {
                        "entity_type": "Claim",
                        "entity_id": "claim_studio_v2_a",
                        "field": "status",
                        "before": "draft",
                        "after": "supported",
                    },
                    {
                        "entity_type": "Evidence",
                        "entity_id": "evidence_studio_v2",
                        "field": "status",
                        "before": "proposed",
                        "after": "verified",
                    },
                ],
            },
        ).model_copy(update={"task_id": "task_evidence"})
        events.append(evidence_changed)
        model_started = _event(
            5,
            EventType.MODEL_STARTED,
            run_id=run_id,
            event_id="event_studio_v2_model_started",
            span_id="span_studio_v2_model",
            parent_span_id="span_root",
            span_kind=SpanKind.MODEL,
        )
        events.append(model_started)
        model_completed = _event(
            6,
            EventType.MODEL_COMPLETED,
            run_id=run_id,
            event_id="event_studio_v2_model_completed",
            span_id="span_studio_v2_model",
            parent_span_id="span_root",
            span_kind=SpanKind.MODEL,
        ).model_copy(
            update={
                "usage": BudgetUsage(
                    input_tokens=120,
                    output_tokens=40,
                    cost_usd=0.08,
                    model_calls=1,
                ),
                "latency_ms": 320.0,
                "task_id": "task_evidence",
            }
        )
        events.append(model_completed)
        events.append(
            _event(
                7,
                EventType.RUN_COMPLETED,
                run_id=run_id,
                event_id="event_studio_v2_completed",
                status=RunStatus.SUCCEEDED,
            )
        )

        all_nodes = []
        all_edges = []
        cursor = None
        while True:
            page = service.evidence_graph(
                run_id,
                cursor=cursor,
                limit=3,
            )
            all_nodes.extend(page.nodes)
            all_edges.extend(page.edges)
            if page.next_cursor is None:
                break
            cursor = page.next_cursor
        assert {
            "Source",
            "SourceSnapshot",
            "Passage",
            "Evidence",
            "Claim",
            "Citation",
            "Conflict",
            "Section",
        } <= {item.node_type for item in all_nodes}
        assert {
            "claim_evidence",
            "citation_snapshot",
            "conflict_claim",
            "section_claim",
        } <= {item.edge_type for item in all_edges}
        assert all(item.links for item in (*all_nodes, *all_edges))
        assert "never-project-this-value" not in json.dumps(
            [
                item.model_dump(mode="json")
                for item in all_nodes
            ]
        )

        navigation = service.snapshot_navigation(snapshot.snapshot_id)
        assert navigation["source"]["source_id"] == "source_studio_v2"
        assert navigation["content_url"].endswith("/content")
        content, media_type, _ = service.artifact_content(
            snapshot.artifact_id
        )
        assert b"primary source" in content
        assert media_type.startswith("text/plain")

        conflicts = service.conflict_navigation(run_id)
        assert conflict.conflict_id in {
            item.node_id for item in conflicts.nodes
        }
        assert "claim_studio_v2_a" in {
            item.node_id for item in conflicts.nodes
        }

        diffs = service.evidence_state_diffs(run_id)
        assert {
            (change.entity_kind, change.field, change.after)
            for item in diffs.items
            for change in item.changes
        } >= {
            ("Claim", "status", "supported"),
            ("Evidence", "status", "verified"),
        }
        assert service.evidence_state_diffs(
            run_id,
            after_sequence=diffs.items[-1].sequence_no,
        ).items == ()

        chain = service.error_retry_chain(run_id)
        assert [item.event_type for item in chain.items] == [
            "task_state_changed",
            "retry_scheduled",
        ]
        assert chain.items[1].retry_of_event_id == failed.event_id
        assert all(item.links for item in chain.items)
        first_error_page = service.error_retry_chain(run_id, limit=1)
        assert first_error_page.next_after_sequence is not None
        assert service.error_retry_chain(
            run_id,
            after_sequence=first_error_page.next_after_sequence,
            limit=1,
        ).items[0].event_id == retried.event_id

        metrics = service.metrics(run_id)
        assert metrics.totals["total_tokens"] == 160
        assert metrics.totals["cost_usd"] == pytest.approx(0.08)
        assert metrics.totals["latency_ms"] == pytest.approx(320.0)
        assert {
            item.kind for item in metrics.component_versions
        } >= {"runtime", "scheduler"}
        assert any(
            item.scope == "registry_active"
            and item.version_id == prompt_ref.version_id
            for item in metrics.component_versions
        )
        assert all(item.links for item in metrics.component_versions)
    finally:
        _close(artifacts, knowledge, events, scheduler_store, versions)


def test_large_task_graph_is_incremental_and_cursor_is_view_bound(tmp_path):
    run_id = "run_studio_v2_large"
    (
        service,
        artifacts,
        knowledge,
        events,
        scheduler_store,
        versions,
        _,
    ) = _service(tmp_path, run_id=run_id)
    try:
        now = utc_now()
        records = tuple(
            TaskRecord(
                envelope=_task(
                    f"studio_large_{index:04d}",
                    run_id=run_id,
                ),
                available_at=now,
                updated_at=now,
            )
            for index in range(1205)
        )
        control = RunControl(
            run_id=run_id,
            projection_revision=1,
            updated_at=now,
            created_at=now,
        )
        scheduler_event = SchedulerEvent(
            event_id="event_studio_v2_large_batch",
            mutation_id="mutation_studio_v2_large_batch",
            fingerprint="a" * 64,
            run_id=run_id,
            sequence_no=1,
            event_type=SchedulerEventType.RUN_CREATED,
            actor_id="agent_supervisor",
            payload={
                "control": control.model_dump(mode="json"),
                "records": [
                    item.model_dump(mode="json") for item in records
                ],
            },
            occurred_at=now,
        )
        with scheduler_store.transaction() as connection:
            scheduler_store.write_control(control, connection=connection)
            for record in records:
                scheduler_store.write_task(record, connection=connection)
            scheduler_store.append_event(
                scheduler_event,
                connection=connection,
            )

        counts = []
        cursor = None
        first_cursor = None
        while True:
            page = service.task_graph(
                run_id,
                cursor=cursor,
                limit=500,
            )
            counts.append(len(page.nodes))
            if first_cursor is None:
                first_cursor = page.next_cursor
            if page.next_cursor is None:
                break
            cursor = page.next_cursor
        assert counts == [500, 500, 205]
        assert first_cursor is not None
        with pytest.raises(ValueError, match="another view"):
            service.evidence_graph(
                run_id,
                cursor=first_cursor,
                limit=1,
            )
    finally:
        _close(artifacts, knowledge, events, scheduler_store, versions)


def test_studio_v2_http_surface_and_shell_are_read_only(tmp_path):
    app = create_app(str(tmp_path / "console"))
    service = app.state.console_service
    run_id = "run_studio_v2_http"
    service.event_store.append(
        _event(
            1,
            EventType.RUN_STARTED,
            run_id=run_id,
            event_id="event_studio_v2_http_started",
        )
    )

    async def seed_scheduler() -> None:
        await service.runtime.scheduler.create_run(
            run_id,
            actor_id="agent_supervisor",
            mutation_id="mutation_studio_v2_http_run",
        )
        await service.runtime.scheduler.submit(
            _task("studio_v2_http", run_id=run_id),
            actor_id="agent_supervisor",
            mutation_id="mutation_studio_v2_http_task",
        )

    import asyncio

    asyncio.run(seed_scheduler())
    with TestClient(app) as client:
        shell = client.get(f"/studio/{run_id}")
        assert shell.status_code == 200
        assert "Task DAG" in shell.text
        assert "/static/studio_v2.js" in shell.text

        task_graph = client.get(
            f"/api/studio/v2/runs/{run_id}/task-graph"
        )
        assert task_graph.status_code == 200
        assert task_graph.json()["nodes"][0]["links"]
        metrics = client.get(
            f"/api/studio/v2/runs/{run_id}/metrics"
        )
        assert metrics.status_code == 200
        scheduler_event = client.get(
            f"/api/studio/v2/runs/{run_id}/scheduler-events/1"
        )
        assert scheduler_event.status_code == 200
        event = client.get(
            "/api/studio/v2/events/event_studio_v2_http_started"
        )
        assert event.status_code == 200

        assert client.post(
            f"/api/studio/v2/runs/{run_id}/task-graph"
        ).status_code == 405
