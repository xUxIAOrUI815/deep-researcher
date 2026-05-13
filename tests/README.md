# Tests

Current test layout is centered on the active `session_knowledge` graph flow.

Recommended fast offline suite:

```bash
pytest tests/test_researcher_offline.py tests/test_distiller_offline.py tests/test_writer_offline.py tests/test_graph_offline_smoke.py -q
```

Session knowledge suite:

```bash
pytest tests/test_session_knowledge_store.py tests/test_knowledge_manager_integration.py tests/test_session_knowledge_flow.py tests/test_planner_session_consumption.py tests/test_researcher_session_dedup.py tests/test_writer_session_pack_priority.py tests/test_session_resume_recovery.py tests/test_graph_session_e2e_offline.py -q
```

Notes:

- Legacy tests tied to `core/knowledge.py` and the pre-session graph contract were removed.
- `test_graph_session_e2e_offline.py` and `test_session_resume_recovery.py` cover the session-backed graph path.
- `test_offline_pipeline.py` and agent offline tests remain useful as smoke coverage for isolated stages.
