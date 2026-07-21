from __future__ import annotations

import importlib
import sys
from types import ModuleType

import pytest

from core.context_builders import PlannerContextBuilder, ResearcherContextBuilder, WriterContextBuilder
from core.observability import NoopObserver, get_observer, set_observer
from core.session_knowledge import KnowledgeManager
from core.session_retrieval import SessionRetrievalService


@pytest.fixture(autouse=True)
def isolate_process_global_runtime(tmp_path):
    """Prevent one test's graph/observer globals from contaminating the next."""

    graph_module = importlib.import_module("core.graph")
    graph_global_dicts: list[dict] = [graph_module.__dict__]
    seen_global_dict_ids = {id(graph_module.__dict__)}
    for loaded_module in list(sys.modules.values()):
        if loaded_module is None:
            continue
        for value in list(vars(loaded_module).values()):
            candidate: dict | None = None
            if isinstance(value, ModuleType) and value.__name__ == "core.graph":
                candidate = value.__dict__
            elif getattr(value, "__module__", None) == "core.graph":
                candidate = getattr(value, "__globals__", None)
            if candidate is not None and id(candidate) not in seen_global_dict_ids:
                seen_global_dict_ids.add(id(candidate))
                graph_global_dicts.append(candidate)
    dependency_names = (
        "SESSION_KNOWLEDGE_MANAGER",
        "SESSION_RETRIEVAL_SERVICE",
        "PLANNER_CONTEXT_BUILDER",
        "RESEARCHER_CONTEXT_BUILDER",
        "WRITER_CONTEXT_BUILDER",
    )
    original_graph_dependencies = [
        (global_dict, tuple(global_dict[name] for name in dependency_names))
        for global_dict in graph_global_dicts
    ]
    original_observer = get_observer()
    manager = KnowledgeManager(base_storage_path=str(tmp_path), sqlite_filename=":memory:")
    retrieval = SessionRetrievalService(manager)
    dependencies = (
        manager,
        retrieval,
        PlannerContextBuilder(retrieval),
        ResearcherContextBuilder(retrieval),
        WriterContextBuilder(retrieval),
    )
    for global_dict in graph_global_dicts:
        global_dict.update(zip(dependency_names, dependencies))
    set_observer(NoopObserver())
    try:
        yield
    finally:
        for global_dict, original_dependencies in original_graph_dependencies:
            global_dict.update(zip(dependency_names, original_dependencies))
        set_observer(original_observer)
        manager.close()
