# `core/graph.py` 调度流程图

下面的流程图按当前 `core/graph.py` v2 调度逻辑整理，起点从前端用户提交研究问题开始。

```mermaid
flowchart TD
    A[前端用户输入研究问题] --> B[后端接口接收 user_query]
    B --> C[调用 run_research_cycle(initial_query)]
    C --> D["init_sqlite_saver 初始化 SQLite Checkpointer"]
    D --> E[create_research_graph 构建 LangGraph]
    E --> F[组装 initial_state 和 run_metadata]
    F --> G[graph.ainvoke(initial_state, config)]

    G --> H[planner 节点]
    H --> H1[_ensure_state_defaults 补齐状态]
    H1 --> H2[_ensure_root_task 创建或确认根任务]
    H2 --> H3[_call_planner_agent 调用 agents/planner.py]
    H3 --> H4[_apply_task_tree_patches 写回任务树]
    H4 --> I{route_after_planner}

    I -->|CONTINUE_RESEARCH 且有 pending task| J[researcher 节点]
    I -->|START_WRITING| N[writer 节点]
    I -->|STOP| Z[结束]

    J --> J1[选择 active_task_id]
    J1 --> J2[_set_task_status -> running]
    J2 --> J3[_call_researcher_agent 调用 agents/researcher.py]
    J3 --> K[distiller 节点]

    K --> K1[_call_distiller_agent 调用 agents/distiller.py]
    K1 --> K2[SESSION_KNOWLEDGE_MANAGER.process_distiller_output]
    K2 --> K3[写回 atomic_facts / section_evidence_packs / knowledge_refs]
    K3 --> K4[_set_task_status -> completed]
    K4 --> H

    N --> N1[_call_writer_agent 调用 agents/writer.py]
    N1 --> N2[生成 final_report]
    N2 --> N3[记录 run completed 事件]
    N3 --> O[返回最终状态]
```

说明：

- `planner -> researcher -> distiller -> planner` 构成主循环，直到规划器判断“证据已足够写作”或“停止”。
- `writer` 只在两种情况下进入：一是 `planner` 明确返回 `START_WRITING`，二是没有待研究任务但仍可基于已有材料成稿。
- 这个流程图以 `core/graph.py` 为中心抽象了“前端 -> 后端 -> 图调度”的入口；真正的 HTTP / WebSocket 控制器如果在别处实现，需要再把那层接到 `run_research_cycle(...)` 或 `graph.ainvoke(...)` 上。
