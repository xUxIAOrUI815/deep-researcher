# Observability

可观测性模块，追踪/记录系统运行过程中的各类事件(如任务执行、节点运行、证据生成等)，方便后续排查问题。

## EventLevel 事件级别

设置日志的常见级别，标识事件的严重程度

## EventType 事件类型

每个类型对应一个业务类型，便于追踪

## ObervabilityEvent

封装单次事件的所有元数据

### 核心方法

1. from_context
    从 RunContext 运行上下文中自动填充 research_id/run_id等链路信息
2. to_dict
    事件对象转为字典，自动处理枚举值、时间，方便存储/传输

## Observer 协议类

定义 事件观察者 的接口规范，实际观察者需要实现这些方法，保证接口一致性。

### emit

基础方法，发送任务 ObservabilityEvent 事件

### record_run_event
 
记录 运行 相关事件，自动关联 RunContext

### record_node_event

记录 节点 相关事件，需指定 node_name

### record_task_event

记录 任务 相关事件，需指定 task_id

### record_evidence_event

记录 证据 相关事件，关联证据/事实/来源

## NoopObserver

空操作观察者