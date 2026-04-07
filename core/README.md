# core 核心逻辑模块说明

## 文件目录
├── config.py          # 配置管理
├── graph.py           # LangGraph 工作流图
├── knowledge.py       # 知识管理器
├── planner_logic.py   # 规划逻辑
├── router.py          # 路由器
├── state_manager.py   # 状态管理
├── content_transformer.py  # 结构化清洗模块
├── semantic_chunker.py     # 语义分块模块 
└── vector_store_qdrant.py  # 向量存储

## 模块功能介绍

### config.py
  配置管理模块，分为三个类：ResearchConfig、QdrantConfig、ModelConfig
  - ResearchConfig：研究相关配置，包括：
    - MAX_DEPTH: 最大深度, 研究的最大深度, 控制知识图谱的层级深度
    - MAX_NODE: 最大节点数, 最大节点数, 控制规模
    - MAX_FACTS: 最大事实数, 最大事实数, 控制收集的事实总量
    - SATURATION_SIMILARITY_THRESHOLD: 饱和度相似度阈值, 当新信息与现有信息相似度超过阈值时, 认为信息已饱和, 和MAX_DEPTH共同限制研究深度, 避免无限循环
    - INFO_GAIN_THRESHOLD: 信息增益阈值, 控制新信息的价值评估
    - TASK_SIMILARITY_THRESHOLD: 任务相似度阈值, 判断任务之间的相似性
    - FACT_SIMILARITY_THRESHOLD: 事实相似度阈值, 判断事实之间的相似性
    - MIN_SEARCH_RESULTS: 最小搜索结果, 确保由足够的信息来源
    - MAX_SEARCH_RESULTS: 最大搜索结果, 控制搜索结果的数量
    - MAX_FACTS_PER_CYCLE: 每个周期的最大事实数, 控制每个研究周期的信息处理量
  - QdrantConfig：Qdrant相关配置: 定义Qdrant向量数据库的配置参数，支持通过环境变量进行配置
    - HOST: 主机名
    - PORT: 端口号, 可以通过环境变量QDRANT_PORT覆盖
    - GRPC_PORT: GRPC端口号, 可以通过环境变量QDRANT_GRPC_PORT覆盖
    - USE_QDRANT: 是否使用Qdrant向量数据库, 可通过环境变量USE_QDRANT覆盖
    - COLLECTION_VECTOR_SIZE: 向量集合的维度大小, 用于存储嵌入向量