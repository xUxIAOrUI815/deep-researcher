# core 核心逻辑模块说明

## 文件目录
```
├── config.py          # 配置管理
├── graph.py           # LangGraph 工作流图
├── knowledge.py       # 知识管理器
├── planner_logic.py   # 规划逻辑
├── router.py          # 路由器
├── state_manager.py   # 状态管理
├── content_transformer.py  # 结构化清洗模块
├── semantic_chunker.py     # 语义分块模块 
└── vector_store_qdrant.py  # 向量存储
```

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
  - ModelConfig：模型相关配置, 支持通过环境变量进行配置, 包含两个子配置: 
    - 主配置: DEFAULT_MODEL: 默认使用的模型
    - GLM子配置/DeepSeek子配置:
      - API_KEY: 从环境配置中获取
      - API_BASE: 
      - MODEL_NAME: 模型名称

### ContentTransformer模块
  结构化清洗与视图转换模块, 将原始的Markdown文本转换为结构化的内容, 包括元数据提取、内容清洗和章节提取
  - ContentMetadata: 存储内容的元数据信息
  - StructuredContent: 存储结构化的内容
  - ContentTransformer: 初始化转换器, 
    - 将原始的Markdown文本转换为结构化的内容
    - 从Markdown文本提取元数据
    - 清洗Markdown文本, 定义一系列需要跳过的模式
    - 提取Markdown文本中的章节标题
  
### Convergence
  收敛检查器，用于判断研究过程中是否应该结束(收敛)。通过分析当前研究状态，基于预设的规则和配置参数，判断研究是否收敛。
  - ConvergenceDecision: 存储收敛决策的结果
    - should_converge: 布尔值，是否收敛
    - reason: 字符串，说明收敛或不收敛的原因
    - action: 字符串，表示应该采取的行动 finish或continue  这个判断来自于planner_action 
    - skip_pending_tasks: 布尔值，表示是否跳过待办任务
  - ConvergenceChecker: 收敛检查器
    - check 方法: 判断条件
      修改planner的设计，先设置大纲，结合大纲和任务树共同确定任务是否终止。

### graph
  - GraphState: 研究图的状态结构
    - 任务树
    - 事实池
    - 原子事实
    - 令牌使用
    - 当前焦点任务
    - 根任务ID
    - 已完成的任务
    - 失败的任务
    - 消息列表
    - 原始抓取数据
    - 搜索结果
    - 最终报告
  - 节点函数
    - planner 节点
      - 和Agents/planner对比
        1. 这部分代码只初始化根节点，模拟消耗token，固定+100，假装做了规划，没有智能决策，只是初始化占位函数
    - researcher 节点
      - 执行研究任务，包括搜索和网页抓取
      - 实现
        - 查找待处理任务
        - 使用 MCPGateway 进行搜索
        - 使用 SmartScraper 抓取网页内容
        - 更新任务状态和收集数据
    - distiller 节点
      当前的任务是：从文章、网页中提取原子事实，生成摘要
      不依赖框架，可以自己执行
      - 核心模块
        - 初始化配置 信号量控制并发，线程锁保证日志/异常安全打印
        - 提示词工程
          - 只提取可验证事实
          - 实体明确
          - balabala
        - API 调用异常处理
        - 结果解析 
    - writer
      根据提取的原子事实生成调研简报 × 
      应该是深度调研报告
      - 处理原子事实，生成上下文
      - 构建提示词生成报告 调用 DeepSeek API 生成报告
      - 处理API调用错误和速率限制
      - 没有API密钥时生成模拟报告
  - 控制函数
    决定工作流是否应该继续
    检查是否存在待处理的任务
  - 创建研究图
    添加节点和编译图
  - SQLite存储
    保存工作流状态
    创建并返回 AsyncSqliteSaver
  - 研究周期
    - 初始化SQLite存储
    - 创建研究图
    - 设置初始状态
    - 运行工作流
    - 打印结果统计
    - 关闭数据库连接并返回结果

### knowledge
知识管理系统，用于存储、管理和检索研究过程中提取的原子事实，并提供了事实冲突检测、相似性搜索等功能

用distillerAgent来调用knowledge?
- FactStatus 定义了事实的枚举状态
  - ACTIVE ：活跃状态，新添加的事实
  - VERIFIED ：已验证状态，由多个来源确认的事实
  - CONFLICTING ：冲突状态，与其他事实存在矛盾
  - SUPERSEDED ：被取代状态，被新事实取代的旧事实
- EmbeddingModel 嵌入模型
  - 生成文本的向量嵌入，用于相似性计算
- FactConflict
  - 字段设计有问题
  - fact_id_1 ：第一个冲突事实的 ID
  - fact_id_2 ：第二个冲突事实的 ID
  - conflict_description ：冲突描述
  - detected_at ：冲突检测时间
- KnowledgeStatus
  - 存储知识管理系统的统计信息
  - total_facts ：总事实数
  - verified_facts ：已验证事实数
  - conflicting_facts ：冲突事实数
  - conflicts_detected ：检测到的冲突数
  - duplicates_merged ：合并的重复数
- StoredFact
  - 存储带有嵌入的原子事实
  - 生成唯一 ID
  - 存储原子事实和其嵌入向量
  - 跟踪事实状态
  - 提供转换为字典的方法，用于存储
- KnowledgeManager 知识管理类
  - 管理事实的存储、检索、冲突检测
    - 设置存储路径
    - 初始化嵌入模型
    - 创建事实字典
    - 设置并发限制信号量
    - 初始化统计信息
    - 加载现有数据 
  - 加载和保存方法
    - JSON 文件存储数据
    - 加载时重建StoredFact对象
    - 保存时将StoredFact对象转换为字典
  - 相似性计算方法
    - 利用向量相似度并查找相似事实
    - 遍历？
    - 为什么不用向量数据库
  - 添加事实方法
    - 添加事实并处理重复和冲突
      - 批量添加事实，用信号量控制并发
      - 处理重复事实
      - 处理验证事实
      - 处理冲突事实
      - 添加新事实
  - 搜索和查询方法
    - 当前阈值设置为 0 ？
  - 管理方法
    - 删除单个事实
    - 清空整个事实集合

### planner_logic
信息饱和度检查
- SaturationResult 数据类
  - 存储信息饱和度检查的结果
    - is_sataturated: 是否饱和
    - repetition_rate: 重复率 相似事实数/总检查事实数
    - new_facts_count: 新事物的数量
    - similar_count: 相似事实数
    - forced_finish: 是否强制结束
  - SaturationChecker 饱和度检查器类
    - 接受 KnowledgeManeger 实例，用于查找相似事实
    - 检查新收集的事实是否与已有的事实重复，判断是否达到信息饱和
      - new_facts: 新收集的事实列表
      - collection_name: 集合名称
      - is_user_triggered: 是否由用户触发
    - 实现逻辑
      如果没有新事实，返回未饱和状态
      如果是用户触发的任务，跳过饱和度检查
      遍历每个新事实，使用 KnowledgeManager 查找相似事实 (这里是遍历会有什么问题？)
      计算重复率
      如果重复率超过阈值，认为达到信息饱和(即当前能找到的信息已经没有新的价值)
      返回饱和度检查结果

### router
路由器功能，决定研究工作流的下一步操作，基于ConvergenceChecker的决策结果
- should_continue
  - 根据当前研究状态，决定工作流的下一步操作
- graph.py 也有 should_continue 方法
  - 比较
  
  