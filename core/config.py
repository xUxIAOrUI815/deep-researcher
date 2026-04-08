import os


class ResearchConfig:
    MAX_DEPTH: int = 3
    MAX_NODES: int = 15
    MAX_FACTS: int = 30

    SATURATION_SIMILARITY_THRESHOLD: float = 0.85

    INFO_GAIN_THRESHOLD: float = 0.20

    TASK_SIMILARITY_THRESHOLD: float = 0.70

    FACT_SIMILARITY_THRESHOLD: float = 0.70

    MIN_SEARCH_RESULTS: int = 3
    MAX_SEARCH_RESULTS: int = 5
    MAX_FACTS_PER_CYCLE: int = 10


class QdrantConfig:
    HOST: str = os.getenv("QDRANT_HOST", "localhost")
    PORT: int = int(os.getenv("QDRANT_PORT", "6333"))
    GRPC_PORT: int = int(os.getenv("QDRANT_GRPC_PORT", "6334"))
    USE_QDRANT: bool = os.getenv("USE_QDRANT", "false").lower() == "true"
    COLLECTION_VECTOR_SIZE: int = 1024


class ModelConfig:
    DEFAULT_MODEL: str = os.getenv("DEFAULT_MODEL", "deepseek")
    
    class GLM:
        API_KEY: str = os.getenv("GLM_API_KEY", "")
        API_BASE: str = os.getenv("GLM_API_BASE", "https://open.bigmodel.cn/api/paas/v4")
        MODEL_NAME: str = os.getenv("GLM_MODEL_NAME", "glm-4-flash")
    
    class DeepSeek:
        API_KEY: str = os.getenv("DEEPSEEK_API_KEY", "")
        API_BASE: str = os.getenv("DEEPSEEK_API_BASE", "https://api.deepseek.com")
        MODEL_NAME: str = os.getenv("DEEPSEEK_MODEL_NAME", "deepseek-chat")
