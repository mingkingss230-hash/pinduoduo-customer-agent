"""
Agent 配置模块。
管理默认配置和运行时配置参数。
"""
from dataclasses import dataclass, field

from config import get_config
from utils.logger_loguru import get_logger

logger = get_logger("AgentConfig")

# 默认参数
DEFAULT_DB_PATH = "./temp/agent.db"
DEFAULT_TOKEN_WINDOW = 18432
DEFAULT_COMPRESS_RATIO = 16384 / 18432
DEFAULT_RETAIN_COUNT = 10
DEFAULT_MAX_LOOPS = 5
DEFAULT_TEMPERATURE = 0.3
DEFAULT_MAX_TOKENS = 60
DEFAULT_TOOL_CALL_MAX_TOKENS = 512
DEFAULT_LLM_TIMEOUT_SECONDS = 20.0
DEFAULT_LLM_MAX_CONCURRENT_REQUESTS = 2


@dataclass
class AgentConfig:
    """Agent 配置数据类。"""

    db_path: str = field(default_factory=lambda: get_config("db_path", DEFAULT_DB_PATH))
    token_window: int = field(default_factory=lambda: int(get_config("agent.token_window", DEFAULT_TOKEN_WINDOW)))
    compress_ratio: float = field(default_factory=lambda: float(get_config("agent.compress_ratio", DEFAULT_COMPRESS_RATIO)))
    retain_count: int = field(default_factory=lambda: int(get_config("agent.retain_count", DEFAULT_RETAIN_COUNT)))
    max_loops: int = field(default_factory=lambda: int(get_config("agent.max_loops", DEFAULT_MAX_LOOPS)))
    temperature: float = field(default_factory=lambda: float(get_config("agent.temperature", DEFAULT_TEMPERATURE)))
    max_tokens: int = field(default_factory=lambda: int(get_config("llm.max_tokens", DEFAULT_MAX_TOKENS)))
    tool_call_max_tokens: int = field(default_factory=lambda: int(get_config("llm.tool_call_max_tokens", DEFAULT_TOOL_CALL_MAX_TOKENS)))
    request_timeout_seconds: float = field(default_factory=lambda: float(get_config("llm.request_timeout_seconds", DEFAULT_LLM_TIMEOUT_SECONDS)))
    max_concurrent_requests: int = field(default_factory=lambda: int(get_config("llm.max_concurrent_requests", DEFAULT_LLM_MAX_CONCURRENT_REQUESTS)))

    model_name: str = field(default_factory=lambda: get_config("llm.model_name", "gpt-3.5-turbo"))
    api_key: str = field(default_factory=lambda: get_config("llm.api_key", ""))
    api_base: str = field(default_factory=lambda: get_config("llm.api_base", ""))
    fallback_enabled: bool = field(default_factory=lambda: bool(get_config("llm.fallback.enabled", False)))
    fallback_model_name: str = field(default_factory=lambda: get_config("llm.fallback.model_name", ""))
    fallback_api_key: str = field(default_factory=lambda: get_config("llm.fallback.api_key", ""))
    fallback_api_base: str = field(default_factory=lambda: get_config("llm.fallback.api_base", ""))
    fallback_timeout_seconds: float = field(default_factory=lambda: float(get_config("llm.fallback.timeout_seconds", DEFAULT_LLM_TIMEOUT_SECONDS)))

    @classmethod
    def load_from_config(cls) -> "AgentConfig":
        """从配置文件加载配置。"""
        config = cls()
        logger.debug("Agent 配置加载完成")
        return config

    def validate(self) -> bool:
        """验证配置有效性。"""
        if not self.api_key:
            logger.error("LLM API 密钥未配置")
            return False
        return True
