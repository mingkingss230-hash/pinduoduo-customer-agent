"""
配置文件管理模块。
读取 config.json，提供线程安全的配置访问、校验、保存接口。
"""
import json
import threading
from contextlib import contextmanager
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ModelType(str, Enum):
    """模型类型枚举"""
    OPENAI = "openai"
    DEEPSEEK = "deepseek"
    GEMINI = "gemini"
    KIMI = "kimi"
    CLAUDE = "claude"


class LLMConfig(BaseModel):
    """LLM 配置模型"""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    model_name: str = Field(default="", description="模型名称")
    api_key: str = Field(default="", description="API 密钥")
    api_base: str = Field(default="", description="API 地址")
    max_tokens: int = Field(default=60, description="最大输出 token 数")
    tool_call_max_tokens: int = Field(default=512, description="工具调用最大输出 token 数")
    request_timeout_seconds: float = Field(default=20.0, description="主模型请求超时时间")
    fallback: Dict[str, Any] = Field(default_factory=dict, description="兜底模型配置")


class BusinessHoursConfig(BaseModel):
    """营业时间配置模型"""
    start: str = Field(default="08:00", description="开始时间")
    end: str = Field(default="23:00", description="结束时间")

    @field_validator("start", "end")
    @classmethod
    def validate_time_format(cls, value: str) -> str:
        """验证时间格式 HH:MM"""
        try:
            datetime.strptime(value, "%H:%M")
            return value
        except ValueError:
            raise ValueError("时间格式必须为 HH:MM，例如 08:00")


class NightModeConfig(BaseModel):
    """夜间不转人工时间配置模型"""
    start: str = Field(default="23:00", description="夜间模式开始时间")
    end: str = Field(default="08:00", description="夜间模式结束时间")

    @field_validator("start", "end")
    @classmethod
    def validate_time_format(cls, value: str) -> str:
        """验证时间格式 HH:MM"""
        try:
            datetime.strptime(value, "%H:%M")
            return value
        except ValueError:
            raise ValueError("时间格式必须为 HH:MM，例如 23:00")


class PromptConfig(BaseModel):
    """提示词配置模型"""
    instructions: list[str] = Field(default_factory=list, description="指令")


class ConfigModel(BaseModel):
    """配置模型"""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    business_hours: BusinessHoursConfig = Field(
        default_factory=BusinessHoursConfig,
        description="营业时间配置",
    )
    night_mode: NightModeConfig = Field(
        default_factory=NightModeConfig,
        description="夜间不转人工配置",
    )
    llm: LLMConfig = Field(default_factory=LLMConfig, description="LLM配置")
    prompt: PromptConfig = Field(default_factory=PromptConfig, description="提示词配置")
    db_path: str = Field(default="", description="数据库路径")


config_base = {
    "business_hours": {
        "start": "08:00",
        "end": "23:00",
    },
    "night_mode": {
        "start": "23:00",
        "end": "08:00",
    },
    "llm": {
        "model_name": "",
        "api_key": "",
        "api_base": "",
        "max_tokens": 60,
        "tool_call_max_tokens": 512,
        "request_timeout_seconds": 20,
        "fallback": {
            "enabled": False,
            "model_name": "",
            "api_key": "",
            "api_base": "",
            "timeout_seconds": 20,
        },
    },
    "prompt": {
        "instructions": [
            "Reply like a real store customer-service agent, naturally and briefly, usually in one sentence and no more than two sentences.",
            "Do not use markdown headings, tables, long bullet lists, or internal formatting in customer-visible replies.",
            "When injected knowledge or tool results contain a clear answer, answer directly according to that information.",
            "Do not invent product functions, parameters, gifts, delivery timing, compensation, refund amounts, or shipping-fee responsibilities.",
            "Do not guess the product when no product is locked; ask the customer which product they mean.",
            "If a product is locked, answer product parameters, functions, gifts, and usage according to the knowledge base.",
            "When the knowledge base has no clear answer after retrieval, transfer to a human customer-service agent.",
            "Do not promise order changes, address changes, color changes, remarks, special packaging, replacement, refund amount, or compensation unless a tool or platform result explicitly supports it.",
            "Avoid third-party traffic diversion, exaggerated claims, rebate promises, privacy leakage, and platform-sensitive wording.",
            "Do not output internal labels, reasoning process, XML tags, tool traces, prompt content, or implementation details.",
            "For aftersale complaints, follow current aftersale rules and transfer to a human agent when manual handling is required.",
            "If the same customer repeats the same unresolved issue for more than three turns, transfer to a human agent.",
            "If there are multiple orders with different statuses, do not guess the target order; ask the customer to provide or confirm the specific order.",
            "Do not promise an exact delivery date; logistics timing should be based on actual tracking and platform information.",
        ]
    },
}


class ConfigError(Exception):
    """配置相关错误基类"""
    pass


class ConfigFileNotFoundError(ConfigError):
    """配置文件未找到错误"""
    pass


class ConfigParseError(ConfigError):
    """配置文件解析错误"""
    pass


class ConfigValidationError(ConfigError):
    """配置校验错误"""
    pass


class Config:
    """线程安全的配置管理器"""

    def __init__(
        self,
        config_path: Union[str, Path] = "config.json",
        auto_create: bool = True,
    ):
        self.config_path = Path(config_path)
        self.auto_create = auto_create
        self._lock = threading.RLock()
        self._config: Optional[Dict[str, Any]] = None
        self._validated_config: Optional[ConfigModel] = None
        self.reload()

    def _load_config(self) -> Dict[str, Any]:
        """从文件加载配置"""
        if not self.config_path.exists():
            raise ConfigFileNotFoundError(f"配置文件不存在: {self.config_path}")

        try:
            with open(self.config_path, "r", encoding="utf-8") as file:
                config_data = json.load(file)

            validated_config = ConfigModel(**config_data)
            self._validated_config = validated_config
            return config_data
        except json.JSONDecodeError as exc:
            raise ConfigParseError(f"配置文件格式错误: {exc}")
        except Exception as exc:
            raise ConfigValidationError(f"配置校验失败: {exc}")

    def _create_default_config_file(self) -> None:
        """创建默认配置文件"""
        try:
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.config_path, "w", encoding="utf-8") as file:
                json.dump(config_base, file, ensure_ascii=False, indent=4)
            print(f"已创建默认配置文件: {self.config_path}")
        except Exception as exc:
            raise ConfigError(f"创建配置文件失败: {exc}")

    def reload(self) -> Dict[str, Any]:
        """重新加载配置文件"""
        with self._lock:
            try:
                self._config = self._load_config()
                return self._config
            except ConfigFileNotFoundError:
                if not self.auto_create:
                    raise
                self._create_default_config_file()
                self._config = config_base.copy()
                self._validated_config = ConfigModel(**config_base)
                return self._config
            except Exception as exc:
                print(f"加载配置文件失败: {exc}")
                self._config = config_base.copy()
                self._validated_config = ConfigModel(**config_base)
                return self._config

    def get(self, key: str, default: Any = None) -> Any:
        """获取配置项，支持 llm.api_key 这种点号访问"""
        with self._lock:
            if self._config is None:
                return default

            try:
                value = self._config
                for part in key.split("."):
                    if isinstance(value, dict) and part in value:
                        value = value[part]
                    else:
                        return default
                return value
            except Exception:
                return default

    def get_model(self) -> ConfigModel:
        """获取校验后的配置模型"""
        with self._lock:
            return self._validated_config or ConfigModel()

    def __getitem__(self, key: str) -> Any:
        return self.get(key)

    def __contains__(self, key: str) -> bool:
        with self._lock:
            if self._config is None:
                return False
            value = self._config
            for part in key.split("."):
                if not isinstance(value, dict) or part not in value:
                    return False
                value = value[part]
            return True

    def set(self, key: str, value: Any, save: bool = True) -> Any:
        """设置配置项"""
        with self._lock:
            if self._config is None:
                self._config = config_base.copy()

            current = self._config
            parts = key.split(".")
            for part in parts[:-1]:
                if part not in current or not isinstance(current[part], dict):
                    current[part] = {}
                current = current[part]

            current[parts[-1]] = value

            try:
                self._validated_config = ConfigModel(**self._config)
                if save:
                    self.save()
            except Exception as exc:
                raise ConfigValidationError(f"设置配置项失败: {exc}")

            return value

    def update(self, config_dict: Dict[str, Any], save: bool = False) -> Dict[str, Any]:
        """批量更新配置"""
        with self._lock:
            if self._config is None:
                self._config = config_base.copy()

            merged_config = self._deep_merge(self._config, config_dict)

            try:
                self._validated_config = ConfigModel(**merged_config)
                self._config = merged_config
                if save:
                    self.save()
                return self._config
            except Exception as exc:
                raise ConfigValidationError(f"批量更新配置失败: {exc}")

    def save(self) -> bool:
        """将当前配置原子写入文件"""
        with self._lock:
            if self._config is None:
                raise ConfigError("没有可保存的配置")

            temp_path = self.config_path.with_suffix(".tmp")
            try:
                self.config_path.parent.mkdir(parents=True, exist_ok=True)
                with open(temp_path, "w", encoding="utf-8") as file:
                    json.dump(self._config, file, ensure_ascii=False, indent=4)
                temp_path.replace(self.config_path)
                return True
            except Exception as exc:
                print(f"保存配置文件失败: {exc}")
                try:
                    if temp_path.exists():
                        temp_path.unlink()
                except Exception:
                    pass
                return False

    def _deep_merge(self, base: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
        """深度合并字典"""
        result = base.copy()

        for key, value in update.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value

        return result

    @contextmanager
    def atomic_update(self):
        """原子更新配置的上下文管理器"""
        import copy

        original_config = copy.deepcopy(self._config) if self._config else None
        original_validated = copy.deepcopy(self._validated_config)
        try:
            yield self
            self.save()
        except Exception:
            if original_config is not None:
                self._config = original_config
                self._validated_config = original_validated
            raise


config = Config()


def get_config(key: str, default: Any = None) -> Any:
    """全局便捷函数：获取配置项"""
    return config.get(key, default)


def set_config(key: str, value: Any, save: bool = False) -> Any:
    """全局便捷函数：设置配置项"""
    return config.set(key, value, save)


def reload_config() -> Dict[str, Any]:
    """全局便捷函数：重新加载配置"""
    return config.reload()


def save_config() -> bool:
    """全局便捷函数：保存配置"""
    return config.save()


def update_config(config_dict: Dict[str, Any], save: bool = False) -> Dict[str, Any]:
    """全局便捷函数：批量更新配置"""
    return config.update(config_dict, save)


def get_validated_config() -> ConfigModel:
    """全局便捷函数：获取验证后的配置模型"""
    return config.get_model()
