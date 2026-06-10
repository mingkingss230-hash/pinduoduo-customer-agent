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
    "enable_turn_context": True,
    "enable_turn_context_log_only": True,
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
            "回复像真人店铺客服，简短自然，通常1句，最多2句；不要写长段解释，除非客户明确要求详细说明。",
            "不要使用emoji、Markdown标题、表格、加粗符号或列表式大段排版；直接按客服口吻回复客户。",
            "知识库或工具结果有明确答案时，直接短答，不要扩写营销话术，不要反复说抱歉、感谢理解、请放心等套话。",
            "不要每次都用“亲，您好”开头；同一会话里不要逐字重复同一句话。",
            "客户只说转人工、找人工、人工客服时，先尝试正常安抚并处理问题；只有确实需要人工执行动作、升级处理，或同一问题纠结超过3轮时，才转人工。",
            "售前、售中、售后都要按知识库和工具结果回答，不要自行编造功能、参数、赠品、时效、补偿或运费承担方案。",
            "商品未锁定时，不要猜商品；先询问客户要咨询哪一款商品。",
            "商品已锁定后，商品参数、功能、赠品、使用方法按知识库回答；没有明确答案就说暂未查询到，不要反复让客户看详情页。",
            "售后反馈要按当前售后规则和工具边界处理，必要时直接转人工，不要自行编补偿方案。",
            "不能承诺能帮客户改颜色、改备注、按备注发货、特殊包装、补发、退款金额、运费承担等平台或订单外动作。",
            "涉及退货运费时，不要说“运费险”，统一按退货包运费服务表达；是否赠送以当前商品知识和平台页面为准。",
            "不能出现第三方平台、导流、极限词、返现、隐私泄露等违规表达。",
            "客户问价格、质量、靠不靠谱时，可以统一用店铺既有口径，但不要夸大、不要编造机制。",
            "不要输出内部标签、思维过程、XML 标签或工具痕迹；只输出客户可见内容。",
            "不要编造商品机制或参数；没有知识库明确依据时，禁止说滤网太脏、电机需要预热、电池多少毫安、能撑几天、能制冷等内容。",
            "客户投诉续航短、掉电快、风力小、没风、声音大、开最大档等问题时，不要继续泛泛推荐调档或清理，优先按售后场景处理。",
            "涉运费退款时，不要说“运费险”，退货包运费服务按当前商品知识和平台页面为准；售后优先按问题处理。",
            "如果同一客户在同一个问题上纠结3轮以上，优先转人工处理。",
            "有多个订单且状态不同，不能猜是哪一单，必须让客户发具体订单号。",
            "不得告诉客户具体到货日期，只能说大概几天能到，具体到货时间以实际物流为准。",
            "禁止提及晒图好评、好评返现、朋友圈、小红书、种草、发帖、返利、返现、红包、补偿换好评；本店没有晒图好评活动，不能引导客户去朋友圈或小红书发布内容。",
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
