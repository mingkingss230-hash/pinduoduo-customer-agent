from PyQt6.QtCore import Qt, QTime
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QFrame,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)
from qfluentwidgets import (
    CardWidget,
    CaptionLabel,
    ComboBox,
    FluentIcon as FIF,
    InfoBar,
    InfoBarPosition,
    PasswordLineEdit,
    PrimaryPushButton,
    PushButton,
    ScrollArea,
    StrongBodyLabel,
    SubtitleLabel,
    TimePicker,
    LineEdit,
)

from config import config, config_base
from utils.logger_loguru import get_logger
from utils.runtime_path import get_resource_path


class LLMConfigCard(CardWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUI()

    def setupUI(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 16, 20, 16)
        layout.setSpacing(16)

        title_label = StrongBodyLabel("LLM模型配置")
        title_label.setFont(QFont("Microsoft YaHei", 12, QFont.Weight.Bold))
        layout.addWidget(title_label)

        form_layout = QFormLayout()
        form_layout.setSpacing(12)
        form_layout.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)
        form_layout.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)
        form_layout.setFormAlignment(Qt.AlignmentFlag.AlignLeft)

        self.api_base_edit = LineEdit()
        self.api_base_edit.setPlaceholderText("https://ark.cn-beijing.volces.com/api/v3")
        form_layout.addRow("API Base URL:", self.api_base_edit)

        self.api_key_edit = PasswordLineEdit()
        self.api_key_edit.setPlaceholderText("输入您的 API Key")
        form_layout.addRow("API Key:", self.api_key_edit)

        self.model_name_edit = LineEdit()
        self.model_name_edit.setPlaceholderText("输入模型名称")
        form_layout.addRow("模型名称:", self.model_name_edit)

        layout.addLayout(form_layout)

        description_label = CaptionLabel(
            "配置LLM模型的连接参数。支持OpenAI兼容的API接口。"
        )
        description_label.setStyleSheet("color: #666; padding: 8px 0;")
        layout.addWidget(description_label)

    def getConfig(self) -> dict:
        return {
            "api_base": self.api_base_edit.text().strip(),
            "api_key": self.api_key_edit.text().strip(),
            "model_name": self.model_name_edit.text().strip(),
        }

    def setConfig(self, config_data: dict):
        self.api_base_edit.setText(config_data.get("api_base", ""))
        self.api_key_edit.setText(config_data.get("api_key", ""))
        self.model_name_edit.setText(config_data.get("model_name", ""))


class LLMFallbackConfigCard(CardWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUI()

    def setupUI(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 16, 20, 16)
        layout.setSpacing(16)

        title_label = StrongBodyLabel("LLM超时和兜底模型")
        title_label.setFont(QFont("Microsoft YaHei", 12, QFont.Weight.Bold))
        layout.addWidget(title_label)

        form_layout = QFormLayout()
        form_layout.setSpacing(12)
        form_layout.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)
        form_layout.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)
        form_layout.setFormAlignment(Qt.AlignmentFlag.AlignLeft)

        self.timeout_edit = LineEdit()
        self.timeout_edit.setPlaceholderText("20")
        form_layout.addRow("主模型超时(秒):", self.timeout_edit)

        self.fallback_enabled_combo = ComboBox()
        self.fallback_enabled_combo.addItems(["关闭", "开启"])
        form_layout.addRow("启用兜底模型:", self.fallback_enabled_combo)

        self.fallback_api_base_edit = LineEdit()
        self.fallback_api_base_edit.setPlaceholderText("OpenAI兼容接口地址")
        form_layout.addRow("兜底API地址:", self.fallback_api_base_edit)

        self.fallback_api_key_edit = PasswordLineEdit()
        self.fallback_api_key_edit.setPlaceholderText("兜底模型API Key")
        form_layout.addRow("兜底API Key:", self.fallback_api_key_edit)

        self.fallback_model_name_edit = LineEdit()
        self.fallback_model_name_edit.setPlaceholderText("兜底模型名称")
        form_layout.addRow("兜底模型名称:", self.fallback_model_name_edit)

        self.fallback_timeout_edit = LineEdit()
        self.fallback_timeout_edit.setPlaceholderText("20")
        form_layout.addRow("兜底超时(秒):", self.fallback_timeout_edit)

        layout.addLayout(form_layout)

        description_label = CaptionLabel(
            "主模型超时、报错或返回空内容时，会自动切换到兜底模型。"
        )
        description_label.setStyleSheet("color: #666; padding: 8px 0;")
        layout.addWidget(description_label)

    @staticmethod
    def _parse_timeout(value: str, default: int = 20) -> int:
        try:
            seconds = int(float(str(value or "").strip()))
        except Exception:
            return default
        return max(5, min(seconds, 120))

    def getConfig(self) -> dict:
        return {
            "request_timeout_seconds": self._parse_timeout(self.timeout_edit.text(), 20),
            "fallback": {
                "enabled": self.fallback_enabled_combo.currentIndex() == 1,
                "api_base": self.fallback_api_base_edit.text().strip(),
                "api_key": self.fallback_api_key_edit.text().strip(),
                "model_name": self.fallback_model_name_edit.text().strip(),
                "timeout_seconds": self._parse_timeout(self.fallback_timeout_edit.text(), 20),
            },
        }

    def setConfig(self, config_data: dict):
        self.timeout_edit.setText(str(config_data.get("request_timeout_seconds", 20)))
        fallback_config = config_data.get("fallback", {})
        if not isinstance(fallback_config, dict):
            fallback_config = {}
        self.fallback_enabled_combo.setCurrentIndex(1 if fallback_config.get("enabled", False) else 0)
        self.fallback_api_base_edit.setText(fallback_config.get("api_base", ""))
        self.fallback_api_key_edit.setText(fallback_config.get("api_key", ""))
        self.fallback_model_name_edit.setText(fallback_config.get("model_name", ""))
        self.fallback_timeout_edit.setText(str(fallback_config.get("timeout_seconds", 20)))


class PromptConfigCard(CardWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUI()

    def setupUI(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 16, 20, 16)
        layout.setSpacing(16)

        title_label = StrongBodyLabel("Prompt配置")
        title_label.setFont(QFont("Microsoft YaHei", 12, QFont.Weight.Bold))
        layout.addWidget(title_label)

        self.instructions_edit = QTextEdit()
        self.instructions_edit.setPlaceholderText("每行一条提示词规则")
        self.instructions_edit.setMinimumHeight(280)
        self.instructions_edit.setAcceptRichText(False)
        self.instructions_edit.setStyleSheet(
            "QTextEdit { color: #111111; background-color: #ffffff; }"
        )
        layout.addWidget(self.instructions_edit)

        description_label = CaptionLabel(
            "这里填写会追加到系统提示词里的业务规则。每行一条，保存后立即生效。"
        )
        description_label.setStyleSheet("color: #666; padding: 8px 0;")
        layout.addWidget(description_label)

    def getConfig(self) -> dict:
        lines = [
            line.strip()
            for line in self.instructions_edit.toPlainText().splitlines()
            if line.strip()
        ]
        return {"instructions": lines}

    def setConfig(self, config_data: dict):
        instructions = config_data.get("instructions", [])
        if not isinstance(instructions, list):
            instructions = []
        self.instructions_edit.setPlainText("\n".join(str(item).strip() for item in instructions if str(item).strip()))


class ScenePromptConfigCard(CardWidget):
    PROMPT_FILES = {
        "presale": "runtime/scene_prompts_review/m11_售前场景prompt_待审.txt",
        "insale": "runtime/scene_prompts_review/m11_售中场景prompt_待审.txt",
        "aftersale": "runtime/scene_prompts_review/m11_售后场景prompt_待审.txt",
    }

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUI()

    def setupUI(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 16, 20, 16)
        layout.setSpacing(16)

        title_label = StrongBodyLabel("场景Prompt配置")
        title_label.setFont(QFont("Microsoft YaHei", 12, QFont.Weight.Bold))
        layout.addWidget(title_label)

        self.presale_edit = self._create_prompt_editor("售前场景Prompt")
        self.insale_edit = self._create_prompt_editor("售中场景Prompt")
        self.aftersale_edit = self._create_prompt_editor("售后场景Prompt")

        layout.addWidget(QLabel("售前场景Prompt"))
        layout.addWidget(self.presale_edit)
        layout.addWidget(QLabel("售中场景Prompt"))
        layout.addWidget(self.insale_edit)
        layout.addWidget(QLabel("售后场景Prompt"))
        layout.addWidget(self.aftersale_edit)

        description_label = CaptionLabel(
            "这里直接编辑三场景Prompt文件，保存后写入 runtime/scene_prompts_review。"
        )
        description_label.setStyleSheet("color: #666; padding: 8px 0;")
        layout.addWidget(description_label)

    @staticmethod
    def _create_prompt_editor(placeholder: str) -> QTextEdit:
        editor = QTextEdit()
        editor.setPlaceholderText(placeholder)
        editor.setMinimumHeight(220)
        editor.setAcceptRichText(False)
        editor.setStyleSheet("QTextEdit { color: #111111; background-color: #ffffff; }")
        return editor

    @staticmethod
    def _read_prompt(relative_path: str) -> str:
        path = get_resource_path(relative_path)
        try:
            return path.read_text(encoding="utf-8")
        except Exception:
            return ""

    @staticmethod
    def _write_prompt(relative_path: str, content: str) -> None:
        path = get_resource_path(relative_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(str(content or "").strip() + "\n", encoding="utf-8")

    def loadPrompts(self):
        self.presale_edit.setPlainText(self._read_prompt(self.PROMPT_FILES["presale"]))
        self.insale_edit.setPlainText(self._read_prompt(self.PROMPT_FILES["insale"]))
        self.aftersale_edit.setPlainText(self._read_prompt(self.PROMPT_FILES["aftersale"]))

    def savePrompts(self):
        self._write_prompt(self.PROMPT_FILES["presale"], self.presale_edit.toPlainText())
        self._write_prompt(self.PROMPT_FILES["insale"], self.insale_edit.toPlainText())
        self._write_prompt(self.PROMPT_FILES["aftersale"], self.aftersale_edit.toPlainText())


class BusinessHoursCard(CardWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUI()

    def setupUI(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 16, 20, 16)
        layout.setSpacing(16)

        title_label = StrongBodyLabel("营业时间设置")
        title_label.setFont(QFont("Microsoft YaHei", 12, QFont.Weight.Bold))
        layout.addWidget(title_label)

        form_layout = QFormLayout()
        form_layout.setSpacing(12)
        form_layout.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)
        form_layout.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)
        form_layout.setFormAlignment(Qt.AlignmentFlag.AlignLeft)

        self.start_time_picker = TimePicker()
        self.start_time_picker.setTime(QTime(8, 0))
        form_layout.addRow("开始时间:", self.start_time_picker)

        self.end_time_picker = TimePicker()
        self.end_time_picker.setTime(QTime(23, 0))
        form_layout.addRow("结束时间:", self.end_time_picker)

        layout.addLayout(form_layout)

        description_label = CaptionLabel(
            "设置AI客服的工作时间。在非工作时间，系统不会自动回复。"
        )
        description_label.setStyleSheet("color: #666; padding: 8px 0;")
        layout.addWidget(description_label)

    def getConfig(self) -> dict:
        return {
            "businessHours": {
                "start": self.start_time_picker.getTime().toString("HH:mm"),
                "end": self.end_time_picker.getTime().toString("HH:mm"),
            },
            "business_hours": {
                "start": self.start_time_picker.getTime().toString("HH:mm"),
                "end": self.end_time_picker.getTime().toString("HH:mm"),
            },
        }

    def setConfig(self, config_data: dict):
        business_hours = config_data.get("businessHours", config_data.get("business_hours", {}))
        start_time = QTime.fromString(business_hours.get("start", "08:00"), "HH:mm")
        if start_time.isValid():
            self.start_time_picker.setTime(start_time)
        end_time = QTime.fromString(business_hours.get("end", "23:00"), "HH:mm")
        if end_time.isValid():
            self.end_time_picker.setTime(end_time)


class NightModeCard(CardWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUI()

    def setupUI(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 16, 20, 16)
        layout.setSpacing(16)

        title_label = StrongBodyLabel("夜间不转人工设置")
        title_label.setFont(QFont("Microsoft YaHei", 12, QFont.Weight.Bold))
        layout.addWidget(title_label)

        form_layout = QFormLayout()
        form_layout.setSpacing(12)
        form_layout.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)
        form_layout.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)
        form_layout.setFormAlignment(Qt.AlignmentFlag.AlignLeft)

        self.start_time_picker = TimePicker()
        self.start_time_picker.setTime(QTime(23, 0))
        form_layout.addRow("开始时间:", self.start_time_picker)

        self.end_time_picker = TimePicker()
        self.end_time_picker.setTime(QTime(8, 0))
        form_layout.addRow("结束时间:", self.end_time_picker)

        layout.addLayout(form_layout)

        description_label = CaptionLabel(
            "设置夜间不转人工时间。该时间段内系统仍会回复客户，但不会真实转接人工。"
        )
        description_label.setStyleSheet("color: #666; padding: 8px 0;")
        layout.addWidget(description_label)

    def getConfig(self) -> dict:
        return {
            "night_mode": {
                "start": self.start_time_picker.getTime().toString("HH:mm"),
                "end": self.end_time_picker.getTime().toString("HH:mm"),
            }
        }

    def setConfig(self, config_data: dict):
        night_mode = config_data.get("night_mode", {})
        start_time = QTime.fromString(night_mode.get("start", "23:00"), "HH:mm")
        if start_time.isValid():
            self.start_time_picker.setTime(start_time)
        end_time = QTime.fromString(night_mode.get("end", "08:00"), "HH:mm")
        if end_time.isValid():
            self.end_time_picker.setTime(end_time)


class SettingUI(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.logger = get_logger("SettingUI")
        self.setupUI()
        self.loadConfig()
        self.setObjectName("设置")

    def setupUI(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(30, 30, 30, 30)
        main_layout.setSpacing(25)

        header_widget = self.createHeaderWidget()
        content_widget = self.createContentWidget()

        self.save_btn.clicked.connect(self.onSaveConfig)
        self.reset_btn.clicked.connect(self.onResetConfig)

        main_layout.addWidget(header_widget)
        main_layout.addWidget(content_widget, 1)

    def createHeaderWidget(self):
        header_widget = QWidget()
        header_layout = QHBoxLayout(header_widget)
        header_layout.setContentsMargins(0, 0, 0, 0)
        header_layout.setSpacing(20)

        title_label = SubtitleLabel("系统设置")
        title_label.setFont(QFont("Microsoft YaHei", 18, QFont.Weight.Bold))

        description_label = CaptionLabel("配置AI客服的基本参数、Prompt和工作时间")
        description_label.setStyleSheet("color: #666;")

        title_area = QWidget()
        title_layout = QVBoxLayout(title_area)
        title_layout.setContentsMargins(0, 0, 0, 0)
        title_layout.setSpacing(5)
        title_layout.addWidget(title_label)
        title_layout.addWidget(description_label)

        buttons_widget = QWidget()
        buttons_layout = QHBoxLayout(buttons_widget)
        buttons_layout.setContentsMargins(0, 0, 0, 0)
        buttons_layout.setSpacing(10)

        self.reset_btn = PushButton("重置")
        self.reset_btn.setIcon(FIF.UPDATE)
        self.reset_btn.setFixedSize(80, 40)

        self.save_btn = PrimaryPushButton("保存")
        self.save_btn.setIcon(FIF.SAVE)
        self.save_btn.setFixedSize(100, 40)

        buttons_layout.addWidget(self.reset_btn)
        buttons_layout.addWidget(self.save_btn)

        header_layout.addWidget(title_area)
        header_layout.addStretch()
        header_layout.addWidget(buttons_widget)
        return header_widget

    def createContentWidget(self):
        scroll_area = ScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll_area.setStyleSheet("""
            ScrollArea {
                border: none;
                background-color: transparent;
            }
        """)

        content_container = QWidget()
        content_layout = QVBoxLayout(content_container)
        content_layout.setSpacing(20)
        content_layout.setContentsMargins(20, 20, 20, 20)
        content_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        self.llm_config_card = LLMConfigCard()
        self.llm_fallback_config_card = LLMFallbackConfigCard()
        self.prompt_config_card = PromptConfigCard()
        self.scene_prompt_config_card = ScenePromptConfigCard()
        self.business_hours_card = BusinessHoursCard()
        self.night_mode_card = NightModeCard()

        content_layout.addWidget(self.llm_config_card)
        content_layout.addWidget(self.llm_fallback_config_card)
        content_layout.addWidget(self.prompt_config_card)
        content_layout.addWidget(self.scene_prompt_config_card)
        content_layout.addWidget(self.business_hours_card)
        content_layout.addWidget(self.night_mode_card)
        content_layout.addStretch()

        content_container.setStyleSheet("""
            QWidget {
                background-color: transparent;
                border: none;
            }
        """)
        scroll_area.setWidget(content_container)
        return scroll_area

    def loadConfig(self):
        try:
            loaded_config = {
                "llm": {
                    "api_base": config.get("llm.api_base", "https://ark.cn-beijing.volces.com/api/v3"),
                    "api_key": config.get("llm.api_key", ""),
                    "model_name": config.get("llm.model_name", ""),
                    "request_timeout_seconds": config.get("llm.request_timeout_seconds", 20),
                    "fallback": {
                        "enabled": config.get("llm.fallback.enabled", False),
                        "api_base": config.get("llm.fallback.api_base", ""),
                        "api_key": config.get("llm.fallback.api_key", ""),
                        "model_name": config.get("llm.fallback.model_name", ""),
                        "timeout_seconds": config.get("llm.fallback.timeout_seconds", 20),
                    },
                },
                "prompt": {
                    "instructions": config.get("prompt.instructions", []),
                },
                "business_hours": {
                    "start": config.get("business_hours.start", "08:00"),
                    "end": config.get("business_hours.end", "23:00"),
                },
                "night_mode": {
                    "start": config.get("night_mode.start", "23:00"),
                    "end": config.get("night_mode.end", "08:00"),
                },
            }
            self._validateAndSetConfig(loaded_config)
            self.logger.info("配置加载成功")
        except Exception as e:
            self.logger.error(f"加载配置失败: {e}")
            QMessageBox.warning(self, "加载失败", f"加载配置失败：{str(e)}")
            self._loadDefaultConfig()

    def _loadDefaultConfig(self):
        default_config = config_base.copy()
        self._validateAndSetConfig(default_config)
        self.logger.info("已加载默认配置")

    def _validateAndSetConfig(self, config_data):
        validated_config = {
            "llm": config_data.get("llm", {}),
            "prompt": config_data.get("prompt", {"instructions": []}),
            "business_hours": config_data.get("business_hours", {"start": "08:00", "end": "23:00"}),
            "night_mode": config_data.get("night_mode", {"start": "23:00", "end": "08:00"}),
        }

        business_hours = validated_config["business_hours"]
        if not isinstance(business_hours, dict):
            business_hours = {"start": "08:00", "end": "23:00"}
            validated_config["business_hours"] = business_hours

        night_mode = validated_config["night_mode"]
        if not isinstance(night_mode, dict):
            night_mode = {"start": "23:00", "end": "08:00"}
            validated_config["night_mode"] = night_mode

        self.llm_config_card.setConfig(validated_config["llm"])
        self.llm_fallback_config_card.setConfig(validated_config["llm"])
        self.prompt_config_card.setConfig(validated_config["prompt"])
        self.scene_prompt_config_card.loadPrompts()
        self.business_hours_card.setConfig({"business_hours": business_hours})
        self.night_mode_card.setConfig({"night_mode": night_mode})

    def onSaveConfig(self):
        try:
            llm_config = self.llm_config_card.getConfig()
            llm_config.update(self.llm_fallback_config_card.getConfig())
            prompt_config = self.prompt_config_card.getConfig()
            business_config = self.business_hours_card.getConfig()
            night_mode_config = self.night_mode_card.getConfig()

            if not llm_config.get("api_key"):
                QMessageBox.warning(self, "配置错误", "请输入LLM API Key")
                return
            if not llm_config.get("model_name"):
                QMessageBox.warning(self, "配置错误", "请输入LLM模型名称")
                return

            start_time = self.business_hours_card.start_time_picker.getTime()
            end_time = self.business_hours_card.end_time_picker.getTime()
            if start_time >= end_time:
                QMessageBox.warning(self, "时间设置错误", "开始时间必须早于结束时间！")
                return

            night_start_time = self.night_mode_card.start_time_picker.getTime()
            night_end_time = self.night_mode_card.end_time_picker.getTime()
            if night_start_time == night_end_time:
                QMessageBox.warning(self, "夜间模式时间错误", "夜间模式开始时间和结束时间不能相同！")
                return

            new_config = {
                "llm": llm_config,
                "prompt": prompt_config,
                "business_hours": business_config.get("businessHours", {"start": "08:00", "end": "23:00"}),
                "night_mode": night_mode_config.get("night_mode", {"start": "23:00", "end": "08:00"}),
                "db_path": config.get("db_path", ""),
            }
            config.update(new_config, save=True)
            self.scene_prompt_config_card.savePrompts()
            self.logger.info("配置保存成功")

            InfoBar.success(
                title="保存成功",
                content="配置已保存",
                orient=Qt.Orientation.Horizontal,
                isClosable=True,
                position=InfoBarPosition.TOP,
                duration=2000,
                parent=self,
            )
        except Exception as e:
            self.logger.error(f"保存配置失败: {e}")
            QMessageBox.critical(self, "保存失败", f"保存配置时发生错误：{str(e)}")

    def onResetConfig(self):
        reply = QMessageBox.question(
            self,
            "确认重置",
            "确定要重置所有配置吗？\n这将重新加载配置文件中的原始设置。",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )

        if reply == QMessageBox.StandardButton.Yes:
            try:
                config.reload()
                self.loadConfig()
                self.logger.info("配置已重置")

                InfoBar.success(
                    title="重置成功",
                    content="配置已重置为配置文件中的设置",
                    orient=Qt.Orientation.Horizontal,
                    isClosable=True,
                    position=InfoBarPosition.TOP,
                    duration=2000,
                    parent=self,
                )
            except Exception as e:
                self.logger.error(f"重置配置失败: {e}")
                QMessageBox.critical(self, "重置失败", f"重置配置失败：{str(e)}")
