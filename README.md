# 拼多多客服 Agent

面向拼多多商家的 AI 客服桌面应用，重点服务售前、售中、售后三类客服场景。项目基于 PyQt6 构建桌面工作台，通过拼多多 WebSocket 接收实时会话消息，结合商品、订单、物流、知识库和人工兜底策略生成客服回复。

当前版本主要围绕“可控客服工作流”设计：大模型不直接凭经验猜答案，而是在结构化上下文、场景判断、知识检索、工具调用和安全规则约束下回复。复杂售后、图片/视频、找不到充电口/插电口等高风险场景会直接转人工。

本项目基于原作者 [@JC0v0](https://github.com/JC0v0) 的 `Customer-Agent` 二次开发。

- 原项目地址：`https://github.com/JC0v0/Customer-Agent`
- 原项目许可证：MIT License
- 当前仓库保留原作者来源说明，并在原项目基础上增加拼多多客服业务、三场景知识库、订单上下文、商品卡/订单卡解析、受控工具调用和桌面端管理能力。

## 当前版本

`v2026.06.09`

主要变化见 [CHANGELOG.md](CHANGELOG.md)。本版本重点引入 `TurnContext` 结构化上下文，并修复商品卡、订单卡元数据污染客户问题的问题。

## 业务能力

- 拼多多 WebSocket 实时消息接入和自动重连。
- 多店铺账号接入，按账号启动独立自动回复线程。
- 售前、售中、售后三场景识别与规则加载。
- 商品卡、订单卡、客户文本、图片、视频分离解析。
- 商品知识、场景知识和订单上下文联合检索。
- 自动发送文本回复、商品卡片，必要时转人工。
- 商品未锁定时避免猜商品，优先询问或返回候选商品。
- 支持订单状态、物流状态、签收状态等上下文约束。
- 售后图片/视频直接转人工，避免看图臆断。
- 多订单待确认默认按售后处理，降低误判售前风险。
- 支持价格分转元，例如 `1161` 解析为 `11.61`。
- 夜间模式、禁词过滤、回复清洗和兜底回复。

## 核心流程

```text
拼多多 WebSocket
    -> PDDChannel
    -> Message Queue
    -> Handler Chain
       -> MessagePreprocessor
       -> KeywordHandler
       -> AIReplyHandler
    -> CustomerAgent
       -> TurnContext 解析
       -> 场景判定
       -> 订单上下文刷新
       -> 预检索知识注入
       -> LLM 工具调用循环
       -> 回复清洗 / 转人工
    -> SendMessage / transfer_conversation
```

关键原则：

- 客户真实文本和商品/订单元数据分开处理。
- 商品卡和订单卡只作为上下文，不直接污染客户问题。
- 场景规则优先于通用提示词。
- 知识库或工具没有明确答案时，不编造，转人工或追问。
- 图片/视频不是商品功能依据，高风险问题交给人工。

## 架构分层

```text
Agent/                  自研 CustomerAgent 和工具注册
  CustomerAgent/custom/ LLM 客户端、消息构建、工具执行、会话管理、TurnContext
  CustomerAgent/tools/  search_knowledge、send_product_card、transfer_conversation

Channel/                渠道接入
  pinduoduo/            拼多多 WebSocket、登录、消息解析和 API 封装

Message/                消息队列和处理器链
  core/                 queue、consumer、handler 管理
  handlers/             预处理、关键词、AI 回复处理

bridge/                 Context / Reply 抽象
core/                   DI 容器、服务注册、连接状态、缓存
database/               SQLAlchemy 模型、知识库服务、商品同步、向量检索
ui/                     PyQt6 桌面界面
utils/                  日志、路径、运行时资源、夜间模式等工具
scripts/                Windows 构建脚本
app.py                  应用入口
```

## Agent 工具

| 工具 | 作用 |
| --- | --- |
| `search_knowledge` | 按店铺、商品、场景检索商品知识和客服知识。 |
| `send_product_card` | 发送当前商品卡；没有锁定商品时返回候选商品。 |
| `transfer_conversation` | 将会话转接给人工客服。 |

工具调用由 `ToolExecutor` 统一执行，CustomerAgent 会把工具结果回填给模型继续推理，直到不再需要工具或达到循环上限。

## TurnContext

`TurnContext` 是当前版本的上下文解析底座，负责把一轮原始消息拆成稳定结构：

- `customer_text`：客户真实文本。
- `product_card`：商品 ID、商品名、规格、价格等。
- `order_card`：订单号、订单状态、物流状态、支付状态、快递单号等。
- `media`：图片、视频标记和图片链接。
- `turn_type`：当前 turn 是否包含文本、商品卡、订单卡、媒体。
- `parse_warnings`：解析异常提示。

它不做 embedding、不做意图路由、不改知识库，只负责把输入清理成可控结构。

## 配置

仓库只提供 `config.example.json`，本地运行时需要复制为 `config.json`：

```powershell
Copy-Item config.example.json config.json
```

常用配置项：

| 配置 | 说明 |
| --- | --- |
| `llm.model_name` | 主模型名称。 |
| `llm.api_key` | OpenAI 兼容接口密钥，本地填写，不提交。 |
| `llm.api_base` | OpenAI 兼容接口地址。 |
| `llm.max_tokens` | 普通回复最大输出长度。 |
| `llm.tool_call_max_tokens` | 工具调用场景最大输出长度。 |
| `llm.fallback` | 兜底模型配置。 |
| `agent.token_window` | 会话上下文窗口。 |
| `agent.compress_ratio` | 触发上下文压缩的比例。 |
| `agent.max_loops` | 工具调用最大循环次数。 |
| `business_hours` | 人工客服工作时间。 |
| `night_mode` | 夜间模式时间。 |
| `db_path` | 本地 SQLite 数据库路径。 |

`config.json`、数据库、日志、runtime 数据和浏览器用户数据都属于本地运行数据，不应提交到仓库。

## 本地运行

环境要求：

- Windows
- Python 3.11+
- uv

安装依赖：

```powershell
uv sync
```

安装 Playwright 浏览器：

```powershell
uv run playwright install chromium
```

启动桌面应用：

```powershell
uv run python app.py
```

也可以使用项目提供的启动脚本：

```powershell
.\start_local.ps1
```

## 构建 Windows 可执行文件

构建脚本位于 `scripts/`。常用命令：

```powershell
uv run python scripts/build_win_exe.py --clean
```

更多构建说明见 [scripts/README.md](scripts/README.md)。

## 测试

当前重点测试 `TurnContext` 解析：

```powershell
uv run pytest tests/test_turn_context.py
```

发布前建议至少检查：

```powershell
uv run python -m compileall Agent Channel Message bridge core database ui utils
uv run pytest tests/test_turn_context.py
```

## License

MIT License。原项目版权归原作者所有，当前仓库在保留原许可证和来源说明的基础上进行二次开发。
