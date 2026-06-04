# 拼多多客服 Agent

一个面向电商客服场景的 AI 客服桌面应用，基于 PyQt6 和 OpenAI 兼容接口构建。

本项目基于原作者 **JC0v0** 的 `Customer-Agent` 二次开发：

- 原项目地址：`https://github.com/JC0v0/Customer-Agent`
- 原项目许可证：MIT License

本仓库保留原作者来源说明，并在原项目基础上增加了更偏生产使用的客服 Agent 能力，包括工具调用、场景知识库、订单/物流上下文、回复安全控制和桌面端知识管理。

## 功能特性

- 桌面端客服工作台，基于 PyQt6
- 支持 OpenAI 兼容格式的大模型接口
- 自研 Agent 循环，支持受控工具调用
- 商品知识库和场景知识库
- 售前、售中、售后三场景检索
- 首轮回复前自动预检索相关知识
- 可结合订单状态和物流状态生成上下文
- 支持转人工工具
- 支持夜间模式话术
- 支持禁词替换和回复安全过滤
- 支持 token、调用次数和回复日志记录

## Agent 工具

| 工具 | 说明 |
| --- | --- |
| `search_knowledge` | 查询当前商品的商品知识和场景知识 |
| `send_product_card` | 发送当前商品卡片；未锁定商品时返回候选商品 |
| `transfer_conversation` | 将会话转接给人工客服 |

## 运行环境

- Python >= 3.11
- 推荐 Windows 环境运行桌面端

## 安装依赖

```bash
uv sync
```

## 启动

```bash
python app.py
```

首次运行会在项目根目录生成 `config.json`。请不要提交真实 API Key、cookies、账号信息、聊天记录、订单号、日志或本地数据库。

## 配置说明

主要配置项：

| 配置项 | 说明 |
| --- | --- |
| `llm` | 大模型名称、API 地址、API Key |
| `embedder` | 向量模型配置 |
| `knowledge_base` | 本地知识库配置 |
| `business_hours` | 人工客服工作时间 |
| `prompt` | 全局客服规则和场景规则 |

可以参考 `config.example.json` 创建自己的配置。

## 项目结构

```text
Agent-Customer-AI/
├── Agent/                  # Agent 循环、LLM 客户端、会话管理和工具
├── Channel/                # 渠道接入
├── Message/                # 消息队列和处理器链
├── bridge/                 # Context / Reply 抽象
├── core/                   # 通用服务和依赖注入
├── database/               # SQLAlchemy 模型和知识库服务
├── ui/                     # PyQt6 桌面界面
├── utils/                  # 运行时工具
├── scripts/                # 构建和通用维护脚本
└── app.py                  # 应用入口
```

## 发布版说明

本公开版本已移除私有运行数据和店铺业务数据，包括：

- `config.json`
- 本地 SQLite 数据库
- 日志文件
- cookies 和浏览器数据
- 客户聊天记录
- 订单号和物流单号
- 私有知识库导出
- 店铺专属迁移脚本
- 商品型号专项补库脚本

## 二次开发方向

这个项目适合继续扩展：

- 多平台客服接入
- 更完善的知识库评测 harness
- 回复质量自动巡检
- 场景知识库可视化编辑
- 多模型切换和成本统计
- 私有化本地模型部署

## 许可证

MIT License。详见 [LICENSE](LICENSE)。
