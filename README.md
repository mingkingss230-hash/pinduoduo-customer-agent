# Agent Customer AI

An AI customer-service desktop application based on PyQt6 and an OpenAI-compatible LLM API.

This project is a secondary development based on the original **Customer-Agent** project by **JC0v0**:

- Original project: `https://github.com/JC0v0/Customer-Agent`
- Original license: MIT License

This repository keeps the original project attribution and focuses on a more production-oriented customer-service agent workflow: tool calling, scene-aware knowledge retrieval, order/logistics context, response safety controls, and a desktop management UI.

## Features

- Multi-channel customer-service message handling
- PyQt6 desktop UI
- OpenAI-compatible LLM client
- Agent loop with controlled tool calling
- Product-level knowledge base
- Scene knowledge for presale, insale, and aftersale workflows
- Pre-retrieval of relevant knowledge before the first LLM response
- Order and logistics context injection
- Manual transfer routing
- Night-mode response policy
- Forbidden-word replacement and output safety filtering
- Token and call logging

## Available Agent Tools

| Tool | Description |
| --- | --- |
| `search_knowledge` | Search product and scene knowledge for the current customer question. |
| `send_product_card` | Send the current product card or return candidate products when no product is locked. |
| `transfer_conversation` | Transfer the conversation to a human customer-service account. |

## Requirements

- Python >= 3.11
- Windows is recommended for the desktop workflow

## Installation

```bash
uv sync
```

## Run

```bash
python app.py
```

On first run, the app creates a local `config.json`. Do not commit real API keys, cookies, account data, logs, or database files.

## Configuration

Main configuration groups:

| Key | Description |
| --- | --- |
| `llm` | LLM model, API base URL, and API key. |
| `embedder` | Embedding model settings. |
| `knowledge_base` | Local knowledge-base storage settings. |
| `business_hours` | Human-service working hours. |
| `prompt` | Global and scene-level customer-service rules. |

## Project Structure

```text
Agent-Customer-AI/
├── Agent/                  # Agent loop, LLM client, session and tools
├── Channel/                # Channel integrations
├── Message/                # Message queue and handler chain
├── bridge/                 # Context and reply abstractions
├── core/                   # Shared services and dependency injection
├── database/               # SQLAlchemy models and knowledge service
├── ui/                     # PyQt6 desktop UI
├── utils/                  # Runtime utilities
├── scripts/                # Build and generic maintenance scripts
└── app.py                  # Application entry
```

## Publishing Notes

This public version intentionally excludes private runtime data:

- `config.json`
- local SQLite databases
- logs
- cookies and browser data
- customer chat records
- order IDs and logistics numbers
- private knowledge-base exports
- shop-specific migration artifacts

## License

MIT License. See [LICENSE](LICENSE).
