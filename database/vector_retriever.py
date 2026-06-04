from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass
from math import sqrt
from pathlib import Path
from typing import Any, Iterable, Sequence

import requests

from utils.logger_loguru import get_logger


logger = get_logger("VectorRetriever")


@dataclass(frozen=True)
class VectorItem:
    item_id: str
    text: str
    payload: Any


class VectorRetriever:
    def __init__(self) -> None:
        self.embedding_url = os.getenv(
            "CUSTOMER_AGENT_EMBEDDING_URL",
            os.getenv("CALLBACK_SERVER_EMBEDDING_URL", "http://127.0.0.1:8081/v1/embeddings"),
        )
        self.embedding_model = os.getenv(
            "CUSTOMER_AGENT_EMBEDDING_MODEL",
            os.getenv("CALLBACK_SERVER_EMBEDDING_MODEL", "bge-large-zh-v1.5-q8_0.gguf"),
        )
        self.timeout_seconds = float(os.getenv("CUSTOMER_AGENT_EMBEDDING_TIMEOUT", "5"))
        self.score_threshold = float(os.getenv("CUSTOMER_AGENT_VECTOR_SCORE_THRESHOLD", "0.25"))
        self.max_text_chars = int(os.getenv("CUSTOMER_AGENT_EMBEDDING_MAX_TEXT_CHARS", "450"))
        self.disable_seconds = float(os.getenv("CUSTOMER_AGENT_EMBEDDING_DISABLE_SECONDS", "60"))
        self._disabled_until = 0.0
        self.cache_dir = Path(
            os.getenv(
                "CUSTOMER_AGENT_VECTOR_INDEX_DIR",
                str(Path(__file__).resolve().parents[1] / "temp" / "vector_index"),
            )
        )
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def rank(
        self,
        namespace: str,
        shop_id: int | str,
        query: str,
        items: Sequence[VectorItem],
        limit: int,
    ) -> list[Any]:
        clean_query = query.strip()
        if not clean_query or not items:
            return []
        if time.monotonic() < self._disabled_until:
            return []

        try:
            query_vector = self._embed(clean_query)
            vectors = self._load_or_build_vectors(namespace, shop_id, items)
        except Exception as exc:
            self._disabled_until = time.monotonic() + self.disable_seconds
            logger.warning(f"向量检索不可用，回退关键词检索: namespace={namespace}, error={exc}")
            return []

        item_map = {item.item_id: item for item in items}
        scored: list[tuple[float, VectorItem]] = []
        for item_id, vector in vectors.items():
            item = item_map.get(item_id)
            if item is None:
                continue
            score = self._cosine_similarity(query_vector, vector)
            if score >= self.score_threshold:
                scored.append((score, item))

        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [item.payload for _, item in scored[:limit]]

    def _load_or_build_vectors(
        self,
        namespace: str,
        shop_id: int | str,
        items: Sequence[VectorItem],
    ) -> dict[str, list[float]]:
        cache_path = self._cache_path(namespace, shop_id)
        content_hash = self._content_hash(items)
        cached = self._load_cache(cache_path)
        if (
            cached.get("content_hash") == content_hash
            and cached.get("model") == self.embedding_model
            and isinstance(cached.get("vectors"), dict)
        ):
            return {
                str(item_id): [float(value) for value in vector]
                for item_id, vector in cached["vectors"].items()
                if isinstance(vector, list)
            }

        vectors = {}
        for item in items:
            try:
                vectors[item.item_id] = self._embed(item.text)
            except Exception as exc:
                logger.warning(f"单条知识向量化失败，已跳过: item_id={item.item_id}, error={exc}")
        cache_path.write_text(
            json.dumps(
                {
                    "model": self.embedding_model,
                    "content_hash": content_hash,
                    "vectors": vectors,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        return vectors

    def _embed(self, text: str) -> list[float]:
        input_text = text[: self.max_text_chars]
        response = requests.post(
            self.embedding_url,
            json={"model": self.embedding_model, "input": input_text},
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        body = response.json()
        vector = (
            body.get("embedding")
            or body.get("data", [{}])[0].get("embedding")
            or body.get("embeddings", [[]])[0]
        )
        if not isinstance(vector, list):
            raise RuntimeError(f"embedding response missing vector: {body!r}")
        return [float(value) for value in vector]

    def _cache_path(self, namespace: str, shop_id: int | str) -> Path:
        key = hashlib.sha1(f"{shop_id}:{namespace}".encode("utf-8")).hexdigest()[:16]
        return self.cache_dir / f"{key}.json"

    def _content_hash(self, items: Sequence[VectorItem]) -> str:
        digest = hashlib.sha1()
        digest.update(self.embedding_model.encode("utf-8"))
        digest.update(str(self.max_text_chars).encode("utf-8"))
        for item in items:
            digest.update(item.item_id.encode("utf-8"))
            digest.update(item.text.encode("utf-8"))
        return digest.hexdigest()

    @staticmethod
    def _load_cache(path: Path) -> dict[str, Any]:
        if not path.exists():
            return {}
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}

    @staticmethod
    def _cosine_similarity(left: Iterable[float], right: Iterable[float]) -> float:
        left_values = list(left)
        right_values = list(right)
        if not left_values or not right_values or len(left_values) != len(right_values):
            return 0.0
        numerator = sum(a * b for a, b in zip(left_values, right_values))
        left_norm = sqrt(sum(value * value for value in left_values))
        right_norm = sqrt(sum(value * value for value in right_values))
        if left_norm == 0.0 or right_norm == 0.0:
            return 0.0
        return numerator / (left_norm * right_norm)
