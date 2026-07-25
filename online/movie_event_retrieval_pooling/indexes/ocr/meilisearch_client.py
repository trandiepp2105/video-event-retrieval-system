from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import meilisearch


@dataclass(frozen=True)
class MeiliSearchClient:
    base_url: str
    api_key: str | None = None
    timeout_sec: int = 600

    def __post_init__(self) -> None:
        client = meilisearch.Client(self.base_url, self.api_key)
        object.__setattr__(self, "_client", client)

    def create_index(self, index_uid: str, primary_key: str) -> dict[str, Any]:
        try:
            task = self._client.create_index(index_uid, {"primaryKey": primary_key})
            return self._to_dict(task)
        except Exception as exc:
            message = str(exc)
            if "already exists" in message.lower() or "index_already_exists" in message.lower():
                return {"status": 409}
            raise

    def update_settings(self, index_uid: str, settings: dict[str, Any]) -> dict[str, Any]:
        index = self._client.index(index_uid)
        task = index.update_settings(settings)
        return self._to_dict(task)

    def add_documents(self, index_uid: str, documents: list[dict[str, Any]]) -> dict[str, Any]:
        index = self._client.index(index_uid)
        task = index.add_documents(documents)
        return self._to_dict(task)

    def search(
        self,
        index_uid: str,
        query: str,
        limit: int,
        *,
        matching_strategy: str | None = None,
        attributes_to_retrieve: list[str] | None = None,
        show_ranking_score: bool | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {"limit": int(limit)}
        if matching_strategy is not None:
            payload["matchingStrategy"] = str(matching_strategy)
        if attributes_to_retrieve is not None:
            payload["attributesToRetrieve"] = list(attributes_to_retrieve)
        if show_ranking_score is not None:
            payload["showRankingScore"] = bool(show_ranking_score)
        index = self._client.index(index_uid)
        result = index.search(query, payload)
        return self._to_dict(result)

    def get_index(self, index_uid: str) -> dict[str, Any]:
        index = self._client.get_index(index_uid)
        return self._to_dict(index)

    def get_index_stats(self, index_uid: str) -> dict[str, Any]:
        index = self._client.index(index_uid)
        stats = index.get_stats()
        return self._to_dict(stats)

    def get_task(self, task_uid: int) -> dict[str, Any]:
        task = self._client.get_task(int(task_uid))
        return self._to_dict(task)

    def wait_for_task(self, task_uid: int, poll_interval_sec: float = 0.5) -> dict[str, Any]:
        task = self._client.wait_for_task(
            int(task_uid),
            timeout_in_ms=int(self.timeout_sec * 1000),
            interval_in_ms=int(max(poll_interval_sec, 0.1) * 1000),
        )
        return self._to_dict(task)

    @staticmethod
    def _to_dict(payload: Any) -> dict[str, Any]:
        if isinstance(payload, dict):
            raw = payload
        elif hasattr(payload, "__dict__"):
            raw = {
                key: value
                for key, value in vars(payload).items()
                if not key.startswith("_")
            }
        elif hasattr(payload, "dict"):
            raw = payload.dict()
        else:
            raise TypeError(f"Khong the chuyen doi payload sang dict: {type(payload)!r}")
        aliases = {
            "task_uid": "taskUid",
            "primary_key": "primaryKey",
            "number_of_documents": "numberOfDocuments",
            "is_indexing": "isIndexing",
        }
        normalized: dict[str, Any] = {}
        for key, value in raw.items():
            normalized[aliases.get(key, key)] = value
        return normalized
