from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any
from urllib import error, parse, request


@dataclass(frozen=True)
class MeiliSearchClient:
    base_url: str
    api_key: str | None = None
    timeout_sec: int = 120

    def create_index(self, index_uid: str, primary_key: str) -> dict[str, Any]:
        return self._request(
            "POST",
            "/indexes",
            {"uid": index_uid, "primaryKey": primary_key},
            allow_conflict=True,
        )

    def update_settings(self, index_uid: str, settings: dict[str, Any]) -> dict[str, Any]:
        return self._request("PATCH", f"/indexes/{index_uid}/settings", settings)

    def add_documents(self, index_uid: str, documents: list[dict[str, Any]]) -> dict[str, Any]:
        return self._request("POST", f"/indexes/{index_uid}/documents", documents)

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
        payload: dict[str, Any] = {"q": query, "limit": int(limit)}
        if matching_strategy is not None:
            payload["matchingStrategy"] = str(matching_strategy)
        if attributes_to_retrieve is not None:
            payload["attributesToRetrieve"] = list(attributes_to_retrieve)
        if show_ranking_score is not None:
            payload["showRankingScore"] = bool(show_ranking_score)
        return self._request("POST", f"/indexes/{index_uid}/search", payload)

    def get_task(self, task_uid: int) -> dict[str, Any]:
        return self._request("GET", f"/tasks/{task_uid}")

    def wait_for_task(self, task_uid: int, poll_interval_sec: float = 0.5) -> dict[str, Any]:
        while True:
            payload = self.get_task(task_uid)
            status = payload.get("status")
            if status in {"succeeded", "failed", "canceled"}:
                return payload
            time.sleep(poll_interval_sec)

    def _request(
        self,
        method: str,
        path: str,
        payload: Any | None = None,
        *,
        allow_conflict: bool = False,
    ) -> dict[str, Any]:
        url = self.base_url.rstrip("/") + path
        data = None
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        if payload is not None:
            data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req = request.Request(url, data=data, headers=headers, method=method)
        try:
            with request.urlopen(req, timeout=self.timeout_sec) as response:
                return json.loads(response.read().decode("utf-8"))
        except error.HTTPError as exc:
            if allow_conflict and exc.code == 409:
                body = exc.read().decode("utf-8")
                return json.loads(body) if body else {"status": 409}
            body = exc.read().decode("utf-8")
            raise RuntimeError(
                f"Meilisearch request failed: method={method} path={path} status={exc.code} body={body}"
            ) from exc
        except error.URLError as exc:
            raise RuntimeError(f"Khong ket noi duoc Meilisearch tai {url}: {exc}") from exc
