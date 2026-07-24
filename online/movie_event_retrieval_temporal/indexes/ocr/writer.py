from __future__ import annotations

from collections.abc import Sequence

from .meilisearch_client import MeiliSearchClient


class OCRIndexWriter:
    def __init__(self, client: MeiliSearchClient, *, batch_size: int = 1000) -> None:
        self.client = client
        self.batch_size = max(int(batch_size), 1)

    def add_documents(self, index_uid: str, documents: Sequence[dict]) -> None:
        batch: list[dict] = []
        for document in documents:
            batch.append(dict(document))
            if len(batch) >= self.batch_size:
                self._flush(index_uid, batch)
                batch = []
        if batch:
            self._flush(index_uid, batch)

    def _flush(self, index_uid: str, batch: list[dict]) -> None:
        task = self.client.add_documents(index_uid=index_uid, documents=batch)
        self.client.wait_for_task(int(task["taskUid"]))
