from __future__ import annotations

from collections.abc import Sequence

from tqdm import tqdm

from .meilisearch_client import MeiliSearchClient


class OCRIndexWriter:
    def __init__(self, client: MeiliSearchClient, *, batch_size: int = 1000) -> None:
        self.client = client
        self.batch_size = max(int(batch_size), 1)

    def add_documents(self, index_uid: str, documents: Sequence[dict]) -> None:
        total_batches = (len(documents) + self.batch_size - 1) // self.batch_size
        print(
            f"[OCR] Start pushing {len(documents)} OCR documents to Meilisearch "
            f"(index={index_uid}, batch_size={self.batch_size}, total_batches={total_batches})"
        )
        batch: list[dict] = []
        batch_index = 0
        for document in tqdm(documents, desc="Push OCR documents"):
            batch.append(dict(document))
            if len(batch) >= self.batch_size:
                batch_index += 1
                self._flush(index_uid, batch, batch_index=batch_index, total_batches=total_batches)
                batch = []
        if batch:
            batch_index += 1
            self._flush(index_uid, batch, batch_index=batch_index, total_batches=total_batches)
        print("[OCR] Finished pushing OCR documents to Meilisearch")

    def _flush(self, index_uid: str, batch: list[dict], *, batch_index: int, total_batches: int) -> None:
        print(
            f"[OCR] Upload batch {batch_index}/{total_batches} "
            f"with {len(batch)} documents to index={index_uid}"
        )
        task = self.client.add_documents(index_uid=index_uid, documents=batch)
        self.client.wait_for_task(int(task["taskUid"]))
        print(f"[OCR] Batch {batch_index}/{total_batches} indexed successfully")
