from __future__ import annotations

from collections.abc import Sequence

from tqdm import tqdm

from .meilisearch_client import MeiliSearchClient


class OCRIndexWriter:
    def __init__(
        self,
        client: MeiliSearchClient,
        *,
        batch_size: int = 10000,
        wait_each_batch: bool = False,
    ) -> None:
        self.client = client
        self.batch_size = max(int(batch_size), 1)
        self.wait_each_batch = bool(wait_each_batch)

    def add_documents(self, index_uid: str, documents: Sequence[dict]) -> None:
        total_batches = (len(documents) + self.batch_size - 1) // self.batch_size
        print(
            f"[OCR] Start pushing {len(documents)} OCR documents to Meilisearch "
            f"(index={index_uid}, batch_size={self.batch_size}, total_batches={total_batches}, "
            f"wait_each_batch={self.wait_each_batch})"
        )
        batch: list[dict] = []
        batch_index = 0
        last_task_uid: int | None = None
        for document in tqdm(documents, desc="Push OCR documents"):
            batch.append(dict(document))
            if len(batch) >= self.batch_size:
                batch_index += 1
                last_task_uid = self._flush(index_uid, batch, batch_index=batch_index, total_batches=total_batches)
                batch = []
        if batch:
            batch_index += 1
            last_task_uid = self._flush(index_uid, batch, batch_index=batch_index, total_batches=total_batches)
        print(f"[OCR] All batches submitted to Meilisearch for index={index_uid}")
        if not self.wait_each_batch and last_task_uid is not None:
            print(f"[OCR] Waiting final task only: taskUid={last_task_uid}")
            result = self.client.wait_for_task(int(last_task_uid))
            status = result.get("status")
            if status != "succeeded":
                raise RuntimeError(
                    f"Meilisearch final task failed for index={index_uid}, "
                    f"taskUid={last_task_uid}, status={status}, payload={result}"
                )
            print(f"[OCR] Final task done: taskUid={last_task_uid}")
        print("[OCR] Finished pushing OCR documents to Meilisearch")

    def _flush(self, index_uid: str, batch: list[dict], *, batch_index: int, total_batches: int) -> int:
        print(
            f"[OCR] Upload batch {batch_index}/{total_batches} "
            f"with {len(batch)} documents to index={index_uid}"
        )
        task = self.client.add_documents(index_uid=index_uid, documents=batch)
        task_uid = int(task["taskUid"])
        print(f"[OCR] Batch {batch_index}/{total_batches} submitted with taskUid={task_uid}")
        if self.wait_each_batch:
            print(f"[OCR] Waiting batch {batch_index}/{total_batches} taskUid={task_uid}")
            result = self.client.wait_for_task(task_uid)
            status = result.get("status")
            if status != "succeeded":
                raise RuntimeError(
                    f"Meilisearch task failed for index={index_uid}, "
                    f"batch={batch_index}/{total_batches}, taskUid={task_uid}, status={status}, payload={result}"
                )
            print(f"[OCR] Done batch {batch_index}/{total_batches} taskUid={task_uid}")
        return task_uid
