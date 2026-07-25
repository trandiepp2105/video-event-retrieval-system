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
        task_uids: list[tuple[int, int]] = []
        for document in tqdm(documents, desc="Push OCR documents"):
            batch.append(dict(document))
            if len(batch) >= self.batch_size:
                batch_index += 1
                task_uid = self._flush(index_uid, batch, batch_index=batch_index, total_batches=total_batches)
                task_uids.append((batch_index, task_uid))
                batch = []
        if batch:
            batch_index += 1
            task_uid = self._flush(index_uid, batch, batch_index=batch_index, total_batches=total_batches)
            task_uids.append((batch_index, task_uid))
        print(f"[OCR] All batches submitted to Meilisearch for index={index_uid}. Waiting for {len(task_uids)} tasks ...")
        for waited, (submitted_batch_index, task_uid) in enumerate(task_uids, start=1):
            result = self.client.wait_for_task(int(task_uid))
            status = result.get("status")
            if status != "succeeded":
                raise RuntimeError(
                    f"Meilisearch task failed for index={index_uid}, "
                    f"batch={submitted_batch_index}/{total_batches}, taskUid={task_uid}, status={status}, payload={result}"
                )
            if waited <= 5 or waited == len(task_uids) or waited % 25 == 0:
                print(
                    f"[OCR] Task {waited}/{len(task_uids)} done "
                    f"(batch={submitted_batch_index}/{total_batches}, taskUid={task_uid})"
                )
        print("[OCR] Finished pushing OCR documents to Meilisearch")

    def _flush(self, index_uid: str, batch: list[dict], *, batch_index: int, total_batches: int) -> int:
        print(
            f"[OCR] Upload batch {batch_index}/{total_batches} "
            f"with {len(batch)} documents to index={index_uid}"
        )
        task = self.client.add_documents(index_uid=index_uid, documents=batch)
        task_uid = int(task["taskUid"])
        print(f"[OCR] Batch {batch_index}/{total_batches} submitted with taskUid={task_uid}")
        return task_uid
