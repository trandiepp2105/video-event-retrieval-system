from __future__ import annotations

from .meilisearch_client import MeiliSearchClient


class OCRIndexConfigurator:
    def __init__(self, client: MeiliSearchClient) -> None:
        self.client = client

    def configure(self, index_uid: str) -> None:
        self.client.create_index(index_uid=index_uid, primary_key="id")
        task = self.client.update_settings(
            index_uid=index_uid,
            settings={
                "rankingRules": ["typo", "proximity", "words", "exactness", "attribute", "sort"],
                "pagination": {"maxTotalHits": 5000},
                "distinctAttribute": None,
                "typoTolerance": {
                    "enabled": True,
                    "minWordSizeForTypos": {"oneTypo": 2, "twoTypos": 3},
                },
                "searchableAttributes": ["text_clean", "text_raw"],
                "filterableAttributes": ["video_id", "event_id", "shot_id"],
                "displayedAttributes": [
                    "id",
                    "ocr_id",
                    "video_id",
                    "event_id",
                    "shot_id",
                    "timestamp_sec",
                    "text_raw",
                    "text_clean",
                    "confidence",
                ],
            },
        )
        self.client.wait_for_task(int(task["taskUid"]))
