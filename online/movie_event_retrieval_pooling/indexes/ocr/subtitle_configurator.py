from __future__ import annotations

from .meilisearch_client import MeiliSearchClient


class SubtitleIndexConfigurator:
    def __init__(self, client: MeiliSearchClient) -> None:
        self.client = client

    def configure(self, index_uid: str) -> None:
        self.client.create_index(index_uid=index_uid, primary_key="subtitle_id")
        task = self.client.update_settings(
            index_uid=index_uid,
            settings={
                "searchableAttributes": ["text"],
                "filterableAttributes": ["video_id"],
                "displayedAttributes": [
                    "subtitle_id",
                    "video_id",
                    "start_time_sec",
                    "end_time_sec",
                    "frame_start",
                    "frame_end",
                    "text",
                ],
            },
        )
        self.client.wait_for_task(int(task["taskUid"]))
