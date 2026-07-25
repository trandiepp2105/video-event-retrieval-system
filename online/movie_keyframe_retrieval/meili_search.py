from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import meilisearch
from rapidfuzz import fuzz
from tqdm.auto import tqdm


def normalize_text(text: str) -> str:
    return " ".join(str(text).split()).lower()


class Score2Text:
    def __init__(self, w_partial: float = 1, w_ngrams: float = 1, w_token: float = 1):
        self.w_partial = w_partial
        self.w_ngrams = w_ngrams
        self.w_token = w_token

    def custom_partial_ratio(self, q_no_space: str, d_no_space: str) -> float:
        len_q = len(q_no_space)
        len_d = len(d_no_space)
        if len_q == 0:
            return 0.0
        if len_q >= len_d:
            return fuzz.ratio(q_no_space, d_no_space) / 100
        return fuzz.partial_ratio(q_no_space, d_no_space) / 100

    @staticmethod
    def generate_ngrams(tokens: list[str], n_values=(1,)) -> dict[int, list[str]]:
        return {
            n: [" ".join(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]
            for n in n_values
        }

    def custom_ngrams_ratio(self, q_split: list[str], d_split: list[str]) -> float:
        len_q = len(q_split)
        q = " ".join(q_split)
        if len_q == 0:
            return 0.0
        n_values = (len_q - 1, len_q) if len_q > 1 else (len_q,)
        ngrams = self.generate_ngrams(d_split, n_values)
        best = 0.0
        for ngram_values in ngrams.values():
            for text in ngram_values:
                best = max(best, fuzz.ratio(q, text))
        return best / 100

    def custom_token_ratio(self, q: str, d: str) -> float:
        return fuzz.token_set_ratio(str(q or ""), str(d or "")) / 100

    def w_score(self, q: str, d: str) -> float:
        q_split = q.split()
        d_split = d.split()
        q_no_space = "".join(q_split)
        d_no_space = "".join(d_split)
        partial = self.custom_partial_ratio(q_no_space, d_no_space)
        ngrams = self.custom_ngrams_ratio(q_split, d_split)
        token = self.custom_token_ratio(q, d)
        return (
            self.w_partial * partial
            + self.w_ngrams * ngrams
            + self.w_token * token
        ) / (self.w_partial + self.w_ngrams + self.w_token)


class MeiliSearchService:
    def __init__(
        self,
        *,
        url: str,
        api_key: str,
        ocr_index_name: str,
        subtitle_index_name: str,
        limit_search: int = 500,
    ) -> None:
        self.url = str(url)
        self.api_key = str(api_key)
        self.client = meilisearch.Client(self.url, self.api_key)
        self.ocr_index_name = str(ocr_index_name)
        self.subtitle_index_name = str(subtitle_index_name)
        self.limit_search = int(limit_search)
        self.scoring = Score2Text()

    def _wait_for_task(self, task_info: Any) -> None:
        task_uid = None
        if hasattr(task_info, "task_uid"):
            task_uid = task_info.task_uid
        elif isinstance(task_info, dict):
            task_uid = task_info.get("taskUid") or task_info.get("uid")
        if task_uid is None:
            return
        self.client.wait_for_task(task_uid)

    def create_indices(self) -> None:
        self._create_index(
            self.ocr_index_name,
            {
                "searchableAttributes": ["text"],
                "displayedAttributes": ["id", "video_name", "frame_index", "text"],
                "filterableAttributes": ["video_name", "frame_index"],
            },
        )
        self._create_index(
            self.subtitle_index_name,
            {
                "searchableAttributes": ["text"],
                "displayedAttributes": ["id", "video_name", "frame_start", "frame_end", "text"],
                "filterableAttributes": ["video_name", "frame_start", "frame_end"],
            },
        )

    def _create_index(self, index_name: str, extra_settings: dict[str, Any]) -> None:
        try:
            index = self.client.get_index(index_name)
        except Exception:
            task = self.client.create_index(index_name, {"primaryKey": "id"})
            self._wait_for_task(task)
            index = self.client.get_index(index_name)
        settings = {
            "rankingRules": ["typo", "proximity", "words", "exactness", "attribute", "sort"],
            "pagination": {"maxTotalHits": 5000},
            "distinctAttribute": None,
            "typoTolerance": {
                "enabled": True,
                "minWordSizeForTypos": {"oneTypo": 2, "twoTypos": 3},
            },
            **extra_settings,
        }
        task = index.update_settings(settings)
        self._wait_for_task(task)

    def index_ocr_dataset(self, data_path: Path) -> None:
        index = self.client.get_index(self.ocr_index_name)
        json_files = list(Path(data_path).rglob("*.json"))
        batch = []
        for i, jf in enumerate(tqdm(json_files, desc=self.ocr_index_name)):
            with open(jf, "r", encoding="utf-8") as file:
                data = json.load(file)
            video_name = jf.stem
            if isinstance(data, dict):
                for frame_index, text in data.items():
                    normalized_text = normalize_text(text)
                    if normalized_text:
                        batch.append(
                            {
                                "id": f"{self.ocr_index_name}_{video_name}_{frame_index}",
                                "video_name": video_name,
                                "frame_index": int(frame_index),
                                "text": normalized_text,
                            }
                        )
            elif isinstance(data, list):
                for item in data:
                    text = normalize_text(item.get("text", ""))
                    frame_index = item.get("frame_id", item.get("frame_index", -1))
                    if text and frame_index != -1:
                        batch.append(
                            {
                                "id": f"{self.ocr_index_name}_{video_name}_{frame_index}",
                                "video_name": video_name,
                                "frame_index": int(frame_index),
                                "text": text,
                            }
                        )
            if (i + 1) % 100 == 0 or (i + 1) == len(json_files):
                if batch:
                    index.add_documents(batch)
                    batch = []

    def index_subtitle_dataset(self, data_path: Path) -> None:
        index = self.client.get_index(self.subtitle_index_name)
        json_files = list(Path(data_path).rglob("*.json"))
        batch = []
        for i, jf in enumerate(tqdm(json_files, desc=self.subtitle_index_name)):
            with open(jf, "r", encoding="utf-8") as file:
                subs = json.load(file)
            video_name = jf.stem
            if not isinstance(subs, list):
                continue
            for sub in subs:
                normalized_text = normalize_text(sub.get("text", ""))
                if normalized_text:
                    batch.append(
                        {
                            "id": f"{self.subtitle_index_name}_{video_name}_{sub['frame_start']}",
                            "video_name": video_name,
                            "frame_start": int(sub["frame_start"]),
                            "frame_end": int(sub["frame_end"]),
                            "text": normalized_text,
                        }
                    )
            if (i + 1) % 100 == 0 or (i + 1) == len(json_files):
                if batch:
                    index.add_documents(batch)
                    batch = []

    def search_ocr(self, query: str, size: int = 1000) -> list[dict[str, Any]]:
        return self._search_text_index(query, self.ocr_index_name, size)

    def search_subtitle(self, query: str, size: int = 1000) -> list[dict[str, Any]]:
        return self._search_text_index(query, self.subtitle_index_name, size)

    def _search_text_index(self, query: str, index_name: str, size: int) -> list[dict[str, Any]]:
        normalized_query = normalize_text(query)
        if not normalized_query:
            return []
        index = self.client.get_index(index_name)
        response = index.search(
            normalized_query,
            {
                "limit": self.limit_search,
                "attributesToRetrieve": ["*"],
                "showRankingScore": False,
                "matchingStrategy": "last",
            },
        )
        results = response.get("hits", [])
        for result in results:
            text = normalize_text(result.get("text", ""))
            result["_rankingScore"] = self._scoring_matching(text, normalized_query)
        results.sort(key=lambda item: item["_rankingScore"], reverse=True)
        return results[: int(size)]

    def _scoring_matching(self, text: str, query: str) -> float:
        text_norm = normalize_text(text)
        query_norm = normalize_text(query)
        if not query_norm or not text_norm:
            return 0.0
        if query_norm in text_norm:
            return 1.0
        return self.scoring.w_score(q=query_norm, d=text_norm)
