from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class SubtitleJsonLoader:
    OPTIONAL_FIELDS = ("frame_start", "frame_end", "start_time_sec", "end_time_sec")

    def __init__(self, text_field: str = "text") -> None:
        self.text_field = text_field

    def load(self, json_path: Path) -> list[dict[str, Any]]:
        with json_path.open("r", encoding="utf-8") as file:
            items = json.load(file)

        if not isinstance(items, list):
            raise ValueError(f"Subtitle JSON must be a list: {json_path}")

        normalized: list[dict[str, Any]] = []
        for index, item in enumerate(items):
            if not isinstance(item, dict):
                raise ValueError(f"Subtitle item {index} in {json_path} must be a dict")
            if self.text_field not in item:
                raise ValueError(f"Subtitle item {index} in {json_path} missing field: {self.text_field}")

            text = str(item[self.text_field]).strip()
            if not text:
                raise ValueError(f"Subtitle item {index} in {json_path} has empty text")

            payload = dict(item)
            payload[self.text_field] = text
            for field in self.OPTIONAL_FIELDS:
                if field in payload and payload[field] is not None:
                    payload[field] = payload[field]
            normalized.append(payload)
        return normalized


class CaptionJsonLoader:
    OPTIONAL_FIELDS = ("frame_start", "frame_end", "start_time_sec", "end_time_sec")

    def __init__(self, caption_field: str = "caption") -> None:
        self.caption_field = caption_field

    def load(self, json_path: Path) -> list[dict[str, Any]]:
        with json_path.open("r", encoding="utf-8") as file:
            items = json.load(file)

        if not isinstance(items, list):
            raise ValueError(f"Caption JSON must be a list: {json_path}")

        normalized: list[dict[str, Any]] = []
        for index, item in enumerate(items):
            if not isinstance(item, dict):
                raise ValueError(f"Caption item {index} in {json_path} must be a dict")
            if "event_id" not in item:
                raise ValueError(f"Caption item {index} in {json_path} missing field: event_id")
            if self.caption_field not in item:
                raise ValueError(f"Caption item {index} in {json_path} missing field: {self.caption_field}")

            caption = str(item[self.caption_field]).strip()
            if not caption:
                raise ValueError(f"Caption item {index} in {json_path} has empty caption")

            payload = dict(item)
            payload[self.caption_field] = caption
            normalized.append(payload)
        return normalized
