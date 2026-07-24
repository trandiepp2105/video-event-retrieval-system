import json
from typing import Any


def collect_event_subtitles(
    subtitles: list[dict[str, Any]],
    event_start_sec: float,
    event_end_sec: float,
    video_fps: float,
) -> list[dict[str, Any]]:
    selected = []
    for item in subtitles:
        if "start_time_sec" in item and "end_time_sec" in item:
            abs_start_sec = float(item["start_time_sec"])
            abs_end_sec = float(item["end_time_sec"])
        elif "frame_start" in item and "frame_end" in item:
            sub_start_frame = float(item["frame_start"])
            sub_end_frame = float(item["frame_end"])
            abs_start_sec = sub_start_frame / video_fps
            abs_end_sec = sub_end_frame / video_fps
        else:
            continue

        text = str(item.get("text", "")).replace("\n", " ").strip()
        if not text:
            continue

        if abs_end_sec < abs_start_sec:
            abs_end_sec = abs_start_sec

        if abs_end_sec < event_start_sec or abs_start_sec > event_end_sec:
            continue

        selected.append(
            {
                "start_time_sec": round(abs_start_sec, 3),
                "end_time_sec": round(abs_end_sec, 3),
                "relative_start_sec": round(abs_start_sec - event_start_sec, 3),
                "relative_end_sec": round(abs_end_sec - event_start_sec, 3),
                "text": text,
            }
        )

    return selected


def format_subtitle_block(event_subtitles: list[dict[str, Any]]) -> str:
    if not event_subtitles:
        return "[]"

    compact = []
    for item in event_subtitles:
        compact.append(
            {
                "relative_time": f'{item["relative_start_sec"]:.3f}s - {item["relative_end_sec"]:.3f}s',
                "text": item["text"],
            }
        )

    return json.dumps(compact, ensure_ascii=False, indent=2)
