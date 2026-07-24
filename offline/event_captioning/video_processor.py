from pathlib import Path
from typing import Any
import traceback

from tqdm import tqdm

from .config import CaptionConfig
from .io_utils import (
    get_event_json_from_output,
    get_sidecar_json,
    load_json,
    resolve_video_path,
    save_json,
    validate_event_item,
)
from .model import Qwen3EventCaptioner
from .prompt_builder import CaptionPromptBuilder
from .subtitles import collect_event_subtitles, format_subtitle_block
from .video_utils import cut_event_clip, get_video_fps, make_tmp_dir


class EventCaptionVideoProcessor:
    def __init__(
        self,
        config: CaptionConfig,
        prompt_builder: CaptionPromptBuilder,
        captioner: Qwen3EventCaptioner,
    ):
        self.config = config
        self.prompt_builder = prompt_builder
        self.captioner = captioner

    def process_one_video(self, video_idx: int, video_id: str) -> None:
        event_path = get_event_json_from_output(self.config.event_output_dir, video_id)
        video_path = resolve_video_path(self.config.videos_dir, video_id)
        if video_path is None:
            raise FileNotFoundError(
                f"Video file not found for video_id={video_id} in {self.config.videos_dir}"
            )

        subtitle_path = get_sidecar_json(self.config.subtitles_dir, video_path)
        output_path = self.config.output_dir / f"{video_id}.json"

        if output_path.exists() and not self.config.overwrite:
            print(f"[SKIP] index={video_idx} video={video_path.name}, output exists: {output_path}")
            return

        if not event_path.exists():
            raise FileNotFoundError(f"Event file not found for video {video_path.name}: {event_path}")

        events = load_json(event_path)
        if not isinstance(events, list):
            raise ValueError(f"Event file must contain a list: {event_path}")

        subtitles: list[dict[str, Any]] = []
        if subtitle_path.exists():
            subtitles = load_json(subtitle_path)
            if not isinstance(subtitles, list):
                raise ValueError(f"Subtitle file must contain a list: {subtitle_path}")
        else:
            print(f"[WARN] Subtitle file not found for {video_path.name}: {subtitle_path}; using empty subtitles.")

        fps = get_video_fps(video_path)
        print(f"\n[VIDEO] index={video_idx} name={video_path.name} events={len(events)} fps={fps:.4f}")

        results: list[dict[str, Any]] = []
        with make_tmp_dir(self.config.tmp_dir) as tmp:
            tmp_dir = Path(tmp)
            for event_index, event in enumerate(tqdm(events, desc=f"Captioning {video_id}")):
                self._process_one_event(
                    video_id=video_id,
                    video_path=video_path,
                    event_path=event_path,
                    output_path=output_path,
                    subtitles=subtitles,
                    fps=fps,
                    tmp_dir=tmp_dir,
                    event_index=event_index,
                    event=event,
                    results=results,
                )

        save_json(results, output_path)
        print(f"[DONE] Saved: {output_path}")

    def _process_one_event(
        self,
        video_id: str,
        video_path: Path,
        event_path: Path,
        output_path: Path,
        subtitles: list[dict[str, Any]],
        fps: float,
        tmp_dir: Path,
        event_index: int,
        event: dict[str, Any],
        results: list[dict[str, Any]],
    ) -> None:
        stage = "init"
        clip_path: Path | None = None
        try:
            stage = "validate_event"
            validate_event_item(event, event_path, event_index)

            event_id = event.get("event_id", event_index)
            start_sec = float(event["start_time_sec"])
            end_sec = float(event["end_time_sec"])
            if end_sec <= start_sec:
                raise ValueError(f"Invalid event time: start={start_sec}, end={end_sec}")

            stage = "collect_subtitles"
            event_subtitles = collect_event_subtitles(
                subtitles=subtitles,
                event_start_sec=start_sec,
                event_end_sec=end_sec,
                video_fps=fps,
            )

            stage = "build_prompt"
            subtitle_block = format_subtitle_block(event_subtitles)
            prompt = self.prompt_builder.build(subtitle_block)

            clip_path = tmp_dir / f"{video_id}_event_{event_id}.mp4"
            stage = "cut_event_clip"
            cut_event_clip(
                video_path=video_path,
                start_sec=start_sec,
                end_sec=end_sec,
                output_path=clip_path,
                codec=self.config.clip_codec,
                loglevel=self.config.ffmpeg_loglevel,
            )

            stage = "caption_clip"
            caption, _raw_response = self.captioner.caption_clip(clip_path, prompt)

            results.append(
                {
                    "event_id": event_id,
                    "start_time_sec": start_sec,
                    "end_time_sec": end_sec,
                    "caption": caption,
                }
            )

            if self.config.save_every_event:
                stage = "save_json_success"
                save_json(results, output_path)

        except Exception as error:
            if not self.config.continue_on_error:
                raise

            results.append(
                {
                    "event_id": event.get("event_id", event_index),
                    "start_time_sec": event.get("start_time_sec"),
                    "end_time_sec": event.get("end_time_sec"),
                    "caption": "",
                    "error": repr(error),
                }
            )

            if self.config.save_every_event:
                try:
                    save_json(results, output_path)
                except Exception as save_error:
                    print(
                        f"[ERROR] video={video_id}, event_index={event_index}, "
                        f"stage=save_json_error_item, output_path={output_path}: {repr(save_error)}"
                    )

            clip_exists = bool(clip_path is not None and clip_path.exists())
            clip_size = clip_path.stat().st_size if clip_exists and clip_path is not None else None
            print(
                f"[ERROR] video={video_id}, event_index={event_index}, stage={stage}, "
                f"video_path={video_path}, output_path={output_path}, "
                f"clip_path={clip_path}, clip_exists={clip_exists}, clip_size={clip_size}: {repr(error)}"
            )
            if isinstance(error, PermissionError):
                print("[TRACEBACK]")
                print(traceback.format_exc())
