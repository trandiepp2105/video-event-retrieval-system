from .config import CaptionConfig
from .model import Qwen3EventCaptioner
from .prompt_builder import CaptionPromptBuilder
from .video_processor import EventCaptionVideoProcessor
from .video_selection import print_selection_summary, select_video_ids


class EventCaptionPipeline:
    def __init__(self, config: CaptionConfig):
        self.config = config
        self.prompt_builder = CaptionPromptBuilder()
        self.captioner = Qwen3EventCaptioner(config)
        self.video_processor = EventCaptionVideoProcessor(
            config=config,
            prompt_builder=self.prompt_builder,
            captioner=self.captioner,
        )

    def run(self) -> None:
        available_video_ids, selected_ids = select_video_ids(self.config)
        print_selection_summary(self.config, available_video_ids, selected_ids)
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

        for video_idx, video_id in enumerate(selected_ids):
            try:
                self.video_processor.process_one_video(video_idx, video_id)
            except Exception as error:
                if self.config.continue_on_error:
                    print(f"[ERROR] Failed video index={video_idx}, video_id={video_id}: {repr(error)}")
                    continue
                raise
