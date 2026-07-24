from .config import CaptionConfig
from .io_utils import list_event_video_ids


def select_video_ids(config: CaptionConfig) -> tuple[list[str], list[str]]:
    available_video_ids = list_event_video_ids(config.event_output_dir)
    if len(available_video_ids) == 0:
        raise RuntimeError(
            f"No event video directories with events.json found in: {config.event_output_dir}"
        )

    if config.video_ids:
        selected_ids: list[str] = []
        missing_ids: list[str] = []
        available_set = set(available_video_ids)
        for video_id in config.video_ids:
            video_id = str(video_id)
            if video_id in available_set:
                selected_ids.append(video_id)
            else:
                missing_ids.append(video_id)

        if missing_ids:
            print(f"[WARN] Missing event_output folders for video IDs: {missing_ids}")
        if not selected_ids:
            raise RuntimeError("No valid video_ids matched any folder inside event_output_dir.")
        return available_video_ids, selected_ids

    start = max(0, int(config.start_index))
    end = config.end_index
    if end is None:
        selected_ids = available_video_ids[start:]
    else:
        selected_ids = available_video_ids[start:int(end)]
    return available_video_ids, selected_ids


def print_selection_summary(
    config: CaptionConfig,
    available_video_ids: list[str],
    selected_ids: list[str],
) -> None:
    print(f"Found {len(available_video_ids)} event-output video folder(s).")
    if config.video_ids:
        print(f"Processing {len(selected_ids)} video(s) from explicit video_ids.")
        return

    print(
        f"Processing {len(selected_ids)} video(s), "
        f"index range [{config.start_index}, {config.end_index})."
    )
