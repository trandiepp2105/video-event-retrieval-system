from __future__ import annotations

from dataclasses import asdict

from tqdm import tqdm

from .config import PrepareTranslationsConfig
from .io_utils import ensure_dir, load_json, save_json
from .translation import CaptionTranslator, TranslationCache, TranslationConfig


def prepare_translations(config: PrepareTranslationsConfig):
    ensure_dir(config.output_dir)
    translator = CaptionTranslator(
        TranslationConfig(
            model_path=config.translation_model_path,
            device=config.device,
        )
    )
    cache = TranslationCache()
    summary = {
        "config": asdict(config),
        "videos": [],
        "num_videos": 0,
        "num_items": 0,
        "num_translated": 0,
        "num_reused_existing": 0,
    }

    caption_paths = sorted(config.event_captions_dir.glob("*.json"))
    start_index = 0 if config.start_index is None else max(int(config.start_index), 0)
    if config.end_index is None:
        selected_caption_paths = caption_paths[start_index:]
    else:
        selected_caption_paths = caption_paths[start_index : int(config.end_index) + 1]

    for caption_path in tqdm(caption_paths, desc="Videos"):
        if caption_path not in selected_caption_paths:
            continue
        items = load_json(caption_path)
        if not isinstance(items, list):
            continue

        output_path = config.output_dir / caption_path.name
        if output_path.exists() and not config.overwrite:
            existing_items = load_json(output_path)
            if isinstance(existing_items, list) and len(existing_items) == len(items):
                translated_count = sum(1 for item in existing_items if str(item.get("translated_caption", "")).strip())
                summary["videos"].append(
                    {
                        "video_id": caption_path.stem,
                        "num_items": len(existing_items),
                        "num_translated": translated_count,
                        "reused_existing_file": True,
                    }
                )
                summary["num_videos"] += 1
                summary["num_items"] += len(existing_items)
                summary["num_reused_existing"] += translated_count
                continue

        output_items = []
        translated_count = 0
        pending_requests = []
        for item_index, item in enumerate(items):
            caption = str(item.get("caption", "")).strip()
            translated_caption = str(item.get("translated_caption", "")).strip()
            output_item = dict(item)
            output_item["caption"] = caption
            output_item["translated_caption"] = translated_caption
            output_items.append(output_item)
            if translated_caption:
                translated_count += 1
                continue
            if not caption:
                continue
            cached = cache.get(caption)
            if cached is not None:
                output_items[item_index]["translated_caption"] = cached
                translated_count += 1
                continue
            pending_requests.append((item_index, caption))

        if pending_requests:
            for batch_start in tqdm(
                range(0, len(pending_requests), max(int(config.translation_batch_size), 1)),
                desc=f"Translating {caption_path.stem}",
                leave=False,
            ):
                batch_requests = pending_requests[batch_start : batch_start + max(int(config.translation_batch_size), 1)]
                batch_captions = [caption for _, caption in batch_requests]
                batch_outputs = translator.translate_batch(batch_captions)
                for (item_index, caption), translated_caption in zip(batch_requests, batch_outputs):
                    output_items[item_index]["translated_caption"] = translated_caption
                    cache.add(caption, translated_caption)
                    translated_count += 1

        save_json(output_items, output_path)
        summary["videos"].append(
            {
                "video_id": caption_path.stem,
                "num_items": len(output_items),
                "num_translated": translated_count,
                "reused_existing_file": False,
            }
        )
        summary["num_videos"] += 1
        summary["num_items"] += len(output_items)
        summary["num_translated"] += translated_count

    summary["selected_start_index"] = config.start_index
    summary["selected_end_index"] = config.end_index
    save_json(summary, config.output_dir / "translation_summary.json")
    return config.output_dir
