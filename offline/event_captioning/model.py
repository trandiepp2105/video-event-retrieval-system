import json
import os
import re
from pathlib import Path
from typing import Any

import torch
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from .config import CaptionConfig


def _dtype_from_string(name: str):
    name = str(name).lower()
    if name == "auto":
        return "auto"
    if name in {"float16", "fp16", "half"}:
        return torch.float16
    if name in {"bfloat16", "bf16"}:
        return torch.bfloat16
    if name in {"float32", "fp32"}:
        return torch.float32
    raise ValueError(f"Unsupported torch_dtype: {name}")


def _first_real_device(model) -> torch.device:
    for param in model.parameters():
        if param.device.type != "meta":
            return param.device
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


def _extract_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        candidate = text[start:end + 1]
        return json.loads(candidate)

    raise ValueError(f"Could not parse JSON from model output: {text[:500]}")


def _normalize_process_vision_output(out):
    if not isinstance(out, tuple):
        raise TypeError(f"Unexpected process_vision_info output type: {type(out)}")

    image_inputs = None
    video_inputs = None
    video_kwargs = {}

    if len(out) == 2:
        image_inputs, video_inputs = out
    elif len(out) == 3:
        image_inputs, video_inputs, video_kwargs = out
    elif len(out) == 4:
        image_inputs, video_inputs, video_kwargs, video_metadata = out
        if video_metadata is not None:
            video_kwargs = dict(video_kwargs or {})
            video_kwargs["video_metadata"] = video_metadata
    else:
        raise ValueError(f"Unexpected process_vision_info output length: {len(out)}")

    if video_inputs and isinstance(video_inputs, (list, tuple)):
        if len(video_inputs) > 0 and isinstance(video_inputs[0], tuple) and len(video_inputs[0]) == 2:
            videos = []
            metadatas = []
            for video_tensor, metadata in video_inputs:
                videos.append(video_tensor)
                metadatas.append(metadata)
            video_inputs = videos
            video_kwargs = dict(video_kwargs or {})
            video_kwargs["video_metadata"] = metadatas

    return image_inputs, video_inputs, video_kwargs or {}


def _prepare_triton_ptxas_permissions() -> None:
    try:
        import triton  # type: ignore
    except Exception:
        return

    triton_root = Path(triton.__file__).resolve().parent
    bin_dir = triton_root / "backends" / "nvidia" / "bin"
    if not bin_dir.exists():
        return

    executable_candidates = []
    for path in sorted(bin_dir.glob("ptxas*")):
        try:
            current_mode = path.stat().st_mode
            desired_mode = current_mode | 0o111
            if desired_mode != current_mode:
                os.chmod(path, desired_mode)
            if os.access(path, os.X_OK):
                executable_candidates.append(path)
        except OSError:
            continue

    preferred = None
    for name in ("ptxas", "ptxas-blackwell"):
        candidate = bin_dir / name
        if candidate.exists() and os.access(candidate, os.X_OK):
            preferred = candidate
            break
    if preferred is None and executable_candidates:
        preferred = executable_candidates[0]

    if preferred is not None:
        os.environ.setdefault("TRITON_PTXAS_PATH", str(preferred))


class Qwen3EventCaptioner:
    def __init__(self, config: CaptionConfig):
        self.config = config
        _prepare_triton_ptxas_permissions()

        dtype = _dtype_from_string(config.torch_dtype)
        self.processor = AutoProcessor.from_pretrained(
            str(config.model_path),
            trust_remote_code=config.trust_remote_code,
            local_files_only=config.local_files_only,
        )
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            str(config.model_path),
            torch_dtype=dtype,
            device_map=config.device_map,
            trust_remote_code=config.trust_remote_code,
            local_files_only=config.local_files_only,
        )
        self.model.eval()
        self.input_device = _first_real_device(self.model)

    @torch.inference_mode()
    def caption_clip(self, clip_path: Path, prompt: str) -> tuple[str, str]:
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": str(clip_path),
                        "fps": float(self.config.video_sample_fps),
                        "min_pixels": int(self.config.min_pixels),
                        "max_pixels": int(self.config.max_pixels),
                    },
                    {
                        "type": "text",
                        "text": prompt,
                    },
                ],
            }
        ]

        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        vision_out = process_vision_info(
            messages,
            image_patch_size=int(self.config.image_patch_size),
            return_video_kwargs=True,
            return_video_metadata=True,
        )
        image_inputs, video_inputs, video_kwargs = _normalize_process_vision_output(vision_out)

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
            **video_kwargs,
        )
        inputs = inputs.to(self.input_device)

        gen_kwargs = {
            "max_new_tokens": int(self.config.max_new_tokens),
            "do_sample": bool(self.config.do_sample),
        }
        if self.config.do_sample:
            if self.config.temperature is not None:
                gen_kwargs["temperature"] = float(self.config.temperature)
            if self.config.top_p is not None:
                gen_kwargs["top_p"] = float(self.config.top_p)

        generated_ids = self.model.generate(**inputs, **gen_kwargs)
        input_len = inputs["input_ids"].shape[1]
        generated_trimmed = generated_ids[:, input_len:]

        raw_text = self.processor.batch_decode(
            generated_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()

        obj = _extract_json_object(raw_text)
        caption = str(obj.get("retrieval_caption", "")).strip()
        if not caption:
            raise ValueError(f"Missing retrieval_caption in output: {raw_text}")

        return caption, raw_text
