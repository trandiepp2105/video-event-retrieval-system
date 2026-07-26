from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any


DEFAULT_SYSTEM_PROMPT = r"""
You are an AI assistant for Vietnamese movie retrieval query analysis.

Your task:
1. Translate the full Vietnamese query into English as `en_query`.
2. Decompose the query into chronological retrieval stages.
3. Each stage must contain only these fields:
   - visual: English visual description
   - ocr: original visible written text if explicitly provided, otherwise ""
   - subtitle: original spoken/subtitle text if explicitly provided or strongly implied, otherwise ""

Return exactly one valid JSON object:
{
  "en_query": "",
  "stages": [
    {
      "visual": "",
      "ocr": "",
      "subtitle": ""
    }
  ]
}

Rules:
- Output JSON only.
- `en_query` must be a faithful English translation of the whole query.
- `visual` must be in English.
- `ocr` and `subtitle` must keep the original language from the user query.
- A stage is one temporal action or one independently retrievable visual moment.
- Split stages chronologically when the query contains temporal progression like "rồi", "sau đó", "tiếp theo", "then", "after that".
- If later stages happen in the same place/scene, repeat the shared scene context in the later stage visual description.
- If someone says/tells/asks/shouts/repeats some words, those words should usually go into `subtitle`.
- If exact visible written text appears, put it in `ocr`.
- Do not put spoken words into `visual`.
- Do not invent missing OCR/subtitle text.
- If repeated spoken text appears like "lag rồi, lag rồi", keep it in a single stage subtitle field, do not create duplicated stages just because of repetition.

Example:
Input:
Một cô gái gọi điện video với bố khi đang ở trên một chuyến xe bus đông đúc, trong điện thoại người bố liên tục nói với con gái rằng lag rồi, lag rồi.

Output:
{
  "en_query": "A girl is on a crowded bus making a video call with her father, and on the phone her father repeatedly tells her that it is lagging.",
  "stages": [
    {
      "visual": "a girl on a crowded bus making a video call on her phone",
      "ocr": "",
      "subtitle": "lag rồi, lag rồi"
    }
  ]
}
"""


@dataclass(frozen=True)
class StageQuery:
    visual: str = ""
    ocr: str = ""
    subtitle: str = ""

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "StageQuery":
        return cls(
            visual=str(payload.get("visual", "")).strip(),
            ocr=str(payload.get("ocr", "")).strip(),
            subtitle=str(payload.get("subtitle", "")).strip(),
        )

    def is_empty(self) -> bool:
        return not (self.visual or self.ocr or self.subtitle)


class SoftTemporalShotQueryAnalyzer:
    def __init__(
        self,
        model_id: str,
        *,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        torch_dtype: str = "auto",
        device_map: str = "auto",
        max_new_tokens: int = 768,
        load_in_4bit: bool = False,
        load_in_8bit: bool = False,
        bnb_8bit_cpu_offload: bool = False,
    ) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        self.torch = torch
        self.model_id = model_id
        self.system_prompt = system_prompt
        self.max_new_tokens = int(max_new_tokens)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True)
        if getattr(self.tokenizer, "padding_side", None) != "left":
            self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        quantization_config = None
        if load_in_4bit:
            quantization_config = BitsAndBytesConfig(load_in_4bit=True)
        elif load_in_8bit:
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_enable_fp32_cpu_offload=bnb_8bit_cpu_offload,
            )

        model_kwargs: dict[str, Any] = {
            "device_map": device_map,
            "trust_remote_code": True,
        }
        if quantization_config is not None:
            model_kwargs["quantization_config"] = quantization_config
        else:
            model_kwargs["torch_dtype"] = torch_dtype

        self.model = AutoModelForCausalLM.from_pretrained(self.model_id, **model_kwargs)
        self.model.eval()

    def build_prompt(self, query: str) -> str:
        messages = [
            {
                "role": "system",
                "content": self.system_prompt + "\n\nOutput the final JSON object only. Do not output analysis or markdown.",
            },
            {"role": "user", "content": query.strip() + "\n\n/no_think"},
        ]
        if hasattr(self.tokenizer, "apply_chat_template") and self.tokenizer.chat_template is not None:
            try:
                return self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
            except TypeError:
                return self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
        text = ""
        for msg in messages:
            text += f"{str(msg['role']).upper()}:\n{str(msg['content'])}\n\n"
        text += "ASSISTANT:\n"
        return text

    def _input_device(self):
        if hasattr(self.model, "hf_device_map"):
            for device in self.model.hf_device_map.values():
                if isinstance(device, int):
                    return self.torch.device(f"cuda:{device}")
                if isinstance(device, str) and device not in {"cpu", "disk"}:
                    return self.torch.device(device)
        return self.model.device

    def generate_text(self, prompt_text: str) -> str:
        inputs = self.tokenizer(prompt_text, return_tensors="pt")
        device = self._input_device()
        inputs = {k: v.to(device) for k, v in inputs.items()}
        eos_ids = []
        if self.tokenizer.eos_token_id is not None:
            eos_ids.append(self.tokenizer.eos_token_id)
        im_end_id = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
        if isinstance(im_end_id, int) and im_end_id >= 0:
            eos_ids.append(im_end_id)
        eos_ids = list(dict.fromkeys(eos_ids)) or None
        with self.torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=eos_ids,
            )
        generated_ids = output_ids[0][inputs["input_ids"].shape[-1] :]
        return self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    @staticmethod
    def extract_json_payload(text: str) -> str:
        cleaned = str(text).strip().replace("```json", "").replace("```", "").strip()
        if "</think>" in cleaned:
            cleaned = cleaned.split("</think>", 1)[-1].strip()
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1 and end > start:
            return cleaned[start : end + 1].strip()
        return cleaned

    @staticmethod
    def _as_str(value: Any) -> str:
        return "" if value is None else str(value).strip()

    def _normalize_stage(self, stage: Any) -> StageQuery | None:
        if not isinstance(stage, dict):
            return None
        stage_query = StageQuery(
            visual=self._as_str(stage.get("visual", "")),
            ocr=self._as_str(stage.get("ocr", "")),
            subtitle=self._as_str(stage.get("subtitle", "")),
        )
        if stage_query.is_empty():
            return None
        return stage_query

    @staticmethod
    def _merge_adjacent_duplicate_stages(stages: list[StageQuery]) -> list[StageQuery]:
        if not stages:
            return []
        merged = [stages[0]]
        for stage in stages[1:]:
            prev = merged[-1]
            if stage == prev:
                continue
            merged.append(stage)
        return merged

    def normalize_result(self, parsed: Any) -> dict[str, Any] | None:
        if not isinstance(parsed, dict):
            return None
        en_query = self._as_str(parsed.get("en_query", ""))
        stages_payload = parsed.get("stages", [])
        if not isinstance(stages_payload, list):
            return None
        stages = []
        for item in stages_payload:
            normalized = self._normalize_stage(item)
            if normalized is not None:
                stages.append(normalized)
        stages = self._merge_adjacent_duplicate_stages(stages)
        if not en_query and not stages:
            return None
        return {
            "en_query": en_query,
            "stages": [
                {
                    "visual": stage.visual,
                    "ocr": stage.ocr,
                    "subtitle": stage.subtitle,
                }
                for stage in stages
            ],
        }

    def analyze(self, query: str, return_raw: bool = False) -> dict[str, Any] | None:
        prompt_text = self.build_prompt(query)
        raw_output = self.generate_text(prompt_text)
        json_text = self.extract_json_payload(raw_output)
        parsed = json.loads(json_text)
        normalized = self.normalize_result(parsed)
        if normalized is None:
            raise ValueError("Query analyzer returned no usable JSON result.")
        if return_raw:
            normalized["_raw_output"] = raw_output
        return normalized
