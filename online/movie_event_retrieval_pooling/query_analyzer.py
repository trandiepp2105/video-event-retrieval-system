from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any


QUERY_ANALYSIS_SYSTEM_PROMPT = r"""
You are an expert multilingual movie retrieval query analyzer.

Analyze one Vietnamese movie/video search query for shot-level soft temporal retrieval.
In one response, translate the full query into English and split it into chronological retrieval stages.

Return exactly one valid JSON object with this schema:
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

Field meaning:
- en_query: natural English translation of the complete Vietnamese query.
- visual: only visible and searchable content for one stage, written in English.
- ocr: exact visible written text only, kept in the original language.
- subtitle: exact spoken words, subtitle text, or dialogue only, kept in the original language.

Translation rules:
1. Preserve all retrieval-critical meaning and chronological order in en_query.
2. Keep quoted dialogue, visible text, names, titles, and proper nouns unchanged when appropriate.
3. Do not add information that is not present in the query.

Stage rules:
1. Output JSON only. Do not use markdown or explanations.
2. Each stage represents one single action, event, or visual moment that can be independently searched.
3. Split temporal sequences into stages in chronological order. Connectors such as 'sau đó', 'rồi', 'tiếp theo', 'before', 'after', and 'then' usually imply a stage boundary.
4. If the query describes one ongoing moment, return one stage only.
5. Every stage must be self-contained and concrete.
6. visual must not contain spoken words or OCR text as dialogue.
7. If visible text is important to recognizing an object, visual may describe that object and its visible label naturally, while ocr must still contain the exact text.
8. ocr must contain only exact visible written text. If no visible text is explicitly given, leave it empty.
9. subtitle must contain only dialogue or spoken-language cues. If none is given, leave it empty.
10. If a person says, tells, asks, shouts, whispers, reads aloud, or repeats words, put those words in subtitle even when they are not quoted.
11. Preserve explicitly provided spoken content in its original language.
12. Repeated speech such as 'lag rồi, lag rồi' remains in one subtitle string and must not create duplicate stages.
13. Use simple natural English in visual.
14. If consecutive stages remain in the same location, room, vehicle, setting, or scene, later stages must inherit and restate that context in visual.
15. If there is no explicit scene or location change, assume the later action remains in the previous setting.
16. Do not drop active context such as cafe, bus, classroom, office, bedroom, hospital room, art studio, restaurant, street, camp gate, or military camp.
17. Resolve references to people and objects concretely instead of using vague pronouns.
18. Do not create duplicate adjacent stages.

Example 1:
Input: Người đàn ông nhìn vào điện thoại có dòng chữ "I miss you", rồi bật khóc.
Output:
{
  "en_query": "A man looks at a phone displaying the words I miss you, then starts crying.",
  "stages": [
    {
      "visual": "a man looking at a phone screen",
      "ocr": "I miss you",
      "subtitle": ""
    },
    {
      "visual": "a man crying while looking at a phone screen",
      "ocr": "I miss you",
      "subtitle": ""
    }
  ]
}

Example 2:
Input: Một cặp nam nữ ngồi đối diện nhau trong phòng vẽ, rồi cô gái nhận được tin nhắn và mở điện thoại ra xem.
Output:
{
  "en_query": "A man and a woman sit facing each other in an art studio, then the woman receives a message and opens her phone to read it.",
  "stages": [
    {
      "visual": "a man and a woman sitting facing each other in an art studio",
      "ocr": "",
      "subtitle": ""
    },
    {
      "visual": "a woman in an art studio opening her phone to read a message",
      "ocr": "",
      "subtitle": ""
    }
  ]
}

Example 3:
Input: Một cô gái gọi điện video với bố khi đang ở trên một chuyến xe bus đông đúc, trong điện thoại người bố liên tục nói với con gái rằng lag rồi, lag rồi.
Output:
{
  "en_query": "A girl has a video call with her father on a crowded bus while he repeatedly tells her that it is lagging.",
  "stages": [
    {
      "visual": "a girl on a crowded bus having a video call with her father on a phone",
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
        self.torch_dtype = torch_dtype
        self.device_map = device_map
        self.max_new_tokens = int(max_new_tokens)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True)
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

        model_kwargs: dict[str, Any] = {"device_map": self.device_map, "trust_remote_code": True}
        if quantization_config is not None:
            model_kwargs["quantization_config"] = quantization_config
        else:
            model_kwargs["torch_dtype"] = self.torch_dtype

        self.model = AutoModelForCausalLM.from_pretrained(self.model_id, **model_kwargs)
        self.model.eval()

    def _input_device(self):
        if hasattr(self.model, "hf_device_map"):
            for device in self.model.hf_device_map.values():
                if isinstance(device, int):
                    return self.torch.device(f"cuda:{device}")
                if isinstance(device, str) and device not in {"cpu", "disk"}:
                    return self.torch.device(device)
        return self.model.device

    def _build_prompt(self, user_text: str) -> str:
        messages = [
            {
                "role": "system",
                "content": QUERY_ANALYSIS_SYSTEM_PROMPT
                + "\n\nDo not think step by step. Do not output <think>. Output the final JSON object only.",
            },
            {"role": "user", "content": user_text.strip() + "\n\n/no_think"},
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

    def _generate(self, prompt_text: str) -> str:
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
    def _extract_json_payload(text: str) -> str:
        cleaned = str(text).strip().replace("```json", "").replace("```", "").strip()
        if "</think>" in cleaned:
            cleaned = cleaned.split("</think>", 1)[-1].strip()
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1 and end > start:
            return cleaned[start : end + 1].strip()
        return cleaned

    def _run_json_prompt(self, user_text: str) -> tuple[dict[str, Any], str]:
        prompt = self._build_prompt(user_text)
        raw_output = self._generate(prompt)
        json_text = self._extract_json_payload(raw_output)
        return json.loads(json_text), raw_output

    @staticmethod
    def _normalize_stage_item(stage: Any) -> dict[str, str] | None:
        if not isinstance(stage, dict):
            return None
        visual = str(stage.get("visual", "")).strip()
        ocr = str(stage.get("ocr", "")).strip()
        subtitle = str(stage.get("subtitle", stage.get("dialogue", ""))).strip()
        if not any([visual, ocr, subtitle]):
            return None
        return {
            "visual": visual,
            "ocr": ocr,
            "subtitle": subtitle,
        }

    @staticmethod
    def _merge_adjacent_duplicate_stages(stages: list[dict[str, str]]) -> list[dict[str, str]]:
        if not stages:
            return []
        merged = [dict(stages[0])]
        for stage in stages[1:]:
            prev = merged[-1]
            if stage == prev:
                continue
            if (
                stage.get("visual", "") == prev.get("visual", "")
                and stage.get("ocr", "") == prev.get("ocr", "")
                and stage.get("subtitle", "")
                and prev.get("subtitle", "")
                and stage.get("subtitle", "") == prev.get("subtitle", "")
            ):
                prev["subtitle"] = f"{prev['subtitle']}, {stage['subtitle']}"
                continue
            merged.append(dict(stage))
        return merged

    def analyze(self, vi_query: str, return_raw: bool = False) -> dict[str, Any]:
        payload, raw_output = self._run_json_prompt(vi_query)
        en_query = str(payload.get("en_query", "")).strip()
        if not en_query:
            raise ValueError("Model output does not contain a valid en_query.")

        raw_stages = payload.get("stages", [])
        if not isinstance(raw_stages, list):
            raise ValueError("Model output does not contain a valid stages list.")

        normalized_stages = []
        for stage in raw_stages:
            item = self._normalize_stage_item(stage)
            if item is not None:
                normalized_stages.append(item)

        result: dict[str, Any] = {
            "en_query": en_query,
            "stages": self._merge_adjacent_duplicate_stages(normalized_stages),
        }
        if return_raw:
            result["_raw_output"] = raw_output
        return result
