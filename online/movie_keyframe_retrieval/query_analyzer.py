from __future__ import annotations

import json
from typing import Any


TRANSLATION_SYSTEM_PROMPT = r"""
You are an expert Vietnamese-to-English translator for movie retrieval queries.

Your job is to translate the full Vietnamese user query into natural English while preserving retrieval-critical meaning.

Rules:
1. Output JSON only.
2. Return exactly one object with one field: {"en_query": "..."}.
3. Translate the full query to English.
4. Keep quoted dialogue, OCR text, names, titles, and proper nouns unchanged when appropriate.
5. Do not explain anything.
6. Do not add information not present in the query.
7. If the query contains multiple actions in sequence, keep that sequence in the English translation.
"""


STAGE_SYSTEM_PROMPT = r"""
You are an AI assistant for multilingual video retrieval query analysis.

Convert one Vietnamese movie/video search query into chronological retrieval stages for shot-level soft temporal retrieval.

Each stage must represent one single action, one single event, or one single visual moment that can be independently searched.

Return exactly one valid JSON object with this schema:
{
  "stages": [
    {
      "visual": "",
      "ocr": "",
      "subtitle": ""
    }
  ]
}

Field meaning:
- visual: only visible content, written in English. This is for visual retrieval.
- ocr: exact visible written text only, keep original language, do not translate.
- subtitle: exact spoken words, subtitle text, or dialogue only, keep original language, do not translate.

Rules:
1. Output JSON only.
2. Do not use markdown.
3. Do not invent details.
4. Split temporal sequences into multiple stages in chronological order.
5. Temporal connectors such as "sau đó", "rồi", "tiếp theo", "before", "after", "then" usually imply a stage boundary.
6. A stage must be self-contained and concrete.
7. visual must never contain spoken words or OCR text verbatim.
8. subtitle must contain only dialogue/subtitle cues.
9. If the query says that a person says, tells, asks, shouts, whispers, reads aloud, or repeats some words, then those spoken words must be put into subtitle, even if they are not surrounded by quotation marks.
10. If the query explicitly gives the spoken content, preserve it in subtitle in the original language.
11. Do not split one single visual situation into multiple stages only because the same person repeats the same or similar dialogue.
12. Repeated speech such as "lag rồi, lag rồi" should remain inside one stage as one subtitle string, not be split into multiple duplicate stages.
13. ocr must contain only exact visible written text. If not explicitly given, leave it empty.
14. Use simple natural English in visual.
15. If the query describes one single ongoing moment, return one stage only.
16. If multiple consecutive stages happen in the same location, setting, room, vehicle, or scene context, later stages must inherit and restate that shared spatial context in visual.
17. If there is no explicit change of place or scene, assume the later action still happens in the same space as the previous stage.
18. A later stage must not drop important active scene context such as cafe, bus, classroom, office, bedroom, hospital room, art studio, restaurant, or street if that context still applies.
19. When a person performs a new action inside the same scene, write the action together with the inherited context, for example: "a girl in an art studio opening her phone to read a message" instead of only "a girl opening her phone".
20. If a later stage refers to an object or person from an earlier stage, rewrite that context concretely instead of using vague references.

Example:
Input: Người đàn ông nhìn vào điện thoại có dòng chữ "I miss you", rồi bật khóc.
Output:
{
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

Example:
Input: Một cặp nam nữ ngồi đối diện nhau trong phòng vẽ, rồi cô gái nhận được tin nhắn và mở điện thoại ra xem.
Output:
{
  "stages": [
    {
      "visual": "a couple sitting facing each other in an art studio",
      "ocr": "",
      "subtitle": ""
    },
    {
      "visual": "a girl in an art studio opening her phone to read a message",
      "ocr": "",
      "subtitle": ""
    }
  ]
}

Example:
Input: Một cô gái gọi điện video với bố khi đang ở trên một chuyến xe bus đông đúc, trong điện thoại người bố liên tục nói với con gái rằng lag rồi, lag rồi.
Output:
{
  "stages": [
    {
      "visual": "a girl on a crowded bus having a video call with her father on a phone",
      "ocr": "",
      "subtitle": "lag rồi, lag rồi"
    }
  ]
}
"""


DEFAULT_SYSTEM_PROMPT = STAGE_SYSTEM_PROMPT


class MovieQueryAnalyzer:
    def __init__(
        self,
        model_id: str,
        *,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        translation_system_prompt: str = TRANSLATION_SYSTEM_PROMPT,
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
        self.stage_system_prompt = system_prompt
        self.translation_system_prompt = translation_system_prompt
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

        model_kwargs = {"device_map": device_map, "trust_remote_code": True}
        if quantization_config is not None:
            model_kwargs["quantization_config"] = quantization_config
        else:
            model_kwargs["torch_dtype"] = torch_dtype

        self.model = AutoModelForCausalLM.from_pretrained(self.model_id, **model_kwargs)
        self.model.eval()

    def _build_prompt(self, system_prompt: str, user_text: str) -> str:
        messages = [
            {
                "role": "system",
                "content": system_prompt + "\n\nDo not think step by step. Do not output <think>. Output the final JSON object only.",
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

    def _input_device(self):
        if hasattr(self.model, "hf_device_map"):
            for device in self.model.hf_device_map.values():
                if isinstance(device, int):
                    return self.torch.device(f"cuda:{device}")
                if isinstance(device, str) and device not in {"cpu", "disk"}:
                    return self.torch.device(device)
        return self.model.device

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
            return cleaned[start:end + 1].strip()
        return cleaned

    def _run_json_prompt(self, system_prompt: str, user_text: str) -> dict[str, Any]:
        prompt = self._build_prompt(system_prompt, user_text)
        raw_output = self._generate(prompt)
        json_text = self._extract_json_payload(raw_output)
        return json.loads(json_text)

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

    def translate_query(self, vi_query: str) -> str:
        result = self._run_json_prompt(self.translation_system_prompt, vi_query)
        return str(result.get("en_query", "")).strip()

    def analyze_stages(self, vi_query: str) -> list[dict[str, str]]:
        result = self._run_json_prompt(self.stage_system_prompt, vi_query)
        stages = result.get("stages", [])
        if not isinstance(stages, list):
            raise ValueError("Model output does not contain a valid stages list.")
        normalized = []
        for stage in stages:
            item = self._normalize_stage_item(stage)
            if item is not None:
                normalized.append(item)
        return self._merge_adjacent_duplicate_stages(normalized)

    def analyze(self, query: str, return_raw: bool = False) -> dict[str, Any] | None:
        try:
            result = {
                "en_query": self.translate_query(query),
                "stages": self.analyze_stages(query),
            }
            if return_raw:
                result["_raw_output"] = None
            return result
        except Exception as exc:
            print(f"⚠️ Query analysis failed: {exc}")
            return None

    def to_search_queries(self, analyzed: dict[str, Any]) -> list[dict[str, str]]:
        stages = analyzed.get("stages", []) if isinstance(analyzed, dict) else []
        if not isinstance(stages, list):
            return []
        normalized: list[dict[str, str]] = []
        for stage in stages:
            item = self._normalize_stage_item(stage)
            if item is not None:
                normalized.append(item)
        return normalized
