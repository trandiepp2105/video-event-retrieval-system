from __future__ import annotations

import json
from typing import Any, Optional


DEFAULT_SYSTEM_PROMPT = r"""
You are an AI assistant for multilingual video retrieval query analysis.

Analyze a natural-language movie/video search query and convert it into chronological retrieval stages for a multimodal retrieval system.

The system has the following channels:

* visual: visible content for vision-language models such as CLIP or SigLIP.
* dialogue: spoken words, subtitles, or meaningful human speech/audio.
* ocr: static visible written text on objects, signs, papers, screens, phones, computers, logos, labels, etc.
* metadata: explicitly named entities such as character names, actor names, movie titles, locations, organizations, genres, countries, or time periods.

Return exactly one valid JSON object:
{
"stages": [
{
"visual": "",
"dialogue": "",
"ocr": "",
"metadata": {
"character_names": [],
"actor_names": [],
"movie_title": "",
"location": "",
"organization": "",
"genre": "",
"country": "",
"time_period": "",
"other": []
}
}
]
}

General rules:

1. Output JSON only. Do not use markdown or explanations.
2. Use English for visual and metadata values.
3. Do not translate dialogue, OCR text, or proper nouns.
4. Use an empty string for missing string fields and an empty list for missing list fields.
5. Do not invent information that is not stated or clearly implied by the query.

Field rules:

6. The visual field must contain only information that can be directly observed visually, like a human describing an image. It should describe people, objects, actions, scenes, appearance, spatial relations, object states, facial expressions, body poses, and visible screen/media content.

7. The dialogue field contains only spoken words, subtitles, quoted speech, or meaningful human speech/audio. If the exact words are provided, copy them verbatim. If the exact words are not provided, leave the dialogue field empty.

8. The ocr field contains only exact visible written text. If the query says that text exists but does not provide the exact words, leave the ocr field empty and describe the text-bearing object in the visual field.

9. The metadata field contains only explicitly named entities. Do not put generic descriptions such as "a man", "a woman", or "a girl" into metadata.

10. Strictly separate visual, dialogue, and ocr. The visual field may describe visible speaking actions such as "shouting", "talking", "whispering", "singing", or "reading aloud", but it must never include the actual spoken words, quoted speech, subtitle text, OCR text, or their translation/paraphrase. Put exact spoken words only in dialogue and exact written words only in ocr.

Stage splitting and context propagation:

11. A stage is one independently retrievable visual moment. It should usually correspond to one main action, one scene state, or one keyframe-level event.

12. If the query describes one single visual moment, return one stage. If it describes a temporal sequence, split it into multiple stages in chronological order.

13. Temporal connectors such as "sau đó", "rồi", "tiếp theo", "kế tiếp", "trước khi", "sau khi", "then", "next", "after", or "before" usually indicate a stage boundary. When such connectors appear, default to splitting at that point unless the connected clauses clearly describe the same single visual moment. Do not keep temporally ordered actions in one visual field using words like "then", "after that", "rồi", or "sau đó".

14. Split into a new stage when there is a clear change in the main visual moment, including changes in action, location, setting, visual focus, body pose, physical state, emotional state, object state, object location, ownership, visibility, camera view, close-up, flashback, memory, dream, imagination, or hallucinated scene.

15. Split cause-result sequences into separate stages when the result is visually different from the cause, such as "vì ... nên ...", "dở quá nên ...", "thấy vậy nên ...", "sợ quá nên ...", or similar expressions.

16. A stage may contain multiple objects or small simultaneous actions only if they belong to the same visual moment and support the same main action.

17. Each stage must be self-contained and understandable without reading previous stages. If a later stage refers to something from an earlier stage, repeat only the necessary concrete context, including people, objects, actions, events, locations, ongoing situations, or visible media content.

18. Replace vague references such as "it", "that", "this", "the scene", "the situation", "the event", "đó", "nó", "cảnh đó", "sự việc đó", "chỗ đó", "tấm ảnh đó", "dòng chữ đó", "món đó", "người đó", or "cái đó" with concrete visual descriptions.

19. Do not make later stages too generic. Avoid outputs such as "a man speaking", "a girl crying", "a person entering the scene", or "someone reacting" if the query provides more concrete context.

Translation and object accuracy:

20. Translate Vietnamese visual descriptions into simple, common English.

21. Preserve important concrete nouns such as tools, food, drinks, clothes, vehicles, animals, weapons, documents, screens, containers, and furniture.

22. Do not replace a specific object with a different object or an overly generic object.

23. If uncertain about an exact translation, use a broader but still correct phrase instead of an incorrect specific word.

24. Avoid rare, fabricated, or hallucinated English words.

Metadata classification:

25. character_names: fictional character names explicitly mentioned.
26. actor_names: actor / actress / public figure names explicitly mentioned.
27. movie_title: movie title explicitly mentioned.
28. location: explicit place or setting name.
29. organization: explicit institution / sect / palace / clan / group / company name.
30. genre: genre or style explicitly mentioned.
31. country: country or nationality context explicitly mentioned.
32. time_period: dynasty, era, historical period, modern / ancient context explicitly mentioned.
33. other: other named metadata not covered above.

Example 1

Input: Tìm cảnh người đàn ông đang dùng điện thoại để chụp một cây nến đang cháy trên bàn, sau đó gửi tấm ảnh đó qua đoạn chat

Output:
{
"stages": [
{
"visual": "a man using a phone to take a photo of a burning candle on a table",
"dialogue": "",
"ocr": "",
"metadata": {
"character_names": [],
"actor_names": [],
"movie_title": "",
"location": "",
"organization": "",
"genre": "",
"country": "",
"time_period": "",
"other": []
}
},
{
"visual": "a phone chat screen showing a photo of a burning candle on a table being sent",
"dialogue": "",
"ocr": "",
"metadata": {
"character_names": [],
"actor_names": [],
"movie_title": "",
"location": "",
"organization": "",
"genre": "",
"country": "",
"time_period": "",
"other": []
}
}
]
}

Example 2

Input: Người phụ nữ nhìn thấy trên màn hình điện thoại có dòng chữ "I miss you", rồi cô bật khóc

Output:
{
"stages": [
{
"visual": "a woman looking at a phone screen",
"dialogue": "",
"ocr": "I miss you",
"metadata": {
"character_names": [],
"actor_names": [],
"movie_title": "",
"location": "",
"organization": "",
"genre": "",
"country": "",
"time_period": "",
"other": []
}
},
{
"visual": "a woman crying while looking at a phone screen showing the message text",
"dialogue": "",
"ocr": "I miss you",
"metadata": {
"character_names": [],
"actor_names": [],
"movie_title": "",
"location": "",
"organization": "",
"genre": "",
"country": "",
"time_period": "",
"other": []
}
}
]
}

Example 3

Input: Người đàn ông chỉ vào tờ giấy có chữ "CONFIDENTIAL", sau đó đọc to dòng chữ đó

Output:
{
"stages": [
{
"visual": "a man pointing at a paper document",
"dialogue": "",
"ocr": "CONFIDENTIAL",
"metadata": {
"character_names": [],
"actor_names": [],
"movie_title": "",
"location": "",
"organization": "",
"genre": "",
"country": "",
"time_period": "",
"other": []
}
},
{
"visual": "a man reading aloud from a paper document",
"dialogue": "CONFIDENTIAL",
"ocr": "CONFIDENTIAL",
"metadata": {
"character_names": [],
"actor_names": [],
"movie_title": "",
"location": "",
"organization": "",
"genre": "",
"country": "",
"time_period": "",
"other": []
}
}
]
}
"""


class MovieQueryAnalyzer:
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
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    @staticmethod
    def empty_metadata() -> dict[str, Any]:
        return {
            "character_names": [],
            "actor_names": [],
            "movie_title": "",
            "location": "",
            "organization": "",
            "genre": "",
            "country": "",
            "time_period": "",
            "other": [],
        }

    def build_prompt(self, query: str) -> str:
        messages = [
            {
                "role": "system",
                "content": self.system_prompt + "\n\nDo not think step by step. Do not output <think>. Output the final JSON object only.",
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
        generated_ids = output_ids[0][inputs["input_ids"].shape[-1]:]
        return self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    @staticmethod
    def extract_json_payload(text: str) -> str:
        cleaned = str(text).strip().replace("```json", "").replace("```", "").strip()
        if "</think>" in cleaned:
            cleaned = cleaned.split("</think>", 1)[-1].strip()
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1 and end > start:
            return cleaned[start:end + 1].strip()
        return cleaned

    @staticmethod
    def _as_list(value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, list):
            return [str(x).strip() for x in value if str(x).strip()]
        if isinstance(value, str):
            return [value.strip()] if value.strip() else []
        value = str(value).strip()
        return [value] if value else []

    @staticmethod
    def _as_str(value: Any) -> str:
        return "" if value is None else str(value).strip()

    def normalize_stage(self, stage: Any) -> Optional[dict[str, Any]]:
        if not isinstance(stage, dict):
            return None
        metadata = stage.get("metadata", {})
        if not isinstance(metadata, dict):
            metadata = {**self.empty_metadata(), "other": self._as_list(metadata)}
        else:
            metadata = {
                "character_names": self._as_list(metadata.get("character_names", [])),
                "actor_names": self._as_list(metadata.get("actor_names", [])),
                "movie_title": self._as_str(metadata.get("movie_title", "")),
                "location": self._as_str(metadata.get("location", "")),
                "organization": self._as_str(metadata.get("organization", "")),
                "genre": self._as_str(metadata.get("genre", "")),
                "country": self._as_str(metadata.get("country", "")),
                "time_period": self._as_str(metadata.get("time_period", "")),
                "other": self._as_list(metadata.get("other", [])),
            }
        normalized = {
            "visual": self._as_str(stage.get("visual", stage.get("text", ""))),
            "dialogue": self._as_str(stage.get("dialogue", stage.get("subtitle", ""))),
            "ocr": self._as_str(stage.get("ocr", "")),
            "metadata": metadata,
        }
        has_main_content = any(normalized[k] for k in ["visual", "dialogue", "ocr"])
        has_metadata = any(
            bool(v) if isinstance(v, list) else bool(str(v).strip())
            for v in metadata.values()
        )
        if not (has_main_content or has_metadata):
            return None
        return normalized

    def normalize_result(self, parsed: Any) -> Optional[dict[str, Any]]:
        if isinstance(parsed, dict) and isinstance(parsed.get("stages"), list):
            stages = parsed["stages"]
        elif isinstance(parsed, dict) and isinstance(parsed.get("states"), list):
            stages = parsed["states"]
        elif isinstance(parsed, list):
            stages = parsed
        elif isinstance(parsed, dict):
            stages = [parsed]
        else:
            return None
        normalized_stages = []
        for stage in stages:
            normalized = self.normalize_stage(stage)
            if normalized is not None:
                normalized_stages.append(normalized)
        if not normalized_stages:
            return None
        return {"stages": normalized_stages}

    def analyze(self, query: str, return_raw: bool = False) -> Optional[dict[str, Any]]:
        try:
            prompt_text = self.build_prompt(query)
            raw_output = self.generate_text(prompt_text)
            json_text = self.extract_json_payload(raw_output)
            parsed = json.loads(json_text)
            result = self.normalize_result(parsed)
            if result is None:
                raise ValueError("Model returned no usable stage.")
            if return_raw:
                result["_raw_output"] = raw_output
            return result
        except Exception as exc:
            print(f"⚠️ Query analysis failed: {exc}")
            return None

    def to_search_queries(self, analyzed: dict[str, Any]) -> list[dict[str, str]]:
        normalized = self.normalize_result(analyzed)
        if not normalized:
            return []
        search_queries = []
        for stage in normalized["stages"]:
            query = {
                "text": stage.get("visual", "").strip(),
                "subtitle": stage.get("dialogue", "").strip(),
                "ocr": stage.get("ocr", "").strip(),
            }
            query = {k: v for k, v in query.items() if v}
            if query:
                search_queries.append(query)
        return search_queries
