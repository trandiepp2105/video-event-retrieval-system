from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


SYSTEM_PROMPT = """You are a precise translation assistant.

Task:
- Translate a Vietnamese event caption into natural English.
- Keep any dialogue or quoted speech in Vietnamese exactly unchanged.
- Translate only the narrative description around the dialogue.
- Return exactly one plain English caption.
- Do not return JSON.
- Do not explain your answer.

Rules:
- If there is dialogue inside quotation marks, keep that dialogue in Vietnamese exactly as written.
- Do not translate Vietnamese spoken lines.
- Do not add new information.
- Keep the final sentence fluent and retrieval-friendly.
"""


def build_user_prompt(text: str) -> str:
    return (
        "Translate this Vietnamese event caption into English, but keep the spoken dialogue "
        f"in Vietnamese exactly unchanged:\n\n{text.strip()}"
    )


def cleanup_output(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(?:text)?", "", text).strip()
    text = re.sub(r"```$", "", text).strip()
    text = re.sub(r"^Output\s*:\s*", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"^Translation\s*:\s*", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"\s+", " ", text).strip()
    return text


@dataclass
class TranslationConfig:
    model_path: str | Path
    device: str = "cuda"
    max_new_tokens: int = 256
    temperature: float = 0.0
    top_p: float = 1.0
    trust_remote_code: bool = True


class CaptionTranslator:
    def __init__(self, config: TranslationConfig) -> None:
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(config.model_path),
            trust_remote_code=config.trust_remote_code,
        )
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        torch_dtype = torch.bfloat16 if config.device.startswith("cuda") else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            str(config.model_path),
            trust_remote_code=config.trust_remote_code,
            dtype=torch_dtype,
            device_map=config.device,
        )
        self.model.eval()

    def _build_prompts(self, texts: list[str]) -> list[str]:
        prompts = []
        for text in texts:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_user_prompt(text)},
            ]
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            prompts.append(prompt)
        return prompts

    def translate_batch(self, texts: list[str]) -> list[str]:
        if not texts:
            return []
        prompts = self._build_prompts(texts)
        model_inputs = self.tokenizer(prompts, return_tensors="pt", padding=True)
        model_inputs = {k: v.to(self.model.device) for k, v in model_inputs.items()}
        prompt_length = int(model_inputs["input_ids"].shape[1])

        do_sample = self.config.temperature > 0
        with torch.no_grad():
            output_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=self.config.max_new_tokens,
                do_sample=do_sample,
                temperature=self.config.temperature if do_sample else None,
                top_p=self.config.top_p if do_sample else None,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        outputs: list[str] = []
        for row_index in range(len(texts)):
            generated_ids = output_ids[row_index, prompt_length:]
            raw_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
            outputs.append(cleanup_output(raw_text))
        return outputs

    def translate(self, text: str) -> str:
        return self.translate_batch([text])[0]


class TranslationCache:
    def __init__(self) -> None:
        self.cache: Dict[str, str] = {}

    def get(self, text: str) -> str | None:
        return self.cache.get(text)

    def add(self, text: str, translated_text: str) -> None:
        self.cache[text] = translated_text
