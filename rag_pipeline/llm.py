from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import requests
from dotenv import load_dotenv

from rag_pipeline.utils import summarize_text


@dataclass(frozen=True)
class LlmConfig:
    api_url: str
    api_key: Optional[str] = None
    model: Optional[str] = None
    timeout_s: int = 30
    summary_template: str = (
        "Summarize the following legal section in 1-2 sentences without adding new facts:\n\n{text}"
    )


class LlmSummarizer:
    def __init__(self, config: LlmConfig) -> None:
        self.config = config

    def summarize(self, text: str) -> str:
        prompt = self.config.summary_template.format(text=text)
        headers = {"Content-Type": "application/json"}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        payload = {"prompt": prompt}
        if self.config.model:
            payload["model"] = self.config.model
        response = requests.post(
            self.config.api_url, json=payload, headers=headers, timeout=self.config.timeout_s
        )
        response.raise_for_status()
        data = response.json()
        if isinstance(data, dict):
            for key in ("summary", "text", "output"):
                if key in data and isinstance(data[key], str):
                    return data[key].strip()
            if "choices" in data and data["choices"]:
                choice = data["choices"][0]
                if isinstance(choice, dict):
                    for key in ("text", "message"):
                        if key in choice and isinstance(choice[key], str):
                            return choice[key].strip()
                    if "message" in choice and isinstance(choice["message"], dict):
                        content = choice["message"].get("content")
                        if isinstance(content, str):
                            return content.strip()
        return summarize_text(text)


def load_summarizer_from_env() -> Optional[LlmSummarizer]:
    load_dotenv()
    api_url = os.getenv("LLM_API_URL")
    if not api_url:
        return None
    api_key = os.getenv("LLM_API_KEY")
    model = os.getenv("LLM_MODEL")
    template = os.getenv(
        "LLM_SUMMARY_TEMPLATE",
        "Summarize the following legal section in 1-2 sentences without adding new facts:\n\n{text}",
    )
    timeout_s = int(os.getenv("LLM_TIMEOUT_S", "30"))
    config = LlmConfig(
        api_url=api_url,
        api_key=api_key,
        model=model,
        timeout_s=timeout_s,
        summary_template=template,
    )
    return LlmSummarizer(config)

