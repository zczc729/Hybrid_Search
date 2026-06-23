"""LLM client for answer generation through LM Studio/OpenAI-compatible API."""

from __future__ import annotations

from openai import OpenAI

from config import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL_NAME
from prompts import RAG_PROMPT_TEMPLATE, SYS_PROMPT_TEMPLATE


def get_client() -> OpenAI:
    return OpenAI(base_url=LLM_BASE_URL, api_key=LLM_API_KEY)


def clean_generation(text: str) -> str:
    """Remove common special tokens that may appear in local LLM output."""
    return (
        text.replace("</s>", "")
        .replace("<s>", "")
        .replace("[INST]", "")
        .replace("[/INST]", "")
        .strip()
    )


def call_llm_with_context(query: str, context: str, temperature: float = 0.1, max_tokens: int = 160) -> str:
    user_content = RAG_PROMPT_TEMPLATE.format(query=query, context=context)
    client = get_client()
    completion = client.chat.completions.create(
        model=LLM_MODEL_NAME,
        messages=[
            {"role": "system", "content": SYS_PROMPT_TEMPLATE},
            {"role": "user", "content": user_content},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=0.8,
        frequency_penalty=0.4,
        presence_penalty=0.2,
    )
    return clean_generation(completion.choices[0].message.content or "")
