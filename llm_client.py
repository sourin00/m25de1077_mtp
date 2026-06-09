"""Swappable LLM backend for decomposition + routing.

Default: local Ollama (free). Switch to a paid API by changing the LLM_* env vars
in config.py's docstring — this file never needs editing.
"""
import json
import re
from openai import OpenAI

from config import CFG

_client = None


def client():
    global _client
    if _client is None:
        _client = OpenAI(base_url=CFG.llm_base_url, api_key=CFG.llm_api_key)
    return _client


def chat_json(system, user, model=None, temperature=None):
    """Call the chat model and parse a JSON response (tolerates fences / trailing text)."""
    resp = client().chat.completions.create(
        model=model or CFG.llm_model,
        temperature=CFG.llm_temperature if temperature is None else temperature,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    return _parse_json(resp.choices[0].message.content)


def _parse_json(text):
    """Decode the FIRST JSON value in the text, ignoring code fences and any trailing
    prose. raw_decode stops at the end of the first value, so 'Extra data' (the model
    appending text after the array) no longer breaks parsing."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?", "", text)
    text = re.sub(r"```$", "", text).strip()

    candidates = [i for i in (text.find("["), text.find("{")) if i != -1]
    start = min(candidates) if candidates else 0
    obj, _ = json.JSONDecoder().raw_decode(text[start:])
    return obj