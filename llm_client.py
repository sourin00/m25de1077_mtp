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


def _complete(system, user, model=None, temperature=None):
    resp = client().chat.completions.create(
        model=model or CFG.llm_model,
        temperature=CFG.llm_temperature if temperature is None else temperature,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    return resp.choices[0].message.content


def chat_json(system, user, model=None, temperature=None):
    """Call the chat model and parse a JSON response (tolerates fences / trailing text)."""
    return _parse_json(_complete(system, user, model, temperature))


def chat_text(system, user, model=None, temperature=None):
    """Raw text response — for outputs that carry verbatim source text and must not be
    forced through JSON escaping (e.g. claim+evidence in a line-delimited format)."""
    return _complete(system, user, model, temperature)


def _parse_json(text):
    """Decode the FIRST JSON value in the text, ignoring code fences and any trailing
    prose. raw_decode stops at the end of the first value, so 'Extra data' (the model
    appending text after the array) no longer breaks parsing. On a decode error, retry
    once after repairing invalid backslash escapes (verbatim source text copied into a
    JSON string often carries stray '\\' or incomplete '\\u' sequences)."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?", "", text)
    text = re.sub(r"```$", "", text).strip()

    candidates = [i for i in (text.find("["), text.find("{")) if i != -1]
    start = min(candidates) if candidates else 0
    body = text[start:]
    try:
        obj, _ = json.JSONDecoder().raw_decode(body)
        return obj
    except json.JSONDecodeError:
        obj, _ = json.JSONDecoder().raw_decode(_repair_escapes(body))
        return obj


def _repair_escapes(s):
    s = re.sub(r"\\u(?![0-9a-fA-F]{4})", r"\\\\u", s)   # incomplete \u -> literal backslash+u
    s = re.sub(r'\\(?!["\\/bfnrtu])', r"\\\\", s)        # any backslash not starting a valid escape
    return s