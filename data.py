"""Load a small, language-balanced sample of Mu-SHROOM.

Mu-SHROOM on HF is organized as one config PER LANGUAGE ('cs', 'en', 'es', 'zh', ...).
We fetch ONLY each language's test parquet directly via the hf:// filesystem.

Why not load_dataset(config, split="test")? That prepares the whole config, which pulls
the large train_unlabeled parquet too (and it 403'd through the proxy). Targeting the
test file surgically avoids downloading anything we don't need.

The test split is the 100-item official eval set the published leaderboard IoU numbers
are computed on, so pilot results here are directly comparable to UCSC / Deloitte / etc.

Field names can drift, so normalization is defensive and prints the real columns once.
If 'question'/'answer' come back empty, add the real key to the _first(...) calls below.
"""
import random
import time
from dataclasses import dataclass, field
from typing import Optional

from datasets import load_dataset

from config import CFG


def _load_test_parquet(lang, attempts=3):
    """Fetch one language's test parquet, retrying through transient proxy 403s."""
    pattern = f"hf://datasets/{CFG.hf_dataset}/{lang}/test-*.parquet"
    last = None
    for i in range(attempts):
        try:
            return load_dataset("parquet", data_files=pattern, split="train")
        except Exception as e:  # noqa: BLE001
            last = e
            print(f"[data] {lang}: attempt {i + 1}/{attempts} failed; retrying...")
            time.sleep(2 * (i + 1))
    print(f"[data] {lang}: giving up after {attempts} attempts ({last})")
    return None


@dataclass
class Item:
    id: str
    lang: str
    question: str
    answer: str
    tokens: Optional[list] = None
    logits: Optional[list] = None
    soft_labels: list = field(default_factory=list)
    hard_labels: list = field(default_factory=list)
    wikipedia_url: str = ""


def _first(record, *keys, default=None):
    for k in keys:
        if k in record and record[k] is not None:
            return record[k]
    return default


def _normalize(r, lang):
    return Item(
        id=str(_first(r, "id", "datapoint_id", default="")),
        lang=lang,  # authoritative: from the config name, not a record field
        question=_first(r, "model_input", "input", "question", default=""),
        answer=_first(r, "model_output_text", "output_text", "model_output", default=""),
        tokens=_first(r, "model_output_tokens", "output_tokens", "tokens"),
        logits=_first(r, "model_output_logits", "output_logits", "logits"),
        soft_labels=_first(r, "soft_labels", "soft", default=[]),
        hard_labels=_first(r, "hard_labels", "hard", default=[]),
        wikipedia_url=_first(r, "wikipedia_url", "wiki_url", "url", default="") or "",
    )


def load_sample(n=None):
    rng = random.Random(CFG.seed)
    k = n or CFG.sample_per_lang
    out = []
    printed_cols = False

    for lang in CFG.langs:
        # Surgically pull only this language's test parquet (avoids train_unlabeled).
        ds = _load_test_parquet(lang)
        if ds is None:
            continue

        if not printed_cols:
            print(f"[data] columns: {ds.column_names}")
            printed_cols = True

        items = [_normalize(r, lang) for r in ds]
        rng.shuffle(items)
        picked = items[:k]
        labeled = sum(1 for it in picked if it.hard_labels)
        print(f"[data] {lang}: {len(picked)} sampled "
              f"(test, {labeled}/{len(picked)} have hard_labels)")
        out.extend(picked)

    return out


if __name__ == "__main__":
    sample = load_sample()
    print(f"\nTotal sampled: {len(sample)}")
    if sample:
        ex = sample[0]
        print(f"\nExample [{ex.lang}] id={ex.id}")
        print("Q:", (ex.question or "")[:200])
        print("A:", (ex.answer or "")[:300])
        print("hard_labels (first 3):", (ex.hard_labels or [])[:3])
        print("has tokens:", ex.tokens is not None, "| has logits:", ex.logits is not None)