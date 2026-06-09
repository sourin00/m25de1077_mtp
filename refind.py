"""REFIND baseline (External route).

CSR(t) = log p(t | D, q, t<) / (log p(t | q, t<) + eps), D = retrieved Wikipedia evidence
(gold wikipedia_url, with a search fallback). Reports BOTH:
  - token-level AUROC over context-bearing items (threshold-free, density-free -> the clean
    number to compare against the probe's AUROC), and
  - char-level IoU at a sample-calibrated threshold (comparable to FLAG-ALL).

Run:  python refind.py
"""
import os
import time
import urllib.parse

import numpy as np
import requests
from sklearn.metrics import roc_auc_score

from config import CFG
from data import load_sample
from wb import load_model, tokenize_with_offsets, token_logprobs, token_labels, merge_flagged_spans
import metrics

_CA = os.environ.get("SSL_CERT_FILE") or os.environ.get("REQUESTS_CA_BUNDLE") or True
_SESSION = requests.Session()
_SESSION.headers.update({"User-Agent": "MuSHROOM-pilot/0.1 (MTech hallucination-detection research)"})
_cache = {}
_last = [0.0]
_MIN_INTERVAL = 0.7   # seconds between API calls (politeness / avoid 429)


def _get(api, params):
    last = RuntimeError("no response after retries")
    for attempt in range(5):
        dt = time.time() - _last[0]
        if dt < _MIN_INTERVAL:
            time.sleep(_MIN_INTERVAL - dt)
        try:
            r = _SESSION.get(api, params=params, timeout=20, verify=_CA)
            _last[0] = time.time()
            if r.status_code == 429:
                last = RuntimeError("429 rate limited")
                time.sleep(6 * (attempt + 1))      # longer back-off on rate limit
                continue
            r.raise_for_status()
            return r.json()
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(1.5 * (attempt + 1))
    raise last


def _extract(lang, title):
    data = _get(f"https://{lang}.wikipedia.org/w/api.php",
                {"action": "query", "prop": "extracts", "explaintext": 1,
                 "format": "json", "redirects": 1, "titles": title})
    pages = data["query"]["pages"]
    return next(iter(pages.values())).get("extract") or ""


def _search_title(lang, query):
    data = _get(f"https://{lang}.wikipedia.org/w/api.php",
                {"action": "query", "list": "search", "srsearch": query,
                 "format": "json", "srlimit": 1})
    hits = data["query"]["search"]
    return hits[0]["title"] if hits else None


def retrieve(item):
    url = (item.wikipedia_url or "").strip()
    key = url or f"search:{item.id}"
    if key in _cache:
        return _cache[key]
    lang = item.lang or "en"
    extract, how = "", ""
    try:
        if url:
            netloc = urllib.parse.urlparse(url).netloc
            lang = netloc.split(".")[0] or lang
            # robust to /wiki/Title AND variant paths like /zh-cn/Title
            seg = urllib.parse.urlparse(url).path.rstrip("/").rsplit("/", 1)[-1]
            title = urllib.parse.unquote(seg)
            extract = _extract(lang, title)
            how = "gold-url"
        if not extract:                       # no url, or gold page had no extract
            t2 = _search_title(lang, (item.question or item.answer or "")[:200])
            if t2:
                extract = _extract(lang, t2)
                how = f"search:{t2[:24]}"
            elif not how:
                how = "search-miss"
    except Exception as e:  # noqa: BLE001
        how = f"error:{type(e).__name__}"
    extract = (extract or "")[: CFG.retrieval_max_chars]
    if not extract:
        print(f"  [retrieve] {item.id}: empty ({how})")
    _cache[key] = (extract, how)
    return extract, how


def run():
    model = load_model()
    items = load_sample()

    per_item = []   # (offsets, csr, labels, has_D) or None
    used = 0
    for it in items:
        if not (it.answer or "").strip():
            per_item.append(None)
            continue
        ans_ids, offsets = tokenize_with_offsets(model, it.answer)
        if not ans_ids:
            per_item.append(None)
            continue
        D, how = retrieve(it)
        q_ids, _ = tokenize_with_offsets(model, (it.question or "") + "\n")
        ctx_ids = (tokenize_with_offsets(model, D + "\n\n" + (it.question or "") + "\n")[0]
                   if D else q_ids)
        lp_with = token_logprobs(model, ctx_ids, ans_ids)
        lp_without = token_logprobs(model, q_ids, ans_ids)
        csr = np.array([w / (wo + CFG.csr_eps) for w, wo in zip(lp_with, lp_without)])
        labels = np.array(token_labels(offsets, it.hard_labels))
        per_item.append((offsets, csr, labels, bool(D)))
        used += 1 if D else 0
        print(f"[{it.lang}] {it.id}: {len(ans_ids)} toks, D={how if D else 'NONE'}")

    # --- token-level AUROC over context-bearing items (clean, threshold-free) ---
    ctx = [(p[1], p[2]) for p in per_item if p is not None and p[3]]
    if ctx:
        csr_c = np.concatenate([c for c, _ in ctx])
        lab_c = np.concatenate([l for _, l in ctx])
        if len(set(lab_c.tolist())) > 1:
            # hallucinated tokens are LESS context-sensitive -> score with -CSR; report
            # the direction-agnostic discriminative power.
            a = roc_auc_score(lab_c, -csr_c)
            auroc = max(a, 1 - a)
            print(f"\nToken-level AUROC over {len(ctx)} context-bearing items: {auroc:.3f} "
                  f"(raw {a:.3f}; {'low-CSR=halluc' if a >= 0.5 else 'high-CSR=halluc'})")

    # --- char-level IoU at a calibrated threshold (comparable to FLAG-ALL) ---
    def preds_at(direction, d):
        out = []
        for p in per_item:
            if p is None:
                out.append([]); continue
            offsets, csr, _, _ = p
            flags = (csr <= d) if direction == "low" else (csr >= d)
            out.append(merge_flagged_spans(offsets, flags.tolist()))
        return out

    valid = [p[1] for p in per_item if p is not None]
    csr_all = np.concatenate(valid) if valid else np.array([0.0])
    grid = np.unique(np.quantile(csr_all, np.linspace(0.05, 0.95, 19)))
    best = (-1.0, "low", float(grid[0]))
    for direction in ("low", "high"):
        for d in grid:
            overall, _ = metrics.mean_iou(items, preds_at(direction, float(d)))
            if overall > best[0]:
                best = (overall, direction, float(d))
    _, direction, d = best
    print(f"\nItems scored: {sum(1 for p in per_item if p)}/{len(items)} | with real evidence: {used}")
    print(f"Calibrated: flag CSR {'<=' if direction == 'low' else '>='} {d:.4f}")
    metrics.report("REFIND (CSR)", items, preds_at(direction, d))


if __name__ == "__main__":
    run()