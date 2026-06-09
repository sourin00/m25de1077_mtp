"""Character-level Intersection-over-Union — the Mu-SHROOM metric. Every method
(REFIND, the Deloitte-style probe, and later TRACE/SAGE) reports through this, so the
numbers are directly comparable to the published leaderboard.
"""
from collections import defaultdict


def _charset(spans):
    s = set()
    for a, b in spans:
        s.update(range(int(a), int(b)))
    return s


def iou(pred_spans, gold_spans):
    P, G = _charset(pred_spans), _charset(gold_spans)
    if not P and not G:
        return 1.0          # correctly predicted "no hallucination"
    if not P or not G:
        return 0.0
    return len(P & G) / len(P | G)


def mean_iou(items, preds):
    """items: objects with .hard_labels and .lang ; preds: predicted spans (aligned)."""
    vals = [iou(p, it.hard_labels) for it, p in zip(items, preds)]
    overall = sum(vals) / len(vals) if vals else 0.0
    by_lang = defaultdict(list)
    for it, v in zip(items, vals):
        by_lang[it.lang].append(v)
    return overall, {l: sum(vs) / len(vs) for l, vs in by_lang.items()}


def report(name, items, preds):
    overall, per = mean_iou(items, preds)
    print(f"\n=== {name}: char-level IoU ===")
    for l, v in sorted(per.items()):
        print(f"  {l}: {v:.3f}")
    print(f"  MEAN: {overall:.3f}")
    return overall
