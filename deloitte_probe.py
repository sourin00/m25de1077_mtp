"""Deloitte-style residual-probe baseline (white-box).

Extract resid_post features per answer token from the probe model, train a logistic head
to classify each token as hallucinated, and report token-level AUROC + char-level IoU on
held-out items. This is the white-box ground truth a SAGE/Hybrid surrogate would try to
approximate, so its MARGIN OVER REFIND is the number that informs whether the Hybrid bet
is worth it.

Reduced from Deloitte's full 172k-feature setup: we use a few resid_post layers
(config.probe_layers) + logistic regression for a fast, fair pilot baseline.

Run:  python deloitte_probe.py
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from config import CFG
from data import load_sample
from wb import (load_model, tokenize_with_offsets, resid_features,
                token_labels, merge_flagged_spans)
import metrics


def build(model, items):
    """Per item -> (offsets, X[na,d], y[na]) or None (no usable answer)."""
    per_item = []
    for it in items:
        if not (it.answer or "").strip():
            per_item.append(None)
            continue
        ans_ids, offsets = tokenize_with_offsets(model, it.answer)
        if not ans_ids:
            per_item.append(None)
            continue
        q_ids, _ = tokenize_with_offsets(model, (it.question or "") + "\n")
        X = resid_features(model, q_ids, ans_ids, CFG.probe_layers)
        y = np.array(token_labels(offsets, it.hard_labels))
        per_item.append((offsets, X, y))
        print(f"[{it.lang}] {it.id}: {len(ans_ids)} toks, {int(y.sum())} gold-hallucinated")
    return per_item


def run():
    model = load_model()
    items = load_sample()
    per_item = build(model, items)

    rng = np.random.RandomState(CFG.seed)
    valid = [i for i, p in enumerate(per_item) if p is not None]
    rng.shuffle(valid)
    n_test = max(1, int(len(valid) * CFG.probe_test_frac))
    test_idx = sorted(valid[:n_test])
    train_idx = valid[n_test:]

    Xtr = np.concatenate([per_item[i][1] for i in train_idx])
    ytr = np.concatenate([per_item[i][2] for i in train_idx])
    print(f"\nTrain tokens: {len(ytr)} ({int(ytr.sum())} hallucinated) | test items: {len(test_idx)}")
    if ytr.sum() == 0 or ytr.sum() == len(ytr):
        print("Degenerate training labels (need both classes). Increase the sample.")
        return

    clf = LogisticRegression(max_iter=2000, class_weight="balanced")
    clf.fit(Xtr, ytr)

    Xte = np.concatenate([per_item[i][1] for i in test_idx])
    yte = np.concatenate([per_item[i][2] for i in test_idx])
    proba = clf.predict_proba(Xte)[:, 1]
    auroc = roc_auc_score(yte, proba) if len(set(yte.tolist())) > 1 else float("nan")
    print(f"Token-level AUROC (test): {auroc:.3f}")

    test_items = [items[i] for i in test_idx]

    def preds_at(thr):
        out = []
        for i in test_idx:
            offsets, X, _ = per_item[i]
            flags = (clf.predict_proba(X)[:, 1] >= thr).tolist()
            out.append(merge_flagged_spans(offsets, flags))
        return out

    best = (-1.0, 0.5)
    for thr in np.linspace(0.1, 0.9, 17):
        overall, _ = metrics.mean_iou(test_items, preds_at(float(thr)))
        if overall > best[0]:
            best = (overall, float(thr))
    _, thr = best
    print(f"Calibrated probability threshold: {thr:.2f}")
    metrics.report("Deloitte residual-probe", test_items, preds_at(thr))


if __name__ == "__main__":
    run()
