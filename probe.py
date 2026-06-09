"""The hallucination probe: train, predict, and per-language threshold calibration.

One place for the train/threshold/apply logic that was copy-pasted across the experiment
scripts. Works on any per-token feature matrix (SAE features or raw residuals).

Per-language calibration is the cheap fix for zh: a single global threshold, set by the
less-dense languages, under-flags Chinese (where answers are far more saturated). Calibrating
one threshold per language on the TRAIN items recovers most of that gap.

    probe = train(Xtr, ytr)                       # feature selection + scaler + logreg
    prob  = [predict(probe, f) for f in feats]    # per-item token probabilities
    thr   = calibrate(train_items, prob_tr, offs_tr, per_language=True)
    spans = predict_spans(test_items, prob_te, offs_te, thr)
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

import metrics
from wb import merge_flagged_spans


def train(Xtr, ytr, min_freq=0.005, C=0.5):
    """Select features that fire on >=min_freq of train tokens, standardize, fit logreg."""
    keep = (Xtr > 0).mean(0) >= min_freq
    if keep.sum() < 50:
        keep = (Xtr > 0).mean(0) > 0
    sc = StandardScaler().fit(Xtr[:, keep])
    clf = LogisticRegression(max_iter=2000, class_weight="balanced", C=C)
    clf.fit(sc.transform(Xtr[:, keep]), ytr)
    return {"keep": keep, "keep_idx": np.where(keep)[0], "sc": sc, "clf": clf}


def predict(probe, X):
    """Per-token P(hallucinated) for a [na, d] feature matrix."""
    return probe["clf"].predict_proba(probe["sc"].transform(X[:, probe["keep"]]))[:, 1]


def _mean_iou_at(items, probs, offs, idxs, thr):
    preds = [merge_flagged_spans(offs[i], [p > thr for p in probs[i]]) for i in idxs]
    return metrics.mean_iou([items[i] for i in idxs], preds)[0]


def calibrate(items, probs, offs, grid=None, per_language=True):
    """Threshold(s) maximizing IoU on the given (train) items. Returns {lang: thr, '_global': thr}."""
    grid = np.linspace(0.05, 0.9, 18) if grid is None else grid
    thresholds = {}
    if per_language:
        for lang in sorted(set(it.lang for it in items)):
            idxs = [i for i, it in enumerate(items) if it.lang == lang]
            best = (-1.0, 0.5)
            for thr in grid:
                o = _mean_iou_at(items, probs, offs, idxs, float(thr))
                if o > best[0]:
                    best = (o, float(thr))
            thresholds[lang] = best[1]
    best = (-1.0, 0.5)
    for thr in grid:
        o = _mean_iou_at(items, probs, offs, list(range(len(items))), float(thr))
        if o > best[0]:
            best = (o, float(thr))
    thresholds["_global"] = best[1]
    return thresholds


def predict_spans(items, probs, offs, thresholds):
    """Merged char-spans per item, each item flagged at its language's threshold."""
    out = []
    for it, p, o in zip(items, probs, offs):
        thr = thresholds.get(it.lang, thresholds.get("_global", 0.5))
        out.append(merge_flagged_spans(o, [pi > thr for pi in p]))
    return out
