"""TRACE integrator (the thesis experiment): does route-conditioned verification beat the
flat probe?

v1 isolates what ROUTING adds on top of the validated SAE-feature probe. It trains one
SAE-feature probe (the Relational verifier) on train-item tokens, tags every answer token with
a route via the anchored atom spans from decompose_route.json, then compares three predictors
on the SAME held-out items at the SAME train-calibrated threshold:

  FLAT      probe flags every token above threshold  (the 0.439-style baseline, on Gemma)
  TRACE-A   FLAT, but tokens routed Subjective are never flagged (opinions aren't factual
            hallucinations); uncovered tokens still use the probe
  TRACE-B   flag ONLY tokens inside Relational/External claims (suppress Subjective AND
            uncovered) — tests whether decomposition covers the answer well enough to trust

Also reports: answer-char COVERAGE (does decomposition actually cover the answer?), and
per-route token-AUROC (does the probe discriminate better on Relational than External? -> a
weak External row is the case for adding a retrieval verifier there next).

The External route still uses the probe here; faithful retrieval (REFIND) is the next
iteration, deferred because the pilot REFIND is proxy-model-limited (~chance AUROC).

Run:  python trace.py     (after decompose_route.py and a Gemma MPS fidelity check)
"""
import json
import os
from collections import Counter

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

from config import CFG
from data import load_sample
from wb import load_model, tokenize_with_offsets, token_labels, merge_flagged_spans
import metrics
from sage import load_sae, extract


def load_routes():
    with open(os.path.join(CFG.out_dir, "decompose_route.json"), encoding="utf-8") as f:
        return {d["id"]: d for d in json.load(f)}


def token_routes(offsets, atoms):
    """Route per token = type of the anchored atom whose span overlaps it most ('Uncovered' if none)."""
    out = []
    for (ts, te) in offsets:
        best, best_ov = "Uncovered", 0
        for a in atoms:
            sp = a.get("span")
            if not sp:
                continue
            ov = min(te, sp[1]) - max(ts, sp[0])
            if ov > best_ov:
                best_ov, best = ov, a["type"]
        out.append(best)
    return out


def preds_from(prob, thr, offs, routes, mode):
    """Build merged char-spans per item under a flagging rule."""
    out = []
    for p, o, r in zip(prob, offs, routes):
        if mode == "flat":
            flags = [pi > thr for pi in p]
        elif mode == "trace_a":                       # suppress Subjective only
            flags = [(pi > thr) and (ri != "Subjective") for pi, ri in zip(p, r)]
        else:                                          # trace_b: only Relational/External
            flags = [(pi > thr) and (ri in ("Relational", "External")) for pi, ri in zip(p, r)]
        out.append(merge_flagged_spans(o, flags))
    return out


def run():
    routes_by_id = load_routes()
    model = load_model(CFG.sage_model)
    sae = load_sae()
    items = load_sample()

    kept, feats, offs, routes_all, labels = [], [], [], [], []
    cov_num, cov_den = Counter(), Counter()
    for it in items:
        if not (it.answer or "").strip():
            continue
        ans_ids, offsets = tokenize_with_offsets(model, it.answer)
        if not ans_ids:
            continue
        atoms = routes_by_id.get(it.id, {}).get("atoms", [])
        q_ids, _ = tokenize_with_offsets(model, (it.question or "") + "\n")
        resid, _ = extract(model, q_ids, ans_ids, CFG.sage_layer)
        with torch.no_grad():
            feat = sae.encode(resid.to(CFG.sae_device)).float().cpu().numpy()

        kept.append(it)
        feats.append(feat)
        offs.append(offsets)
        routes_all.append(token_routes(offsets, atoms))
        labels.append(np.array(token_labels(offsets, it.hard_labels)))

        covered = np.zeros(len(it.answer), dtype=bool)
        for a in atoms:
            if a.get("span"):
                covered[a["span"][0]:a["span"][1]] = True
        cov_num[it.lang] += int(covered.sum())
        cov_den[it.lang] += len(it.answer)
        print(f"[{it.lang}] {it.id}: {len(ans_ids)} toks, "
              f"{int(labels[-1].sum())} halluc, {100*covered.mean():.0f}% covered")

    # item-level split (same recipe as the probe)
    rng = np.random.default_rng(CFG.seed)
    order = np.arange(len(kept)); rng.shuffle(order)
    n_test = max(1, int(round(CFG.probe_test_frac * len(kept))))
    test = set(order[:n_test].tolist())
    tr_items = [i for i in range(len(kept)) if i not in test]
    te_items = [i for i in range(len(kept)) if i in test]

    Xtr = np.concatenate([feats[i] for i in tr_items])
    ytr = np.concatenate([labels[i] for i in tr_items])
    keep = (Xtr > 0).mean(0) >= 0.005
    if keep.sum() < 50:
        keep = (Xtr > 0).mean(0) > 0
    sc = StandardScaler().fit(Xtr[:, keep])
    clf = LogisticRegression(max_iter=2000, class_weight="balanced", C=0.5)
    clf.fit(sc.transform(Xtr[:, keep]), ytr)
    print(f"\nTrain tokens {len(ytr)} ({ytr.sum()} halluc) | "
          f"test items {len(te_items)} | SAE feats kept {int(keep.sum())}/{Xtr.shape[1]}")

    prob = [clf.predict_proba(sc.transform(f[:, keep]))[:, 1] for f in feats]

    # calibrate one threshold on TRAIN flat-IoU; apply to all predictors on TEST
    tr_obj = [kept[i] for i in tr_items]
    best_thr, best_iou = 0.5, -1.0
    for thr in np.linspace(0.05, 0.9, 18):
        p = preds_from([prob[i] for i in tr_items], thr,
                       [offs[i] for i in tr_items], [routes_all[i] for i in tr_items], "flat")
        o, _ = metrics.mean_iou(tr_obj, p)
        if o > best_iou:
            best_iou, best_thr = o, float(thr)
    print(f"Calibrated probe threshold (train): {best_thr:.2f}")

    te_obj = [kept[i] for i in te_items]
    te_prob = [prob[i] for i in te_items]
    te_offs = [offs[i] for i in te_items]
    te_rts = [routes_all[i] for i in te_items]
    for mode, name in [("flat", "FLAT probe"),
                       ("trace_a", "TRACE-A (suppress Subjective)"),
                       ("trace_b", "TRACE-B (only Relational/External)")]:
        metrics.report(name, te_obj, preds_from(te_prob, best_thr, te_offs, te_rts, mode))

    # per-route token-level AUROC on test
    pr = np.concatenate(te_prob)
    lb = np.concatenate([labels[i] for i in te_items])
    rt = np.array([r for rs in te_rts for r in rs])
    print("\n=== per-route token-AUROC (test) ===")
    for route in ("Relational", "External", "Subjective"):
        m = rt == route
        if m.sum() > 5 and len(set(lb[m].tolist())) > 1:
            print(f"  {route:<12} n={int(m.sum()):5d}  AUROC={roc_auc_score(lb[m], pr[m]):.3f}  "
                  f"(halluc rate {lb[m].mean():.2f})")
        else:
            print(f"  {route:<12} n={int(m.sum()):5d}  (too few / single-class)")

    print("\n=== answer-char coverage by decomposition ===")
    for lang in sorted(cov_den):
        print(f"  {lang}: {100*cov_num[lang]/max(1,cov_den[lang]):.0f}% of answer chars under some atom")
    print("\nRead: TRACE-A>FLAT -> suppressing opinions helps. TRACE-B>=FLAT only if coverage")
    print("is high. Weak External AUROC -> that route wants a retrieval verifier next.")


if __name__ == "__main__":
    run()
