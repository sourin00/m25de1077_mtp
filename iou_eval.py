"""Honest IoU on a representative sample.

The seed-13 80-item pilot sample is hallucination-saturated, so flag-all scored ~0.336 and the
probe could barely clear it on IoU even with AUROC 0.78 — the metric had no room. This widens
the draw (CFG.eval_per_lang per language) so density regresses toward the true Mu-SHROOM test
distribution, trains the SAE-feature probe on a train split, calibrates its threshold on TRAIN
(no leakage), and reports on held-out TEST:

  - hallucination DENSITY of the sample (to confirm it's less saturated than the pilot draw)
  - FLAG-ALL and FLAG-NONE IoU (the floors)
  - SAE-probe IoU and its LIFT over flag-all  <- the headline: does AUROC convert to IoU?
  - token-level AUROC (leakage-free)

If the probe clears flag-all by a real margin here, the white-box signal is leaderboard-relevant
and the dense-sample IoU was just base-rate compression. If it still hugs flag-all, the IoU
story is genuinely weak regardless of sample.

Run:  python iou_eval.py     (Gemma on MPS; scales with CFG.eval_per_lang — minutes)
"""
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


def run():
    model = load_model(CFG.sage_model)
    sae = load_sae()
    items = load_sample(CFG.eval_per_lang)

    kept, feats, offs, labels = [], [], [], []
    dens = []
    for it in items:
        if not (it.answer or "").strip():
            continue
        ans_ids, offsets = tokenize_with_offsets(model, it.answer)
        if not ans_ids:
            continue
        q_ids, _ = tokenize_with_offsets(model, (it.question or "") + "\n")
        resid, _ = extract(model, q_ids, ans_ids, CFG.sage_layer)
        with torch.no_grad():
            feat = sae.encode(resid.to(CFG.sae_device)).float().cpu().numpy()
        lab = np.array(token_labels(offsets, it.hard_labels))
        kept.append(it); feats.append(feat); offs.append(offsets); labels.append(lab)
        gold_chars = sum(int(e) - int(s) for s, e in (it.hard_labels or []))
        dens.append(gold_chars / max(1, len(it.answer)))
    print(f"\n{len(kept)} items | mean hallucination density "
          f"{np.mean(dens):.2f} (pilot seed-13 draw was ~0.45+)")

    rng = np.random.default_rng(CFG.seed)
    order = np.arange(len(kept)); rng.shuffle(order)
    n_test = max(1, int(round(CFG.probe_test_frac * len(kept))))
    test = set(order[:n_test].tolist())
    tr = [i for i in range(len(kept)) if i not in test]
    te = [i for i in range(len(kept)) if i in test]

    Xtr = np.concatenate([feats[i] for i in tr]); ytr = np.concatenate([labels[i] for i in tr])
    keep = (Xtr > 0).mean(0) >= 0.005
    if keep.sum() < 50:
        keep = (Xtr > 0).mean(0) > 0
    sc = StandardScaler().fit(Xtr[:, keep])
    clf = LogisticRegression(max_iter=2000, class_weight="balanced", C=0.5)
    clf.fit(sc.transform(Xtr[:, keep]), ytr)
    prob = [clf.predict_proba(sc.transform(f[:, keep]))[:, 1] for f in feats]
    print(f"Train tokens {len(ytr)} ({int(ytr.sum())} halluc) | test items {len(te)} | "
          f"SAE feats kept {int(keep.sum())}/{Xtr.shape[1]}")

    def probe_preds(idxs, thr):
        return [merge_flagged_spans(offs[i], [p > thr for p in prob[i]]) for i in idxs]

    tr_obj = [kept[i] for i in tr]
    best_thr, best = 0.5, -1.0
    for thr in np.linspace(0.05, 0.9, 18):
        o, _ = metrics.mean_iou(tr_obj, probe_preds(tr, thr))
        if o > best:
            best, best_thr = o, float(thr)

    te_obj = [kept[i] for i in te]
    flag_all = [[[0, len(kept[i].answer)]] if kept[i].answer.strip() else [] for i in te]
    flag_none = [[] for _ in te]
    fa, _ = metrics.mean_iou(te_obj, flag_all)
    metrics.report("FLAG-ALL", te_obj, flag_all)
    metrics.report("FLAG-NONE", te_obj, flag_none)
    pr_overall = metrics.report(f"SAE probe (thr={best_thr:.2f}, train-calibrated)",
                                te_obj, probe_preds(te, best_thr))

    pr = np.concatenate([prob[i] for i in te]); lb = np.concatenate([labels[i] for i in te])
    auroc = roc_auc_score(lb, pr) if len(set(lb.tolist())) > 1 else float("nan")
    print(f"\nToken-level AUROC (test): {auroc:.3f}")
    print(f"IoU LIFT over flag-all: {pr_overall - fa:+.3f}   "
          f"(probe {pr_overall:.3f} vs flag-all {fa:.3f})")
    print("\nRead: a clearly positive lift here = the 0.78 AUROC converts to real IoU once the")
    print("sample isn't base-rate-saturated, and the white-box probe is leaderboard-relevant.")


if __name__ == "__main__":
    run()
