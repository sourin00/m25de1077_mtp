"""Headline detection result: SAE-feature probe vs flag-all, honest and multilingual.

Representative draw (CFG.eval_per_lang) so density isn't saturated; SAE-feature probe trained
and PER-LANGUAGE-calibrated on a train split (per-lang thresholds rescue zh); evaluated on
held-out test. Repeated over several seeds for mean +/- std. Reports floors, probe IoU,
per-language IoU, token-AUROC, and the lift over flag-all.

Features are extracted once (cached by features.py); each seed only re-splits, re-fits the
logistic head, re-calibrates, and re-scores — so the seeds are cheap.

Run:  python iou_eval.py
"""
import numpy as np
from sklearn.metrics import roc_auc_score

from config import CFG
from data import load_sample
from wb import load_model
import metrics
import features as F
import probe as P

SEEDS = [13, 41, 97]


def _ms(xs):
    a = np.array(xs, dtype=float)
    return f"{a.mean():.3f} +/- {a.std():.3f}"


def run():
    model = load_model(CFG.sage_model)
    sae = F.load_sae()
    items = load_sample(CFG.eval_per_lang)

    data, dens = [], []
    for i, it in enumerate(items):
        f = F.item_features(model, it)
        if f is None:
            continue
        acts = F.sae_encode(sae, f["resid"])
        data.append({"item": it, "acts": acts, "offs": f["offsets"], "lab": f["labels"]})
        gold = sum(int(e) - int(s) for s, e in (it.hard_labels or []))
        dens.append(gold / max(1, len(it.answer or "")))
        if (i + 1) % 40 == 0:
            print(f"  ...features for {i + 1}/{len(items)} items")
    n = len(data)
    print(f"\n{n} items | mean hallucination density {np.mean(dens):.2f}")

    langs = sorted(set(d["item"].lang for d in data))
    agg = {"flag_all": [], "probe": [], "auroc": [], "lift": []}
    per_lang = {l: {"flag_all": [], "probe": []} for l in langs}

    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        order = np.arange(n); rng.shuffle(order)
        n_test = max(1, int(round(CFG.probe_test_frac * n)))
        test = set(order[:n_test].tolist())
        tr = [i for i in range(n) if i not in test]
        te = [i for i in range(n) if i in test]

        Xtr = np.concatenate([data[i]["acts"] for i in tr])
        ytr = np.concatenate([data[i]["lab"] for i in tr]).astype(int)
        pb = P.train(Xtr, ytr)
        prob = [P.predict(pb, d["acts"]) for d in data]

        tr_items = [data[i]["item"] for i in tr]
        tr_prob = [prob[i] for i in tr]
        tr_offs = [data[i]["offs"] for i in tr]
        thr = P.calibrate(tr_items, tr_prob, tr_offs, per_language=True)

        te_items = [data[i]["item"] for i in te]
        te_prob = [prob[i] for i in te]
        te_offs = [data[i]["offs"] for i in te]

        flag_all = [[[0, len(it.answer)]] if (it.answer or "").strip() else [] for it in te_items]
        probe_spans = P.predict_spans(te_items, te_prob, te_offs, thr)

        fa_o, fa_by = metrics.mean_iou(te_items, flag_all)
        pr_o, pr_by = metrics.mean_iou(te_items, probe_spans)
        lb = np.concatenate([data[i]["lab"] for i in te]).astype(int)
        pr_tok = np.concatenate(te_prob)
        au = roc_auc_score(lb, pr_tok) if len(set(lb.tolist())) > 1 else float("nan")

        agg["flag_all"].append(fa_o); agg["probe"].append(pr_o)
        agg["auroc"].append(au); agg["lift"].append(pr_o - fa_o)
        for l in langs:
            if l in pr_by:
                per_lang[l]["flag_all"].append(fa_by.get(l, 0.0))
                per_lang[l]["probe"].append(pr_by[l])
        print(f"seed {seed}: probe IoU {pr_o:.3f} | flag-all {fa_o:.3f} | "
              f"AUROC {au:.3f} | thr {({k: round(v,2) for k,v in thr.items()})}")

    print("\n=== detection result over seeds " + str(SEEDS) + " (mean +/- std) ===")
    print(f"  token-AUROC      {_ms(agg['auroc'])}")
    print(f"  FLAG-ALL IoU     {_ms(agg['flag_all'])}")
    print(f"  SAE-probe IoU    {_ms(agg['probe'])}")
    print(f"  LIFT over flag-all {_ms(agg['lift'])}")
    print("\n  per-language IoU (probe vs flag-all):")
    for l in langs:
        print(f"    {l}: probe {_ms(per_lang[l]['probe'])}  |  flag-all {_ms(per_lang[l]['flag_all'])}")
    print("\nThis is Table 1. Lead the writeup with AUROC + lift; report the UCSC IoU gap plainly.")


if __name__ == "__main__":
    run()