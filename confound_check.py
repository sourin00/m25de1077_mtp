"""Confound check — is the probe's signal hallucination, or topic + surface form?

On the same tokens, same item-level split, same seeds, compares token-AUROC of:
  surface       per-token surface features ONLY (length, punct, whitespace, digit, position,
                language) — no model. If this matches the SAE probe, the signal isn't semantic.
  sae_full      the SAE-feature probe (reproduces iou_eval's ~0.69)
  sae_rare      SAE features EXCLUDING always-on ones (train-freq > 0.2 dropped)
  sae_highfreq  ONLY the always-on SAE features (train-freq > 0.2)
  surface+rare  surface + rare SAE features (does anything semantic add over surface?)
Plus a PER-LANGUAGE surface-vs-sae_full table, so the result can't be blamed on one language.

Reads: surface ~ sae_full        -> signal is surface/topic confound, not hallucination.
       surface+rare > surface    -> some genuine semantic signal survives.

Set eval_per_lang=20 in config.py to re-run on the dense seed-13 regime (robustness check).

Run:  python confound_check.py     (reuses the feature cache; fast)
"""
import string

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

from config import CFG
from data import load_sample
from wb import load_model
import features as F

SEEDS = [13, 41, 97]
LANGS = ("es", "cs", "zh", "en")
_PUNCT = set(string.punctuation + "。，、？！；：（）《》「」·…—")


def surface_feats(answer, offsets, lang):
    n = len(offsets)
    rows = []
    for i, (s, e) in enumerate(offsets):
        t = answer[s:e]
        ts = t.strip()
        row = [
            float(len(t)),
            1.0 if ts and all(c in _PUNCT for c in ts) else 0.0,   # all-punctuation
            1.0 if ts == "" else 0.0,                              # whitespace / newline
            1.0 if any(c.isdigit() for c in t) else 0.0,           # has digit
            1.0 if any(c.isalpha() for c in t) else 0.0,           # has letter
            1.0 if t[:1] == " " else 0.0,                          # word-initial
            1.0 if ts[:1].isupper() else 0.0,                      # capitalized
            i / max(1, n - 1),                                     # relative position
        ]
        row += [1.0 if lang == L else 0.0 for L in LANGS]
        rows.append(row)
    return np.array(rows, dtype=float)


def _ms(xs):
    a = np.array([x for x in xs if not np.isnan(x)], float)
    return f"{a.mean():.3f} +/- {a.std():.3f}" if len(a) else "n/a"


def auroc(y, p):
    return roc_auc_score(y, p) if len(set(y.tolist())) > 1 else float("nan")


def fit_predict(Xtr, ytr, Xte):
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(max_iter=2000, class_weight="balanced", C=0.5)
    clf.fit(sc.transform(Xtr), ytr)
    return clf.predict_proba(sc.transform(Xte))[:, 1]


def run():
    model = load_model(CFG.sage_model)
    sae = F.load_sae()
    items = load_sample(CFG.eval_per_lang)

    SAE, SURF, Y, ITEM, LANGTOK = [], [], [], [], []
    for idx, it in enumerate(items):
        f = F.item_features(model, it)
        if f is None:
            continue
        SAE.append(F.sae_encode(sae, f["resid"]))
        SURF.append(surface_feats(it.answer or "", f["offsets"], it.lang))
        Y.append(f["labels"])
        ITEM.append(np.full(len(f["labels"]), idx))
        LANGTOK.append(np.array([it.lang] * len(f["labels"])))
    Xsae = np.concatenate(SAE); Xsurf = np.concatenate(SURF)
    y = np.concatenate(Y).astype(int); item = np.concatenate(ITEM)
    lang_tok = np.concatenate(LANGTOK)
    n_items = len(SAE)
    print(f"{n_items} items | {len(y)} tokens ({int(y.sum())} halluc) | "
          f"surface dims {Xsurf.shape[1]} | SAE dims {Xsae.shape[1]}")

    res = {k: [] for k in ("surface", "sae_full", "sae_rare", "sae_highfreq", "surface+rare")}
    per_lang = {L: {"surface": [], "sae_full": []} for L in LANGS}

    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        order = np.arange(n_items); rng.shuffle(order)
        nt = max(1, int(round(CFG.probe_test_frac * n_items)))
        test = set(order[:nt].tolist())
        te = np.isin(item, list(test)); tr = ~te
        yte = y[te]

        freq = (Xsae[tr] > 0).mean(0)

        def sel(lo, hi):
            m = (freq >= lo) & (freq <= hi)
            return m if m.sum() >= 10 else (freq > 0)

        full, rare, high = sel(0.005, 1.0), sel(0.005, 0.2), sel(0.2, 1.0)
        p_surf = fit_predict(Xsurf[tr], y[tr], Xsurf[te])
        p_full = fit_predict(Xsae[tr][:, full], y[tr], Xsae[te][:, full])
        res["surface"].append(auroc(yte, p_surf))
        res["sae_full"].append(auroc(yte, p_full))
        res["sae_rare"].append(auroc(yte, fit_predict(Xsae[tr][:, rare], y[tr], Xsae[te][:, rare])))
        res["sae_highfreq"].append(auroc(yte, fit_predict(Xsae[tr][:, high], y[tr], Xsae[te][:, high])))
        comb_tr = np.concatenate([Xsurf[tr], Xsae[tr][:, rare]], axis=1)
        comb_te = np.concatenate([Xsurf[te], Xsae[te][:, rare]], axis=1)
        res["surface+rare"].append(auroc(yte, fit_predict(comb_tr, y[tr], comb_te)))

        lte = lang_tok[te]
        for L in LANGS:
            m = lte == L
            if m.sum() > 5:
                per_lang[L]["surface"].append(auroc(yte[m], p_surf[m]))
                per_lang[L]["sae_full"].append(auroc(yte[m], p_full[m]))

    print("\n=== token-AUROC over seeds " + str(SEEDS) + " (mean +/- std) ===")
    for k in ("surface", "sae_highfreq", "sae_full", "sae_rare", "surface+rare"):
        print(f"  {k:<14} {_ms(res[k])}")

    print("\n=== per-language: surface vs sae_full (mean +/- std) ===")
    for L in LANGS:
        print(f"  {L}: surface {_ms(per_lang[L]['surface'])}  |  sae_full {_ms(per_lang[L]['sae_full'])}")

    print("\nRead: surface ~ sae_full (overall AND per language) -> the probe's edge is")
    print("surface/topic, not hallucination. surface+rare ~ surface -> no semantic signal survives.")


if __name__ == "__main__":
    run()