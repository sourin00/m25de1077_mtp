"""Within-item token-AUROC — the control that separates "detects hallucination" from
"detects topic".

The pooled AUROC (~0.745) mixes two things: telling hallucinated items from faithful ones
(topic/item-level, a confound) and telling hallucinated tokens from faithful tokens inside the
same answer (genuine token-level signal). This isolates the second: for each TEST item that has
both hallucinated and faithful tokens, rank that item's tokens by the probe and take AUROC
WITHIN the item, then macro-average. Topic, language, and answer are constant within an item, so
within-item discrimination cannot be topic or surface confound.

Reports surface vs SAE within-item AUROC (and pooled, for contrast), multi-seed, per language.

Reads: sae within-item >> 0.5  -> real token-level hallucination signal (robust to both confounds)
       sae within-item ~ 0.5    -> the cross-item signal was topic/item confound all along
       sae within-item > surface within-item -> the token-level signal is beyond surface form

Run:  python within_item.py     (reuses the feature cache)
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

from config import CFG
from data import load_sample
from wb import load_model
import features as F
from confound_check import surface_feats

SEEDS = [13, 41, 97, 7, 123]
LANGS = ("es", "cs", "zh", "en")


def _ms(xs):
    a = np.array([x for x in xs if x is not None and not np.isnan(x)], float)
    return f"{a.mean():.3f} +/- {a.std():.3f}" if len(a) else "n/a"


def fit(Xtr, ytr):
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(max_iter=2000, class_weight="balanced", C=0.5)
    clf.fit(sc.transform(Xtr), ytr)
    return sc, clf


def pred(m, X):
    sc, clf = m
    return clf.predict_proba(sc.transform(X))[:, 1]


def run():
    model = load_model(CFG.sage_model)
    sae = F.load_sae()
    items = load_sample(10_000)

    per = []
    for it in items:
        f = F.item_features(model, it)
        if f is None:
            continue
        per.append({"sae": F.sae_encode(sae, f["resid"]),
                    "surf": surface_feats(it.answer or "", f["offsets"], it.lang),
                    "lab": f["labels"].astype(int), "lang": it.lang})
    n = len(per)

    def qualifies(i):
        y = per[i]["lab"]
        return y.sum() >= 2 and (y == 0).sum() >= 2 and len(y) >= 6

    qual = [i for i in range(n) if qualifies(i)]
    print(f"{n} items | {len(qual)} qualify for within-item AUROC (>=2 halluc & >=2 faithful tokens)")

    agg = {"surface_within": [], "sae_within": [], "sae_pooled": []}
    pl = {L: {"surf": [], "sae": []} for L in LANGS}

    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        order = np.arange(n); rng.shuffle(order)
        nt = max(1, int(round(CFG.probe_test_frac * n)))
        test = set(order[:nt].tolist())
        tr = [i for i in range(n) if i not in test]
        te = [i for i in range(n) if i in test]
        te_q = [i for i in te if qualifies(i)]

        Xtr_sae = np.concatenate([per[i]["sae"] for i in tr])
        ytr = np.concatenate([per[i]["lab"] for i in tr])
        Xtr_surf = np.concatenate([per[i]["surf"] for i in tr])
        freq = (Xtr_sae > 0).mean(0)
        keep = freq >= 0.005
        if keep.sum() < 50:
            keep = freq > 0
        m_sae = fit(Xtr_sae[:, keep], ytr)
        m_surf = fit(Xtr_surf, ytr)

        # pooled (cross-item) SAE AUROC on all test tokens, for contrast
        y_all = np.concatenate([per[i]["lab"] for i in te])
        p_all = np.concatenate([pred(m_sae, per[i]["sae"][:, keep]) for i in te])
        agg["sae_pooled"].append(roc_auc_score(y_all, p_all))

        wi_s, wi_a = [], []
        wl = {L: {"s": [], "a": []} for L in LANGS}
        for i in te_q:
            y = per[i]["lab"]
            a_s = roc_auc_score(y, pred(m_surf, per[i]["surf"]))
            a_a = roc_auc_score(y, pred(m_sae, per[i]["sae"][:, keep]))
            wi_s.append(a_s); wi_a.append(a_a)
            wl[per[i]["lang"]]["s"].append(a_s); wl[per[i]["lang"]]["a"].append(a_a)
        agg["surface_within"].append(np.mean(wi_s))
        agg["sae_within"].append(np.mean(wi_a))
        for L in LANGS:
            if wl[L]["a"]:
                pl[L]["surf"].append(np.mean(wl[L]["s"]))
                pl[L]["sae"].append(np.mean(wl[L]["a"]))

    print("\n=== AUROC over seeds " + str(SEEDS) + " (mean +/- std) ===")
    print(f"  SAE pooled (cross-item, includes topic) {_ms(agg['sae_pooled'])}")
    print(f"  SAE WITHIN-item (topic held constant)   {_ms(agg['sae_within'])}   <- the decider")
    print(f"  surface WITHIN-item                     {_ms(agg['surface_within'])}")
    print("\n  within-item per language (SAE | surface):")
    for L in LANGS:
        print(f"    {L}: {_ms(pl[L]['sae'])}  |  {_ms(pl[L]['surf'])}")
    print("\nRead: SAE within-item >> 0.5 and > surface within-item -> genuine token-level")
    print("hallucination signal, robust to topic AND surface. Near 0.5 -> it was item-level confound.")


if __name__ == "__main__":
    run()
