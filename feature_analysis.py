"""Feature interpretability — which SAE features signal hallucination?

The thesis's interpretability core: detection via NAMEABLE sparse features, not opaque residual
directions. Trains the SAE-feature probe on the representative sample, then for the most
predictive features reports the signed probe weight, mean activation on hallucinated vs
faithful tokens, activation frequency, the top max-activating answer tokens (with their text),
and a Neuronpedia link to the Gemma Scope auto-interpretation. You read those examples + the
auto-interp and give each top feature a human label in the writeup.

Run:  python feature_analysis.py     (uses the feature cache, so it's fast after the first pass)
"""
import json
import os

import numpy as np

from config import CFG
from data import load_sample
from wb import load_model
import features as F
import probe as P

# Gemma Scope 16k residual SAEs on Neuronpedia (adjust the source slug if a link 404s):
NEURONPEDIA = "https://www.neuronpedia.org/gemma-2-2b/{layer}-gemmascope-res-16k/{idx}"
TOP_FEATURES = 25
TOP_EXAMPLES = 8


def run():
    model = load_model(CFG.sage_model)
    sae = F.load_sae()
    items = load_sample(CFG.eval_per_lang)

    X, Y, toks = [], [], []
    kept = 0
    for it in items:
        f = F.item_features(model, it)
        if f is None:
            continue
        acts = F.sae_encode(sae, f["resid"])
        X.append(acts)
        Y.append(f["labels"])
        for (s, e) in f["offsets"]:
            toks.append((it.lang, (it.answer or "")[s:e]))
        kept += 1
    X = np.concatenate(X)
    y = np.concatenate(Y).astype(int)
    print(f"{kept} items | {len(y)} tokens ({int(y.sum())} hallucinated)")

    pr = P.train(X, y)
    coef = pr["clf"].coef_[0]                       # aligned to kept features
    order = np.argsort(-np.abs(coef))[:TOP_FEATURES]

    results = []
    for rank, j in enumerate(order):
        fid = int(pr["keep_idx"][j])
        col = X[:, fid]
        w = float(coef[j])
        hm = float(col[y == 1].mean()) if (y == 1).any() else 0.0
        fm = float(col[y == 0].mean()) if (y == 0).any() else 0.0
        freq = float((col > 0).mean())
        top = np.argsort(-col)[:TOP_EXAMPLES]
        examples = [{"lang": toks[t][0], "token": toks[t][1], "act": float(col[t])} for t in top]
        results.append({
            "rank": rank + 1, "feature": fid, "weight": round(w, 3),
            "halluc_mean": round(hm, 3), "faithful_mean": round(fm, 3),
            "direction": "↑halluc" if hm > fm else "↓halluc", "freq": round(freq, 3),
            "neuronpedia": NEURONPEDIA.format(layer=CFG.sage_layer, idx=fid),
            "examples": examples,
        })
        print(f"#{rank+1:>2} feat {fid:6d}  w={w:+.2f}  halluc={hm:.2f} faithful={fm:.2f} "
              f"freq={freq:.2f}  {results[-1]['direction']}")
        print("       ex:", ", ".join(repr(e["token"]) for e in examples[:5]))

    out = os.path.join(CFG.out_dir, "feature_analysis.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(results, fh, ensure_ascii=False, indent=2)
    print(f"\nSaved -> {out}")
    print("Open the Neuronpedia links for the top +weight / ↑halluc features and name them in")
    print("the thesis: these are the model's hallucination-associated directions.")


if __name__ == "__main__":
    run()
