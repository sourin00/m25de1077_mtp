"""SAGE substrate experiment (the thesis-deciding head-to-head).

On one Gemma-2-2b layer, over the same 80-item sample and the same item-level split, compare
the token-level AUROC of four hallucination signals:

  1. CONF        external assertion only: the model's own token log-prob (a logit baseline)
  2. RAW         internal representation: raw resid_post  (Deloitte-style probe, on Gemma)
  3. SAE         SAE features of that residual  (does sparse coding preserve the signal?)
  4. SAE+CONF    internal SAE features + external confidence  (the internal-vs-external GAP)

Read it like this:
  - SAE >= RAW           -> SAEs are a viable, interpretable substrate (differentiates SAGE
                            from Deloitte's raw residuals even at parity).
  - SAE+CONF > max(...)  -> combining internal awareness with external assertion adds signal;
                            the gap framing has legs -> SAGE/Hybrid worth building out.
  - SAE+CONF ~= SAE      -> no gap lift here; lean TRACE-only.

This is the first-cut gap (concatenation). The explicit knowledge-feature gap (Ferrando-style
known-entity directions minus confidence) is the deeper instantiation to build IF this is
promising. Establish substrate viability before investing in the specific gap definition.

PREREQS:
  - Validate Gemma-2 MPS fidelity first: run check_mps_fidelity.py with MODEL="google/gemma-2-2b".
  - sae_lens installed; Gemma license accepted (hf auth login).

Run:  python sage.py
"""
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

from config import CFG
from data import load_sample
from wb import load_model, tokenize_with_offsets, token_labels, _with_bos, _ids_tensor


def load_sae():
    from sae_lens import SAE
    loaded = SAE.from_pretrained(CFG.sae_release, CFG.sae_id, device=CFG.sae_device)
    sae = loaded[0] if isinstance(loaded, (tuple, list)) else loaded
    sae.eval()
    return sae


def extract(model, q_ids, ans_ids, layer):
    """One forward: raw resid_post[layer] and teacher-forced log-prob per answer token."""
    ids = _with_bos(model, q_ids, ans_ids)
    name = f"blocks.{layer}.hook_resid_post"
    with torch.no_grad():
        logits, cache = model.run_with_cache(_ids_tensor(ids), names_filter=lambda n: n == name)
    na = len(ans_ids)
    start = len(ids) - na
    resid = cache["resid_post", layer][0, start:start + na, :].float()          # [na, d_model]
    lp = torch.log_softmax(logits[0].float(), dim=-1)
    logp = np.array([lp[start + j - 1, ans_ids[j]].item() for j in range(na)])   # [na]
    return resid, logp


def auroc(labels, score):
    if len(set(labels.tolist())) < 2:
        return float("nan")
    a = roc_auc_score(labels, score)
    return max(a, 1 - a)


def fit_auroc(Xtr, ytr, Xte, yte):
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(max_iter=2000, class_weight="balanced", C=0.5)
    clf.fit(sc.transform(Xtr), ytr)
    p = clf.predict_proba(sc.transform(Xte))[:, 1]
    return roc_auc_score(yte, p) if len(set(yte.tolist())) > 1 else float("nan")


def run():
    model = load_model(CFG.sage_model)
    sae = load_sae()
    items = load_sample()

    rows_resid, rows_sae, rows_logp, rows_lab, rows_item = [], [], [], [], []
    sanity_done = False
    for idx, it in enumerate(items):
        if not (it.answer or "").strip():
            continue
        ans_ids, offsets = tokenize_with_offsets(model, it.answer)
        if not ans_ids:
            continue
        q_ids, _ = tokenize_with_offsets(model, (it.question or "") + "\n")
        resid, logp = extract(model, q_ids, ans_ids, CFG.sage_layer)
        with torch.no_grad():
            sae_acts = sae.encode(resid.to(CFG.sae_device))                      # [na, d_sae]
        resid = resid.cpu().numpy()
        sae_acts = sae_acts.float().cpu().numpy()
        labels = np.array(token_labels(offsets, it.hard_labels))

        if not sanity_done:
            print(f"[sanity] layer {CFG.sage_layer}: resid {resid.shape}, SAE {sae_acts.shape}, "
                  f"mean active feats/token {(sae_acts > 0).sum(1).mean():.1f} "
                  f"({100 * (sae_acts > 0).mean():.2f}% of {sae_acts.shape[1]})")
            if resid.shape[1] != model.cfg.d_model:
                print(f"[sanity] WARNING resid dim {resid.shape[1]} != d_model {model.cfg.d_model}")
            sanity_done = True

        rows_resid.append(resid); rows_sae.append(sae_acts)
        rows_logp.append(logp); rows_lab.append(labels)
        rows_item.append(np.full(len(labels), idx))
        print(f"[{it.lang}] {it.id}: {len(ans_ids)} toks, {int(labels.sum())} gold-halluc")

    X_resid = np.concatenate(rows_resid)
    X_sae = np.concatenate(rows_sae)
    logp = np.concatenate(rows_logp)
    y = np.concatenate(rows_lab)
    item_of = np.concatenate(rows_item)

    # item-level held-out split (same recipe as the Deloitte probe)
    rng = np.random.default_rng(CFG.seed)
    uniq = np.unique(item_of)
    rng.shuffle(uniq)
    n_test = max(1, int(round(CFG.probe_test_frac * len(uniq))))
    test_items = set(uniq[:n_test].tolist())
    te = np.array([i in test_items for i in item_of])
    tr = ~te
    print(f"\nTrain tokens: {tr.sum()} ({y[tr].sum()} halluc) | test tokens: {te.sum()} "
          f"({y[te].sum()} halluc) over {len(test_items)} held-out items")

    # keep SAE features that actually fire on train (cuts 16k dims -> overfitting control)
    freq = (X_sae[tr] > 0).mean(0)
    keep = freq >= 0.005
    if keep.sum() < 50:
        keep = freq > 0
    Xsae_k = X_sae[:, keep]
    print(f"SAE features kept (fire >=0.5% of train tokens): {int(keep.sum())}/{X_sae.shape[1]}")

    conf = logp.reshape(-1, 1)                       # external assertion (log-confidence)
    Xsae_conf = np.concatenate([Xsae_k, conf], axis=1)

    res = {
        "CONF (external only)":      auroc(y[te], logp[te]),
        "RAW resid (Deloitte/Gemma)": fit_auroc(X_resid[tr], y[tr], X_resid[te], y[te]),
        "SAE features":              fit_auroc(Xsae_k[tr], y[tr], Xsae_k[te], y[te]),
        "SAE + CONF (gap)":          fit_auroc(Xsae_conf[tr], y[tr], Xsae_conf[te], y[te]),
    }

    print("\n=== SAGE substrate: token-level AUROC (held-out items) ===")
    for k, v in res.items():
        print(f"  {k:<28} {v:.3f}")
    print("\nRead: SAE>=RAW -> SAE viable substrate; SAE+CONF>max -> gap has lift -> build SAGE.")


if __name__ == "__main__":
    run()
