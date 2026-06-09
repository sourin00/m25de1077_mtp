"""SAE features over the residual stream, with an on-disk cache.

The expensive step is the Gemma forward pass. We cache its output per item — resid_post at the
SAE layer, teacher-forced log-probs, token char-offsets, and gold labels — keyed by
(model, layer, item-id), under runs/feat_cache/. SAE encoding is a cheap matmul and runs fresh
each call, so you can iterate on the analysis without recomputing forwards.

    model = load_model(CFG.sage_model); sae = load_sae()
    f = item_features(model, item)          # dict(offsets, resid, logp, labels) — cached
    acts = sae_encode(sae, f["resid"])      # [n_tokens, d_sae]
"""
import hashlib
import os

import numpy as np
import torch

from config import CFG
from wb import tokenize_with_offsets, token_labels, _with_bos, _ids_tensor

_SAE = None
CACHE_DIR = os.path.join(CFG.out_dir, "feat_cache")


def load_sae():
    """Load the Gemma Scope SAE once (cached in-process)."""
    global _SAE
    if _SAE is None:
        from sae_lens import SAE
        loaded = SAE.from_pretrained(CFG.sae_release, CFG.sae_id, device=CFG.sae_device)
        _SAE = loaded[0] if isinstance(loaded, (tuple, list)) else loaded
        _SAE.eval()
    return _SAE


def _forward(model, q_ids, ans_ids, layer):
    """One forward: resid_post[layer] per answer token + teacher-forced log-prob per token."""
    ids = _with_bos(model, q_ids, ans_ids)
    name = f"blocks.{layer}.hook_resid_post"
    with torch.no_grad():
        logits, cache = model.run_with_cache(_ids_tensor(ids), names_filter=lambda n: n == name)
    na = len(ans_ids)
    start = len(ids) - na
    resid = cache["resid_post", layer][0, start:start + na, :].float().cpu().numpy()
    lp = torch.log_softmax(logits[0].float(), dim=-1)
    logp = np.array([lp[start + j - 1, ans_ids[j]].item() for j in range(na)], dtype=np.float32)
    return resid, logp


def _key(item, layer):
    raw = f"{CFG.sage_model}|{layer}|{item.id}|{len(item.answer or '')}"
    return hashlib.md5(raw.encode()).hexdigest()


def item_features(model, item, layer=None, use_cache=True):
    """dict(offsets, resid[na,d], logp[na], labels[na]) for one item, or None if no tokens.
    Caches the Gemma forward output (resid as fp16) so repeat runs skip it."""
    layer = CFG.sage_layer if layer is None else layer
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, _key(item, layer) + ".npz")
    if use_cache and os.path.exists(path):
        z = np.load(path, allow_pickle=False)
        return {"offsets": [tuple(o) for o in z["offsets"]],
                "resid": z["resid"].astype(np.float32),
                "logp": z["logp"], "labels": z["labels"]}

    ans_ids, offsets = tokenize_with_offsets(model, item.answer or "")
    if not ans_ids:
        return None
    q_ids, _ = tokenize_with_offsets(model, (item.question or "") + "\n")
    resid, logp = _forward(model, q_ids, ans_ids, layer)
    labels = np.array(token_labels(offsets, item.hard_labels), dtype=np.int8)
    if use_cache:
        np.savez_compressed(path, resid=resid.astype(np.float16), logp=logp,
                            labels=labels, offsets=np.array(offsets, dtype=np.int32))
    return {"offsets": offsets, "resid": resid, "logp": logp, "labels": labels}


def sae_encode(sae, resid_np):
    """SAE feature activations for a [na, d_model] residual array -> np [na, d_sae]."""
    with torch.no_grad():
        t = torch.tensor(resid_np, dtype=torch.float32, device=CFG.sae_device)
        acts = sae.encode(t)
    return acts.float().cpu().numpy()
