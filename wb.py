"""Shared white-box plumbing for the Milestone-2 baselines.

Loads the probe model once (config.whitebox_model on config.whitebox_device, which you
validated as numerically faithful on MPS), tokenizes the answer WITH character offsets so
token-level signals map back to the character spans Mu-SHROOM scores on, and provides
teacher-forced token log-probs (REFIND) and residual-stream features (Deloitte-style probe).

Both baselines re-process the answer text through this fixed model rather than trusting the
dataset's heterogeneous per-generator logits.
"""
import functools
import os

import numpy as np
import torch

from config import CFG

os.environ.setdefault("TRANSFORMERLENS_ALLOW_MPS", "1")


@functools.lru_cache(maxsize=2)
def load_model(name=None):
    from transformer_lens import HookedTransformer
    name = name or CFG.whitebox_model
    model = HookedTransformer.from_pretrained(name, device=CFG.whitebox_device)
    model.eval()
    if model.tokenizer is None or not getattr(model.tokenizer, "is_fast", False):
        print("[wb] WARNING: tokenizer is not 'fast'; character offsets may be unavailable.")
    return model


def tokenize_with_offsets(model, text):
    """(token_ids, offsets) for text, no special tokens. offsets[i] = (char_start, char_end)."""
    enc = model.tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
    return enc["input_ids"], [tuple(o) for o in enc["offset_mapping"]]


def _ids_tensor(ids):
    return torch.tensor([ids], device=CFG.whitebox_device)


def _with_bos(model, prefix_ids, ans_ids):
    bos = model.tokenizer.bos_token_id
    head = [bos] if bos is not None else []
    return head + list(prefix_ids) + list(ans_ids)


def token_logprobs(model, prefix_ids, ans_ids):
    """Teacher-forced log p(ans_token_j | everything before it) for each answer token."""
    ids = _with_bos(model, prefix_ids, ans_ids)
    with torch.no_grad():
        logits = model(_ids_tensor(ids))                  # [1, seq, vocab]
    logprobs = torch.log_softmax(logits[0].float(), dim=-1)
    na = len(ans_ids)
    start = len(ids) - na                                 # first answer-token position
    return [logprobs[start + j - 1, ans_ids[j]].item() for j in range(na)]


def resid_features(model, prefix_ids, ans_ids, layers):
    """Concatenated resid_post features at `layers` for each answer token -> np.array [na, d*L]."""
    ids = _with_bos(model, prefix_ids, ans_ids)
    want = {f"blocks.{L}.hook_resid_post" for L in layers}
    with torch.no_grad():
        _, cache = model.run_with_cache(_ids_tensor(ids), names_filter=lambda n: n in want)
    na = len(ans_ids)
    start = len(ids) - na
    parts = [cache["resid_post", L][0, start:start + na, :].float().cpu().numpy() for L in layers]
    return np.concatenate(parts, axis=-1)


def resid_at_layer(model, prefix_ids, ans_ids, layer):
    """Raw resid_post at one layer for each answer token -> torch tensor [na, d_model] (on device)."""
    ids = _with_bos(model, prefix_ids, ans_ids)
    name = f"blocks.{layer}.hook_resid_post"
    with torch.no_grad():
        _, cache = model.run_with_cache(_ids_tensor(ids), names_filter=lambda n: n == name)
    na = len(ans_ids)
    start = len(ids) - na
    return cache["resid_post", layer][0, start:start + na, :].float()


def token_labels(offsets, hard_labels):
    """1 if the token's char span overlaps any gold hallucination span, else 0."""
    gold = [(int(s), int(e)) for s, e in (hard_labels or [])]
    return [1 if any(ts < ge and gs < te for (gs, ge) in gold) else 0 for (ts, te) in offsets]


def merge_flagged_spans(offsets, flags):
    """Merge char ranges of consecutive flagged tokens into [[start, end], ...]."""
    spans, cur = [], None
    for (s, e), f in zip(offsets, flags):
        if not f:
            if cur:
                spans.append(list(cur)); cur = None
            continue
        if cur and s <= cur[1]:
            cur = (cur[0], max(cur[1], e))
        else:
            if cur:
                spans.append(list(cur))
            cur = (s, e)
    if cur:
        spans.append(list(cur))
    return spans