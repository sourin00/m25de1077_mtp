#!/usr/bin/env python3
"""
check_mps_fidelity.py — does MPS give the SAME numbers as CPU on THIS machine?

Known issue: PyTorch's MPS backend can silently corrupt TransformerLens logits
(TransformerLens #1178 + several PyTorch 'module: correctness (silent)' bugs).
Shapes look fine; values can be wrong. This script makes the divergence visible
so you know whether to trust MPS for the white-box parts of the pilot.

Run:  python check_mps_fidelity.py
"""
import os
# Only silences the warning text — does NOT fix the underlying bug.
os.environ["TRANSFORMERLENS_ALLOW_MPS"] = "1"
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch
from transformer_lens import HookedTransformer

MODEL = "google/gemma-2-2b"          # swap for "meta-llama/Llama-3.2-1B-Instruct" to test your real backbone
LAYER = 12
PROMPT = "The Eiffel Tower is located in the city of"

def run(device):
    model = HookedTransformer.from_pretrained(MODEL, device=device)
    tokens = model.to_tokens(PROMPT)
    with torch.no_grad():
        logits, cache = model.run_with_cache(tokens)
    last_logits = logits[0, -1].float().cpu()
    resid = cache["resid_post", LAYER][0, -1].float().cpu()
    del model
    return last_logits, resid

print(f"Model: {MODEL} | comparing CPU vs MPS final-token logits and resid_post[{LAYER}]\n")

cpu_logits, cpu_resid = run("cpu")
mps_logits, mps_resid = run("mps")

# Logit agreement
logit_max_abs = (cpu_logits - mps_logits).abs().max().item()
logit_cos = torch.nn.functional.cosine_similarity(cpu_logits, mps_logits, dim=0).item()
top_cpu = cpu_logits.argmax().item()
top_mps = mps_logits.argmax().item()

# Residual agreement
resid_max_abs = (cpu_resid - mps_resid).abs().max().item()
resid_cos = torch.nn.functional.cosine_similarity(cpu_resid, mps_resid, dim=0).item()

print(f"  logits   : max|Δ| = {logit_max_abs:.4f} | cosine = {logit_cos:.6f} | "
      f"argmax cpu={top_cpu} mps={top_mps} {'(MATCH)' if top_cpu == top_mps else '(MISMATCH!)'}")
print(f"  resid[{LAYER}] : max|Δ| = {resid_max_abs:.4f} | cosine = {resid_cos:.6f}")

# Verdict — tolerances are generous; real corruption blows way past these.
ok = (logit_cos > 0.999) and (resid_cos > 0.999) and (top_cpu == top_mps)
print("\nVERDICT:", "MPS matches CPU closely — probably safe here." if ok
      else "MPS DIVERGES from CPU — run white-box extraction on CPU/CUDA, NOT MPS.")
