#!/usr/bin/env python3
"""
smoke_test.py — verify the Mu-SHROOM pilot stack on Apple Silicon before
investing time in the actual experiments.

Each check is isolated in try/except so one failure doesn't hide the others.
Run:  python smoke_test.py
"""
import os
import sys
import traceback

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

PASS, FAIL = "PASS", "FAIL"
results = {}


def check(name):
    """Decorator: run a check, record PASS/FAIL, never crash the whole script."""
    def wrap(fn):
        print(f"\n=== {name} ===")
        try:
            fn()
            results[name] = PASS
            print(f"[{PASS}] {name}")
        except Exception as e:  # noqa: BLE001
            results[name] = FAIL
            print(f"[{FAIL}] {name}: {e}")
            traceback.print_exc(limit=2)
    return wrap


# --- 1. PyTorch + MPS --------------------------------------------------------
@check("torch / MPS backend")
def _():
    import torch
    print("torch:", torch.__version__)
    avail = torch.backends.mps.is_available()
    built = torch.backends.mps.is_built()
    print("mps available:", avail, "| mps built:", built)
    assert avail and built, "MPS not available — check macOS >= 12.3 and arm64 build"
    # quick op on the GPU
    x = torch.randn(1024, 1024, device="mps")
    y = (x @ x).sum().item()
    print("sample matmul on mps ok, scalar:", round(y, 2))


# --- 2. TransformerLens hook capture (the SAGE / Deloitte substrate) ---------
@check("TransformerLens resid_post hook capture")
def _():
    import torch
    from transformer_lens import HookedTransformer
    # gpt2 is ungated — proves the hook mechanism without HF-login friction.
    # Swap to "meta-llama/Llama-3.2-1B-Instruct" after huggingface-cli login.
    model = HookedTransformer.from_pretrained("gpt2", device="mps")
    tokens = model.to_tokens("The Eiffel Tower is located in")
    logits, cache = model.run_with_cache(tokens)
    resid = cache["resid_post", 6]          # residual stream after block 6
    print("logits:", tuple(logits.shape), "| resid_post[6]:", tuple(resid.shape))
    assert resid.ndim == 3, "unexpected residual shape"
    print("NOTE: for the pilot, re-run this with Llama-3.2-1B-Instruct.")


# --- 3. SAE load (Gemma Scope / LlamaScope via SAELens) ----------------------
@check("SAELens pretrained SAE load")
def _():
    from sae_lens import SAE
    # Loads just the SAE weights (lightweight); does not run the full LM.
    sae = SAE.from_pretrained(
        release="gemma-scope-2b-pt-res-canonical",
        sae_id="layer_12/width_16k/canonical",
        device="mps",
    )
    # API returns either an SAE or a (sae, cfg, sparsity) tuple across versions
    obj = sae[0] if isinstance(sae, tuple) else sae
    print("SAE d_in:", obj.cfg.d_in, "| d_sae:", obj.cfg.d_sae)


# --- 4. Sentence embeddings (External route retrieval) -----------------------
@check("SBERT embeddings")
def _():
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="mps")
    emb = m.encode(["Sándor Palace is a presidential residence.",
                    "Buda Castle is a royal palace and museum complex."])
    print("embedding matrix:", emb.shape)


# --- 5. NLI (Relational route cross-claim entailment) ------------------------
@check("DeBERTa NLI cross-encoder")
def _():
    from sentence_transformers import CrossEncoder
    nli = CrossEncoder("cross-encoder/nli-deberta-v3-base", device="mps")
    scores = nli.predict([("Sándor Palace is a presidential residence.",
                           "Sándor Palace is also known as Buda Castle.")])
    print("NLI logits [contradiction, entailment, neutral]:", scores)


# --- 6. FAISS (retrieval index) ---------------------------------------------
@check("FAISS index build/search")
def _():
    import numpy as np
    import faiss
    d, n = 384, 100
    xb = np.random.random((n, d)).astype("float32")
    index = faiss.IndexHNSWFlat(d, 32)      # HNSW, matching the proposal
    index.add(xb)
    D, I = index.search(xb[:1], 5)
    print("HNSW search returned neighbours:", I[0].tolist())


# --- 7. API clients (atomic decomposition / route classification) -----------
@check("LLM API key present")
def _():
    has_anthropic = bool(os.environ.get("ANTHROPIC_API_KEY"))
    has_openai = bool(os.environ.get("OPENAI_API_KEY"))
    print("ANTHROPIC_API_KEY set:", has_anthropic, "| OPENAI_API_KEY set:", has_openai)
    assert has_anthropic or has_openai, "set at least one API key for decomposition"


# --- summary -----------------------------------------------------------------
def main():
    print("\n" + "=" * 50)
    print("SMOKE TEST SUMMARY")
    print("=" * 50)
    for name, status in results.items():
        print(f"  {status:4}  {name}")
    failed = [n for n, s in results.items() if s == FAIL]
    if failed:
        print(f"\n{len(failed)} check(s) failed. Most failures here are either a "
              f"missing HF login/API key or an MPS op gap — read the trace above.")
        sys.exit(1)
    print("\nAll checks passed. You're ready to build the pilot.")


if __name__ == "__main__":
    main()
