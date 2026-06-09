"""Central config for the Mu-SHROOM pilot. Edit values here; nothing else changes.

Migration note: the decomposition/routing LLM is OpenAI-compatible. To go from
the free local model to a paid API later, set these env vars (no code change):
  OpenAI:  LLM_BASE_URL=https://api.openai.com/v1  LLM_API_KEY=sk-...  LLM_MODEL=gpt-4o
  (Anthropic uses a different SDK; ask and I'll add an adapter when you're ready.)
"""
import os
from dataclasses import dataclass


@dataclass
class Config:
    # --- data ---
    hf_dataset: str = "Helsinki-NLP/mu-shroom"
    # split: str = "validation"               # validation carries gold soft/hard labels
    langs: tuple = ("es", "cs", "zh", "en")  # Spanish, Czech, Mandarin + English anchor
    sample_per_lang: int = 20
    eval_per_lang: int = 60                  # larger draw for the honest (representative) IoU run
    seed: int = 13

    # --- white-box (MPS validated faithful vs CPU for Llama-3.2 on your machine) ---
    whitebox_model: str = "meta-llama/Llama-3.2-1B-Instruct"
    whitebox_device: str = "mps"            # flip to "cpu" or "cuda" without touching code
    spot_check_every: int = 25              # periodic CPU re-check during extraction (0 = off)

    # --- decomposition / routing LLM (OpenAI-compatible; Ollama by default) ---
    llm_base_url: str = os.environ.get("LLM_BASE_URL", "http://localhost:11434/v1")
    llm_api_key: str = os.environ.get("LLM_API_KEY", "ollama")   # dummy value for Ollama
    llm_model: str = os.environ.get("LLM_MODEL", "qwen2.5:14b")
    llm_temperature: float = 0.0

    # --- baselines (Milestone 2) ---
    probe_layers: tuple = (6, 9, 12)         # resid_post layers for the Deloitte-style probe
    retrieval_max_chars: int = 4000          # truncate fetched Wikipedia context
    csr_eps: float = 1e-6                    # REFIND denominator epsilon
    probe_test_frac: float = 0.30            # item-level held-out fraction for the probe

    # --- SAGE substrate (Gemma-2-2b + Gemma Scope SAEs) ---
    # Gemma has pretrained SAEs (Llama-3.2-1B does not). MPS fidelity on Gemma-2 is NOT yet
    # validated (different arch: logit soft-capping, alternating attention) -> run
    # check_mps_fidelity.py with this model BEFORE trusting any SAGE numbers.
    sage_model: str = "google/gemma-2-2b"
    sage_layer: int = 12                     # single layer for the raw-vs-SAE head-to-head
    sae_release: str = "gemma-scope-2b-pt-res-canonical"
    sae_id: str = "layer_12/width_16k/canonical"   # must match sage_layer
    sae_device: str = "cpu"                  # encode SAE on CPU to dodge sae_lens+MPS edges

    # --- output ---
    out_dir: str = "runs"


CFG = Config()