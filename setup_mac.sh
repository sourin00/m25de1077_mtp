#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# setup_mac.sh — environment for the Mu-SHROOM hallucination-detection pilot
# Target: Apple M4 Max, 64 GB, macOS Tahoe (arm64)
#
# Usage:
#   chmod +x setup_mac.sh
#   ./setup_mac.sh
# ---------------------------------------------------------------------------
set -euo pipefail

# --- 0. Sanity: confirm we're on Apple Silicon -----------------------------
ARCH="$(uname -m)"
if [[ "$ARCH" != "arm64" ]]; then
  echo "WARNING: expected arm64, got $ARCH. Are you in a Rosetta shell?"
fi

# --- 1. Python ---------------------------------------------------------------
# Use Python 3.11: broad library compatibility (some ML libs still lag on 3.13).
# If you don't have it: `brew install python@3.11`  (install Homebrew first).
PYTHON="${PYTHON:-python3.11}"
echo ">> Using interpreter: $($PYTHON --version)"

# --- 2. Virtual environment --------------------------------------------------
VENV_DIR="${VENV_DIR:-.venv-mushroom}"
if [[ ! -d "$VENV_DIR" ]]; then
  $PYTHON -m venv "$VENV_DIR"
fi
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip wheel setuptools

# --- 3. Core stack -----------------------------------------------------------
# PyTorch: the default macOS arm64 wheels already ship the MPS (Metal) backend.
# Do NOT add a CUDA index URL — there is no CUDA on Mac.
pip install torch torchvision torchaudio

# White-box analysis
pip install transformer-lens sae-lens

# HF model loading / data
pip install "transformers>=4.44" accelerate datasets huggingface_hub

# Retrieval + NLI + embeddings (TRACE External route, Relational NLI)
pip install sentence-transformers faiss-cpu       # faiss-cpu: no CUDA needed at pilot scale

# Classical heads: Deloitte-style residual MLP, surrogate, route classifier
pip install scikit-learn xgboost

# API clients for atomic decomposition / route classification
pip install anthropic openai

# Plotting / bookkeeping
pip install pandas matplotlib tqdm

echo ">> pip install complete."

# --- 4. Environment variables -----------------------------------------------
# MPS doesn't implement every op; fall back to CPU instead of crashing.
PROFILE="${PROFILE:-$HOME/.zshrc}"
if ! grep -q "PYTORCH_ENABLE_MPS_FALLBACK" "$PROFILE" 2>/dev/null; then
  {
    echo ""
    echo "# Mu-SHROOM pilot: let unsupported MPS ops fall back to CPU"
    echo "export PYTORCH_ENABLE_MPS_FALLBACK=1"
  } >> "$PROFILE"
  echo ">> Added PYTORCH_ENABLE_MPS_FALLBACK=1 to $PROFILE (open a new shell to pick it up)."
fi
export PYTORCH_ENABLE_MPS_FALLBACK=1

# --- 5. Reminders (not automated on purpose) --------------------------------
cat <<'EOF'

------------------------------------------------------------------
NEXT, DO THESE MANUALLY:

1. Hugging Face auth + model licenses (Llama & Gemma are gated):
     huggingface-cli login
   Then accept the licenses on the model pages:
     - meta-llama/Llama-3.2-1B-Instruct   (Deloitte's backbone)
     - meta-llama/Llama-3.2-3B-Instruct
     - google/gemma-2-2b                  (for Gemma Scope SAEs)

2. API keys for atomic decomposition / route classification:
     export ANTHROPIC_API_KEY=...     # or
     export OPENAI_API_KEY=...

3. (Optional) For running larger decomposition models locally instead of API,
   install Ollama or LM Studio. NOTE: these expose no residual-stream hooks,
   so they're only for the API-style TRACE routes, never for SAGE.

4. Run the smoke test:
     python smoke_test.py
------------------------------------------------------------------
EOF

echo ">> Done. Activate later with: source $VENV_DIR/bin/activate"
