# Interpretable white-box hallucination detection on multilingual Mu-SHROOM

A token-level hallucination detector built from **sparse autoencoder (SAE) features** over a
model's residual stream, evaluated on Mu-SHROOM (SemEval-2025 Task 3) in Spanish, Czech,
Mandarin, and English. The thesis claim is not "beat the leaderboard" but: **white-box signal
detects hallucinations multilingually, and SAE features let us name *which* internal directions
carry that signal** — interpretability no SemEval system offered — plus a set of honest negative
results that map the design space.

> Status: pilot complete; detector and interpretability validated; assembling the analysis and
> writeup. A leaderboard-competitiveness push (multi-layer features, larger backbone) is a
> scoped Phase 2, not the core claim.

## What's validated (pilot)

| Component | Result |
|---|---|
| Decomposition + span-mapping | faithful, multilingual; ~94% of atoms anchor to char spans (zh 82%) |
| Claim routing (κ vs reference) | ~0.72 — usable, but routes are **not** differentially verifiable |
| White-box probe (token AUROC) | ~0.70–0.78 across languages; beats flag-all in es/cs/en |
| SAE vs raw residuals | **parity** (0.770 vs 0.773 AUROC) — interpretability at no accuracy cost |
| Honest IoU (representative) | probe 0.322 vs flag-all 0.286 (+0.04 overall; +0.07–0.10 es/cs/en; −0.13 zh) |

Negative results (reported as findings): routing by claim type does not improve detection
(per-route AUROC is flat: Rel 0.78 / Ext 0.79 / Subj 0.71); the internal-vs-external "gap"
cannot be tested on heterogeneous external generations (proxy-model confound); retrieval-CSR
(REFIND) collapses to chance on a proxy model.

## Repo (flat)

**Library**
- `config.py` — central config (model, layers, SAE id, sample sizes)
- `data.py` — load Mu-SHROOM test parquets per language
- `wb.py` — model load, tokenize+offsets, residuals, log-probs, token labels, span-merge
- `features.py` — SAE load/encode + **on-disk cache** of Gemma forwards
- `probe.py` — train / predict / **per-language threshold calibration**
- `metrics.py` — char-level IoU (the Mu-SHROOM metric)
- `llm_client.py`, `decompose_route.py` — decomposition + routing + span-mapping (explanation layer)

**Experiments**
- `iou_eval.py` — MAIN: probe vs floors, AUROC, IoU on a representative sample
- `feature_analysis.py` — interpretability core: which SAE features signal hallucination
- `trace.py` — ablation: routing does not help
- `sage.py` — ablation: SAE==raw, gap shows no lift
- `deloitte_probe.py`, `refind.py`, `baseline_trivial.py` — baselines
- `label_routes.py`, `score_vs_claude.py` — routing-agreement (κ)

**Checks**: `check_mps_fidelity.py`, `smoke_test.py`, `setup_mac.sh`

## Run

```bash
# env: Python 3.11 venv; Ollama for decomposition; HF login for Gemma/Llama; MPS on Apple Silicon
python check_mps_fidelity.py            # verify MPS == CPU for the backbone (set MODEL inside)
python iou_eval.py                      # detection result (probe vs flag-all, per-language)
python feature_analysis.py              # interpretability: top hallucination features
python decompose_route.py               # decomposition + routing + span anchoring
python trace.py                         # routing ablation
python sage.py                          # SAE-vs-raw + gap ablation
```

Outputs and the feature cache live under `runs/`. See `decisions.md` for why the architecture
is what it is.
