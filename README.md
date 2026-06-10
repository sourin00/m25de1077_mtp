# Interpretable white-box hallucination detection on multilingual Mu-SHROOM

A token-level hallucination detector built from **sparse autoencoder (SAE) features** over a
model's residual stream, evaluated on Mu-SHROOM (SemEval-2025 Task 3) in Spanish, Czech,
Mandarin, and English. The claim: **a white-box signal genuinely detects hallucination at the
token level, multilingually — validated against surface AND topic confounds** (a within-item
control most of the literature skips), with interpretable SAE features and an honest set of
negative results.

> Status: pilot complete; signal validated through the full control gauntlet; writing.

## What's validated

| Result | Number |
|---|---|
| SAE probe token-AUROC (full test split, 5 seeds) | 0.745 ± 0.012 |
| Surface-only baseline (the confound control) | 0.698 ± 0.026 — recovers ~80% but not the signal |
| **Within-item AUROC (topic held constant — the decider)** | **0.818 ± 0.011** (vs surface 0.730), all 4 langs |
| SAE vs raw residuals | parity — interpretability at no accuracy cost |
| Char-level IoU (full split, 5 seeds, per-lang threshold) | 0.396 vs FLAG-ALL 0.313 (+0.084); clears the floor in all 4 langs; UCSC ~0.55 |

The within-item control is the headline: inside a single answer (topic, language, generator
fixed), the probe still separates hallucinated from faithful tokens at 0.82, ~0.09 above a
surface control, in every language — so the signal is real token-level detection, not surface or
topic confound.

Negative results (honest scoping): routing by claim type does not improve detection (per-route
AUROC flat); the internal-vs-external "gap" cannot be tested on heterogeneous generations
(proxy-model confound); retrieval-CSR (REFIND) collapses to chance on a proxy model.

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
- `iou_eval.py` — probe vs floors, AUROC, IoU on the full test split (per-language calibrated)
- `confound_check.py` — surface baseline vs SAE probe, full test split, per language (confound control)
- `within_item.py` — within-item AUROC: the topic-held-constant control (the decider)
- `feature_analysis.py` — which SAE features the probe uses
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