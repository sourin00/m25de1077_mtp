# Decision log

Dated record of direction changes and the evidence that drove them. These are thesis assets:
the "why not X" sections write themselves from here.

## D0 — Original framing
Multi-signal trust score (sampling consistency + NLI + retrieval) over TruthfulQA / FEVER /
HotpotQA. Archived under `docs/archive/`. Dropped in favor of a span-level multilingual
benchmark with gold labels (Mu-SHROOM), which is directly comparable to a public leaderboard.

## D1 — TRACE + SAGE
Decompose answers into atomic claims, route by epistemic type (External / Relational /
Subjective) to specialized verifiers; SAGE = white-box internal-vs-external "gap" as the
Relational verifier. Chosen for novelty (no SemEval team used SAEs or the gap framing).

## D2 — Pilot findings that reshaped the thesis

- **Substrate works.** Residual probe token-AUROC ≈ 0.778 (Llama) / 0.773 (Gemma); beats
  flag-all in es/cs/en. White-box signal is real and model-agnostic.
- **SAEs at parity.** SAE-feature probe 0.770 vs raw 0.773 — interpretability at no accuracy
  cost. This becomes the core contribution.
- **Gap untestable here.** External "assertion" measured on a proxy model (Gemma over text it
  didn't generate) is near-chance (CONF AUROC 0.540); SAE+CONF adds no lift. The gap needs
  self-generated answers, which Mu-SHROOM's heterogeneous generations don't provide. → SAGE
  deferred to future work.
- **Routing doesn't help.** Per-route AUROC is flat (Relational 0.780, External 0.791,
  Subjective 0.708); routed IoU variants under-perform the flat probe. The taxonomy is not
  differentially verifiable. → routing demoted to an *explanation* layer (`trace.py` becomes an
  ablation).
- **Retrieval-CSR collapses on proxy.** REFIND token-AUROC ≈ 0.502. Faithful REFIND would need
  generator-scored probabilities + targeted retrieval; out of scope for the core.

## D3 — Commit to the low-investment thesis
Interpretable SAE-feature detector + feature-interpretation study + honest negatives. Reason:
fully validated, achievable, genuinely novel. Leaderboard competitiveness (IoU 0.32 vs UCSC
0.55 in minimal form) is a **Phase 2** stretch, not the core claim.

## D4 — Codebase: keep flat
Added `features.py` (SAE + forward cache) and `probe.py` (per-language calibration) to de-
duplicate the experiment scripts; `feature_analysis.py` for the interpretability core. No
package restructure for now.

## Open / Phase-2 levers (to test if pursuing competitiveness)
Per-language thresholds (done in `probe.py`); multi-layer features; Gemma-2-9b; span
post-processing beyond merging adjacent flagged tokens; a faithful self-generation experiment
to revisit the gap.
