# Decision log

Dated record of direction changes and the evidence that drove them. These are thesis assets:
the methods narrative and the "why not X" sections write themselves from here.

**Current direction (D5):** a confound analysis — *what do white-box hallucination probes
actually learn on Mu-SHROOM?* The answer, in the proxy-model setting: predominantly surface and
topic structure, with at most a small, fragile residual that doesn't reproduce across
languages or samples.

## D0 — Original framing
Multi-signal trust score (sampling consistency + NLI + retrieval) over TruthfulQA / FEVER /
HotpotQA. Archived under `docs/archive/`. Dropped for a span-level multilingual benchmark with
gold labels (Mu-SHROOM), directly comparable to a public leaderboard.

## D1 — TRACE + SAGE
Decompose answers into atomic claims, route by epistemic type, SAGE = white-box internal-vs-
external "gap" as the Relational verifier. Chosen for novelty (no SemEval team used SAEs or the
gap framing).

## D2 — Pilot findings
- Residual probe token-AUROC ≈ 0.778 (Llama) / 0.773 (Gemma); SAE-feature probe at parity
  (0.770). Looked like real, interpretable white-box signal.
- Gap untestable: external "assertion" on a proxy model is near-chance (CONF AUROC 0.540);
  SAE+CONF adds no lift. Needs self-generated answers Mu-SHROOM doesn't provide. → SAGE deferred.
- Routing doesn't help: per-route AUROC flat (Rel 0.780 / Ext 0.791 / Subj 0.708); routed IoU
  under-performs the flat probe. Taxonomy not differentially verifiable. → routing demoted.
- Retrieval-CSR (REFIND) ≈ chance on a proxy model (0.502).

## D3 — Commit to a low-investment thesis (interpretable detector)
Reasoning at the time: the probe worked (AUROC ~0.7) and SAEs were interpretable; lead with that
+ feature analysis + negatives. Leaderboard competitiveness (IoU 0.32 vs UCSC 0.55) parked as a
Phase-2 stretch.

## D4 — Codebase kept flat
Added `features.py` (SAE + forward cache), `probe.py` (per-language calibration),
`feature_analysis.py`, `confound_check.py`.

## D5 — The confound result (reframes the whole thesis)
- **Honest IoU (representative, multi-seed):** probe 0.330 ± 0.008 vs flag-all 0.310 ± 0.022;
  lift only +0.020 ± 0.015; below UCSC's ~0.55; zh ≤ flag-all (density saturates the metric).
- **Feature interpretation:** the predictive SAE features are TOPICS (boxing, anatomy,
  transport, geography) and SURFACE FORM (periods, "\n\n", function words, morphology
  fragments) — not hallucination directions.
- **Surface-baseline control:** a 12-feature surface-only probe recovers the BULK of the
  apparent performance — AUROC 0.692 (representative) / 0.670 (dense) vs the SAE probe's 0.690 /
  0.709. The overall representative tie is partly averaging: per language the SAE edge is
  +0.05 cs, +0.06 en, +0.02 zh, −0.04 es (representative) and +0.06–0.08 across all four (dense).
- **The residual is small and fragile:** the SAE-over-surface edge is ≤ ~0.06–0.08, inverts in
  es, and does not reproduce across samples (`surface+rare` beats surface on the dense draw but
  not the representative one; error bars are wide, ±0.03–0.13). Not the stable edge a real
  hallucination representation would give.
- **Conclusion:** proxy-model white-box probing on Mu-SHROOM is PREDOMINANTLY surface/topic
  confound — a trivial baseline recovers most of the AUROC — with at most a small, non-robust
  residual. The earlier AUROC "successes" are largely the confound. Empirically supports the
  internal-states skeptic critique (arXiv:2510.09033) on a multilingual benchmark.
- **Scope:** claim is bounded to the PROXY-model setting (prober ≠ generator). Self-generation
  (Option 1 in `scratch.md`) is the boundary and the recommended future direction.

## Open / future
Self-generation white-box probing (prober = generator); whether the confound persists there is
the open question. Surface-baseline + feature-interpretation controls recommended as standard
methodology for any white-box hallucination-detection claim.