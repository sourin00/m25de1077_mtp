# Decision log

Dated record of direction changes and the evidence that drove them — the methods narrative and
the "why not X" sections come straight from here. The arc itself is a contribution: a naive
AUROC looked confounded under weak controls, but proper-powered controls show the signal is real.

**Final direction (D6):** an interpretable multilingual white-box hallucination detector,
validated against surface AND topic confounds. The within-item control is the headline.

## D0 — Original framing
Multi-signal trust score over TruthfulQA/FEVER/HotpotQA. Archived. Dropped for Mu-SHROOM
(span-level, multilingual, gold labels, public leaderboard).

## D1 — TRACE + SAGE
Decompose + route claims; SAGE = internal-vs-external "gap" verifier. Chosen for novelty.

## D2 — Pilot
Residual probe AUROC ~0.78; SAE-feature probe at parity (0.77). Gap untestable on proxy
generations (CONF ~chance). Routing flat across types. REFIND ~chance on proxy.

## D3/D4 — Scope + flat codebase
Committed to the probe + interpretability; added `features.py`, `probe.py`,
`feature_analysis.py`, `confound_check.py`, `within_item.py` (flat layout).

## D5 — Preliminary confound read (SUPERSEDED by D6)
On a 240-item representative sample, a 12-feature surface baseline (0.692) appeared to match the
SAE probe (0.690); read this as "predominantly surface confound." **This was underpowered** —
small sample, and a surface-only control that cannot address topic. Kept as a cautionary step:
weak controls on small samples can fake a confound result just as easily as a positive one.

## D6 — Proper-powered controls resolve it: the signal is real
- **Full test split (556 items), 5 seeds.** SAE probe AUROC 0.745 ± 0.012 vs surface
  0.698 ± 0.026; per language the SAE edge is +0.06–0.13, all four languages, non-overlapping
  bars. Surface form is a large confound (~80% of pooled above-chance signal) but does NOT
  explain the signal away.
- **Within-item control (the decider).** For items with both classes (500/556), AUROC computed
  WITHIN each item (topic/language/generator constant): **SAE 0.818 ± 0.011**, surface
  0.730 ± 0.013, all four languages (es .843, cs .831, en .827, zh .781), each ~+0.09 over
  surface. Within-item > pooled (0.744): cross-item topic variation was depressing the pooled
  metric, not inflating it.
- **Conclusion:** a genuine token-level hallucination signal in Gemma-2-2b SAE features, robust
  to surface AND topic confounds, multilingual. The confound is real and under-reported, but the
  signal survives the controls that diagnose it.
- **Caveats:** char-level IoU stays modest (~0.33 vs UCSC ~0.55) — strong token discrimination
  doesn't convert to span overlap on this saturated benchmark (Phase-2 span extraction).
  Proxy-model setting (notable that it works cross-model). Subtler within-item confounds beyond
  surface/topic not excluded.

## Negatives (kept as honest scoping)
Routing by claim type doesn't help (per-route AUROC flat). The internal/external gap is
unmeasurable on heterogeneous generations (proxy confound) — needs self-generation. REFIND
retrieval ~chance on a proxy model.

## Future
Span extraction to convert the 0.82 token signal into competitive IoU; self-generation to test
the gap; richer within-item confound controls (entity-type, frequency).