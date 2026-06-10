# Thesis proposal — Interpretable white-box hallucination detection on multilingual Mu-SHROOM, validated against surface and topic confounds

**Student:** Sourin Ghosh (M25DE1077) · **Supervisor:** Dr. Subhash Bhagat
**Supersedes:** the multi-signal-trust-score and TRACE+SAGE framings (archived; see
`decisions.md`). Direction converged after a pilot whose controls are themselves a contribution.

## 1. Problem

White-box / internal-states methods are increasingly proposed for hallucination detection (on
Mu-SHROOM, a residual probe won French). But such probes are easy to fool: a classifier that
scores well across held-out items may be exploiting surface form (token length, punctuation,
position) or topic (content that correlates with which items are hallucinated) rather than a
hallucination representation. This thesis asks whether a white-box signal is **real** in the
multilingual span-level setting, and answers it with controls the literature usually omits.

## 2. Approach

A token-level probe over Gemma-2-2b residuals using interpretable Gemma Scope **SAE features**,
trained per token and evaluated on Spanish, Czech, Mandarin, English by token-AUROC and
char-level IoU. Then a gauntlet of confound controls:
- **Surface baseline** — a 12-feature surface-only probe (length, punctuation, whitespace,
  digit, position, language; no model).
- **Within-item AUROC** — discrimination of hallucinated vs faithful tokens *inside the same
  answer*, where topic/language/generator are constant, so it cannot be topic or surface
  confound.
- Plus interpretation (which SAE features the probe uses) and ablations (routing, the gap).

## 3. Findings (pilot, complete)

- **The probe works:** token-AUROC 0.745 ± 0.012 on the full test split (5 seeds), beating the
  flag-all floor in 3/4 languages.
- **Surface is a large, under-reported confound:** a 12-feature surface probe alone reaches
  AUROC 0.698 — ~80% of the pooled above-chance signal — so raw white-box AUROC numbers are
  inflated and must be reported against a surface baseline. But surface does NOT explain it away:
  the SAE retains a +0.06–0.13 per-language edge, all four languages, non-overlapping bars.
- **The signal is genuine, not topic (the decider):** **within-item AUROC = 0.818 ± 0.011**
  (es .843, cs .831, en .827, zh .781), ~+0.09 above the surface within-item (0.730) in every
  language. Within an item, topic can't vary, so this is real token-level hallucination signal,
  robust to both confounds. Within-item exceeds pooled (0.744) — cross-item topic variation was
  depressing the pooled metric, not inflating it.
- **SAE features at parity with raw residuals** — interpretability at no accuracy cost.

Honest negatives (scoping): routing by claim type adds nothing (per-route AUROC flat); the
internal/external gap is unmeasurable on heterogeneous generations (proxy confound); retrieval-
CSR is at chance on a proxy model.

## 4. Contributions

1. A genuine, multilingual, token-level white-box hallucination signal (within-item AUROC 0.82)
   in interpretable SAE features, validated against **both** a surface baseline and a topic
   (within-item) control — controls largely absent from the white-box hallucination literature.
2. Evidence that surface form is a large, under-reported confound (~80% of pooled signal),
   establishing the surface + within-item controls as necessary methodology for such claims.
3. SAE features shown equivalent to raw residuals (interpretability at parity).
4. A map of what does NOT add value here (routing, internal/external gap, proxy retrieval).

## 5. Evaluation

Token-AUROC (pooled and within-item) and char-level IoU; floors; item-level splits; 5 seeds with
error bars; per-language reporting; surface and within-item controls as primary diagnostics.

## 6. Caveats and scope

Char-level IoU is $0.396$ vs a FLAG-ALL floor of $0.313$ (full split, 5 seeds; the probe clears
the floor in all four languages, tying densely-hallucinated Mandarin), still below UCSC ~$0.55$:
strong token discrimination does not fully convert
to competitive span overlap on this base-rate-saturated benchmark — a span-extraction problem,
not a signal problem (Phase 2). Proxy-model setting (the probe detects hallucination in text the
model didn't generate — robust, but the gap method still needs self-generation). Within-item
controls for topic and surface, not every conceivable token-level confound.

## 7. Timeline (remaining — mostly writing)

| Phase | Work |
|---|---|
| 1 | done: probe, surface + within-item controls, interpretation, ablations (all in `runs/`) |
| 2 | write: premise & confound risk → method → apparent performance → control gauntlet → negatives → caveats |
| 3 | figures/tables, references, submission |
| (stretch) | span extraction to lift IoU; self-generation pilot for the gap |

## 8. Risks
- **Over-claiming.** Mitigation: lead with within-item + surface controls and the IoU caveat;
  state the proxy-model scope.
- **Modest IoU invites "so what."** Mitigation: AUROC + the controls are the contribution; IoU
  conversion is explicitly framed as future span-extraction work.

## References (to expand)
Mu-SHROOM / SemEval-2025 Task 3; UCSC (Huang et al. 2025); Deloitte (SemEval-2025);
internal-states skeptic critique (arXiv:2510.09033); Gemma Scope (Lieberum et al. 2024);
Ferrando et al. (ICLR 2025); Orgad et al. (ICLR 2025); REFIND (Lee & Yu 2025).