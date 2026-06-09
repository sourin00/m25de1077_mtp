# Thesis proposal — What do white-box hallucination probes actually learn? A confound analysis on multilingual Mu-SHROOM

**Student:** Sourin Ghosh (M25DE1077) · **Supervisor:** Dr. Subhash Bhagat
**Supersedes:** the multi-signal-trust-score and TRACE+SAGE framings (archived / see
`decisions.md`). The direction converged after a pilot whose every result pointed the same way.

## 1. Problem

White-box / internal-states methods are increasingly proposed for hallucination detection — on
Mu-SHROOM (SemEval-2025 Task 3) a residual-stream probe (Deloitte) won French. The implicit
claim is that a model's hidden states encode whether it is hallucinating. But probes are easy to
fool: a classifier that scores well on held-out items may be exploiting **surface and topic
correlates of the labels** rather than a hallucination representation. This thesis asks, for the
multilingual span-level setting, **whether the white-box signal is real**, and provides the
controls the field largely omits.

## 2. Approach

Build the full apparatus a white-box detector would use, then try to break it with controls:
- **Detector:** a probe over Gemma-2-2b residuals, using interpretable Gemma Scope **SAE
  features**, trained per token and evaluated by char-level IoU (Mu-SHROOM metric) and
  token-AUROC, across Spanish, Czech, Mandarin, English.
- **Interpretation:** rank the SAE features the probe relies on; inspect what they represent.
- **Controls:** (i) a **surface-only baseline** (token length, punctuation, whitespace, digit,
  position, language — no model); (ii) frequency-split probes (always-on vs rare features);
  (iii) per-route AUROC to test whether claim type is differentially verifiable; (iv) the
  internal-vs-external "gap" idea.

## 3. Findings (pilot, complete)

- **Apparent success:** the SAE probe reaches token-AUROC ~0.69–0.78 and beats the flag-all
  IoU floor in 3/4 languages — superficially a working white-box detector.
- **Most of it is confound:** a 12-feature **surface-only probe recovers the bulk** of the
  apparent performance — AUROC 0.69 (representative) / 0.67 (dense) against the SAE probe's
  0.69 / 0.71. So the large majority of the "white-box signal" is surface and topic structure.
- **The residual is small and fragile:** a surface control leaves the SAE probe only a ≤0.06–0.08
  edge; it inverts in Spanish, washes out on the larger representative draw (where `surface+rare`
  no longer beats surface), and sits within wide error bars. Not the stable edge a real
  hallucination representation would give.
- **Interpretation agrees:** the probe's most predictive features are topics (boxing, anatomy,
  transport, geography) and surface tokens (periods, newlines, function words), not
  hallucination directions.
- **Elaborations don't help:** routing by claim type adds nothing (per-route AUROC flat); the
  internal/external gap is unmeasurable on heterogeneous generations (proxy-model confound);
  retrieval-CSR is at chance on a proxy model.

## 4. Contributions

1. Evidence that **proxy-model white-box hallucination probing on Mu-SHROOM is largely
   surface/topic confounded** — a trivial surface baseline recovers most of the apparent AUROC,
   and the residual white-box edge is small (≤0.06–0.08) and not robust across languages or
   samples, shown multilingually.
2. **Feature-level interpretation** localizing the probe's reliance on topic and surface form.
3. A reusable **control protocol** (surface baseline + feature interpretation + frequency split)
   that any white-box hallucination-detection claim should pass.
4. A map of why the popular elaborations (routing, internal/external gap, proxy retrieval) fail
   here — empirical support for the internal-states skeptic critique (arXiv:2510.09033) in a
   multilingual, span-level setting.

## 5. Evaluation

Token-AUROC (leakage-free) and char-level IoU; floors (flag-all/flag-none); item-level splits;
multiple seeds with error bars; per-language reporting; the surface baseline as the primary
control. Comparisons framed as confound diagnosis, not leaderboard ranking.

## 6. Scope and future work

The claim is bounded to the **proxy-model** setting (prober ≠ generator). Whether self-generation
white-box probing (the prober *is* the generating model; `scratch.md` Option 1) escapes the
confound is the open question and the recommended next study; a small self-generation pilot is a
natural extension if time permits.

## 7. Timeline (remaining — mostly writing)

| Phase | Work |
|---|---|
| 1 | robustness: surface baseline on the dense regime + per-language table (done in `confound_check.py`) |
| 2 | write: apparatus & apparent success → controls → scope → relate to skeptic literature |
| 3 | figures/tables, references, submission |
| (stretch) | self-generation pilot to probe the boundary of the confound |

## 8. Risks

- **"Just a negative result."** Mitigation: the surface-baseline control shows a trivial probe
  recovers most of the apparent white-box AUROC, with only a small non-robust residual left —
  a concrete methodological contribution (the missing control) that tempers a common assumption.
- **Over-generalization.** Mitigation: the proxy-model scope line is stated explicitly throughout.

## References (to expand)
Mu-SHROOM / SemEval-2025 Task 3; UCSC (Huang et al. 2025); Deloitte (SemEval-2025);
internal-states skeptic critique (arXiv:2510.09033); Gemma Scope (Lieberum et al. 2024);
Ferrando et al. (ICLR 2025); Orgad et al. (ICLR 2025); REFIND (Lee & Yu 2025).