# Thesis proposal — Interpretable white-box hallucination detection on multilingual Mu-SHROOM

**Student:** Sourin Ghosh (M25DE1077) · **Supervisor:** Dr. Subhash Bhagat
**Supersedes:** the earlier multi-signal-trust-score proposal (TruthfulQA/FEVER/HotpotQA),
archived under `docs/archive/`. The direction changed after a pilot; see `decisions.md`.

## 1. Problem

Large language models hallucinate — assert fluent, confident, factually unsupported content.
Mu-SHROOM (SemEval-2025 Task 3) frames detection at the **character span** level across 14
languages, scored by intersection-over-union (IoU) against human spans. Most strong systems are
black-box (sampling consistency, retrieval, NLI). Whether a model's **own internal states**
betray its hallucinations — and whether that signal is **interpretable** — is the open question
this thesis addresses, with particular attention to the lower-resource languages (Spanish,
Czech, Mandarin) where the leading system (UCSC) is weakest.

## 2. Approach

A token-level detector that reads the residual stream of an open model (Gemma-2-2b) and scores
each answer token for hallucination. The distinctive choice is the **representation**: instead
of raw residual vectors (opaque), the detector uses **sparse autoencoder features** (Gemma
Scope), so the signal decomposes into individually inspectable, nameable directions. A light
decomposition-and-routing front-end (claims typed as External / Relational / Subjective, each
anchored to its character span) provides **explanations** — what *kind* of error each flagged
span is — layered on top of detection.

## 3. What the pilot established

- The white-box substrate carries real, model-agnostic signal: a residual probe reaches token
  AUROC ~0.78 on both Llama-3.2-1B and Gemma-2-2b, beating the flag-all floor in 3/4 languages.
- **SAE features match raw residuals at parity** (0.770 vs 0.773 AUROC) — the interpretability
  win comes at no accuracy cost. This is the core novelty: no SemEval-2025 system used SAEs.
- Span-mapping via verbatim-evidence decomposition anchors ~94% of atomic claims to character
  spans (a reusable technique; weaker on Mandarin at 82%).
- MPS numerical fidelity validated for the backbones used.

Honest negatives, kept as findings: routing by claim type does **not** improve detection (the
probe is uniformly good across routes); the internal-vs-external "gap" idea is **untestable** on
Mu-SHROOM's heterogeneous generations (the external-confidence signal requires the generating
model, which the benchmark does not expose); proxy-model retrieval-CSR collapses to chance.

## 4. Contributions

1. The first **SAE-feature** hallucination detector, evaluated multilingually on Mu-SHROOM.
2. A **feature-level interpretation** of hallucination signal — which sparse features fire on
   hallucinated tokens, and what they represent (via max-activating examples + Gemma Scope
   auto-interpretation).
3. Evidence that SAE features are an interpretable substitute for raw residuals **at parity**.
4. A characterization of the design space: why routing, the internal/external gap, and
   proxy-model retrieval do or don't work on this benchmark.

## 5. Evaluation

- Metric: char-level IoU (primary, Mu-SHROOM) + token-level AUROC (leakage-free discrimination).
- Baselines: flag-all / flag-none floors, raw-residual probe (Deloitte-style), REFIND retrieval.
- Representative (density-controlled) sampling; per-language reporting; per-language threshold
  calibration; multi-seed error bars.
- Comparison framed honestly against UCSC/Deloitte, leading with AUROC and interpretability
  rather than claiming SOTA IoU.

## 6. Timeline (remaining)

| Phase | Work |
|---|---|
| 1a | per-language calibration + representative IoU, multi-seed error bars |
| 1b | feature-interpretation study (the centerpiece) |
| 1c | consolidate ablations (routing, gap), write methods + results |
| 2 (stretch) | competitiveness push: multi-layer features, Gemma-2-9b, span post-processing |
| 3 | writing, figures, submission |

## 7. Risks

- **Absolute IoU below SOTA** (~0.32 vs UCSC ~0.55 in minimal form). Mitigation: lead with
  AUROC + interpretability; Phase 2 tests whether scaling closes the gap.
- **Mandarin** is the hardest case across span-mapping, coverage, and calibration. Treated as a
  documented limitation with a partial per-language fix.
- **SAE availability** ties the detector to models with released SAEs (Gemma). Stated as scope.

## References (to expand)
Mu-SHROOM / SemEval-2025 Task 3; UCSC (Huang et al. 2025); Deloitte (SemEval-2025);
REFIND (Lee & Yu 2025); Gemma Scope (Lieberum et al. 2024); Ferrando et al. (ICLR 2025);
Orgad et al. (ICLR 2025).
