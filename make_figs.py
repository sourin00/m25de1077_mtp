"""Generate the two results figures from the validated numbers.

Outputs (in ./figs):
  fig_detection.pdf/.png  -- per-language detection AUROC, surface vs SAE (full split, 5 seeds)
  fig_within.pdf/.png     -- per-language WITHIN-ITEM AUROC, SAE vs surface (the decider)

Vector PDFs go into the LaTeX paper; PNGs are for quick preview.
Numbers are hard-coded from the runs (no model needed), so this is reproducible anywhere.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.makedirs("figs", exist_ok=True)
LANGS = ["es", "cs", "zh", "en"]

det_surface = {"es": (0.672, 0.014), "cs": (0.631, 0.032), "zh": (0.603, 0.013), "en": (0.655, 0.014)}
det_sae     = {"es": (0.735, 0.017), "cs": (0.759, 0.014), "zh": (0.696, 0.024), "en": (0.757, 0.040)}
wi_surface  = {"es": (0.757, 0.018), "cs": (0.733, 0.027), "zh": (0.696, 0.018), "en": (0.740, 0.024)}
wi_sae      = {"es": (0.843, 0.008), "cs": (0.831, 0.014), "zh": (0.781, 0.017), "en": (0.827, 0.025)}
WI_POOLED = 0.744

C_SURF, C_SAE = "#9aa6b2", "#2f6f8f"
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})


def grouped(ax, surf, sae, title, ytop):
    ax.grid(axis="y", alpha=0.3, linewidth=0.6); ax.set_axisbelow(True)
    x = np.arange(len(LANGS)); w = 0.38
    ax.bar(x - w/2, [surf[l][0] for l in LANGS], w, yerr=[surf[l][1] for l in LANGS],
           capsize=3, color=C_SURF, label="surface", error_kw=dict(lw=0.8))
    ax.bar(x + w/2, [sae[l][0] for l in LANGS], w, yerr=[sae[l][1] for l in LANGS],
           capsize=3, color=C_SAE, label="SAE", error_kw=dict(lw=0.8))
    ax.axhline(0.5, color="#444", lw=0.8, ls=":")
    ax.text(len(LANGS) - 0.55, 0.508, "chance", fontsize=7, color="#444", va="bottom", ha="right")
    ax.set_xticks(x); ax.set_xticklabels([l.upper() for l in LANGS])
    ax.set_ylim(0.5, ytop); ax.set_ylabel("token AUROC"); ax.set_title(title, fontsize=9.5)
    ax.legend(loc="upper center", ncol=2, frameon=False, fontsize=8.5, bbox_to_anchor=(0.5, 0.995))


fig, ax = plt.subplots(figsize=(3.4, 2.5), dpi=200)
grouped(ax, det_surface, det_sae, "Detection AUROC (full split, 5 seeds)", 0.95)
fig.tight_layout(); fig.savefig("figs/fig_detection.pdf"); fig.savefig("figs/fig_detection.png"); plt.close(fig)

fig, ax = plt.subplots(figsize=(3.4, 2.5), dpi=200)
grouped(ax, wi_surface, wi_sae, "Within-item AUROC (topic held constant)", 1.02)
ax.axhline(WI_POOLED, color=C_SAE, lw=0.9, ls="--", alpha=0.8)
ax.set_xlim(-0.5, 4.5)
ax.text(3.62, WI_POOLED, "SAE pooled\n(cross-item) 0.74", fontsize=6.5, color=C_SAE, va="center", ha="left")
fig.tight_layout(); fig.savefig("figs/fig_within.pdf"); fig.savefig("figs/fig_within.png"); plt.close(fig)
print("wrote figs/fig_detection.{pdf,png} and figs/fig_within.{pdf,png}")
