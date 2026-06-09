"""Trivial reference baselines, so the real methods' IoU is interpretable.

On a heavily-hallucinated sample, 'flag the whole answer' scores high on IoU by base rate
alone. Any method (REFIND, the probe, TRACE/SAGE) must be read against these floors:
if it doesn't clear FLAG-ALL, its IoU reflects the base rate, not detection skill.

Run:  python baseline_trivial.py
"""
from data import load_sample
import metrics


def run():
    items = load_sample()
    flag_all = [[[0, len(it.answer or "")]] if (it.answer or "").strip() else [] for it in items]
    flag_none = [[] for _ in items]
    metrics.report("FLAG-ALL  (predict the entire answer hallucinated)", items, flag_all)
    metrics.report("FLAG-NONE (predict nothing hallucinated)", items, flag_none)
    print("\nRead REFIND / probe IoU against FLAG-ALL: clearing it = real detection skill.")


if __name__ == "__main__":
    run()
