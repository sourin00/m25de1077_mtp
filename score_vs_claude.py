"""Score the route classifier against Claude's independent reference labels, and show
the contested claims. Reproduces kappa=0.775, accuracy=0.85 on the seed=13 sample.

This is a REFERENCE pass, not ground truth: Claude co-authored the routing rubric, so
agreement here is optimistic. For an unbiased number, label runs/route_labels.csv
yourself (blind) and run `python label_routes.py score`, then diff your calls against
CLAUDE_GOLD below to separate classifier error from genuine taxonomy ambiguity.

Run:  python score_vs_claude.py
"""
import csv
import json
import os
from collections import Counter

from config import CFG

KEY_PATH = os.path.join(CFG.out_dir, "route_labels_key.json")
CSV_PATH = os.path.join(CFG.out_dir, "route_labels.csv")

EXPAND = {"E": "External", "R": "Relational", "S": "Subjective"}
LABELS = ["External", "Relational", "Subjective"]

# Claude's independent labels for the seed=13, N=60 sample (idx -> E/R/S).
CLAUDE_GOLD = {
    0: "E", 1: "S", 2: "S", 3: "E", 4: "S", 5: "R", 6: "E", 7: "E", 8: "S", 9: "E",
    10: "R", 11: "S", 12: "R", 13: "R", 14: "E", 15: "E", 16: "E", 17: "S", 18: "E",
    19: "S", 20: "E", 21: "R", 22: "S", 23: "E", 24: "S", 25: "R", 26: "S", 27: "E",
    28: "E", 29: "E", 30: "S", 31: "R", 32: "S", 33: "R", 34: "S", 35: "R", 36: "S",
    37: "S", 38: "E", 39: "E", 40: "E", 41: "R", 42: "R", 43: "R", 44: "R", 45: "E",
    46: "E", 47: "E", 48: "S", 49: "S", 50: "R", 51: "R", 52: "E", 53: "R", 54: "E",
    55: "R", 56: "E", 57: "E", 58: "S", 59: "R",
}


def main():
    if not os.path.exists(KEY_PATH):
        print(f"Missing {KEY_PATH}. Run decompose_route.py then label_routes.py prepare.")
        return
    key = json.load(open(KEY_PATH))

    claims = {}
    if os.path.exists(CSV_PATH):
        for row in csv.DictReader(open(CSV_PATH, encoding="utf-8")):
            claims[int(row["idx"])] = (row.get("lang", ""), row.get("claim", ""))

    gold, pred = [], []
    for idx, code in CLAUDE_GOLD.items():
        if str(idx) not in key:
            continue
        gold.append(EXPAND[code])
        pred.append(key[str(idx)])

    n = len(gold)
    acc = sum(g == p for g, p in zip(gold, pred)) / n
    pm, pc = Counter(gold), Counter(pred)
    pe = sum((pm[l] / n) * (pc[l] / n) for l in LABELS)
    kappa = (acc - pe) / (1 - pe) if pe < 1 else 0.0

    print(f"n={n}  accuracy={acc:.3f}  Cohen's kappa={kappa:.3f}")
    conf = {a: Counter() for a in LABELS}
    for g, p in zip(gold, pred):
        conf[g][p] += 1
    print("confusion (rows=Claude, cols=classifier):", LABELS)
    for a in LABELS:
        print(f"  {a:11}", [conf[a][b] for b in LABELS])
    for l in LABELS:
        tp = conf[l][l]
        pp = sum(conf[a][l] for a in LABELS)
        ap = sum(conf[l].values())
        prec = tp / pp if pp else 0
        rec = tp / ap if ap else 0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0
        print(f"  {l:11} precision={prec:.2f} recall={rec:.2f} f1={f1:.2f}")

    print("\nContested claims (Claude vs classifier):")
    for idx, code in CLAUDE_GOLD.items():
        c, p = EXPAND[code], key.get(str(idx))
        if p and c != p:
            lang, claim = claims.get(idx, ("", "<run label_routes.py prepare for text>"))
            print(f"  [{idx:>2}|{lang}] Claude={c:11} clf={p:11} {claim[:70]}")


if __name__ == "__main__":
    main()
