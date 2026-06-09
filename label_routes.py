"""Measure route-classifier quality against your OWN hand labels — the actual
GO/NO-GO #2 metric ("acceptable agreement with manually labeled atoms").

Workflow:
  1) python label_routes.py prepare
       -> writes runs/route_labels.csv (BLIND: no predicted label shown, so you
          aren't anchored) and a hidden key file with the classifier's predictions.
  2) Open runs/route_labels.csv and fill the 'gold' column with E / R / S.
  3) python label_routes.py score
       -> accuracy, Cohen's kappa, per-class precision/recall, confusion matrix.

Sampling is stratified by the classifier's predicted type so the rare classes
(Relational, Subjective) actually appear in what you label.
"""
import csv
import json
import os
import random
import sys
from collections import defaultdict

from config import CFG

ATOMS_PATH = os.path.join(CFG.out_dir, "decompose_route.json")
CSV_PATH = os.path.join(CFG.out_dir, "route_labels.csv")
KEY_PATH = os.path.join(CFG.out_dir, "route_labels_key.json")

CODE = {"E": "External", "R": "Relational", "S": "Subjective"}
LABELS = ["External", "Relational", "Subjective"]
N_SAMPLE = 60


def prepare():
    if not os.path.exists(ATOMS_PATH):
        print(f"Missing {ATOMS_PATH}. Run decompose_route.py first.")
        return
    data = json.load(open(ATOMS_PATH, encoding="utf-8"))

    flat = []
    for item in data:
        for a in item["atoms"]:
            flat.append({"id": item["id"], "lang": item["lang"],
                         "claim": a["claim"], "predicted": a["type"]})

    rng = random.Random(CFG.seed)
    by_type = defaultdict(list)
    for r in flat:
        by_type[r["predicted"]].append(r)

    per = max(1, N_SAMPLE // max(1, len(by_type)))
    sample = []
    for rows in by_type.values():
        rng.shuffle(rows)
        sample.extend(rows[:per])
    rng.shuffle(sample)

    key = {}
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["idx", "lang", "claim", "gold"])
        for i, r in enumerate(sample):
            w.writerow([i, r["lang"], r["claim"], ""])
            key[str(i)] = r["predicted"]
    json.dump(key, open(KEY_PATH, "w"))

    print(f"Wrote {len(sample)} atoms -> {CSV_PATH}")
    print("Fill the 'gold' column with E (External), R (Relational), or S (Subjective).")
    print("Then: python label_routes.py score")


def score():
    if not (os.path.exists(CSV_PATH) and os.path.exists(KEY_PATH)):
        print("Run 'prepare' and fill the labels first.")
        return
    from sklearn.metrics import cohen_kappa_score, classification_report, confusion_matrix

    key = json.load(open(KEY_PATH))
    gold, pred = [], []
    with open(CSV_PATH, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            g = (row.get("gold") or "").strip().upper()[:1]
            if g not in CODE:
                continue
            gold.append(CODE[g])
            pred.append(key[row["idx"]])

    if not gold:
        print(f"No labels found. Fill the 'gold' column in {CSV_PATH}.")
        return

    n = len(gold)
    acc = sum(g == p for g, p in zip(gold, pred)) / n
    kappa = cohen_kappa_score(gold, pred, labels=LABELS)

    print(f"Labeled atoms: {n}")
    print(f"Accuracy (classifier vs your labels): {acc:.2f}")
    print(f"Cohen's kappa: {kappa:.2f}")
    print("\nPer-class (your labels as ground truth):")
    print(classification_report(gold, pred, labels=LABELS, zero_division=0))
    print(f"Confusion matrix (rows = you, cols = classifier): {LABELS}")
    for lab, row in zip(LABELS, confusion_matrix(gold, pred, labels=LABELS)):
        print(f"  {lab:11} {row.tolist()}")
    print("\nKappa guide: <0.4 weak | 0.4-0.6 moderate | 0.6-0.8 substantial | >0.8 strong.")
    print("If Relational precision/recall is poor, the taxonomy boundary needs rework "
          "before TRACE/SAGE relies on that route.")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "prepare"
    {"prepare": prepare, "score": score}.get(cmd, prepare)()
