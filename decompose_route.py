"""TRACE front-end: decompose each answer into atomic claims, then route each claim
by epistemic type (External / Relational / Subjective). Runs entirely on the local model.

Pilot v2 changes (from the first run's error analysis):
  - Decomposition is now FAITHFULNESS-constrained: it may use the question only to
    resolve ellipsis/references, never to import facts the answer doesn't assert
    (fixes the "Ouagadougou -> capital of Burkina Faso" fabrication), and every atom
    must keep its full predicate (fixes the "was a strong proponent" stub).
  - Routing uses ORDERED, mutually-exclusive rules + worked examples so structurally
    identical claims ("founded by", "developed by") route consistently.

Run:  python decompose_route.py     (with `ollama serve` running)
"""
import json
import os
from collections import Counter

from config import CFG
from llm_client import chat_json
from data import load_sample

DECOMPOSE_SYS = (
    "You decompose an ANSWER into atomic claims for fact-checking. Follow every rule:\n"
    "1. FAITHFULNESS: each claim must be asserted BY the answer. Never add facts, "
    "context, or outside knowledge. You may use the QUESTION only to resolve what the "
    "answer refers to (ellipsis, pronouns) — not to introduce information the answer "
    "does not state. If the answer is a bare name or fragment, claim only what it, "
    "combined with the question's framing, literally asserts.\n"
    "2. COMPLETE & ATOMIC: one fact per claim, but keep each claim whole — "
    "subject + predicate + object. Never drop the object "
    "(write 'X was a proponent of evolution', never 'X was a proponent').\n"
    "3. DECONTEXTUALIZE: resolve pronouns/references so each claim stands alone.\n"
    "4. Write claims in the answer's ORIGINAL language.\n"
    "Return ONLY a JSON array of strings. No commentary, no code fences."
)

ROUTE_SYS = (
    "Classify one atomic claim into exactly ONE type. Apply the tests IN ORDER and "
    "assign the FIRST that matches:\n\n"
    "1. Subjective — an opinion, evaluation, aesthetic/importance/popularity judgment, "
    "prediction, or a hedged claim ('probably', 'is considered', 'one of the most ...').\n"
    "2. Relational — asserts a link BETWEEN TWO OR MORE SPECIFIC NAMED ENTITIES "
    "(person / work / organization / place / event): who made, founded, directed, leads, "
    "belongs to, is located in, or is otherwise bound to what. The characteristic error "
    "is binding the wrong entity.\n"
    "3. External — anything else: a standalone property, quantity, date, measurement, "
    "definition, or category of a SINGLE subject, verifiable by looking up one fact.\n\n"
    "TIE-BREAKER for the External/Relational seam: a person's occupation, title, role, or "
    "activity, and an object's material composition, are EXTERNAL — UNLESS the claim names "
    "a SPECIFIC second entity (a particular place, organization, or work), which makes it "
    "Relational. So 'minister of the army' = role = External; 'ambassador to the United "
    "States' names a place = Relational. 'plays guitar' = External; 'member of the band "
    "Kabat' names a group = Relational.\n\n"
    "Worked examples:\n"
    "'Tabor was founded by the Hussites.' -> Relational (town<->group)\n"
    "'Tabor was founded in 1420.' -> External (subject + date)\n"
    "'F.E.A.R. was developed by Monolith Productions.' -> Relational (work<->org)\n"
    "'The boiling point of DOT 4 brake fluid is 312 C.' -> External (subject + quantity)\n"
    "'Stahlberg is located in Wismar.' -> Relational (place<->place)\n"
    "'Karen Percy is a Canadian athlete.' -> External (single subject + category)\n"
    "'X served as Minister of the Army.' -> External (role, no specific second entity)\n"
    "'X was ambassador to the United States.' -> Relational (names a place)\n"
    "'X plays the guitar.' -> External (instrument is a generic category)\n"
    "'X is a member of the band Kabat.' -> Relational (names a specific group)\n"
    "'The material is made of carbon fiber and Kevlar.' -> External (composition = property)\n"
    "'David Cerny is one of the most respected sculptors.' -> Subjective (evaluation)\n"
    "'It is probable the author was on the editorial staff.' -> Subjective (hedged)\n\n"
    'Return ONLY JSON: {"type": "...", "reason": "<=12 words"}. No commentary.'
)

VALID_TYPES = ("External", "Relational", "Subjective")


def decompose(question, answer, lang):
    if not (answer or "").strip():
        return []
    user = (
        f"Language: {lang}\n"
        f"Question (for reference resolution only):\n{question}\n\n"
        f"Answer to decompose:\n{answer}\n\n"
        "Return the JSON array of atomic claims."
    )
    try:
        atoms = chat_json(DECOMPOSE_SYS, user)
        return [a for a in atoms if isinstance(a, str) and a.strip()]
    except Exception as e:  # noqa: BLE001
        print(f"    [decompose] failed: {e}")
        return []


def route(claim):
    try:
        r = chat_json(ROUTE_SYS, f"Claim: {claim}\n\nClassify it.")
        t = r.get("type", "")
        return t if t in VALID_TYPES else "External"   # default unknowns to External
    except Exception as e:  # noqa: BLE001
        print(f"    [route] failed: {e}")
        return "External"


def run():
    os.makedirs(CFG.out_dir, exist_ok=True)
    items = load_sample()

    results = []
    route_counts = Counter()
    for it in items:
        atoms = decompose(it.question, it.answer, it.lang)
        routed = [{"claim": a, "type": route(a)} for a in atoms]
        for r in routed:
            route_counts[r["type"]] += 1
        results.append({
            "id": it.id, "lang": it.lang, "question": it.question, "answer": it.answer,
            "n_atoms": len(atoms), "atoms": routed,
        })
        print(f"[{it.lang}] {it.id}: {len(atoms)} atoms -> {dict(Counter(r['type'] for r in routed))}")

    out_path = os.path.join(CFG.out_dir, "decompose_route.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # ---- pilot signal ----
    print("\n" + "=" * 52)
    print("PILOT SIGNAL")
    print("=" * 52)
    by_lang = {}
    for r in results:
        by_lang.setdefault(r["lang"], []).append(r["n_atoms"])
    for lang, counts in by_lang.items():
        nonzero = sum(1 for c in counts if c > 0)
        avg = sum(counts) / len(counts) if counts else 0.0
        print(f"  {lang}: {nonzero}/{len(counts)} answers decomposed | avg {avg:.1f} atoms/answer")
    print(f"  route distribution: {dict(route_counts)}")
    print(f"\nSaved -> {out_path}")
    print("Next: python label_routes.py prepare  (then hand-label, then 'score')")


if __name__ == "__main__":
    run()