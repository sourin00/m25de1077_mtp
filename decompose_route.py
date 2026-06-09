"""TRACE front-end: decompose each answer into atomic claims, route each by epistemic type
(External / Relational / Subjective), and anchor each claim to the character span of the
answer it came from (span mapping).

Decomposition output is line-delimited (CLAIM:/EVIDENCE:), not JSON: the evidence is a
verbatim copy of arbitrary answer text (quotes, backslashes, unicode, CJK), which is exactly
what breaks JSON string escaping. A line-labeled format has nothing to escape. Routing stays
JSON (a tiny {type, reason} that never carries raw text).

Run:  python decompose_route.py     (with `ollama serve` running)
"""
import difflib
import json
import os
import re
from collections import Counter

from config import CFG
from llm_client import chat_json, chat_text
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
    "4. Write the claim in the answer's ORIGINAL language.\n"
    "5. EVIDENCE: copy the exact, contiguous substring of the ANSWER the claim is based "
    "on — character-for-character VERBATIM, no paraphrase, no translation, on ONE line.\n\n"
    "OUTPUT FORMAT — for each claim, exactly two lines:\n"
    "CLAIM: <the atomic claim>\n"
    "EVIDENCE: <verbatim answer substring>\n"
    "Put a blank line between claims. Output nothing else: no JSON, no numbering, no "
    "commentary, no code fences."
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

_CLAIM_RE = re.compile(r"^\s*(?:[-*\d.)\s]*)?claim\s*:\s*(.*)$", re.I)
_EVID_RE = re.compile(r"^\s*evidence\s*:\s*(.*)$", re.I)


def _parse_atoms(text):
    atoms, cur = [], None
    for line in (text or "").splitlines():
        mc = _CLAIM_RE.match(line)
        if mc and not _EVID_RE.match(line):
            if cur and cur["claim"]:
                atoms.append(cur)
            cur = {"claim": mc.group(1).strip(), "evidence": ""}
        elif cur is not None:
            me = _EVID_RE.match(line)
            if me:
                cur["evidence"] = me.group(1).strip()
    if cur and cur["claim"]:
        atoms.append(cur)
    return [a for a in atoms if a["claim"]]


def decompose(question, answer, lang):
    if not (answer or "").strip():
        return []
    user = (
        f"Language: {lang}\n"
        f"Question (for reference resolution only):\n{question}\n\n"
        f"Answer to decompose:\n{answer}\n\n"
        "Return the CLAIM/EVIDENCE lines."
    )
    try:
        return _parse_atoms(chat_text(DECOMPOSE_SYS, user))
    except Exception as e:  # noqa: BLE001
        print(f"    [decompose] failed: {e}")
        return []


def route(claim):
    try:
        r = chat_json(ROUTE_SYS, f"Claim: {claim}\n\nClassify it.")
        t = r.get("type", "")
        return t if t in VALID_TYPES else "External"
    except Exception as e:  # noqa: BLE001
        print(f"    [route] failed: {e}")
        return "External"


def map_span(answer, evidence):
    """Anchor an evidence substring to [start, end] char offsets in answer. Returns (span, how)."""
    ev = (evidence or "").strip()
    if not ev:
        return None, "empty"
    i = answer.find(ev)
    if i >= 0:
        return [i, i + len(ev)], "exact"
    sm = difflib.SequenceMatcher(None, answer, ev, autojunk=False)
    covered = [b for b in sm.get_matching_blocks() if b.size > 0]
    total = sum(b.size for b in covered)
    if covered and total >= max(4, int(0.6 * len(ev))):
        return [covered[0].a, covered[-1].a + covered[-1].size], "fuzzy"
    return None, "miss"


def run():
    os.makedirs(CFG.out_dir, exist_ok=True)
    items = load_sample()

    results = []
    route_counts = Counter()
    map_counts = Counter()
    by_lang_map = {}
    for it in items:
        atoms = decompose(it.question, it.answer, it.lang)
        routed = []
        for a in atoms:
            span, how = map_span(it.answer or "", a["evidence"])
            routed.append({"claim": a["claim"], "evidence": a["evidence"],
                           "type": route(a["claim"]), "span": span, "map": how})
            route_counts[routed[-1]["type"]] += 1
            map_counts[how] += 1
            by_lang_map.setdefault(it.lang, Counter())[how] += 1
        results.append({
            "id": it.id, "lang": it.lang, "question": it.question, "answer": it.answer,
            "n_atoms": len(atoms), "atoms": routed,
        })
        nmapped = sum(1 for r in routed if r["span"] is not None)
        print(f"[{it.lang}] {it.id}: {len(atoms)} atoms, {nmapped} anchored "
              f"-> {dict(Counter(r['type'] for r in routed))}")

    out_path = os.path.join(CFG.out_dir, "decompose_route.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

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

    print("\n  SPAN-MAPPING COVERAGE:")
    total = sum(map_counts.values()) or 1
    anchored = map_counts['exact'] + map_counts['fuzzy']
    print(f"  overall: {anchored}/{total} anchored ({100*anchored/total:.0f}%)  "
          f"[exact={map_counts['exact']} fuzzy={map_counts['fuzzy']} "
          f"miss={map_counts['miss']} empty={map_counts['empty']}]")
    for lang, c in by_lang_map.items():
        t = sum(c.values()) or 1
        anc = c['exact'] + c['fuzzy']
        print(f"    {lang}: {100*anc/t:.0f}% anchored  (exact={c['exact']} fuzzy={c['fuzzy']} "
              f"miss={c['miss']})")
    print(f"\nSaved -> {out_path}")
    print("Next: python trace.py   (route-conditioned verification vs the flat probe)")


if __name__ == "__main__":
    run()