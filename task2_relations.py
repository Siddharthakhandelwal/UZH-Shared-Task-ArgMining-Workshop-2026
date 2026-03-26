"""
task2_relations.py
──────────────────
Subtask 2 — Argumentative Relation Prediction

For every paragraph in the test set, predicts which preceding paragraphs
it is argumentatively related to, and labels the link with one or more of:
  supporting  |  contradictive  |  complemental  |  modifying

Reads Task-1 predictions from outputs/task1_predictions.json (run task1 first).
Writes the combined Task-1 + Task-2 submission to outputs/submission_final.json.

Usage
-----
  python task2_relations.py                   # full test set
  python task2_relations.py --dry-run 5       # first 5 documents only
"""

import argparse
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

import config as CFG
from src.data_utils import (
    load_json_docs, load_tags, flatten_paragraphs,
    save_json, load_json, group_by_doc,
)
from src.embeddings import build_para_vecs
from src.llm_utils  import llm_json


# ─────────────────────────────────────────────────────────────────────────────
# Prompt template
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are an expert in argument mining for UN/UNESCO resolutions.

TASK
────
Decide whether PARAGRAPH B has an argumentative relation to PARAGRAPH A,
and if so, identify the relation type(s).

RELATION DEFINITIONS
────────────────────
"supporting"
  B reinforces or provides evidence for the claim or decision made in A.
  B is on the SAME side as A.

"contradictive"
  B directly opposes, negates, or conflicts with A.
  B is on the OPPOSITE side of A.

"complemental"
  B adds NEW information, a new sub-theme, or a new dimension to the SAME
  topic as A — without changing A's claim.

"modifying"
  B revises, qualifies, restricts, or adjusts a specific aspect of A,
  while still operating within the same topic.

RULES
─────
• Multiple relation types are allowed for the same pair.
• Pure formatting, ordering, or sequential flow does NOT constitute a relation.
• If no argumentative link exists, set has_relation to false.

MANDATORY OUTPUT FORMAT
───────────────────────
Return a single JSON object and NOTHING ELSE.

{
  "think": "<multi-step reasoning — REQUIRED:
            Step 1: summarise A's core claim or action in one sentence.
            Step 2: summarise B's core claim or action in one sentence.
            Step 3: compare direction, theme, and intent of A vs B.
            Step 4: assign relation type(s) with justification, or state no relation.>",
  "has_relation"  : true | false,
  "relation_types": ["supporting", "complemental"]
}"""


def build_prompt(para_a: dict, para_b: dict) -> str:
    type_a = para_a.get("type", "unknown")
    type_b = para_b.get("type", "unknown")
    return (
        f"{SYSTEM_PROMPT}\n\n"
        f"PARAGRAPH A  (id: {para_a['para_id']}, type: {type_a})\n"
        f"{para_a['text']}\n\n"
        f"PARAGRAPH B  (id: {para_b['para_id']}, type: {type_b})\n"
        f"{para_b['text']}\n\n"
        "Respond with the JSON object only:"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Pair-level relation predictor
# ─────────────────────────────────────────────────────────────────────────────

def predict_pair(para_a: dict, para_b: dict) -> Optional[dict]:
    """
    Predict the argumentative relation from A to B (directed: B relates to A).
    Returns a relation dict, or None if no relation.
    """
    prompt = build_prompt(para_a, para_b)
    result = llm_json(prompt, required_keys=["has_relation", "relation_types"])

    if not result or not result.get("has_relation"):
        return None

    rel_types = [
        r for r in (result.get("relation_types") or [])
        if r in CFG.RELATION_TYPES
    ]
    if not rel_types:
        return None

    return {
        "target_para_id" : para_b["para_id"],
        "relation_types" : rel_types,
        "_think"         : result.get("think", ""),  # kept internally; stripped at output
    }


# ─────────────────────────────────────────────────────────────────────────────
# Document-level relation predictor
# ─────────────────────────────────────────────────────────────────────────────

def predict_relations_for_doc(
    doc_paras : list[dict],
    t1_lookup : dict[str, dict],
    para_vecs : dict[str, np.ndarray],
) -> dict[str, list[dict]]:
    """
    For each paragraph B (index > 0), compare it against a window of
    preceding paragraphs A using:
      1. Cosine-similarity pre-filter  →  skip clearly unrelated pairs (fast)
      2. LLM call                      →  classify surviving pairs (expensive)

    Returns {para_id_of_A: [relation_dicts_pointing_to_B, …]}
    """
    # Attach predicted type from Task 1
    for p in doc_paras:
        t1 = t1_lookup.get(p["para_id"])
        if t1:
            p["type"] = t1["type"]

    out: dict[str, list] = {p["para_id"]: [] for p in doc_paras}

    for b_idx, para_b in enumerate(doc_paras):
        if b_idx == 0:
            continue

        b_vec = para_vecs.get(para_b["para_id"])
        if b_vec is None:
            continue

        # Backward window
        window = doc_paras[max(0, b_idx - CFG.REL_WINDOW): b_idx]

        # Sort window by cosine similarity (descending) — process best candidates first
        scored = []
        for para_a in window:
            a_vec = para_vecs.get(para_a["para_id"])
            if a_vec is not None:
                scored.append((float(np.dot(a_vec, b_vec)), para_a))
        scored.sort(reverse=True)

        for sim, para_a in scored:
            if sim < CFG.REL_SIM_THRESH:
                break   # sorted list → everything remaining is below threshold too
            rel = predict_pair(para_a, para_b)
            if rel:
                out[para_a["para_id"]].append(rel)

    return out


# ─────────────────────────────────────────────────────────────────────────────
# Build final combined submission
# ─────────────────────────────────────────────────────────────────────────────

def build_final_submission(
    test_docs       : list[dict],
    task1_preds     : list[dict],
    task2_by_doc    : dict[str, dict[str, list]],
) -> list[dict]:
    """
    Merge Task-1 paragraph predictions with Task-2 relation predictions.
    Returns the full submission list (one dict per document).
    """
    # Index Task-1 predictions by doc_id → para_id
    t1_by_doc: dict[str, dict] = defaultdict(dict)
    for p in task1_preds:
        t1_by_doc[p["doc_id"]][p["para_id"]] = p

    submission = []
    for doc in test_docs:
        doc_id    = doc.get("doc_id") or doc.get("id") or doc.get("filename", "unknown")
        t1_doc    = t1_by_doc.get(doc_id, {})
        t2_doc    = task2_by_doc.get(doc_id, {})

        para_list = []
        for para_id, t1_pred in t1_doc.items():
            # Clean relation dicts — drop internal _think key
            rels_raw   = t2_doc.get(para_id, [])
            rels_clean = [
                {
                    "target_para_id": r["target_para_id"],
                    "relation_types": r["relation_types"],
                }
                for r in rels_raw
            ]
            para_list.append({
                "para_id"  : para_id,
                "type"     : t1_pred["type"],
                "tags"     : t1_pred["tags"],
                "think"    : t1_pred["think"],   # required for LLM-Judge
                "relations": rels_clean,
            })

        submission.append({"doc_id": doc_id, "paragraphs": para_list})

    return submission


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Subtask 2 — Relation Prediction")
    parser.add_argument("--dry-run", type=int, default=0, metavar="N",
                        help="Only process the first N test documents (0 = all)")
    args = parser.parse_args()

    print("=" * 60)
    print(" Subtask 2 — Argumentative Relation Prediction")
    print("=" * 60)

    # ── Require Task-1 output ─────────────────────────────────────────────────
    t1_path = CFG.OUTPUT_DIR / "task1_predictions.json"
    if not t1_path.exists():
        print(f"\n❌  Task-1 predictions not found at {t1_path}")
        print("    Run  python task1_classify.py  first.")
        sys.exit(1)

    print("\n[1/5] Loading data …")
    test_docs  = load_json_docs(CFG.TEST_DIR)
    task1_preds = load_json(t1_path)
    print(f"  Task-1 predictions loaded: {len(task1_preds):,} paragraphs")

    test_paras = flatten_paragraphs(test_docs)
    by_doc     = group_by_doc(test_paras)

    if args.dry_run > 0:
        doc_ids = list(by_doc.keys())[: args.dry_run]
        by_doc  = {k: by_doc[k] for k in doc_ids}
        print(f"  [DRY RUN] Processing {args.dry_run} documents")

    # ── Build paragraph vectors (cosine pre-filter) ───────────────────────────
    print("\n[2/5] Building paragraph vectors …")
    para_vecs = build_para_vecs(test_paras)

    # ── Build Task-1 lookup {doc_id → {para_id → pred}} ──────────────────────
    t1_by_doc: dict[str, dict] = defaultdict(dict)
    for p in task1_preds:
        t1_by_doc[p["doc_id"]][p["para_id"]] = p

    # ── Load LLM ──────────────────────────────────────────────────────────────
    print("\n[3/5] Loading LLM …")
    from src.llm_utils import load_llm
    load_llm()

    # ── Predict relations ─────────────────────────────────────────────────────
    print(f"\n[4/5] Predicting relations for {len(by_doc):,} documents …")
    task2_by_doc: dict[str, dict] = {}
    t0 = time.time()

    for doc_id, doc_paras in tqdm(by_doc.items(), desc="Task 2", unit="doc"):
        t1_lookup = t1_by_doc.get(doc_id, {})
        task2_by_doc[doc_id] = predict_relations_for_doc(
            doc_paras, t1_lookup, para_vecs
        )

    elapsed = time.time() - t0
    total_rels = sum(
        len(rels)
        for doc_rels in task2_by_doc.values()
        for rels in doc_rels.values()
    )
    print(f"  Done in {elapsed/60:.1f} min  |  Total relations predicted: {total_rels:,}")

    # ── Assemble & save final submission ──────────────────────────────────────
    print("\n[5/5] Assembling final submission …")
    submission = build_final_submission(test_docs, task1_preds, task2_by_doc)

    save_json(task2_by_doc, CFG.OUTPUT_DIR / "task2_predictions.json")
    save_json(submission,   CFG.OUTPUT_DIR / "submission_final.json")

    print(f"\n✅ Task 2 complete.")
    print(f"   → outputs/task2_predictions.json   (Task-2 raw results)")
    print(f"   → outputs/submission_final.json    (FINAL combined submission)")

    # Relation type breakdown
    from collections import Counter
    rel_counter: Counter = Counter()
    for doc_rels in task2_by_doc.values():
        for rels in doc_rels.values():
            for r in rels:
                for rt in r["relation_types"]:
                    rel_counter[rt] += 1
    print("\n  Relation type breakdown:")
    for rt in CFG.RELATION_TYPES:
        print(f"    {rt:<16} : {rel_counter.get(rt, 0):,}")


if __name__ == "__main__":
    main()
