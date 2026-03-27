"""
task2_relations.py  (resumable + chunk-aware)
─────────────────────────────────────────────
Checkpoint granularity: one document at a time (documents are small enough
that a single doc never hits the timeout).

MODES
─────
1. Resume (default)
   python task2_relations.py
   Skips documents already in outputs/task2_checkpoint.json.

2. Chunk mode
   python task2_relations.py --chunk 0 --total-chunks 3
   python task2_relations.py --chunk 1 --total-chunks 3
   python task2_relations.py --chunk 2 --total-chunks 3
   Then: python merge_outputs.py

3. Dry run
   python task2_relations.py --dry-run 5
"""

import argparse
import json
import sys
import time
from collections import defaultdict, Counter
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
# Checkpoint helpers — save after every document
# ─────────────────────────────────────────────────────────────────────────────

def ckpt_path(chunk):
    name = "task2_checkpoint.json" if chunk is None else f"task2_checkpoint_chunk{chunk}.json"
    return CFG.OUTPUT_DIR / name


def load_checkpoint(chunk):
    """Return {doc_id: {para_id: [rel_dicts]}} of completed documents."""
    cp = ckpt_path(chunk)
    if cp.exists():
        with open(cp, encoding="utf-8") as f:
            data = json.load(f)
        print(f"  Checkpoint loaded — {len(data):,} documents already done  ({cp.name})")
        return data
    return {}


def save_checkpoint(done, chunk):
    cp = ckpt_path(chunk)
    cp.parent.mkdir(parents=True, exist_ok=True)
    with open(cp, "w", encoding="utf-8") as f:
        json.dump(done, f, ensure_ascii=False)


# ─────────────────────────────────────────────────────────────────────────────
# Prompt
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are an expert in argument mining for UN/UNESCO resolutions.

TASK
────
Decide whether PARAGRAPH B has an argumentative relation to PARAGRAPH A,
and identify the relation type(s).

RELATION DEFINITIONS
────────────────────
"supporting"    — B reinforces or provides evidence for A's claim/decision.
"contradictive" — B directly opposes, negates, or conflicts with A.
"complemental"  — B adds NEW information or a new sub-theme to the SAME topic as A.
"modifying"     — B revises, qualifies, restricts, or adjusts a specific aspect of A.

RULES
─────
• Multiple relation types are allowed for the same pair.
• Pure formatting or sequential flow is NOT a relation.
• If no argumentative link exists → has_relation: false.

MANDATORY OUTPUT FORMAT — single JSON object, NOTHING ELSE:
{
  "think": "<Step 1: A's core claim. Step 2: B's core claim.
            Step 3: compare direction/theme/intent. Step 4: relation type(s) or none.>",
  "has_relation"  : true | false,
  "relation_types": ["supporting", "complemental"]
}"""


def build_prompt(para_a, para_b):
    return (
        f"{SYSTEM_PROMPT}\n\n"
        f"PARAGRAPH A  (id: {para_a['para_id']}, type: {para_a.get('type','?')})\n"
        f"{para_a['text']}\n\n"
        f"PARAGRAPH B  (id: {para_b['para_id']}, type: {para_b.get('type','?')})\n"
        f"{para_b['text']}\n\n"
        "Respond with JSON only:"
    )


def predict_pair(para_a, para_b) -> Optional[dict]:
    result = llm_json(
        build_prompt(para_a, para_b),
        required_keys=["has_relation", "relation_types"],
    )
    if not result or not result.get("has_relation"):
        return None
    rel_types = [r for r in (result.get("relation_types") or []) if r in CFG.RELATION_TYPES]
    if not rel_types:
        return None
    return {
        "target_para_id" : para_b["para_id"],
        "relation_types" : rel_types,
        "_think"         : result.get("think", ""),
    }


def predict_relations_for_doc(doc_paras, t1_lookup, para_vecs):
    """
    Sliding-window relation prediction for one document.
    Returns {para_id_of_A: [relation_dicts]}
    """
    for p in doc_paras:
        t1 = t1_lookup.get(p["para_id"])
        if t1:
            p["type"] = t1["type"]

    out = {p["para_id"]: [] for p in doc_paras}

    for b_idx, para_b in enumerate(doc_paras):
        if b_idx == 0:
            continue
        b_vec = para_vecs.get(para_b["para_id"])
        if b_vec is None:
            continue

        window = doc_paras[max(0, b_idx - CFG.REL_WINDOW): b_idx]

        scored = []
        for para_a in window:
            a_vec = para_vecs.get(para_a["para_id"])
            if a_vec is not None:
                scored.append((float(np.dot(a_vec, b_vec)), para_a))
        scored.sort(reverse=True)

        for sim, para_a in scored:
            if sim < CFG.REL_SIM_THRESH:
                break
            rel = predict_pair(para_a, para_b)
            if rel:
                out[para_a["para_id"]].append(rel)

    return out


# ─────────────────────────────────────────────────────────────────────────────
# Assemble final submission
# ─────────────────────────────────────────────────────────────────────────────

def build_final_submission(test_docs, task1_preds, task2_by_doc):
    t1_by_doc = defaultdict(dict)
    for p in task1_preds:
        t1_by_doc[p["doc_id"]][p["para_id"]] = p

    submission = []
    for doc in test_docs:
        doc_id = doc.get("doc_id") or doc.get("id") or doc.get("filename", "unknown")
        t1_doc = t1_by_doc.get(doc_id, {})
        t2_doc = task2_by_doc.get(doc_id, {})

        para_list = []
        for para_id, t1_pred in t1_doc.items():
            rels_clean = [
                {"target_para_id": r["target_para_id"], "relation_types": r["relation_types"]}
                for r in t2_doc.get(para_id, [])
            ]
            para_list.append({
                "para_id"  : para_id,
                "type"     : t1_pred["type"],
                "tags"     : t1_pred["tags"],
                "think"    : t1_pred["think"],
                "relations": rels_clean,
            })
        submission.append({"doc_id": doc_id, "paragraphs": para_list})

    return submission


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run",      type=int, default=0,    metavar="N",
                        help="Process only first N documents")
    parser.add_argument("--chunk",        type=int, default=None, metavar="IDX")
    parser.add_argument("--total-chunks", type=int, default=1,    metavar="N")
    args = parser.parse_args()

    print("=" * 60)
    print(" Subtask 2 — Relation Prediction  (resumable)")
    if args.chunk is not None:
        print(f" Chunk {args.chunk} of {args.total_chunks}")
    print("=" * 60)

    # ── Require Task-1 output ─────────────────────────────────────────────────
    # Accept merged or single-run task1 output
    t1_path = CFG.OUTPUT_DIR / "task1_predictions.json"
    if not t1_path.exists():
        print(f"\n❌  {t1_path} not found.")
        print("    Run task1_classify.py (and merge_outputs.py if using chunks) first.")
        sys.exit(1)

    print("\n[1/5] Loading data …")
    test_docs   = load_json_docs(CFG.TEST_DIR)
    task1_preds = load_json(t1_path)
    test_paras  = flatten_paragraphs(test_docs)
    by_doc      = group_by_doc(test_paras)

    print(f"  Task-1 predictions : {len(task1_preds):,} paragraphs")
    print(f"  Test documents     : {len(by_doc):,}")

    # ── Slice by dry-run / chunk ─────────────────────────────────────────────
    doc_ids = list(by_doc.keys())

    if args.dry_run > 0:
        doc_ids = doc_ids[: args.dry_run]
        print(f"  [DRY RUN] {len(doc_ids)} documents")
    elif args.chunk is not None:
        doc_ids = [d for i, d in enumerate(doc_ids) if i % args.total_chunks == args.chunk]
        print(f"  Chunk {args.chunk}: {len(doc_ids):,} documents assigned")

    # ── Checkpoint ───────────────────────────────────────────────────────────
    done      = load_checkpoint(args.chunk)       # {doc_id: {para_id: [rels]}}
    remaining = [d for d in doc_ids if d not in done]
    print(f"  Remaining: {len(remaining):,}  |  Already done: {len(done):,}")

    if not remaining:
        print("  Nothing left — all documents already processed ✅")
    else:
        # ── Para vectors ──────────────────────────────────────────────────────
        print("\n[2/5] Building paragraph vectors …")
        para_vecs = build_para_vecs(test_paras)

        # ── Task-1 lookup ─────────────────────────────────────────────────────
        t1_by_doc = defaultdict(dict)
        for p in task1_preds:
            t1_by_doc[p["doc_id"]][p["para_id"]] = p

        # ── LLM ───────────────────────────────────────────────────────────────
        print("\n[3/5] Loading LLM …")
        from src.llm_utils import load_llm
        load_llm()

        # ── Predict — checkpoint after every document ─────────────────────────
        print(f"\n[4/5] Processing {len(remaining):,} documents …")
        t0 = time.time()

        for doc_id in tqdm(remaining, desc="Task 2", unit="doc"):
            doc_paras = by_doc.get(doc_id, [])
            t1_lookup = t1_by_doc.get(doc_id, {})
            done[doc_id] = predict_relations_for_doc(doc_paras, t1_lookup, para_vecs)
            save_checkpoint(done, args.chunk)      # ← flush after every document

        print(f"  Finished in {(time.time()-t0)/60:.1f} min")

    # ── Write outputs ─────────────────────────────────────────────────────────
    print("\n[5/5] Saving outputs …")
    CFG.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    suffix = f"_chunk{args.chunk}" if args.chunk is not None else ""
    save_json(done, CFG.OUTPUT_DIR / f"task2_predictions{suffix}.json")

    if args.chunk is None:
        # Non-chunked run → build final combined submission
        submission = build_final_submission(test_docs, task1_preds, done)
        save_json(submission, CFG.OUTPUT_DIR / "submission_final.json")

    total_rels = sum(len(rels) for doc in done.values() for rels in doc.values())
    print(f"\n  Documents processed : {len(done):,}")
    print(f"  Total relations     : {total_rels:,}")

    rel_counter: Counter = Counter()
    for doc_rels in done.values():
        for rels in doc_rels.values():
            for r in rels:
                for rt in r.get("relation_types", []):
                    rel_counter[rt] += 1
    for rt in CFG.RELATION_TYPES:
        print(f"    {rt:<16} : {rel_counter.get(rt, 0):,}")

    if args.chunk is not None:
        print(f"\n  ⚡ Chunk {args.chunk} done.")
        print("  When ALL chunks finish → python merge_outputs.py")
    else:
        print("\n✅ Task 2 complete → run python validate_submission.py")


if __name__ == "__main__":
    main()