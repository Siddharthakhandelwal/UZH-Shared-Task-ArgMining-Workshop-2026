"""
merge_outputs.py
────────────────
Combines the outputs from parallel chunk runs into single unified files.
Run this AFTER all chunk notebooks have finished.

Usage
-----
  python merge_outputs.py --total-chunks 3      # merges chunk0, chunk1, chunk2
  python merge_outputs.py --total-chunks 2      # merges chunk0, chunk1

What it does
────────────
  1. Merges task1_predictions_chunk*.json  →  task1_predictions.json
  2. Merges task2_predictions_chunk*.json  →  task2_predictions.json   (if present)
  3. Builds the final submission_final.json from the merged files
  4. Runs schema validation automatically
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import config as CFG
from src.data_utils import load_json_docs, load_json, save_json


def load_chunk(path: Path) -> object:
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def merge_task1(total_chunks: int) -> list[dict]:
    """
    Merge task1_predictions_chunk*.json into a flat list.
    Each chunk file is a list of prediction dicts.
    """
    merged: dict[str, dict] = {}   # para_id → pred (dedup by para_id)

    for c in range(total_chunks):
        path = CFG.OUTPUT_DIR / f"task1_predictions_chunk{c}.json"
        data = load_chunk(path)
        if data is None:
            print(f"  ⚠  {path.name} not found — skipping chunk {c}")
            continue
        for pred in data:
            merged[pred["para_id"]] = pred
        print(f"  chunk {c}: {len(data):,} paragraphs loaded")

    result = list(merged.values())
    print(f"  → {len(result):,} paragraphs total after merge")
    return result


def merge_task2(total_chunks: int) -> dict:
    """
    Merge task2_predictions_chunk*.json.
    Each chunk file is {doc_id: {para_id: [rel_dicts]}}.
    """
    merged: dict = {}

    for c in range(total_chunks):
        path = CFG.OUTPUT_DIR / f"task2_predictions_chunk{c}.json"
        data = load_chunk(path)
        if data is None:
            print(f"  ⚠  {path.name} not found — skipping chunk {c}")
            continue
        merged.update(data)
        print(f"  chunk {c}: {len(data):,} documents loaded")

    print(f"  → {len(merged):,} documents total after merge")
    return merged


def build_final_submission(test_docs, task1_preds, task2_by_doc) -> list[dict]:
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--total-chunks", type=int, required=True,
                        help="Number of chunk files to merge (must match what you ran)")
    args = parser.parse_args()

    print("=" * 60)
    print(f" merge_outputs.py  ({args.total_chunks} chunks)")
    print("=" * 60)

    CFG.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Task 1 merge ──────────────────────────────────────────────────────────
    print(f"\n[1] Merging Task-1 predictions ({args.total_chunks} chunks) …")
    t1_preds = merge_task1(args.total_chunks)
    if not t1_preds:
        print("❌  No Task-1 predictions found. Aborting.")
        sys.exit(1)
    save_json(t1_preds, CFG.OUTPUT_DIR / "task1_predictions.json")

    # ── Task 2 merge (optional) ───────────────────────────────────────────────
    t2_chunks_exist = any(
        (CFG.OUTPUT_DIR / f"task2_predictions_chunk{c}.json").exists()
        for c in range(args.total_chunks)
    )

    task2_by_doc = {}
    if t2_chunks_exist:
        print(f"\n[2] Merging Task-2 predictions ({args.total_chunks} chunks) …")
        task2_by_doc = merge_task2(args.total_chunks)
        save_json(task2_by_doc, CFG.OUTPUT_DIR / "task2_predictions.json")
    else:
        print("\n[2] No Task-2 chunk files found — skipping Task-2 merge.")
        print("    Run task2_relations.py (possibly in chunks) then re-run merge_outputs.py.")

    # ── Final submission ──────────────────────────────────────────────────────
    print("\n[3] Building submission_final.json …")
    test_docs  = load_json_docs(CFG.TEST_DIR)
    submission = build_final_submission(test_docs, t1_preds, task2_by_doc)
    save_json(submission, CFG.OUTPUT_DIR / "submission_final.json")

    n_paras = sum(len(d["paragraphs"]) for d in submission)
    n_rels  = sum(len(p["relations"]) for d in submission for p in d["paragraphs"])
    print(f"\n  Documents  : {len(submission):,}")
    print(f"  Paragraphs : {n_paras:,}")
    print(f"  Relations  : {n_rels:,}")

    # ── Auto-validate ─────────────────────────────────────────────────────────
    print("\n[4] Running schema validation …")
    import subprocess
    result = subprocess.run(
        [sys.executable, "validate_submission.py",
         "--file", str(CFG.OUTPUT_DIR / "submission_final.json")],
        capture_output=False,
    )
    if result.returncode != 0:
        print("\n⚠  Validation reported errors. Check output above.")
    else:
        print("\n🎉 Done! Upload outputs/submission_final.json to the organiser portal.")


if __name__ == "__main__":
    main()