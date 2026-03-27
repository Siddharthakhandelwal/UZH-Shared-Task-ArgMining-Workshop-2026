"""
task1_classify.py  (resumable + chunk-aware)
────────────────────────────────────────────
Saves progress after EVERY paragraph → safe to kill/resume at any time.

MODES
─────
1. Resume (default) — just run again after a timeout, it picks up where it stopped
   python task1_classify.py

2. Chunk mode — run 3 Kaggle notebooks in parallel, each gets 1/3 of paragraphs
   python task1_classify.py --chunk 0 --total-chunks 3   # notebook A
   python task1_classify.py --chunk 1 --total-chunks 3   # notebook B
   python task1_classify.py --chunk 2 --total-chunks 3   # notebook C
   Then: python merge_outputs.py

3. Dry run — smoke test on N paragraphs
   python task1_classify.py --dry-run 10
"""

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

import config as CFG
from src.data_utils import load_json_docs, load_tags, flatten_paragraphs, save_json
from src.embeddings import (
    build_rag_index, build_tag_index,
    retrieve_few_shots, retrieve_tag_candidates,
)
from src.llm_utils import llm_json


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint helpers — save after EVERY paragraph
# ─────────────────────────────────────────────────────────────────────────────

def ckpt_path(chunk):
    name = "task1_checkpoint.json" if chunk is None else f"task1_checkpoint_chunk{chunk}.json"
    return CFG.OUTPUT_DIR / name


def load_checkpoint(chunk):
    """Return {para_id: pred_dict} of already-finished paragraphs."""
    cp = ckpt_path(chunk)
    if cp.exists():
        with open(cp, encoding="utf-8") as f:
            data = json.load(f)
        print(f"  Checkpoint loaded — {len(data):,} paragraphs already done  ({cp.name})")
        return data
    return {}


def save_checkpoint(done, chunk):
    """Write the full done-dict to disk (compact JSON for speed)."""
    cp = ckpt_path(chunk)
    cp.parent.mkdir(parents=True, exist_ok=True)
    with open(cp, "w", encoding="utf-8") as f:
        json.dump(done, f, ensure_ascii=False)   # no indent — faster I/O


# ─────────────────────────────────────────────────────────────────────────────
# Prompt
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are an expert analyst of UN/UNESCO legal-political resolutions.

TASK
────
Classify the paragraph below into ONE type and assign relevant thematic tags.

TYPE DEFINITIONS
────────────────
"preambular"
  Sets context, cites previous decisions, states reasons and justifications.
  Opening words: Recalling, Noting, Recognizing, Bearing in mind, Considering,
  Convinced, Aware, Reaffirming, Acknowledging, Emphasizing, Having regard to,
  Conscious, Mindful ...

"operative"
  Takes concrete action: makes decisions, issues requests, gives mandates.
  Opening words: Decides, Requests, Urges, Recommends, Encourages, Calls upon,
  Invites, Stresses, Affirms, Declares, Authorizes, Welcomes, Endorses,
  Commends, Supports, Expresses ...

TAG RULES
─────────
• Choose ONLY from the candidate list provided.
• Include a tag only if the paragraph CLEARLY and DIRECTLY addresses that theme.
• A paragraph may have 0, 1, or several tags.

MANDATORY OUTPUT FORMAT — single JSON object, NOTHING ELSE:
{
  "think": "<Step 1: identify opening keyword. Step 2: decide type + justify.
            Step 3: evaluate each candidate tag. Step 4: list final tags + why.>",
  "type": "preambular" | "operative",
  "tags": ["tag_name_1", "tag_name_2"]
}"""


def build_prompt(para_text, tag_candidates, few_shots, tag_desc):
    shots = ""
    for i, ex in enumerate(few_shots, 1):
        shots += (
            f"\n=== Example {i} ===\n"
            f"Paragraph : {ex['text'][:500]}\n"
            f"Type      : {ex['type']}\n"
            f"Tags      : {ex.get('tags') or []}\n"
        )
    tags_block = "\n".join(
        f"  • {t} — {tag_desc.get(t, '').strip()}" for t in tag_candidates
    )
    return (
        f"{SYSTEM_PROMPT}\n\n"
        f"## Annotated Examples\n{shots}\n"
        f"## Tag Candidates ({len(tag_candidates)} options — choose only from these)\n"
        f"{tags_block}\n\n"
        f"## Paragraph to classify\n{para_text}\n\n"
        "Respond with JSON only:"
    )


OPERATIVE_WORDS = {
    "decides","requests","urges","recommends","encourages","calls",
    "invites","stresses","affirms","declares","authorizes","endorses",
    "commends","supports","expresses","notes",
}

def heuristic_type(text):
    first = text.strip().lower().split()[0].rstrip(",.")
    return "operative" if first in OPERATIVE_WORDS else "preambular"


def classify_paragraph(para, rag_index, train_paras, tag_index, all_tags, tag_desc):
    text      = para["text"]
    tag_cands = retrieve_tag_candidates(text, tag_index, all_tags)
    few_shots = retrieve_few_shots(text, rag_index, train_paras)
    prompt    = build_prompt(text, tag_cands, few_shots, tag_desc)

    result = llm_json(prompt, required_keys=["type", "tags", "think"])

    if result is None:
        result = {
            "type"  : heuristic_type(text),
            "tags"  : [],
            "think" : "LLM failed after all retries. Heuristic fallback applied.",
        }

    if result.get("type") not in ("preambular", "operative"):
        result["type"] = heuristic_type(text)

    all_tags_set = set(all_tags)
    valid_tags   = [t for t in (result.get("tags") or []) if t in all_tags_set]

    return {
        "para_id" : para["para_id"],
        "doc_id"  : para["doc_id"],
        "type"    : result["type"],
        "tags"    : valid_tags,
        "think"   : result.get("think", ""),
    }


def build_submission(test_docs, predictions):
    pred_by_doc = defaultdict(list)
    for p in predictions:
        pred_by_doc[p["doc_id"]].append(p)

    submission = []
    for doc in test_docs:
        doc_id = doc.get("doc_id") or doc.get("id") or doc.get("filename", "unknown")
        para_list = [
            {
                "para_id"  : pred["para_id"],
                "type"     : pred["type"],
                "tags"     : pred["tags"],
                "think"    : pred["think"],
                "relations": [],
            }
            for pred in pred_by_doc.get(doc_id, [])
        ]
        submission.append({"doc_id": doc_id, "paragraphs": para_list})
    return submission


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run",      type=int, default=0,    metavar="N")
    parser.add_argument("--chunk",        type=int, default=None, metavar="IDX",
                        help="0-based chunk index for parallel runs")
    parser.add_argument("--total-chunks", type=int, default=1,    metavar="N")
    args = parser.parse_args()

    print("=" * 60)
    print(" Subtask 1 — Paragraph Classification  (resumable)")
    if args.chunk is not None:
        print(f" Chunk {args.chunk} of {args.total_chunks}  "
              f"(notebooks run in parallel, merge_outputs.py combines them)")
    print("=" * 60)

    # ── Load ──────────────────────────────────────────────────────────────────
    print("\n[1/5] Loading data …")
    train_docs         = load_json_docs(CFG.TRAIN_DIR)
    test_docs          = load_json_docs(CFG.TEST_DIR)
    all_tags, tag_desc = load_tags(CFG.TAGS_CSV)
    train_paras        = flatten_paragraphs(train_docs)
    test_paras         = flatten_paragraphs(test_docs)
    print(f"  Total test paragraphs: {len(test_paras):,}")

    # ── Slice ─────────────────────────────────────────────────────────────────
    if args.dry_run > 0:
        test_paras = test_paras[: args.dry_run]
        print(f"  [DRY RUN] capped at {len(test_paras)} paragraphs")
    elif args.chunk is not None:
        # Round-robin split — paragraph i goes to chunk (i % total_chunks)
        test_paras = [p for i, p in enumerate(test_paras)
                      if i % args.total_chunks == args.chunk]
        print(f"  Chunk {args.chunk}: {len(test_paras):,} paragraphs assigned")

    # ── Checkpoint — resume from last save ───────────────────────────────────
    done      = load_checkpoint(args.chunk)
    remaining = [p for p in test_paras if p["para_id"] not in done]
    print(f"  Remaining: {len(remaining):,}  |  Already done: {len(done):,}")

    if not remaining:
        print("  Nothing left — all paragraphs already classified ✅")
    else:
        # ── Indices ───────────────────────────────────────────────────────────
        print("\n[2/5] Building / loading FAISS indices …")
        rag_index, train_paras = build_rag_index(train_paras)
        tag_index              = build_tag_index(all_tags, tag_desc)

        # ── LLM ───────────────────────────────────────────────────────────────
        print("\n[3/5] Loading LLM …")
        from src.llm_utils import load_llm
        load_llm()

        # ── Classify — checkpoint after every para ────────────────────────────
        print(f"\n[4/5] Classifying {len(remaining):,} paragraphs …")
        t0 = time.time()

        for para in tqdm(remaining, desc="Task 1", unit="para"):
            pred = classify_paragraph(
                para, rag_index, train_paras, tag_index, all_tags, tag_desc
            )
            done[para["para_id"]] = pred
            save_checkpoint(done, args.chunk)      # ← flush to disk every step

        print(f"  Finished in {(time.time()-t0)/60:.1f} min")

    # ── Write outputs ─────────────────────────────────────────────────────────
    print("\n[5/5] Saving outputs …")
    CFG.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    predictions = list(done.values())

    suffix = f"_chunk{args.chunk}" if args.chunk is not None else ""
    save_json(predictions, CFG.OUTPUT_DIR / f"task1_predictions{suffix}.json")

    if args.chunk is None:
        # Full (non-chunked) run → also write the task2-ready submission
        submission = build_submission(test_docs, predictions)
        save_json(submission, CFG.OUTPUT_DIR / "submission_task1.json")

    types = [p["type"] for p in predictions]
    print(f"\n  preambular : {types.count('preambular'):,}")
    print(f"  operative  : {types.count('operative'):,}")
    print(f"  with tags  : {sum(1 for p in predictions if p['tags']):,} / {len(predictions):,}")

    if args.chunk is not None:
        print(f"\n  ⚡ Chunk {args.chunk} done.")
        print("  When ALL chunks are done → python merge_outputs.py")
    else:
        print("\n✅ Task 1 complete → run python task2_relations.py")


if __name__ == "__main__":
    main()