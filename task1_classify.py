"""
task1_classify.py
─────────────────
Subtask 1 — Argumentative Paragraph Classification

For every paragraph in the test set, predicts:
  • type   : "preambular" | "operative"
  • tags   : list of tags from the 141-tag taxonomy
  • think  : step-by-step chain-of-thought (required for the LLM-Judge score)

Output
------
  outputs/task1_predictions.json   — used directly by task2_relations.py
  outputs/submission_task1.json    — standalone submission if running Task 1 only

Usage
-----
  python task1_classify.py                   # runs on full test set
  python task1_classify.py --dry-run 10      # smoke test on first 10 paragraphs
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from tqdm import tqdm

# Make src/ importable regardless of working directory
sys.path.insert(0, str(Path(__file__).parent))

import config as CFG
from src.data_utils  import load_json_docs, load_tags, flatten_paragraphs, save_json
from src.embeddings  import (
    build_rag_index, build_tag_index,
    retrieve_few_shots, retrieve_tag_candidates,
)
from src.llm_utils   import llm_json


# ─────────────────────────────────────────────────────────────────────────────
# Prompt templates
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
  Conscious, Mindful, Welcoming (when used as a preamble verb) …

"operative"
  Takes concrete action: makes decisions, issues requests, gives mandates.
  Opening words: Decides, Requests, Urges, Recommends, Encourages, Calls upon,
  Invites, Stresses, Affirms, Declares, Authorizes, Welcomes (as an action),
  Endorses, Commends, Supports, Expresses, Notes (as an action) …

TAG RULES
─────────
• Choose ONLY from the candidate list provided.
• Include a tag only if the paragraph CLEARLY and DIRECTLY addresses that theme.
• A paragraph may have 0, 1, or several tags — do not over-tag.

MANDATORY OUTPUT FORMAT
───────────────────────
Return a single JSON object and NOTHING ELSE — no preamble, no prose.

{
  "think": "<multi-step reasoning — REQUIRED:
            Step 1: identify the opening keyword/phrase and sentence structure.
            Step 2: decide preambular vs operative and justify.
            Step 3: evaluate each candidate tag against the paragraph content.
            Step 4: state the final tag list and why those tags were chosen.>",
  "type": "preambular" | "operative",
  "tags": ["tag_name_1", "tag_name_2"]
}"""


def build_prompt(para_text: str, tag_candidates: list[str],
                 few_shots: list[dict], tag_desc: dict[str, str]) -> str:
    """Assemble the full prompt for a single paragraph."""

    # Few-shot block
    shots_block = ""
    for i, ex in enumerate(few_shots, 1):
        ex_tags = ex.get("tags") or []
        shots_block += (
            f"\n=== Example {i} ===\n"
            f"Paragraph : {ex['text'][:500]}\n"
            f"Type      : {ex['type']}\n"
            f"Tags      : {ex_tags}\n"
        )

    # Candidate tag block
    tag_block = "\n".join(
        f"  • {t} — {tag_desc.get(t, '').strip()}"
        for t in tag_candidates
    )

    return (
        f"{SYSTEM_PROMPT}\n\n"
        f"## Annotated Examples (for reference)\n{shots_block}\n"
        f"## Tag Candidates — choose ONLY from these {len(tag_candidates)} tags\n"
        f"{tag_block}\n\n"
        f"## Paragraph to classify\n{para_text}\n\n"
        "Respond with the JSON object only:"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Heuristic fallback (zero LLM cost)
# ─────────────────────────────────────────────────────────────────────────────

PREAMBULAR_WORDS = {
    "recalling","noting","recognizing","bearing","considering","convinced",
    "aware","reaffirming","acknowledging","emphasizing","having","whereas",
    "conscious","mindful","affirming","concerned","stressing","welcoming",
}
OPERATIVE_WORDS = {
    "decides","requests","urges","recommends","encourages","calls",
    "invites","stresses","affirms","declares","authorizes","endorses",
    "commends","supports","expresses","notes",
}

def heuristic_type(text: str) -> str:
    first = text.strip().lower().split()[0].rstrip(",.")
    return "operative" if first in OPERATIVE_WORDS else "preambular"


# ─────────────────────────────────────────────────────────────────────────────
# Single-paragraph classifier
# ─────────────────────────────────────────────────────────────────────────────

def classify_paragraph(
    para       : dict,
    rag_index,
    train_paras: list[dict],
    tag_index,
    all_tags   : list[str],
    tag_desc   : dict[str, str],
) -> dict:
    """
    Classify one paragraph. Returns a dict with para_id, type, tags, think.
    """
    text       = para["text"]
    tag_cands  = retrieve_tag_candidates(text, tag_index, all_tags)
    few_shots  = retrieve_few_shots(text, rag_index, train_paras)
    prompt     = build_prompt(text, tag_cands, few_shots, tag_desc)

    result = llm_json(prompt, required_keys=["type", "tags", "think"])

    # Fallback if LLM fails
    if result is None:
        result = {
            "type"  : heuristic_type(text),
            "tags"  : [],
            "think" : "LLM failed after all retries. Heuristic fallback applied.",
        }

    # Sanitise type
    if result.get("type") not in ("preambular", "operative"):
        result["type"] = heuristic_type(text)

    # Sanitise tags — keep only names that actually exist in the taxonomy
    all_tags_set = set(all_tags)
    valid_tags   = [t for t in (result.get("tags") or []) if t in all_tags_set]

    return {
        "para_id" : para["para_id"],
        "doc_id"  : para["doc_id"],
        "type"    : result["type"],
        "tags"    : valid_tags,
        "think"   : result.get("think", ""),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Build submission structure
# ─────────────────────────────────────────────────────────────────────────────

def build_submission(test_docs: list[dict],
                     predictions: list[dict]) -> list[dict]:
    """
    Merge predictions back into per-document structure.
    predictions: flat list of classify_paragraph() outputs
    """
    from collections import defaultdict
    pred_by_doc: dict[str, list] = defaultdict(list)
    for p in predictions:
        pred_by_doc[p["doc_id"]].append(p)

    submission = []
    for doc in test_docs:
        doc_id = (
            doc.get("doc_id") or doc.get("id") or doc.get("filename", "unknown")
        )
        para_list = []
        for pred in pred_by_doc.get(doc_id, []):
            para_list.append({
                "para_id"  : pred["para_id"],
                "type"     : pred["type"],
                "tags"     : pred["tags"],
                "think"    : pred["think"],
                "relations": [],           # Task 2 will fill this in
            })
        submission.append({"doc_id": doc_id, "paragraphs": para_list})

    return submission


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Subtask 1 — Paragraph Classification")
    parser.add_argument("--dry-run", type=int, default=0, metavar="N",
                        help="Only classify the first N test paragraphs (0 = all)")
    args = parser.parse_args()

    print("=" * 60)
    print(" Subtask 1 — Paragraph Classification")
    print("=" * 60)

    # ── Load data ────────────────────────────────────────────────────────────
    print("\n[1/5] Loading data …")
    train_docs    = load_json_docs(CFG.TRAIN_DIR)
    test_docs     = load_json_docs(CFG.TEST_DIR)
    all_tags, tag_desc = load_tags(CFG.TAGS_CSV)

    from src.data_utils import flatten_paragraphs
    train_paras = flatten_paragraphs(train_docs)
    test_paras  = flatten_paragraphs(test_docs)
    print(f"  Train paragraphs : {len(train_paras):,}")
    print(f"  Test  paragraphs : {len(test_paras):,}")

    if args.dry_run > 0:
        test_paras = test_paras[: args.dry_run]
        print(f"  [DRY RUN] Using first {args.dry_run} paragraphs")

    # ── Build indices ─────────────────────────────────────────────────────────
    print("\n[2/5] Building / loading FAISS indices …")
    rag_index, train_paras = build_rag_index(train_paras)
    tag_index              = build_tag_index(all_tags, tag_desc)

    # ── Load LLM ──────────────────────────────────────────────────────────────
    print("\n[3/5] Loading LLM …")
    from src.llm_utils import load_llm
    load_llm()   # warms up the singleton; subsequent calls reuse it

    # ── Classify ──────────────────────────────────────────────────────────────
    print(f"\n[4/5] Classifying {len(test_paras):,} paragraphs …")
    predictions = []
    t0 = time.time()

    for para in tqdm(test_paras, desc="Task 1", unit="para"):
        pred = classify_paragraph(
            para, rag_index, train_paras, tag_index, all_tags, tag_desc
        )
        predictions.append(pred)

    elapsed = time.time() - t0
    print(f"  Done in {elapsed/60:.1f} min  "
          f"({elapsed/len(test_paras):.1f} s/para avg)")

    # ── Save outputs ──────────────────────────────────────────────────────────
    print("\n[5/5] Saving outputs …")
    CFG.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Flat predictions list (consumed by task2_relations.py)
    save_json(predictions, CFG.OUTPUT_DIR / "task1_predictions.json")

    # Full per-document submission (standalone Task-1 submission)
    submission = build_submission(test_docs, predictions)
    save_json(submission, CFG.OUTPUT_DIR / "submission_task1.json")

    # Stats
    types = [p["type"] for p in predictions]
    print(f"\n  preambular : {types.count('preambular'):,}")
    print(f"  operative  : {types.count('operative'):,}")
    tagged = sum(1 for p in predictions if p["tags"])
    print(f"  paragraphs with ≥1 tag : {tagged:,} / {len(predictions):,}")

    print("\n✅ Task 1 complete.")
    print(f"   → outputs/task1_predictions.json    (for task2)")
    print(f"   → outputs/submission_task1.json     (standalone submission)")


if __name__ == "__main__":
    main()
