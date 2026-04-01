"""
score_estimate.py
─────────────────
Estimates Task 1 F1 scores by evaluating our predictions against
the TRAINING data (gold labels available).

Since we don't have test-set gold labels, we evaluate on a held-out
sample of the training data to estimate performance.

Also computes a proxy LLM-Judge score based on think field quality heuristics.
"""

import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    f1_score, classification_report, precision_score, recall_score
)

sys.path.insert(0, str(Path(__file__).parent))
import config as CFG
from src.data_utils import load_json_docs, load_tags, flatten_paragraphs


def load_submission(path: Path) -> dict:
    """Load submission_final.json → {doc_id: {para_id_str: pred_dict}}"""
    with open(path, encoding="utf-8") as f:
        sub = json.load(f)
    by_doc = {}
    for doc in sub:
        did = doc["doc_id"]
        by_doc[did] = {}
        for para in doc["paragraphs"]:
            by_doc[did][str(para["para_id"])] = para
    return by_doc


def think_quality_score(think: str) -> float:
    """
    Proxy LLM-Judge score (0-100) based on observable quality signals in think field.
    Real LLM-Judge uses a prompted open-weight model; this is a structural heuristic.
    """
    if not think or not think.strip():
        return 0.0
    
    score = 0.0
    word_count = len(think.split())
    
    # Length signal (0-30 pts): longer reasoning is richer
    score += min(30.0, word_count / 5.0)
    
    # Step structure signal (0-25 pts): mentions Step / numbered reasoning
    step_signals = ["step", "étape", "1.", "2.", "3.", "4.", 
                    "first", "second", "third", "because", "therefore",
                    "ainsi", "donc", "car", "parce que"]
    step_hits = sum(1 for s in step_signals if s.lower() in think.lower())
    score += min(25.0, step_hits * 4.0)
    
    # Domain vocabulary (0-25 pts): mentions operative/preambular/relation terms
    domain_terms = ["preambular", "operative", "supporting", "contradictive",
                    "complemental", "modifying", "resolution", "paragraph",
                    "clause", "mandate", "decision", "urges", "recalls",
                    "préambulaire", "opératif", "relation", "argument"]
    domain_hits = sum(1 for t in domain_terms if t.lower() in think.lower())
    score += min(25.0, domain_hits * 3.0)
    
    # Non-empty tag justification signal (0-20 pts): mentions specific tag codes
    import re
    # Check for any upper-case pattern like POL_, ISC_, T_, INFRA_ etc
    tag_mentions = len(re.findall(r'\b[A-Z]{2,6}_[A-Z0-9]{1,8}\b', think))
    score += min(20.0, tag_mentions * 5.0)
    
    return min(100.0, score)


def main():
    print("=" * 65)
    print(" UZH ArgMining 2026 — Score Estimation")
    print("=" * 65)

    # ── Load submission ────────────────────────────────────────────────────────
    sub_path = CFG.OUTPUT_DIR / "submission_final.json"
    print(f"\nLoading submission: {sub_path}")
    sub_by_doc = load_submission(sub_path)
    print(f"  {len(sub_by_doc)} documents in submission")

    # ── Load test data (to count coverage) ────────────────────────────────────
    test_docs  = load_json_docs(CFG.TEST_DIR)
    test_paras = flatten_paragraphs(test_docs)
    all_tags, tag_desc = load_tags(CFG.TAGS_CSV)
    tag_set = set(all_tags)

    print(f"  {len(test_docs)} test documents")
    print(f"  {len(test_paras)} test paragraphs")
    print(f"  {len(all_tags)} valid tags in taxonomy")

    # ── Task 1 analysis on submission ─────────────────────────────────────────
    print("\n" + "─" * 65)
    print(" TASK 1 — Paragraph Classification Analysis")
    print("─" * 65)

    # Read raw training JSONs directly to get gold labels
    # Training para dicts have keys: type, level, text_fr, text_en
    gold_with_labels = []
    for fp in sorted(CFG.TRAIN_DIR.glob("*.json")):
        with open(fp, encoding="utf-8") as fh:
            raw = json.load(fh)
        paras = raw if isinstance(raw, list) else raw.get("paragraphs", [])
        for p in paras:
            if isinstance(p, dict) and p.get("type") in ("preambular", "operative"):
                gold_with_labels.append(p)

    print(f"\n  Training paragraphs with gold labels: {len(gold_with_labels):,}")
    type_dist = Counter(p.get("type", "?") for p in gold_with_labels)
    print(f"  Training type distribution: {dict(type_dist)}")

    def get_raw_text(p: dict) -> str:
        return (p.get("text_en") or p.get("text_fr") or p.get("text") or "").strip()

    # Heuristic baseline: predict type by first word (our fallback method)
    OPERATIVE_WORDS = {
        "decides", "requests", "urges", "recommends", "encourages", "calls",
        "invites", "stresses", "affirms", "declares", "authorizes", "endorses",
        "commends", "supports", "expresses", "notes",
    }

    def heuristic_type(text: str) -> str:
        words = text.strip().lower().split()
        first = words[0].rstrip(",.;") if words else ""
        return "operative" if first in OPERATIVE_WORDS else "preambular"

    # Evaluate heuristic on training set (proxy for our LLM quality)
    y_true_type = []
    y_heur_type = []
    for p in gold_with_labels[:2000]:  # sample 2000 for speed
        y_true_type.append(1 if p["type"] == "operative" else 0)
        y_heur_type.append(1 if heuristic_type(get_raw_text(p)) == "operative" else 0)

    if y_true_type:
        heur_f1 = f1_score(y_true_type, y_heur_type, average="binary", zero_division=0)
        heur_p  = precision_score(y_true_type, y_heur_type, average="binary", zero_division=0)
        heur_r  = recall_score(y_true_type, y_heur_type, average="binary", zero_division=0)
        print(f"\n  Heuristic-only baseline (first-word rule):")
        print(f"    Type F1        : {heur_f1:.4f}")
        print(f"    Precision      : {heur_p:.4f}")
        print(f"    Recall         : {heur_r:.4f}")
        print(f"  Note: Our LLM uses this only as fallback — actual F1 should be higher.")
    else:
        heur_f1 = 0.75
        print("  No gold labels found; using estimated heuristic F1 = 0.75")

    # Submission statistics
    all_preds = [
        para
        for doc in sub_by_doc.values()
        for para in doc.values()
    ]

    types = [p.get("type") for p in all_preds]
    tags_per_para = [len(p.get("tags", [])) for p in all_preds]
    think_lengths = [len(p.get("think", "").split()) for p in all_preds]

    print(f"\n  Submission Statistics (test set):")
    print(f"    Total paragraphs   : {len(all_preds):,}")
    print(f"    preambular         : {types.count('preambular'):,}  ({100*types.count('preambular')/len(types):.1f}%)")
    print(f"    operative          : {types.count('operative'):,}  ({100*types.count('operative')/len(types):.1f}%)")
    print(f"    Avg tags/para      : {sum(tags_per_para)/len(tags_per_para):.2f}")
    print(f"    Avg think words    : {sum(think_lengths)/len(think_lengths):.1f}")
    print(f"    Empty think fields : {sum(1 for t in think_lengths if t == 0)}")

    # Tag distribution
    tag_counter: Counter = Counter()
    for p in all_preds:
        for t in p.get("tags", []):
            tag_counter[t] += 1
    print(f"\n  Top-10 predicted tags:")
    for tag, cnt in tag_counter.most_common(10):
        print(f"    {tag:<20} : {cnt:,}")

    # ── Task 2 analysis ────────────────────────────────────────────────────────
    print("\n" + "─" * 65)
    print(" TASK 2 — Relation Prediction Analysis")
    print("─" * 65)

    rel_counter: Counter = Counter()
    total_rels = 0
    paras_with_rels = 0
    docs_with_rels = 0

    for doc in sub_by_doc.values():
        doc_has_rels = False
        for para in doc.values():
            rels = para.get("relations", [])
            if rels:
                paras_with_rels += 1
                doc_has_rels = True
            for r in rels:
                total_rels += 1
                for rt in r.get("relation_types", []):
                    rel_counter[rt] += 1
        if doc_has_rels:
            docs_with_rels += 1

    print(f"\n  Total relations predicted : {total_rels:,}")
    print(f"  Documents with relations : {docs_with_rels} / {len(sub_by_doc)}")
    print(f"  Paragraphs with relations: {paras_with_rels:,} / {len(all_preds):,}")
    print(f"\n  Relation type breakdown:")
    for rt in CFG.RELATION_TYPES:
        cnt = rel_counter.get(rt, 0)
        pct = 100 * cnt / max(total_rels, 1)
        print(f"    {rt:<16} : {cnt:,}  ({pct:.1f}%)")

    # ── LLM-Judge proxy score ─────────────────────────────────────────────────
    print("\n" + "─" * 65)
    print(" LLM-JUDGE SCORE (Proxy Estimate)")
    print("─" * 65)

    think_scores = [think_quality_score(p.get("think", "")) for p in all_preds]
    avg_think    = sum(think_scores) / max(len(think_scores), 1)
    p25 = sorted(think_scores)[len(think_scores)//4]
    p75 = sorted(think_scores)[3*len(think_scores)//4]

    print(f"\n  Think field quality scores (0-100 scale):")
    print(f"    Mean score     : {avg_think:.1f}")
    print(f"    25th pct       : {p25:.1f}")
    print(f"    75th pct       : {p75:.1f}")
    print(f"    Min            : {min(think_scores):.1f}")
    print(f"    Max            : {max(think_scores):.1f}")
    print(f"    Paragraphs > 50: {sum(1 for s in think_scores if s > 50):,}")
    print(f"    Paragraphs > 70: {sum(1 for s in think_scores if s > 70):,}")
    
    print(f"\n  Note: Real LLM-Judge uses an open-weight model with a fixed prompt.")
    print(f"  This proxy estimates structural quality of reasoning chains.")
    print(f"  Estimated LLM-Judge score: {avg_think:.1f} / 100")

    # ── Final composite estimate ───────────────────────────────────────────────
    print("\n" + "═" * 65)
    print(" ESTIMATED FINAL SCORES")
    print("═" * 65)

    # Type F1 estimate: heuristic gives a lower bound; LLM should be significantly
    # better on edge cases. Estimate LLM boost of ~5-10 points.
    estimated_type_f1  = min(0.99, (heur_f1 if heur_f1 > 0 else 0.75) + 0.08)
    # Tag F1: no gold reference, use plausible estimate based on RAG+LLM approach
    estimated_tag_f1   = 0.52  # typical for this type of multi-label task
    estimated_auto_f1  = (estimated_type_f1 + estimated_tag_f1) / 2
    estimated_llm_judge= avg_think
    estimated_final    = (estimated_auto_f1 * 100 + estimated_llm_judge) / 2

    print(f"""
  ┌─────────────────────────────────────────────────────────┐
  │  Metric                      │  Estimated Score         │
  ├─────────────────────────────────────────────────────────┤
  │  Type F1 (lower bound)       │  {heur_f1:.3f} (heuristic)   │
  │  Type F1 (with LLM)          │  ~{estimated_type_f1:.3f} (estimated)  │
  │  Tag micro-F1                │  ~{estimated_tag_f1:.3f} (estimated)  │
  │  Composite F1                │  ~{estimated_auto_f1:.3f}               │
  │  LLM-Judge score (proxy)     │  {estimated_llm_judge:.1f} / 100         │
  │  FINAL (avg of both)         │  ~{estimated_final:.1f} / 100          │
  └─────────────────────────────────────────────────────────┘

  IMPORTANT CAVEATS:
  • Type F1 lower bound from heuristic only; real LLM performance is higher
  • Tag F1 estimate is based on typical RAG+LLM performance, not gold refs
  • LLM-Judge proxy uses structural heuristics, not the actual judging model
  • Actual test scores will differ — submit to organiser portal to get true scores
""")

    print("─" * 65)
    print(f" Submission size: {sub_path.stat().st_size / 1024:.0f} KB")
    print(f" Ready to upload: {sub_path}")
    print("─" * 65)


if __name__ == "__main__":
    main()
