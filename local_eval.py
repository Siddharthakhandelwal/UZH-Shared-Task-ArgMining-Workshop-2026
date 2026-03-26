"""
local_eval.py
─────────────
Evaluate Task-1 classification on a random sample of TRAINING data.
Gives you a local F1 estimate before committing to the full test run.

Usage
-----
  python local_eval.py             # evaluate on 100 random training paragraphs
  python local_eval.py --n 200     # evaluate on 200 paragraphs
  python local_eval.py --n 50 --seed 99
"""

import argparse
import random
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm
from sklearn.metrics import f1_score, classification_report

sys.path.insert(0, str(Path(__file__).parent))

import config as CFG
from src.data_utils import load_json_docs, load_tags, flatten_paragraphs
from src.embeddings import build_rag_index, build_tag_index
from src.llm_utils  import load_llm
from task1_classify import classify_paragraph


def main():
    parser = argparse.ArgumentParser(description="Local Task-1 evaluation on train split")
    parser.add_argument("--n",    type=int, default=100, help="Number of paragraphs to evaluate")
    parser.add_argument("--seed", type=int, default=CFG.SEED)
    args = parser.parse_args()

    print("=" * 60)
    print(f" Local Task-1 Evaluation  (n={args.n}, seed={args.seed})")
    print("=" * 60)

    random.seed(args.seed)
    np.random.seed(args.seed)

    print("\n[1/4] Loading data …")
    train_docs       = load_json_docs(CFG.TRAIN_DIR)
    all_tags, tag_desc = load_tags(CFG.TAGS_CSV)
    train_paras      = flatten_paragraphs(train_docs)

    # Sample
    sample = random.sample(train_paras, min(args.n, len(train_paras)))
    print(f"  Sample size: {len(sample):,}")

    print("\n[2/4] Building indices …")
    # Build RAG index over the complement of the sample (no data leakage)
    sample_ids  = {p["para_id"] for p in sample}
    train_base  = [p for p in train_paras if p["para_id"] not in sample_ids]
    rag_index, train_base = build_rag_index(train_base, force_rebuild=True)
    tag_index   = build_tag_index(all_tags, tag_desc)

    print("\n[3/4] Loading LLM …")
    load_llm()

    print(f"\n[4/4] Classifying {len(sample)} paragraphs …")
    tag2idx = {t: i for i, t in enumerate(all_tags)}

    y_true_type, y_pred_type = [], []
    y_true_tags, y_pred_tags = [], []

    for para in tqdm(sample, desc="Eval"):
        pred = classify_paragraph(para, rag_index, train_base, tag_index, all_tags, tag_desc)

        # Type
        y_true_type.append(1 if para["type"] == "operative" else 0)
        y_pred_type.append(1 if pred["type"]  == "operative" else 0)

        # Tags (multi-label, 141-d binary vectors)
        gt = np.zeros(len(all_tags), dtype=int)
        pr = np.zeros(len(all_tags), dtype=int)
        for t in (para.get("tags") or []):
            if t in tag2idx:
                gt[tag2idx[t]] = 1
        for t in pred["tags"]:
            if t in tag2idx:
                pr[tag2idx[t]] = 1
        y_true_tags.append(gt)
        y_pred_tags.append(pr)

    type_f1 = f1_score(y_true_type, y_pred_type, average="binary", zero_division=0)
    tag_micro_f1 = f1_score(
        np.array(y_true_tags), np.array(y_pred_tags),
        average="micro", zero_division=0,
    )
    tag_macro_f1 = f1_score(
        np.array(y_true_tags), np.array(y_pred_tags),
        average="macro", zero_division=0,
    )

    print("\n── Local Evaluation Results ─────────────────────────────")
    print(f"  n_eval         : {len(sample)}")
    print(f"  type_f1        : {type_f1:.4f}   (preambular vs operative)")
    print(f"  tag_micro_f1   : {tag_micro_f1:.4f}   (141-class micro-average)")
    print(f"  tag_macro_f1   : {tag_macro_f1:.4f}   (141-class macro-average)")
    print(f"  composite_f1   : {(type_f1 + tag_micro_f1) / 2:.4f}  (avg of type + tag)")
    print("─────────────────────────────────────────────────────────")

    print("\nType classification report:")
    print(classification_report(
        y_true_type, y_pred_type,
        target_names=["preambular", "operative"],
        zero_division=0,
    ))


if __name__ == "__main__":
    main()
