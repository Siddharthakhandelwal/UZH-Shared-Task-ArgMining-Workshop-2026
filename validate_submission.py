"""
validate_submission.py
──────────────────────
Validates the final submission JSON against the organiser's schema.
Run this before uploading to the organiser portal.

Usage
-----
  python validate_submission.py                                  # validates submission_final.json
  python validate_submission.py --file outputs/my_custom.json   # validate any file
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import config as CFG


VALID_TYPES    = {"preambular", "operative"}
VALID_RELS     = set(CFG.RELATION_TYPES)


def validate(sub: list[dict], all_tags: list[str]) -> tuple[bool, list[str]]:
    all_tags_set = set(all_tags)
    errors: list[str] = []

    if not isinstance(sub, list):
        return False, ["Top-level object must be a JSON array"]

    all_para_ids: set[str] = set()

    for doc in sub:
        did = doc.get("doc_id", "MISSING_DOC_ID")
        if "doc_id" not in doc:
            errors.append(f"[{did}] Missing 'doc_id'")
        if "paragraphs" not in doc:
            errors.append(f"[{did}] Missing 'paragraphs'")
            continue

        for para in doc["paragraphs"]:
            pid = para.get("para_id", "MISSING_PARA_ID")
            ref = f"[{did} | {pid}]"

            # Required keys
            for key in ("para_id", "type", "tags", "think", "relations"):
                if key not in para:
                    errors.append(f"{ref} Missing key: '{key}'")

            # Type
            t = para.get("type")
            if t not in VALID_TYPES:
                errors.append(f"{ref} Invalid type: '{t}' (must be preambular or operative)")

            # Tags
            tags = para.get("tags")
            if not isinstance(tags, list):
                errors.append(f"{ref} 'tags' must be a list, got {type(tags).__name__}")
            else:
                unknown = [x for x in tags if x not in all_tags_set]
                if unknown:
                    errors.append(f"{ref} Unknown tag(s): {unknown[:5]}")

            # Think
            think = para.get("think")
            if not isinstance(think, str):
                errors.append(f"{ref} 'think' must be a string")
            elif not think.strip():
                errors.append(f"{ref} 'think' is empty (required for LLM-Judge scoring)")

            # Relations
            rels = para.get("relations")
            if not isinstance(rels, list):
                errors.append(f"{ref} 'relations' must be a list")
            else:
                for r in rels:
                    if "target_para_id" not in r:
                        errors.append(f"{ref} Relation missing 'target_para_id': {r}")
                    if "relation_types" not in r:
                        errors.append(f"{ref} Relation missing 'relation_types': {r}")
                    else:
                        bad = [x for x in r["relation_types"] if x not in VALID_RELS]
                        if bad:
                            errors.append(f"{ref} Unknown relation type(s): {bad}")

            all_para_ids.add(pid)

    # Cross-check: all target_para_ids must refer to known para_ids in the submission
    for doc in sub:
        did = doc.get("doc_id", "?")
        for para in doc.get("paragraphs", []):
            for rel in para.get("relations", []):
                tid = rel.get("target_para_id")
                if tid and tid not in all_para_ids:
                    errors.append(
                        f"[{did} | {para.get('para_id')}] "
                        f"target_para_id '{tid}' not found in submission"
                    )

    return len(errors) == 0, errors


def print_summary(sub: list[dict]) -> None:
    n_docs  = len(sub)
    n_paras = sum(len(d.get("paragraphs", [])) for d in sub)
    n_rels  = sum(
        len(p.get("relations", []))
        for d in sub
        for p in d.get("paragraphs", [])
    )
    n_tags  = sum(
        len(p.get("tags", []))
        for d in sub
        for p in d.get("paragraphs", [])
    )
    types   = [p.get("type") for d in sub for p in d.get("paragraphs", [])]

    print("\n── Submission Summary ──────────────────────────────────")
    print(f"  Documents          : {n_docs:,}")
    print(f"  Paragraphs         : {n_paras:,}")
    print(f"    preambular       : {types.count('preambular'):,}")
    print(f"    operative        : {types.count('operative'):,}")
    print(f"  Total tags         : {n_tags:,}  ({n_tags/max(n_paras,1):.2f} avg/para)")
    print(f"  Total relations    : {n_rels:,}  ({n_rels/max(n_paras,1):.2f} avg/para)")
    from collections import Counter
    rel_counter: Counter = Counter()
    for d in sub:
        for p in d.get("paragraphs", []):
            for r in p.get("relations", []):
                for rt in r.get("relation_types", []):
                    rel_counter[rt] += 1
    for rt in CFG.RELATION_TYPES:
        print(f"    {rt:<16} : {rel_counter.get(rt, 0):,}")
    print("────────────────────────────────────────────────────────")


def main():
    parser = argparse.ArgumentParser(description="Validate submission JSON")
    parser.add_argument(
        "--file", type=Path,
        default=CFG.OUTPUT_DIR / "submission_final.json",
        help="Path to submission JSON to validate",
    )
    args = parser.parse_args()

    print(f"Validating: {args.file}")

    if not args.file.exists():
        print(f"❌  File not found: {args.file}")
        sys.exit(1)

    with open(args.file, encoding="utf-8") as f:
        sub = json.load(f)

    # Load tag list for validation
    import pandas as pd
    df       = pd.read_csv(CFG.TAGS_CSV, sep=';')
    all_tags = df[df.columns[-1]].astype(str).tolist()

    ok, errors = validate(sub, all_tags)
    print_summary(sub)

    if ok:
        print("\n✅  Submission is valid — ready to upload to the organiser portal.")
    else:
        print(f"\n❌  {len(errors)} validation error(s) found:\n")
        for i, e in enumerate(errors[:40], 1):
            print(f"  {i:>3}. {e}")
        if len(errors) > 40:
            print(f"  … and {len(errors) - 40} more errors")
        sys.exit(1)


if __name__ == "__main__":
    main()
