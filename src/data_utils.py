"""
src/data_utils.py
─────────────────
Data loading and normalisation helpers shared by task1 and task2 scripts.
"""

import json
from pathlib import Path

import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# Document / paragraph loading
# ─────────────────────────────────────────────────────────────────────────────

def load_json_docs(folder: Path) -> list[dict]:
    """Load every *.json file in *folder* and return as a list of dicts."""
    docs = []
    for fp in sorted(folder.glob("*.json")):
        with open(fp, encoding="utf-8") as f:
            docs.append(json.load(f))
    print(f"  Loaded {len(docs):,} documents from {folder}")
    return docs


def get_doc_id(doc: dict) -> str:
    return (
        doc.get("doc_id") or doc.get("id") or doc.get("filename") or "unknown"
    )


def get_para_text(para: dict) -> str:
    """Prefer English translation if available, fall back to French."""
    return (
        para.get("text_en") or para.get("text_fr") or para.get("text") or ""
    ).strip()


def flatten_paragraphs(docs: list[dict]) -> list[dict]:
    """
    Flatten all paragraphs from all documents into a single list.
    Each item carries: doc_id, para_id, text, type, tags, relations.
    Gold-label fields (type/tags/relations) are empty strings/lists for test docs.
    """
    rows = []
    for doc in docs:
        doc_id = get_doc_id(doc)
        for para in doc.get("paragraphs", []):
            rows.append({
                "doc_id"   : doc_id,
                "para_id"  : para.get("para_id") or para.get("id"),
                "text"     : get_para_text(para),
                "type"     : para.get("type", ""),
                "tags"     : para.get("tags") or [],
                "relations": para.get("relations") or [],
            })
    return rows


def group_by_doc(paras: list[dict]) -> dict[str, list[dict]]:
    """Return {doc_id: [para, para, …]} preserving original order."""
    from collections import defaultdict
    grouped: dict[str, list] = defaultdict(list)
    for p in paras:
        grouped[p["doc_id"]].append(p)
    return dict(grouped)


# ─────────────────────────────────────────────────────────────────────────────
# Tag metadata
# ─────────────────────────────────────────────────────────────────────────────

def load_tags(csv_path: Path) -> tuple[list[str], dict[str, str]]:
    """
    Returns
    -------
    tag_list : list[str]          — ordered list of all 141 tag names
    tag_desc : dict[str, str]     — {tag_name: human-readable description}
    """
    df       = pd.read_csv(csv_path)
    name_col = df.columns[0]
    desc_col = df.columns[1] if len(df.columns) > 1 else name_col

    tag_list = df[name_col].astype(str).tolist()
    tag_desc = dict(zip(df[name_col].astype(str), df[desc_col].astype(str)))
    print(f"  Loaded {len(tag_list)} tags from {csv_path.name}")
    return tag_list, tag_desc


# ─────────────────────────────────────────────────────────────────────────────
# Submission I/O
# ─────────────────────────────────────────────────────────────────────────────

def save_json(obj: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    print(f"  Saved → {path}")


def load_json(path: Path) -> object:
    with open(path, encoding="utf-8") as f:
        return json.load(f)
