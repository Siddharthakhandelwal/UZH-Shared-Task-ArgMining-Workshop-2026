"""
src/embeddings.py
─────────────────
Sentence embedding + FAISS index helpers used by both task scripts.

Indices are cached to disk so they are only built once per session.
"""

import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
import config as CFG


# ─────────────────────────────────────────────────────────────────────────────
# Embedding model (singleton)
# ─────────────────────────────────────────────────────────────────────────────

_embed_model: Optional[SentenceTransformer] = None

def get_embed_model() -> SentenceTransformer:
    global _embed_model
    if _embed_model is None:
        print(f"  Loading embedding model: {CFG.EMBED_MODEL} …")
        _embed_model = SentenceTransformer(CFG.EMBED_MODEL, device=CFG.DEVICE)
    return _embed_model


def encode(texts: list[str], show_progress: bool = True) -> np.ndarray:
    """
    Encode a list of texts into L2-normalised float32 vectors.
    dot-product on normalised vectors == cosine similarity.
    """
    model = get_embed_model()
    return model.encode(
        texts,
        batch_size        = CFG.EMBED_BATCH,
        show_progress_bar = show_progress,
        normalize_embeddings = True,
        convert_to_numpy  = True,
    ).astype("float32")


def encode_single(text: str) -> np.ndarray:
    """Encode one text string → shape (dim,) float32."""
    return encode([text], show_progress=False)[0]


# ─────────────────────────────────────────────────────────────────────────────
# FAISS index helpers
# ─────────────────────────────────────────────────────────────────────────────

def build_flat_ip_index(vecs: np.ndarray) -> faiss.IndexFlatIP:
    """Build an exact inner-product (cosine) FAISS index from a numpy array."""
    dim   = vecs.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(vecs)
    return index


def _cache_path(name: str) -> Path:
    p = CFG.CACHE_DIR / f"{name}.pkl"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def _load_cache(name: str):
    cp = _cache_path(name)
    if cp.exists():
        with open(cp, "rb") as f:
            return pickle.load(f)
    return None


def _save_cache(name: str, obj) -> None:
    with open(_cache_path(name), "wb") as f:
        pickle.dump(obj, f)


# ─────────────────────────────────────────────────────────────────────────────
# Named index builders (with disk caching)
# ─────────────────────────────────────────────────────────────────────────────

def build_rag_index(
    train_paras: list[dict],
    force_rebuild: bool = False,
) -> tuple[faiss.IndexFlatIP, list[dict]]:
    """
    Build (or load from cache) a FAISS index over all training paragraphs.

    Returns
    -------
    index       : faiss.IndexFlatIP
    train_paras : list[dict]  — same list, used for result lookup by index position
    """
    cached = _load_cache("rag_index") if not force_rebuild else None
    if cached is not None:
        print("  RAG index loaded from cache")
        return cached["index"], cached["paras"]

    print(f"  Building RAG index over {len(train_paras):,} training paragraphs …")
    texts = [p["text"] for p in train_paras]
    vecs  = encode(texts)
    index = build_flat_ip_index(vecs)
    _save_cache("rag_index", {"index": index, "paras": train_paras})
    print(f"  RAG index built: {index.ntotal:,} vectors @ dim={vecs.shape[1]}")
    return index, train_paras


def build_tag_index(
    all_tags: list[str],
    tag_desc: dict[str, str],
    force_rebuild: bool = False,
) -> faiss.IndexFlatIP:
    """
    Build (or load from cache) a FAISS index over tag descriptions.
    """
    cached = _load_cache("tag_index") if not force_rebuild else None
    if cached is not None:
        print("  Tag index loaded from cache")
        return cached["index"]

    print(f"  Building tag index over {len(all_tags)} tags …")
    texts = [f"{t}: {tag_desc.get(t, t)}" for t in all_tags]
    vecs  = encode(texts)
    index = build_flat_ip_index(vecs)
    _save_cache("tag_index", {"index": index})
    print(f"  Tag index built: {index.ntotal} vectors")
    return index


def build_para_vecs(
    paras: list[dict],
    cache_name: str = "test_para_vecs",
    force_rebuild: bool = False,
) -> dict[str, np.ndarray]:
    """
    Encode a list of paragraphs and return {para_id: vector}.
    Used by Task 2 for fast cosine pre-filtering.
    """
    cached = _load_cache(cache_name) if not force_rebuild else None
    if cached is not None:
        print(f"  Para vectors loaded from cache ({cache_name})")
        return cached

    print(f"  Encoding {len(paras):,} paragraphs for similarity pre-filter …")
    texts   = [p["text"] for p in paras]
    vecs    = encode(texts)
    vec_map = {p["para_id"]: vecs[i] for i, p in enumerate(paras)}
    _save_cache(cache_name, vec_map)
    return vec_map


# ─────────────────────────────────────────────────────────────────────────────
# Retrieval helpers
# ─────────────────────────────────────────────────────────────────────────────

def retrieve_few_shots(
    query_text  : str,
    rag_index   : faiss.IndexFlatIP,
    train_paras : list[dict],
    k           : int = CFG.RAG_TOP_K,
) -> list[dict]:
    """Return k training paragraphs most similar to query_text."""
    q_vec = encode_single(query_text).reshape(1, -1)
    _, idxs = rag_index.search(q_vec, k + 2)
    shots = []
    for i in idxs[0]:
        if 0 <= i < len(train_paras):
            p = train_paras[i]
            if p["text"] != query_text:
                shots.append(p)
        if len(shots) == k:
            break
    return shots


def retrieve_tag_candidates(
    query_text : str,
    tag_index  : faiss.IndexFlatIP,
    all_tags   : list[str],
    k          : int = CFG.TAG_CANDIDATE_K,
) -> list[str]:
    """Return top-k tag names most relevant to the paragraph text."""
    q_vec = encode_single(query_text).reshape(1, -1)
    _, idxs = tag_index.search(q_vec, k)
    return [all_tags[i] for i in idxs[0]]
