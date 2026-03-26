"""
config.py
─────────
Central configuration for the UZH ArgMining 2026 shared task pipeline.
All scripts import from here — change one value, it propagates everywhere.
"""

from pathlib import Path
import torch

# ─── Paths ────────────────────────────────────────────────────────────────────
# Repo-local directories (created at runtime if missing)
REPO_ROOT      = Path(__file__).parent
OUTPUT_DIR     = REPO_ROOT / "outputs"
CACHE_DIR      = REPO_ROOT / ".cache"         # stores FAISS indices + embeddings

# Using the cloned dataset in the data/ directory
TRAIN_DIR      = REPO_ROOT / "data" / "train-data"
TEST_DIR       = REPO_ROOT / "data" / "test-data"
TAGS_CSV       = REPO_ROOT / "data" / "education_dimensions_updated.csv"

# ─── GPU / dtype ──────────────────────────────────────────────────────────────
# P100 = CUDA compute 6.0  →  supports INT8 but NOT NF4/4-bit or bfloat16
DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"
TORCH_DTYPE    = torch.float16                # P100 has no bfloat16
LOAD_IN_8BIT   = True                         # ~7 GB VRAM; leaves ~9 GB headroom

# ─── LLM ──────────────────────────────────────────────────────────────────────
MODEL_ID       = "Qwen/Qwen2.5-7B-Instruct"  # best reasoning model ≤ 8B
MAX_NEW_TOKENS = 1536                          # long think chains → better LLM-Judge score
TEMPERATURE    = 0.05                          # near-deterministic for structured JSON
TOP_P          = 0.9
REP_PENALTY    = 1.1
MAX_RETRIES    = 3                             # JSON-parse retries per LLM call

# ─── Embedding model ──────────────────────────────────────────────────────────
EMBED_MODEL    = "sentence-transformers/all-mpnet-base-v2"   # 768-d; stronger than MiniLM
EMBED_BATCH    = 256

# ─── RAG (few-shot retrieval) ─────────────────────────────────────────────────
RAG_TOP_K      = 5                             # annotated examples injected into prompt

# ─── Tag retrieval ────────────────────────────────────────────────────────────
TAG_CANDIDATE_K = 20                           # pre-filter top-20 tags, LLM picks final set

# ─── Relation prediction ──────────────────────────────────────────────────────
REL_WINDOW     = 8                             # backward window (para B vs last N paras)
REL_SIM_THRESH = 0.30                          # cosine-sim floor before LLM is invoked

# ─── Misc ─────────────────────────────────────────────────────────────────────
SEED           = 42
RELATION_TYPES = ["supporting", "contradictive", "complemental", "modifying"]
