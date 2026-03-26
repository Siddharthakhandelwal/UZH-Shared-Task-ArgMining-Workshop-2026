# UZH ArgMining 2026 — Shared Task Solution

**"Reconstructing the Reasoning in United Nations Resolutions"**  
ArgMining Workshop @ ACL 2026 · San Diego, USA

---

## What this repo does

| Script | Task |
|---|---|
| `task1_classify.py` | **Subtask 1** — classify each paragraph as `preambular` / `operative` and assign multi-label tags from a 141-tag taxonomy |
| `task2_relations.py` | **Subtask 2** — predict argumentative relations between paragraphs (`supporting`, `contradictive`, `complemental`, `modifying`) |
| `validate_submission.py` | Schema-validate the final JSON before uploading |
| `local_eval.py` | Local F1 evaluation on the training split (no test-set leakage) |

---

## Architecture

```
Qwen2.5-7B-Instruct  (INT8, float16)   ← main LLM, ~7 GB VRAM on P100
all-mpnet-base-v2    (SentenceTransformer, 768-d)   ← embeddings
FAISS IndexFlatIP    ← cosine-similarity search (RAG + tag retrieval)
```

**Subtask 1 pipeline per paragraph:**
1. Embed paragraph → cosine-search tag index → top-20 candidate tags
2. Cosine-search training index → 5 annotated few-shot examples
3. LLM prompt (system + examples + candidates + paragraph) → `{think, type, tags}`
4. Sanitise: enforce valid type, keep only known tag names

**Subtask 2 pipeline per document:**
1. Embed all test paragraphs → `{para_id: vector}` lookup
2. For each para B, compare against window of 8 preceding paras A
3. Cosine pre-filter (threshold 0.30) eliminates clearly unrelated pairs
4. LLM prompt for surviving pairs → `{think, has_relation, relation_types}`

---

## Repo structure

```
argmining2026/
├── config.py               ← all tunable settings (edit DATASET_SLUG here)
├── task1_classify.py       ← run this first
├── task2_relations.py      ← run this second
├── validate_submission.py  ← run before submitting
├── local_eval.py           ← optional F1 sanity check
├── requirements.txt
├── src/
│   ├── __init__.py
│   ├── data_utils.py       ← JSON loading, paragraph flattening
│   ├── embeddings.py       ← SentenceTransformer + FAISS helpers
│   └── llm_utils.py        ← LLM loading (INT8/P100) + JSON extraction
└── outputs/                ← created at runtime
    ├── task1_predictions.json
    ├── task2_predictions.json
    └── submission_final.json   ← upload this to the organiser portal
```

---

## Kaggle setup (P100 GPU · 16 GB VRAM)

### Step 1 — Upload your dataset to Kaggle

Upload the organiser-provided files as a Kaggle dataset with this layout:

```
uzh-argmining-2026/
├── train/          ← 2,695 JSON resolution files
├── test/           ← 45 held-out JSON files
└── evaluation_dimensions_updated.csv
```

> If you use a different dataset slug, edit `DATASET_SLUG` in `config.py`.

### Step 2 — Open a Kaggle Notebook

- **Accelerator**: GPU → P100
- **Internet**: ON (needed to download model weights from HuggingFace)

### Step 3 — Clone this repo and install dependencies

Paste this into the **first notebook cell** and run it:

```bash
# Clone repo
!git clone https://github.com/<your-username>/argmining2026.git
%cd argmining2026

# Install dependencies
!pip install -q -r requirements.txt
```

### Step 4 — (Optional) Local evaluation on training data

Run this to get a local F1 estimate before the full test run (~15 min for 100 samples):

```bash
!python local_eval.py --n 100
```

### Step 5 — Run Subtask 1

```bash
!python task1_classify.py
```

Outputs:
- `outputs/task1_predictions.json` — consumed by task2
- `outputs/submission_task1.json` — standalone Task-1 submission

Estimated runtime on P100: **~2–4 hours** for all 45 test documents.

For a quick smoke test on 10 paragraphs first:

```bash
!python task1_classify.py --dry-run 10
```

### Step 6 — Run Subtask 2

```bash
!python task2_relations.py
```

Outputs:
- `outputs/task2_predictions.json`
- `outputs/submission_final.json` ← **this is your final submission**

For a smoke test on 3 documents:

```bash
!python task2_relations.py --dry-run 3
```

### Step 7 — Validate before uploading

```bash
!python validate_submission.py
```

---

## Configuration reference (`config.py`)

| Parameter | Default | Effect |
|---|---|---|
| `DATASET_SLUG` | `uzh-argmining-2026` | Kaggle dataset slug |
| `MODEL_ID` | `Qwen/Qwen2.5-7B-Instruct` | HuggingFace model |
| `LOAD_IN_8BIT` | `True` | P100 requires INT8; do not change to 4-bit |
| `MAX_NEW_TOKENS` | `1536` | Longer → richer `think` chains → better LLM-Judge score |
| `TEMPERATURE` | `0.05` | Near-deterministic; raise to 0.3 for more creative tagging |
| `RAG_TOP_K` | `5` | Few-shot examples injected per prompt |
| `TAG_CANDIDATE_K` | `20` | Tag pre-filter width before LLM selection |
| `REL_WINDOW` | `8` | Backward paragraph window for relation search |
| `REL_SIM_THRESH` | `0.30` | Cosine floor; lower = more LLM calls, better recall |
| `MAX_RETRIES` | `3` | JSON-parse retries per LLM call |

---

## VRAM budget on P100 (16 GB)

```
Qwen2.5-7B  INT8 weights       ≈  7.0 GB
all-mpnet-base-v2 on GPU       ≈  0.4 GB
KV-cache (1536 tokens)         ≈  3.0 GB
FAISS + numpy buffers          ≈  0.5 GB
──────────────────────────────────────────
Total                          ≈ 10.9 GB  ✅  safe on 16 GB
```

---

## Important model constraint

The organiser **prohibits closed-source models** (GPT-4, Claude, etc.) and requires **open-weight models ≤ 8B parameters**. This solution uses `Qwen2.5-7B-Instruct` (7B parameters, open-weight on HuggingFace). The `think` field in every paragraph output satisfies the organiser's requirement for visible reasoning, which the LLM-Judge scores from 0–100.

---

## Submission deadline

| Date | Event |
|---|---|
| 18 March 2026 | Evaluation opens |
| **1 April 2026** | **Submission deadline** |
| 15 April 2026 | Results notified |
| 24 April 2026 | System paper due (4 pages) |
| 3 July 2026 | Workshop @ ACL 2026, San Diego |
