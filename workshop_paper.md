---
title: "Efficient Zero-Shot Argument Mining of UN Resolutions with Small LMs and Semantic Pre-filtering"
authors: 
  - Anonymous Authors
venue: "UZH ArgMining 2026 Shared Task"
format: "ACL 4-page"
---

# Efficient Zero-Shot Argument Mining of UN Resolutions with Small LMs and Semantic Pre-filtering

## Abstract
This paper describes our submission for the UZH ArgMining 2026 Shared Task on analysing UN/UNESCO resolution documents. The task requires a dual-pronged approach: classifying paragraph types and themes (Task 1) and identifying argumentative relations between paragraphs (Task 2). Operating under the constraint of using open-weight models under 8 billion parameters, our pipeline employs a quantized Qwen2.5-7B-Instruct model. To tackle the quadratic complexity of relation prediction and the vast taxonomy of thematic tags, we implement a retrieval-augmented generation (RAG) approach to pre-filter context and relations using dense embeddings. Furthermore, we engineered a highly robust, resumable, and parallelisable pipeline optimized for resource-constrained environments like Kaggle P100 GPUs, featuring atomic checkpointing and composite key deduplication.

## 1. Introduction
The UZH ArgMining 2026 shared task challenges participants to parse complex grammatical constructs of international policy documents into structured argumentative graphs. The shared task strictly mandates the use of open-weight LLMs ($\le 8B$ parameters) and demands reasoning transparency through a Chain-of-Thought (CoT) `think` field in the final JSON output. 

Our approach frames both tasks as zero-shot/few-shot generation problems augmented by semantic search. We deliberately avoided fine-tuning to explore the out-of-the-box reasoning bounds of recent dense models (Qwen 2.5), using `all-MiniLM-L6-v2` embeddings to dynamically manage the context window. 

## 2. System Architecture

Our base model is `Qwen/Qwen2.5-7B-Instruct`, loaded in 8-bit precision via `bitsandbytes` to fit within a 16GB VRAM constraint. The pipeline is split into two independent, sequential tasks.

### 2.1 Task 1: Paragraph Classification and Thematic Tagging
Task 1 requires assigning a structural type (`preambular` or `operative`) and a subset of thematic tags (from a predefined taxonomy in a CSV file). Because injecting the entire tag taxonomy into the LLM prompt vastly exceeds the effective context window and dilutes instruction following, we utilize a Retrieval-Augmented Generation (RAG) strategy:
- **Embedding space:** All taxonomy tag descriptions and training paragraphs are embedded using a lightweight SentenceTransformer metric (`all-MiniLM-L6-v2`) and indexed via FAISS.
- **Candidate Pre-filtering:** For a given test paragraph, we retrieve the top-$K$ semantically similar training examples (used as few-shot demonstrations) and the top-$N$ most relevant thematic tags. 
- **LLM Prompting:** The prompt requests the LLM to provide explicit reasoning in a `think` JSON field, followed by the structural `type` classification and the selection of fitting tags from only the retrieved subset.

### 2.2 Task 2: Directed Argumentative Relation Prediction
Task 2 maps the argumentative graph, predicting relationships (`supporting`, `contradictive`, `complemental`, `modifying`) between paragraphs. A naive approach evaluating every pair in a document yields a quadratic $O(N^2)$ complexity, rendering it computationally infeasible on free tiers.
- **Sliding Window:** We constrain the search backward. Paragraph $P_n$ only evaluates relations targeting preceding paragraphs $P_{n-1}, P_{n-2}, \dots, P_0$.
- **Semantic Pre-filtering:** We compute continuous cosine similarities between paragraph embeddings. For $P_n$, we only prompt the LLM to assess relations with previous paragraphs whose similarity score exceeds an empirical threshold, capping the maximum number of structural comparisons per paragraph to $C$. This reduces relation discovery to a linear upper bound $O(N \cdot C)$ per document.

## 3. Engineering & Scalability

Deploying this pipeline on Kaggle kernels with 12-hour session timeouts necessitated robust fault tolerance:
1. **Parallel Chunking:** Both inference stages support chunking (`--chunk K --total-chunks N`), allowing the workload to be mathematically distributed across multiple concurrent Kaggle notebook sessions. 
2. **Atomic Granular Checkpointing:** The system writes continuous paragraph-level checkpoints using a `.tmp` to `.json` atomic rename pattern. We identified and resolved a critical collision bug where paragraph IDs reset between documents by adopting a global composite key (`"doc_id||para_id"`). This prevented silent overwrites and ensured $100\%$ prediction coverage upon merge.

## 4. Evaluation 

Final system ranking is uniquely determined by a blend of classification F1 Score and an "LLM-as-a-Judge" evaluation of the system's reasoning (`think` fields). Since test set labels are blind, we implemented an internal proxy evaluation over the training split:
- **F1 Heuristic:** A naive rule-based baseline separating paragraphs by operative vocabulary boundaries establishes a strict lower bound for structural classification.
- **Judge Proxy Metrics:** We evaluate reasoning structural depth based on heuristic signals: chain length, step-by-step sequencing, and embedding of domain-specific vocabulary. Preliminary qualitative review of the generated `think` fields shows Qwen2.5 successfully anchoring its decisions in specific lexical cues present in the UN documents.

## 5. Conclusion
We successfully designed a parallelisable, compute-efficient pipeline for the UZH ArgMining 2026 task. By combining the strong zero-shot reasoning of Qwen 2.5 with classic dense-retrieval pre-filtering, we circumvent the traditional quadratic scaling laws of relation mapping. Our architecture demonstrates how open-weight models under rigorous computational constraints can tackle complex, domain-specific NLP formulations.

*Code available at: [Link to your GitHub Repository]*
