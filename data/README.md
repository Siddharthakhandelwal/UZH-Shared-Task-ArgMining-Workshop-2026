---
license: cc-by-nc-4.0
language:
- en
- fr
size_categories:
- 1K<n<10K

---

# Reconstructing the Reasoning in United Nations Resolutions

This repository contains the dataset for the **UZH Shared Task @ ArgMining Workshop 2026**, co-located with **ACL 2026**.  
The shared task focuses on recovering **paragraph-level argumentative structure** in highly formal, legal-political documents, specifically **United Nations resolutions and recommendations**.

The dataset supports two subtasks:
1. Argumentative paragraph classification  
2. Argumentative relation prediction between paragraphs

---

## Claimer

**The content of this publication has not been approved by the United Nations and does not reflect the views of the United Nations or its officials or Member States. UN-RES should only be used for research purposes.**

---

## Contact

Please contact the shared task organizers at University of Zurich for questions. 

## Task Overview

United Nations resolutions encode collective reasoning at scale through carefully structured preambles and operative clauses.  
This shared task evaluates how well systems can reconstruct this implicit reasoning structure.

Participants are expected to build systems that:
- Identify whether a paragraph is **preambular** or **operative**
- Assign one or more **argumentative tags**
- Predict **argumentative relations** between paragraphs

Only **open-weight language models with ≤ 8B parameters** are permitted.

---

## Subtasks

### Subtask 1: Argumentative Paragraph Classification

For each paragraph, systems must predict:
- **Paragraph type**: `preambular` or `operative`
- **Argumentative tags**: multi-label classification over a predefined tag set

### Subtask 2: Argumentative Relation Prediction

For each paragraph, systems must:
- Identify related paragraphs (by index)
- Assign one or more relation types:
  - `supporting`
  - `contradictive`
  - `complemental`
  - `modifying`

---

## Dataset Description

### Languages
- English
- French

### Granularity
- Paragraph level

### Splits

#### Training Set (parsed_data_en)
- Source: UN-RES dataset [Gao et al., 2025](https://aclanthology.org/2025.emnlp-demos.3/)
- Size: 2,695 UN resolutions
- Language: French (with machine-generated English translations)
- Annotation: paragraph-level argumentative structure

#### Test Set (parsed_data_fr)
- Source: UNESCO International Conference on Education (1934–2008)
- Size: 45 parsed documents (each may contain up to three resolutions in **JSON**)
- Language: French
- Annotation: paragraph-level (held out for evaluation)
- Validation set: none

#### Tags
- See `education_dimensions_updated.csv`
  
---

## Data Format

All data are provided in **JSON** format following a fixed schema.

### Example (simplified)

```json
"TEXT_ID": "ICPE-25-1962_RES1-FR_res_54",
  "RECOMMENDATION": 54,
  "TITLE": "LA PLANIFICATION DE L'ÉDUCATION",
  "METADATA": {
    "structure": {
      "doc_title": "ICPE-25-1962_RES1-FR",
      "nb_paras": 58,
      "preambular_para": [], 
      "operative_para": []
      "think": ""      
    }
  },
  "body": {
    "paras": [
      {
        "para_number": 1,
        "para": "La Conférence internationale de l'instruction publique, Convoquée à...",
        "type": null,      
        "tags": [],     
        "matched_paras": [],     
        "think": "",
        "para_en": "The International Conference on Education, convened in ..."
      },
      ...
    ]
  }
}