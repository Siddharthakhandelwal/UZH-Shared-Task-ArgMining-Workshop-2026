"""
src/llm_utils.py
────────────────
LLM loading and inference helpers.

P100-specific settings applied here:
  • INT8 quantisation   (NF4 / 4-bit requires CUDA compute ≥ 7.5 — P100 is 6.0)
  • float16 dtype       (P100 has no bfloat16)
  • device_map="auto"   (single-GPU layout)
"""

import json
import re
from typing import Optional

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import config as CFG


# ─────────────────────────────────────────────────────────────────────────────
# Model / tokenizer (singletons — loaded once, reused everywhere)
# ─────────────────────────────────────────────────────────────────────────────

_tokenizer = None
_llm       = None


def load_llm() -> tuple:
    """Load tokenizer + LLM in INT8 on the P100. Returns (tokenizer, model)."""
    global _tokenizer, _llm
    if _llm is not None:
        return _tokenizer, _llm

    print(f"\nLoading LLM: {CFG.MODEL_ID}")
    print(f"  Quantisation : INT8 (P100-compatible)")
    print(f"  dtype        : float16")

    bnb_cfg = BitsAndBytesConfig(load_in_8bit=True) if CFG.LOAD_IN_8BIT else None

    _tokenizer = AutoTokenizer.from_pretrained(
        CFG.MODEL_ID, trust_remote_code=True
    )
    _tokenizer.padding_side = "left"   # required for decoder-only batch decode

    _llm = AutoModelForCausalLM.from_pretrained(
        CFG.MODEL_ID,
        quantization_config = bnb_cfg,
        device_map          = "auto",
        torch_dtype         = CFG.TORCH_DTYPE,
        trust_remote_code   = True,
    )
    _llm.eval()

    used_gb = torch.cuda.memory_allocated() / 1e9
    total_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"  ✅ LLM ready  |  VRAM: {used_gb:.1f} / {total_gb:.1f} GB used")
    return _tokenizer, _llm


# ─────────────────────────────────────────────────────────────────────────────
# Core generation
# ─────────────────────────────────────────────────────────────────────────────

def llm_chat(prompt: str) -> str:
    """
    Single-turn chat completion.
    Wraps the prompt in the model's chat template and returns the reply text.
    """
    tokenizer, model = load_llm()

    messages  = [{"role": "user", "content": prompt}]
    formatted = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(formatted, return_tensors="pt").to(CFG.DEVICE)

    with torch.inference_mode():
        out = model.generate(
            **inputs,
            max_new_tokens     = CFG.MAX_NEW_TOKENS,
            temperature        = CFG.TEMPERATURE,
            top_p              = CFG.TOP_P,
            repetition_penalty = CFG.REP_PENALTY,
            do_sample          = True,
            pad_token_id       = tokenizer.eos_token_id,
        )

    new_tokens = out[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


# ─────────────────────────────────────────────────────────────────────────────
# JSON extraction & retrying
# ─────────────────────────────────────────────────────────────────────────────

def extract_json(text: str) -> Optional[dict]:
    """
    Try three strategies to extract a JSON object from LLM output:
      1. Direct json.loads()
      2. First {...} block via regex
      3. Strip markdown fences then parse
    """
    # Strategy 1 — direct
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Strategy 2 — first {...} block
    m = re.search(r"\{[\s\S]+\}", text)
    if m:
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            pass

    # Strategy 3 — strip markdown
    cleaned = re.sub(r"```(?:json)?", "", text).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return None


def llm_json(prompt: str, required_keys: list[str]) -> Optional[dict]:
    """
    Call the LLM and retry up to MAX_RETRIES times until a valid JSON dict
    containing all *required_keys* is returned.
    Returns None if all attempts fail.
    """
    for attempt in range(CFG.MAX_RETRIES):
        raw    = llm_chat(prompt)
        result = extract_json(raw)
        if result and all(k in result for k in required_keys):
            return result
        if attempt < CFG.MAX_RETRIES - 1:
            print(f"    ⚠  Attempt {attempt + 1}: malformed JSON, retrying …")
    return None
