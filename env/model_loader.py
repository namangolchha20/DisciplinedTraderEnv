"""Load a trained LoRA adapter for inference (PEFT + 4-bit base model)."""

from __future__ import annotations

import json
from pathlib import Path


def load_trained_agent(adapter_dir: str | Path = "./trained_trader_lora"):
    """Return (model, tokenizer) ready for inference."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    adapter_dir = Path(adapter_dir)
    with open(adapter_dir / "adapter_config.json") as f:
        cfg = json.load(f)

    base_model = cfg.get("base_model_name_or_path", "unsloth/Qwen2.5-0.5B-Instruct")
    if "unsloth-bnb" in base_model.lower():
        base_model = "unsloth/Qwen2.5-0.5B-Instruct"

    tokenizer = AutoTokenizer.from_pretrained(str(adapter_dir))
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(base, str(adapter_dir))
    model.eval()
    return model, tokenizer
