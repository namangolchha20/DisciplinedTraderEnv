"""Load a trained LoRA adapter for inference (PEFT + 4-bit or fp16 base model)."""

from __future__ import annotations

import json
import os
from pathlib import Path

DEFAULT_ADAPTER_REPO = "NGGAMER/disciplined-trader-lora"


def _prefer_fp16() -> bool:
    """HF Spaces: skip bitsandbytes 4-bit (CUDA lib mismatch). Local 4GB GPU: use 4-bit."""
    if os.environ.get("LLM_FP16", "").lower() in ("1", "true", "yes"):
        return True
    if os.environ.get("SPACE_ID"):  # set automatically on Hugging Face Spaces
        return True
    return False


def ensure_adapter_downloaded(
    adapter_dir: str | Path = "./trained_trader_lora",
    repo_id: str | None = None,
) -> Path:
    """Download the LoRA adapter from the Hub if missing locally (HF Spaces)."""
    adapter_dir = Path(adapter_dir)
    if (adapter_dir / "adapter_config.json").exists():
        return adapter_dir

    repo_id = (repo_id or os.environ.get("HF_ADAPTER_REPO") or DEFAULT_ADAPTER_REPO).strip()
    if not repo_id:
        return adapter_dir

    from huggingface_hub import snapshot_download

    adapter_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading LoRA adapter from huggingface.co/{repo_id} ...")
    snapshot_download(repo_id=repo_id, local_dir=str(adapter_dir))
    print(f"Adapter saved to {adapter_dir}")
    return adapter_dir


def load_trained_agent(adapter_dir: str | Path = "./trained_trader_lora"):
    """Return (model, tokenizer) ready for inference."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    adapter_dir = Path(adapter_dir)
    with open(adapter_dir / "adapter_config.json") as f:
        cfg = json.load(f)

    base_model = cfg.get("base_model_name_or_path", "unsloth/Qwen2.5-0.5B-Instruct")
    if "unsloth-bnb" in base_model.lower():
        base_model = "unsloth/Qwen2.5-0.5B-Instruct"

    tokenizer = AutoTokenizer.from_pretrained(str(adapter_dir))
    mode = "fp16" if _prefer_fp16() else "4bit"
    print(f"Loading base model {base_model} ({mode})...")

    if mode == "4bit":
        from transformers import BitsAndBytesConfig

        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
        base = AutoModelForCausalLM.from_pretrained(
            base_model,
            quantization_config=bnb,
            device_map="auto",
        )
    else:
        base = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.float16,
            device_map="auto",
        )

    model = PeftModel.from_pretrained(base, str(adapter_dir))
    model.eval()
    return model, tokenizer
