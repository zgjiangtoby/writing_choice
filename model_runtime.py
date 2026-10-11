"""Load local causal models and record the numerical execution settings."""

import hashlib
import json
from pathlib import Path

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer


MODULE_SOURCE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def load_tokenizer(model_path):
    config = json.loads((Path(model_path)/"config.json").read_text())
    # Transformers 4.57.2 has a Qwen fast-tokenizer loading bug; use its slow tokenizer.
    tokenizer = AutoTokenizer.from_pretrained(model_path,local_files_only=True,
                                              use_fast=config.get("model_type")!="qwen3")
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    assert tokenizer.pad_token_id is not None
    return tokenizer


def load_local(model_path, dtype="bfloat16"):
    tokenizer = load_tokenizer(model_path)
    torch.set_float32_matmul_precision("highest")
    model = AutoModelForCausalLM.from_pretrained(
        model_path, local_files_only=True, torch_dtype=getattr(torch,dtype),
        attn_implementation="sdpa").to("cuda").eval()
    return tokenizer, model


def forward_probe_indices(sequences):
    """Choose at most two sequences using only length, including padding if possible."""
    indices = [0]
    for i, sequence in enumerate(sequences[1:], 1):
        if len(sequence) != len(sequences[0]):
            indices.append(i)
            break
    return indices


def metadata(args,tokenizer=None):
    return {"model_path": str(Path(args.model).resolve()), "input": str(Path(args.input).resolve()),
            "torch": torch.__version__, "transformers": transformers.__version__,
            "device": torch.cuda.get_device_name(), "dtype": args.dtype,
            "float32_matmul_precision": "highest", "position_ids": "unpadded cumulative attention mask",
            "attention": "sdpa", "thinking": False, "batch_size": args.batch_size,
            "maximum_prompt_tokens": args.max_tokens, "gold_used": False,
            "runtime_source_sha256": {"model_runtime.py": MODULE_SOURCE_SHA256},
            "tokenizer_class":type(tokenizer).__name__ if tokenizer is not None else None}
