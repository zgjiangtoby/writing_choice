"""Teacher-force exact candidate strings after separately tokenized context strings."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch

from model_runtime import forward_probe_indices, load_local, metadata

MODULE_SOURCE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def prepare(tokenizer, row, maximum, sentinel_id=None):
    prefix = tokenizer.encode(row["prefix"], add_special_tokens=False)
    continuation = tokenizer.encode(row["continuation"], add_special_tokens=False)
    assert continuation, row["id"]
    sentinel = sentinel_id
    if sentinel is None:
        sentinel = tokenizer.bos_token_id
        if sentinel is None:
            sentinel = tokenizer.eos_token_id
    assert sentinel is not None
    retained = prefix[-maximum:]
    sequence = [sentinel] + retained + continuation
    canonical = tokenizer.encode(row["prefix"]+row["continuation"], add_special_tokens=False)
    return {"input_ids": sequence, "start": 1+len(retained), "count": len(continuation),
            "prefix_tokens_original": len(prefix), "prefix_tokens_retained": len(retained),
            "canonical_boundary": canonical == prefix+continuation}


def execution_groups(items, serialize_empty_prefix=False):
    """Use a canonical singleton forward for every genuinely empty prefix."""
    if not serialize_empty_prefix:
        return [list(range(len(items)))]
    contextual = [i for i, item in enumerate(items) if item["prefix_tokens_retained"] > 0]
    empty = [[i] for i, item in enumerate(items) if item["prefix_tokens_retained"] == 0]
    return ([contextual] if contextual else [])+empty


def score_batch(model, tokenizer, items, serialize_empty_prefix=False):
    if serialize_empty_prefix:
        scores = [None]*len(items)
        for indices in execution_groups(items, True):
            values = score_batch(model, tokenizer, [items[i] for i in indices])
            for i, value in zip(indices, values):
                scores[i] = value
        assert all(score is not None for score in scores)
        return scores
    inputs = tokenizer.pad({"input_ids": [x["input_ids"] for x in items]},
                           padding=True, return_tensors="pt").to("cuda")
    positions = (inputs.attention_mask.cumsum(-1)-1).clamp_min(0)
    hidden = model.model(**inputs, position_ids=positions, use_cache=False).last_hidden_state
    width = inputs.input_ids.shape[1]
    answers = []
    for i, item in enumerate(items):
        left = width-len(item["input_ids"])
        begin = left+item["start"]
        token_scores = []
        # Only candidate prediction positions need the expensive vocabulary projection.
        for position in range(begin, width, 64):
            stop = min(position+64, width)
            logits = model.lm_head(hidden[i, position-1:stop-1]).float()
            targets = inputs.input_ids[i, position:stop]
            values = logits.gather(1, targets[:, None]).squeeze(1)-logits.logsumexp(-1)
            token_scores.extend(values.cpu().tolist())
        assert len(token_scores) == item["count"] and all(torch.isfinite(torch.tensor(token_scores)))
        answers.append(token_scores)
    return answers


def check_full_forward(model, tokenizer, items, serialize_empty_prefix=False, batch_size=None):
    indices = forward_probe_indices([item["input_ids"] for item in items])
    if serialize_empty_prefix:
        contextual = [i for i, item in enumerate(items) if item["prefix_tokens_retained"] > 0]
        if contextual:
            additions = forward_probe_indices([items[i]["input_ids"] for i in contextual])
            indices = list(dict.fromkeys(indices+[contextual[i] for i in additions]))
    probes = [items[i] for i in indices]
    differences = []
    groups = [[i] for i in range(len(probes))] if batch_size == 1 else execution_groups(probes, serialize_empty_prefix)
    for group in groups:
        subset = [probes[i] for i in group]
        inputs = tokenizer.pad({"input_ids": [item["input_ids"] for item in subset]},
                               padding=True, return_tensors="pt").to("cuda")
        positions = (inputs.attention_mask.cumsum(-1)-1).clamp_min(0)
        logits = model(**inputs, position_ids=positions, use_cache=False).logits.float()
        optimized = score_batch(model, tokenizer, subset)
        width = inputs.input_ids.shape[1]
        for i, item in enumerate(subset):
            begin = width-len(item["input_ids"])+item["start"]
            targets = inputs.input_ids[i, begin:]
            reference = -torch.nn.functional.cross_entropy(
                logits[i, begin-1:width-1], targets, reduction="none")
            observed = torch.tensor(optimized[i], device=reference.device, dtype=reference.dtype)
            assert reference.shape == observed.shape and torch.isfinite(reference).all()
            differences.append(float((reference-observed).abs().max()))
    difference = max(differences)
    assert difference < 1e-4, difference
    return {"source_row_indices": indices, "sequence_lengths": [len(x["input_ids"]) for x in probes],
            "candidate_token_counts": [x["count"] for x in probes],
            "execution_groups_source_indices": [[indices[i] for i in group] for group in groups],
            "max_abs_token_logprob_difference": difference,
            "reference": "AutoModelForCausalLM full forward and negative cross_entropy at shifted candidate positions",
            "tolerance": 1e-4, "passed": True}


def batch_probe_groups(items, batch_size, serialize_empty_prefix):
    groups = [list(range(min(batch_size, len(items))))]
    if serialize_empty_prefix:
        contextual = [i for i, item in enumerate(items) if item["prefix_tokens_retained"] > 0][:batch_size]
        if contextual and contextual not in groups:
            groups.append(contextual)
    return groups


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=1536, help="Maximum retained prefix tokens")
    parser.add_argument("--sentinel-token", help="Explicit shared document-boundary special token for paired checkpoints")
    parser.add_argument("--dtype", choices=("bfloat16","float32"), default="bfloat16")
    parser.add_argument("--check-batch-equivalence", action="store_true")
    parser.add_argument("--check-full-forward", action="store_true")
    parser.add_argument("--serialize-empty-prefix", action="store_true",
                        help="Canonical singleton model forward for all empty prefixes, including empty C/W inputs")
    args = parser.parse_args()
    assert torch.cuda.is_available() and not args.output.exists()
    input_bytes = args.input.read_bytes()
    rows = [json.loads(line) for line in input_bytes.splitlines() if line.strip()]
    assert rows and len({row["id"] for row in rows}) == len(rows)
    tokenizer, model = load_local(args.model,args.dtype)
    sentinel = None
    if args.sentinel_token is not None:
        token_ids = tokenizer.encode(args.sentinel_token,add_special_tokens=False)
        assert len(token_ids)==1 and token_ids[0] in tokenizer.all_special_ids
        sentinel = token_ids[0]
    prepared = [prepare(tokenizer, row, args.max_tokens,sentinel) for row in rows]
    assert max(len(item["input_ids"]) for item in prepared) <= 4096
    assert all(item["canonical_boundary"] for item in prepared), "Candidate boundary resegments prefix; no silent noncanonical score"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    meta = metadata(args,tokenizer)
    meta["input_sha256"] = hashlib.sha256(input_bytes).hexdigest()
    meta["runtime_source_sha256"]["score_continuations.py"] = MODULE_SOURCE_SHA256
    meta.update(template="raw continuation; no chat", tokenization="prefix and candidate separately; BOS else EOS sentinel",
                likelihood="sum of candidate-token log probabilities; no EOS score",
                sentinel_token=args.sentinel_token,sentinel_id=prepared[0]["input_ids"][0],
                serialize_empty_prefix=args.serialize_empty_prefix,
                empty_prefix_rows=sum(x["prefix_tokens_retained"]==0 for x in prepared))
    if sentinel is not None:
        meta["tokenization"] = "prefix and candidate separately; explicitly matched document-boundary sentinel"
    start = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode(), args.output.open("x") as output:
        if args.check_full_forward:
            meta["full_forward_check"] = check_full_forward(model, tokenizer, prepared,args.serialize_empty_prefix,args.batch_size)
        if args.check_batch_equivalence:
            differences = []
            checks = []
            for indices in batch_probe_groups(prepared,args.batch_size,args.serialize_empty_prefix):
                probes = [prepared[i] for i in indices]
                batch_scores = score_batch(model, tokenizer, probes,args.serialize_empty_prefix)
                current = []
                for i, probe in enumerate(probes):
                    single = score_batch(model, tokenizer, [probe],args.serialize_empty_prefix)[0]
                    current.append(max(abs(a-b) for a, b in zip(batch_scores[i], single)))
                differences.extend(current)
                checks.append({"source_row_indices":indices,"max_abs_token_logprob_difference":max(current),
                    "execution_group_sizes":[len(group) for group in execution_groups(probes,args.serialize_empty_prefix)],
                    "empty_prefix_rows":sum(x["prefix_tokens_retained"]==0 for x in probes)})
            meta["batch_token_logprob_max_abs_difference"] = max(differences)
            meta["batch_equivalence_checks"] = checks
            assert max(differences) < (1e-4 if args.dtype == "float32" else 0.15), differences
        for offset in range(0, len(rows), args.batch_size):
            items = prepared[offset:offset+args.batch_size]
            scores = score_batch(model, tokenizer, items,args.serialize_empty_prefix)
            for j, (item, values) in enumerate(zip(items, scores)):
                record = {"id": rows[offset+j]["id"], "sum_log_probability": sum(values),
                          "mean_log_probability": sum(values)/len(values),
                          "token_log_probabilities": values, **{key: value for key, value in item.items() if key != "input_ids"}}
                output.write(json.dumps(record)+"\n")
            output.flush()
            if offset == 0 or (offset//args.batch_size+1) % 50 == 0:
                print(json.dumps({"completed": min(offset+args.batch_size,len(rows)),
                                  "total": len(rows), "elapsed_seconds": time.monotonic()-start}), flush=True)
    torch.cuda.synchronize()
    meta.update(rows=len(rows), elapsed_seconds=time.monotonic()-start,
                peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                peak_reserved_bytes=torch.cuda.max_memory_reserved(), status="complete")
    args.output.with_suffix(".metadata.json").write_text(json.dumps(meta, indent=2)+"\n")
    print(json.dumps(meta), flush=True)


if __name__ == "__main__":
    main()
