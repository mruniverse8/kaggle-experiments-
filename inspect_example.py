#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dz2_causal.dataset import (  # noqa: E402
    TweetExtractionCausalDataset,
    register_special_tokens,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Inspect one prepared sample")
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen3.5-0.8B")
    parser.add_argument(
        "--prompt",
        type=str,
        default="You are a professional emotion identifier",
    )
    parser.add_argument("--tweet", type=str, default="I love this movie!")
    parser.add_argument("--sentiment", type=str, default="positive")
    parser.add_argument("--selected-text", type=str, default="love this")
    parser.add_argument("--max-len", type=int, default=256)
    parser.add_argument("--trust-remote-code", action="store_true", default=False)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        use_fast=True,
        trust_remote_code=args.trust_remote_code,
    )
    register_special_tokens(tokenizer)

    records = [
        {
            "prompt": args.prompt,
            "text": args.tweet,
            "sentiment": args.sentiment,
            "selected_text": args.selected_text,
        }
    ]
    dataset = TweetExtractionCausalDataset(
        records=records,
        tokenizer=tokenizer,
        prompt_text=args.prompt,
        max_len=args.max_len,
    )
    sample = dataset[0]

    ids = sample["input_ids"].tolist()
    labels = sample["labels"].tolist()
    source_mask = sample["source_mask"].tolist()
    start_soft = sample["start_soft"].tolist()
    end_soft = sample["end_soft"].tolist()
    select_targets = sample["select_targets"].tolist()

    tokens = tokenizer.convert_ids_to_tokens(ids)

    print("\nPrepared sequence (index | token | label | src_mask | start_soft | end_soft | select)")
    print("-" * 100)
    for i, tok in enumerate(tokens):
        label_str = str(labels[i]) if labels[i] != -100 else "-100"
        print(
            f"{i:03d} | {tok:20s} | {label_str:>6s} | "
            f"{source_mask[i]:.0f} | {start_soft[i]:.4f} | {end_soft[i]:.4f} | {select_targets[i]:.0f}"
        )

    ce_target_ids = [x for x in labels if x != -100]
    ce_target_text = tokenizer.decode(ce_target_ids, skip_special_tokens=False)
    selected_from_mask = []
    for i, flag in enumerate(select_targets):
        if flag > 0.5:
            selected_from_mask.append(ids[i])
    selected_from_mask_text = tokenizer.decode(selected_from_mask, skip_special_tokens=False)

    print("\nSummary")
    print("-" * 100)
    print(f"Input length: {len(ids)}")
    print(f"CE target text: {ce_target_text}")
    print(f"Token-selection target text: {selected_from_mask_text}")
    print(f"Sum(start_soft): {float(torch.tensor(start_soft).sum()):.4f}")
    print(f"Sum(end_soft): {float(torch.tensor(end_soft).sum()):.4f}")


if __name__ == "__main__":
    main()

