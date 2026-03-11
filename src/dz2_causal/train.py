from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from .dataset import (
    TweetExtractionCausalDataset,
    causal_collate_fn,
    load_records_from_csv,
    register_special_tokens,
)
from .losses import compute_total_loss
from .modeling import CausalExtractionModel


def default_train_config() -> Dict:
    return {
        "train_csv": "",
        "model_name": "Qwen/Qwen3.5-0.8B",
        "output_dir": "",
        "prompt_text": "You are a professional emotion identifier",
        "prompt_col": None,
        "max_len": 256,
        "soft_alpha": 0.6,
        "epochs": 3,
        "batch_size": 8,
        "lr": 2e-5,
        "weight_decay": 0.01,
        "warmup_ratio": 0.05,
        "lambda_kl": 0.3,
        "lambda_select": 0.2,
        "ce_only_epochs": 1,
        "grad_accum_steps": 1,
        "num_workers": 2,
        "log_every": 20,
        "seed": 42,
        "trust_remote_code": False,
        "no_amp": False,
        "amp_dtype": "auto",
    }


def _resolve_path(value: str, base_dir: Path) -> str:
    p = Path(value).expanduser()
    if not p.is_absolute():
        p = (base_dir / p).resolve()
    return str(p)


def load_train_config(config_path: str) -> argparse.Namespace:
    path = Path(config_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        user_cfg = json.load(f)

    if not isinstance(user_cfg, dict):
        raise ValueError("Config JSON must be an object/dictionary.")

    cfg = default_train_config()
    cfg.update(user_cfg)

    if not cfg["train_csv"]:
        raise ValueError("Config requires non-empty 'train_csv'.")
    if not cfg["output_dir"]:
        raise ValueError("Config requires non-empty 'output_dir'.")

    base_dir = path.parent
    cfg["train_csv"] = _resolve_path(cfg["train_csv"], base_dir)
    cfg["output_dir"] = _resolve_path(cfg["output_dir"], base_dir)

    return argparse.Namespace(**cfg)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def move_batch_to_device(batch: Dict, device: torch.device) -> Dict:
    out = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            out[key] = value.to(device)
        else:
            out[key] = value
    return out


def resolve_amp_settings(args: argparse.Namespace, device: torch.device):
    use_amp = device.type == "cuda" and not bool(getattr(args, "no_amp", False))
    amp_pref = str(getattr(args, "amp_dtype", "auto")).lower()
    if amp_pref not in {"auto", "fp16", "bf16"}:
        raise ValueError("amp_dtype must be one of: auto, fp16, bf16")

    if not use_amp:
        return False, None, False

    bf16_supported = torch.cuda.is_bf16_supported()
    if amp_pref == "fp16":
        amp_dtype = torch.float16
    elif amp_pref == "bf16":
        if not bf16_supported:
            raise ValueError("amp_dtype=bf16 requested but GPU does not support bf16.")
        amp_dtype = torch.bfloat16
    else:
        amp_dtype = torch.bfloat16 if bf16_supported else torch.float16

    # GradScaler should be used with fp16 only; not with bf16.
    use_grad_scaler = amp_dtype == torch.float16
    return True, amp_dtype, use_grad_scaler


def train(args: argparse.Namespace) -> None:
    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        use_fast=True,
        trust_remote_code=args.trust_remote_code,
    )

    model = CausalExtractionModel(
        model_name=args.model_name,
        trust_remote_code=args.trust_remote_code,
    )
    register_special_tokens(tokenizer, model=model.lm)

    train_records = load_records_from_csv(
        csv_path=args.train_csv,
        prompt_text=args.prompt_text,
        prompt_col=args.prompt_col,
    )
    train_dataset = TweetExtractionCausalDataset(
        records=train_records,
        tokenizer=tokenizer,
        prompt_text=args.prompt_text,
        max_len=args.max_len,
        soft_alpha=args.soft_alpha,
    )

    loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=lambda x: causal_collate_fn(x, tokenizer.pad_token_id),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = max(1, args.epochs * len(loader))
    warmup_steps = int(args.warmup_ratio * total_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    use_amp, amp_dtype, use_grad_scaler = resolve_amp_settings(args, device)
    scaler = torch.amp.GradScaler("cuda", enabled=use_grad_scaler)
    optimizer.zero_grad(set_to_none=True)

    model.train()
    global_step = 0
    for epoch in range(args.epochs):
        ce_only = epoch < args.ce_only_epochs
        lambda_kl = 0.0 if ce_only else args.lambda_kl
        lambda_select = 0.0 if ce_only else args.lambda_select

        running = {"loss": 0.0, "ce": 0.0, "kl": 0.0, "sel": 0.0}

        for step, batch in enumerate(loader):
            batch = move_batch_to_device(batch, device)

            with torch.amp.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                )
                losses = compute_total_loss(
                    outputs=outputs,
                    batch=batch,
                    lambda_kl=lambda_kl,
                    lambda_select=lambda_select,
                )
                loss = losses["loss"] / args.grad_accum_steps

            if use_grad_scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if (step + 1) % args.grad_accum_steps == 0:
                if use_grad_scaler:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                global_step += 1

            running["loss"] += losses["loss"].item()
            running["ce"] += losses["ce_loss"].item()
            running["kl"] += losses["kl_loss"].item()
            running["sel"] += losses["select_loss"].item()

            if (step + 1) % args.log_every == 0:
                denom = float(args.log_every)
                print(
                    f"epoch={epoch + 1} step={step + 1}/{len(loader)} "
                    f"loss={running['loss']/denom:.4f} "
                    f"ce={running['ce']/denom:.4f} "
                    f"kl={running['kl']/denom:.4f} "
                    f"sel={running['sel']/denom:.4f} "
                    f"lambda_kl={lambda_kl:.3f} lambda_select={lambda_select:.3f}"
                )
                running = {"loss": 0.0, "ce": 0.0, "kl": 0.0, "sel": 0.0}

        ckpt_path = os.path.join(args.output_dir, f"checkpoint_epoch_{epoch + 1}.pt")
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "model_name": args.model_name,
                "args": vars(args),
                "global_step": global_step,
            },
            ckpt_path,
        )
        print(f"Saved: {ckpt_path}")

    tokenizer_dir = os.path.join(args.output_dir, "tokenizer")
    tokenizer.save_pretrained(tokenizer_dir)
    print(f"Saved tokenizer to: {tokenizer_dir}")

    with open(os.path.join(args.output_dir, "train_args.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("Causal tweet extraction trainer")
    parser.add_argument("--train-csv", type=str, required=True)
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument(
        "--prompt-text",
        type=str,
        default="You are a professional emotion identifier",
    )
    parser.add_argument("--prompt-col", type=str, default=None)
    parser.add_argument("--max-len", type=int, default=256)
    parser.add_argument("--soft-alpha", type=float, default=0.6)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--lambda-kl", type=float, default=0.3)
    parser.add_argument("--lambda-select", type=float, default=0.2)
    parser.add_argument("--ce-only-epochs", type=int, default=1)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--trust-remote-code", action="store_true", default=False)
    parser.add_argument("--no-amp", action="store_true", default=False)
    parser.add_argument("--amp-dtype", type=str, default="auto", choices=["auto", "fp16", "bf16"])
    return parser
