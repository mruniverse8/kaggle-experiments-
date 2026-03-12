from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
import torch
from transformers import AutoTokenizer

from .dataset import register_special_tokens
from .eval_utils import evaluate_dataframe_jaccard
from .modeling import CausalExtractionModel


def default_compare_config() -> Dict:
    return {
        "eval_csv": "",
        "output_dir": "",
        "model_name": "Qwen/Qwen3.5-0.8B",
        "prompt_text": "You are a professional emotion identifier.",
        "v3_checkpoint": "",
        "v4_checkpoint": "",
        "trust_remote_code": False,
        "max_new_tokens": 16,
        "constrain_to_tweet_span": True,
        "eval_subset_fraction": 1.0,
        "max_eval_rows": 0,
        "device": "cuda:0",
        "seed": 42,
    }


def _resolve_path(value: str, base_dir: Path) -> str:
    p = Path(value).expanduser()
    if not p.is_absolute():
        p = (base_dir / p).resolve()
    return str(p)


def load_compare_config(config_path: str) -> argparse.Namespace:
    path = Path(config_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Comparison config file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        user_cfg = json.load(f)

    if not isinstance(user_cfg, dict):
        raise ValueError("Comparison config JSON must be an object/dictionary.")

    cfg = default_compare_config()
    cfg.update(user_cfg)

    required = ("eval_csv", "output_dir", "v3_checkpoint", "v4_checkpoint")
    for key in required:
        if not cfg.get(key):
            raise ValueError(f"Comparison config requires non-empty '{key}'.")

    base_dir = path.parent
    cfg["eval_csv"] = _resolve_path(cfg["eval_csv"], base_dir)
    cfg["output_dir"] = _resolve_path(cfg["output_dir"], base_dir)
    cfg["v3_checkpoint"] = _resolve_path(cfg["v3_checkpoint"], base_dir)
    cfg["v4_checkpoint"] = _resolve_path(cfg["v4_checkpoint"], base_dir)

    return argparse.Namespace(**cfg)


def _normalize_state_dict_keys(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if not state_dict:
        return state_dict
    keys = list(state_dict.keys())
    if all(k.startswith("module.") for k in keys):
        return {k[len("module.") :]: v for k, v in state_dict.items()}
    return state_dict


def _load_checkpoint_model(
    checkpoint_path: str,
    model_name: str,
    trust_remote_code: bool,
    device: torch.device,
):
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state = ckpt["model_state_dict"]
        model_name = str(ckpt.get("model_name", model_name))
    else:
        state = ckpt

    model = CausalExtractionModel(
        model_name=model_name,
        trust_remote_code=trust_remote_code,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_fast=True,
        trust_remote_code=trust_remote_code,
    )
    register_special_tokens(tokenizer, model=model.lm)

    state = _normalize_state_dict_keys(state)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"[warn] Missing keys while loading {checkpoint_path}: {len(missing)}")
    if unexpected:
        print(f"[warn] Unexpected keys while loading {checkpoint_path}: {len(unexpected)}")

    model.to(device)
    model.eval()
    return model, tokenizer


def _cleanup_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        if hasattr(torch.cuda, "ipc_collect"):
            torch.cuda.ipc_collect()


def evaluate_checkpoint(
    label: str,
    checkpoint_path: str,
    model_name: str,
    trust_remote_code: bool,
    eval_df: pd.DataFrame,
    prompt_text: str,
    device: torch.device,
    max_new_tokens: int,
    constrain_to_tweet_span: bool,
) -> Tuple[Dict[str, object], Dict[str, object]]:
    print(f"\n=== Evaluating {label}: {checkpoint_path} ===")
    model, tokenizer = _load_checkpoint_model(
        checkpoint_path=checkpoint_path,
        model_name=model_name,
        trust_remote_code=trust_remote_code,
        device=device,
    )

    out = evaluate_dataframe_jaccard(
        df=eval_df,
        model=model,
        tokenizer=tokenizer,
        prompt_text=prompt_text,
        device=device,
        max_new_tokens=max_new_tokens,
        constrain_to_tweet_span=constrain_to_tweet_span,
    )

    preds = out["predictions"]
    scores = out["scores"]
    mean_jaccard = float(out["mean_jaccard"])
    by_sent = (
        pd.DataFrame({"sentiment": eval_df["sentiment"], "score": scores})
        .groupby("sentiment", as_index=False)["score"]
        .mean()
        .sort_values("sentiment")
    )

    metrics = {
        "label": label,
        "checkpoint": checkpoint_path,
        "rows": int(len(eval_df)),
        "mean_jaccard": mean_jaccard,
        "by_sentiment": [
            {"sentiment": str(r.sentiment), "mean_jaccard": float(r.score)}
            for r in by_sent.itertuples(index=False)
        ],
    }
    artifacts = {"predictions": preds, "scores": scores}
    print(f"{label} mean_jaccard={mean_jaccard:.6f}")

    del model, tokenizer
    _cleanup_cuda()
    return metrics, artifacts


def run_comparison(args: argparse.Namespace) -> Dict[str, object]:
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("Config requests CUDA device but CUDA is not available.")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    df = pd.read_csv(args.eval_csv).dropna(subset=["text", "sentiment", "selected_text"]).reset_index(drop=True)
    eval_subset_fraction = float(getattr(args, "eval_subset_fraction", 1.0))
    if not (0.0 < eval_subset_fraction <= 1.0):
        raise ValueError("Config value 'eval_subset_fraction' must be in the interval (0, 1].")

    if eval_subset_fraction < 1.0:
        subset_rows = max(1, int(round(len(df) * eval_subset_fraction)))
        df = df.sample(n=subset_rows, random_state=int(args.seed)).reset_index(drop=True)
        print(f"Eval subset enabled: fraction={eval_subset_fraction:.3f} rows={len(df)}")

    if int(args.max_eval_rows) > 0 and int(args.max_eval_rows) < len(df):
        df = df.sample(n=int(args.max_eval_rows), random_state=int(args.seed)).reset_index(drop=True)
    print(f"Eval rows: {len(df)} | constrain_to_tweet_span={bool(args.constrain_to_tweet_span)}")

    v3_metrics, v3_artifacts = evaluate_checkpoint(
        label="v3_ce",
        checkpoint_path=args.v3_checkpoint,
        model_name=args.model_name,
        trust_remote_code=bool(args.trust_remote_code),
        eval_df=df,
        prompt_text=args.prompt_text,
        device=device,
        max_new_tokens=int(args.max_new_tokens),
        constrain_to_tweet_span=bool(args.constrain_to_tweet_span),
    )
    v4_metrics, v4_artifacts = evaluate_checkpoint(
        label="v4_kl_jaccard",
        checkpoint_path=args.v4_checkpoint,
        model_name=args.model_name,
        trust_remote_code=bool(args.trust_remote_code),
        eval_df=df,
        prompt_text=args.prompt_text,
        device=device,
        max_new_tokens=int(args.max_new_tokens),
        constrain_to_tweet_span=bool(args.constrain_to_tweet_span),
    )

    compare_df = df[["text", "sentiment", "selected_text"]].copy()
    compare_df["pred_v3"] = v3_artifacts["predictions"]
    compare_df["pred_v4"] = v4_artifacts["predictions"]
    compare_df["jaccard_v3"] = v3_artifacts["scores"]
    compare_df["jaccard_v4"] = v4_artifacts["scores"]
    compare_df["delta_v4_minus_v3"] = compare_df["jaccard_v4"] - compare_df["jaccard_v3"]
    compare_df["winner"] = compare_df["delta_v4_minus_v3"].apply(
        lambda x: "v4" if x > 0 else ("v3" if x < 0 else "tie")
    )

    by_sent_compare = (
        compare_df.groupby("sentiment", as_index=False)[["jaccard_v3", "jaccard_v4", "delta_v4_minus_v3"]]
        .mean()
        .sort_values("sentiment")
    )

    summary = {
        "rows": int(len(compare_df)),
        "mean_jaccard_v3": float(compare_df["jaccard_v3"].mean()),
        "mean_jaccard_v4": float(compare_df["jaccard_v4"].mean()),
        "delta_v4_minus_v3": float(compare_df["delta_v4_minus_v3"].mean()),
        "winner_counts": compare_df["winner"].value_counts().to_dict(),
        "by_sentiment": [
            {
                "sentiment": str(r.sentiment),
                "jaccard_v3": float(r.jaccard_v3),
                "jaccard_v4": float(r.jaccard_v4),
                "delta_v4_minus_v3": float(r.delta_v4_minus_v3),
            }
            for r in by_sent_compare.itertuples(index=False)
        ],
    }

    out_dir = Path(args.output_dir)
    (out_dir / "v3_metrics.json").write_text(json.dumps(v3_metrics, indent=2))
    (out_dir / "v4_metrics.json").write_text(json.dumps(v4_metrics, indent=2))
    (out_dir / "comparison_summary.json").write_text(json.dumps(summary, indent=2))
    compare_df.to_csv(out_dir / "v3_vs_v4_predictions.csv", index=False)

    print("\nSaved:")
    print(out_dir / "v3_metrics.json")
    print(out_dir / "v4_metrics.json")
    print(out_dir / "comparison_summary.json")
    print(out_dir / "v3_vs_v4_predictions.csv")

    return {
        "v3_metrics": v3_metrics,
        "v4_metrics": v4_metrics,
        "summary": summary,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("Compare v3 vs v4 checkpoints with Jaccard metrics")
    parser.add_argument("--config", type=str, required=True, help="Path to comparison JSON config")
    return parser


def main() -> None:
    parser = build_arg_parser()
    cli_args = parser.parse_args()
    args = load_compare_config(cli_args.config)
    run_comparison(args)


if __name__ == "__main__":
    main()
