from __future__ import annotations

from contextlib import contextmanager, nullcontext
from typing import Dict, List

import pandas as pd
import torch

from .dataset import SPECIAL_TOKENS, normalize_kaggle_span_text


def word_jaccard(str1: str, str2: str) -> float:
    a = set(str(str1).lower().split())
    b = set(str(str2).lower().split())
    c = a.intersection(b)
    denom = len(a) + len(b) - len(c)
    return float(len(c)) / float(denom + 1e-8)


def _normalize_ws(text: str) -> str:
    return " ".join(str(text).split())


def best_tweet_span_by_jaccard(tweet: str, generated_text: str) -> str:
    """
    Project generated output onto a contiguous span from tweet words.
    This aligns inference with the competition metric, which is word-level Jaccard.
    """
    tweet_norm = _normalize_ws(tweet)
    pred_norm = _normalize_ws(generated_text)

    if not tweet_norm:
        return pred_norm

    tweet_words = tweet_norm.split()
    pred_words = pred_norm.split()

    if not pred_words:
        return tweet_norm

    pred_len = len(pred_words)
    best_text = tweet_words[0]
    best_rank = (-1.0, float("-inf"), float("-inf"), float("-inf"))

    for i in range(len(tweet_words)):
        for j in range(i, len(tweet_words)):
            candidate_words = tweet_words[i : j + 1]
            candidate_text = " ".join(candidate_words)
            score = word_jaccard(pred_norm, candidate_text)
            len_gap = abs(len(candidate_words) - pred_len)
            # Rank priority: higher Jaccard, closer length to prediction,
            # shorter span, earlier position.
            rank = (score, -float(len_gap), -float(len(candidate_words)), -float(i))
            if rank > best_rank:
                best_rank = rank
                best_text = candidate_text

    return best_text


def build_prompt_text(prompt: str, tweet: str, sentiment: str) -> str:
    tweet_norm = normalize_kaggle_span_text(tweet)
    return (
        f"{SPECIAL_TOKENS.prompt_open} {prompt} {SPECIAL_TOKENS.prompt_close} "
        f"{SPECIAL_TOKENS.tweet_open}{tweet_norm} {SPECIAL_TOKENS.tweet_close} "
        f"{SPECIAL_TOKENS.sentiment_open} {sentiment} {SPECIAL_TOKENS.sentiment_close} "
        f"{SPECIAL_TOKENS.answer_open}"
    )


def decode_generated_answer(
    tokenizer,
    generated_ids: torch.Tensor,
    prompt_len: int,
) -> str:
    continuation_ids = generated_ids[prompt_len:]
    text = tokenizer.decode(continuation_ids, skip_special_tokens=False)
    if SPECIAL_TOKENS.answer_close in text:
        text = text.split(SPECIAL_TOKENS.answer_close, 1)[0]
    return text.strip()


def _resolve_lm_module(model):
    return getattr(model, "lm", model)


@contextmanager
def _generation_mode(model):
    """
    Ensure generation runs in eval mode with cache enabled and without
    gradient-checkpointing side effects.
    """
    lm = _resolve_lm_module(model)
    cfg = getattr(lm, "config", None)

    prev_training = bool(getattr(lm, "training", False))
    prev_use_cache = getattr(cfg, "use_cache", None) if cfg is not None else None
    prev_gc = bool(getattr(lm, "is_gradient_checkpointing", False))

    if hasattr(model, "eval"):
        model.eval()
    if prev_gc and hasattr(lm, "gradient_checkpointing_disable"):
        lm.gradient_checkpointing_disable()
    if prev_use_cache is not None:
        cfg.use_cache = True

    try:
        yield
    finally:
        if prev_use_cache is not None:
            cfg.use_cache = prev_use_cache
        if prev_gc and hasattr(lm, "gradient_checkpointing_enable"):
            lm.gradient_checkpointing_enable()
        if prev_training and hasattr(model, "train"):
            model.train()


@torch.no_grad()
def predict_selected_text(
    model,
    tokenizer,
    prompt: str,
    tweet: str,
    sentiment: str,
    device: torch.device,
    max_new_tokens: int = 24,
    constrain_to_tweet_span: bool = True,
) -> str:
    prompt_text = build_prompt_text(prompt, tweet, sentiment)
    encoded = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False)
    encoded = {k: v.to(device) for k, v in encoded.items()}
    prompt_len = encoded["input_ids"].size(1)

    lm = _resolve_lm_module(model)
    amp_ctx = (
        torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=True)
        if device.type == "cuda"
        else nullcontext()
    )
    with amp_ctx:
        out = lm.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )
    pred = decode_generated_answer(tokenizer, out[0], prompt_len)
    if constrain_to_tweet_span:
        return best_tweet_span_by_jaccard(tweet=tweet, generated_text=pred)
    return pred


@torch.no_grad()
def evaluate_dataframe_jaccard(
    df: pd.DataFrame,
    model,
    tokenizer,
    prompt_text: str,
    device: torch.device,
    max_new_tokens: int = 24,
    constrain_to_tweet_span: bool = True,
) -> Dict[str, object]:
    preds: List[str] = []
    scores: List[float] = []
    with _generation_mode(model):
        for row in df.itertuples(index=False):
            pred = predict_selected_text(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt_text,
                tweet=str(row.text),
                sentiment=str(row.sentiment),
                device=device,
                max_new_tokens=max_new_tokens,
                constrain_to_tweet_span=constrain_to_tweet_span,
            )
            target = str(row.selected_text).strip()
            score = word_jaccard(target, pred)
            preds.append(pred)
            scores.append(score)

    mean_jaccard = float(sum(scores) / max(len(scores), 1))
    return {"mean_jaccard": mean_jaccard, "predictions": preds, "scores": scores}
