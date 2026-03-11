from __future__ import annotations

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


@torch.no_grad()
def predict_selected_text(
    model,
    tokenizer,
    prompt: str,
    tweet: str,
    sentiment: str,
    device: torch.device,
    max_new_tokens: int = 24,
) -> str:
    prompt_text = build_prompt_text(prompt, tweet, sentiment)
    encoded = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False)
    encoded = {k: v.to(device) for k, v in encoded.items()}
    prompt_len = encoded["input_ids"].size(1)

    out = model.lm.generate(
        **encoded,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        num_beams=1,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    return decode_generated_answer(tokenizer, out[0], prompt_len)


@torch.no_grad()
def evaluate_dataframe_jaccard(
    df: pd.DataFrame,
    model,
    tokenizer,
    prompt_text: str,
    device: torch.device,
    max_new_tokens: int = 24,
) -> Dict[str, object]:
    preds: List[str] = []
    scores: List[float] = []
    for row in df.itertuples(index=False):
        pred = predict_selected_text(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt_text,
            tweet=str(row.text),
            sentiment=str(row.sentiment),
            device=device,
            max_new_tokens=max_new_tokens,
        )
        target = str(row.selected_text).strip()
        score = word_jaccard(target, pred)
        preds.append(pred)
        scores.append(score)

    mean_jaccard = float(sum(scores) / max(len(scores), 1))
    return {"mean_jaccard": mean_jaccard, "predictions": preds, "scores": scores}

