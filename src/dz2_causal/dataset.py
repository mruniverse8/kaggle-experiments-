from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class SpecialTokens:
    prompt_open: str = "<PROMPT>"
    prompt_close: str = "</PROMPT>"
    tweet_open: str = "<TWEET>"
    tweet_close: str = "</TWEET>"
    sentiment_open: str = "<SENTIMENT>"
    sentiment_close: str = "</SENTIMENT>"
    answer_open: str = "<ANSWER>"
    answer_close: str = "</ANSWER>"

    def as_list(self) -> List[str]:
        return [
            self.prompt_open,
            self.prompt_close,
            self.tweet_open,
            self.tweet_close,
            self.sentiment_open,
            self.sentiment_close,
            self.answer_open,
            self.answer_close,
        ]


SPECIAL_TOKENS = SpecialTokens()


def normalize_text(text: str) -> str:
    return " ".join(str(text).split())


def normalize_kaggle_span_text(text: str) -> str:
    # Match the original Tweet Sentiment pipeline convention.
    return " " + normalize_text(text)


def register_special_tokens(tokenizer, model=None) -> int:
    added = tokenizer.add_special_tokens(
        {"additional_special_tokens": SPECIAL_TOKENS.as_list()}
    )
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is None:
            raise ValueError("Tokenizer must define eos_token or pad_token.")
        tokenizer.pad_token = tokenizer.eos_token
    if model is not None and added > 0:
        model.resize_token_embeddings(len(tokenizer))
    return added


def jaccard_array(a: Iterable[int], b: Iterable[int]) -> float:
    a_set = set(a)
    b_set = set(b)
    inter = a_set.intersection(b_set)
    denom = len(a_set) + len(b_set) - len(inter)
    return float(len(inter)) / float(denom + 1e-8)


def build_soft_span_labels(n_tokens: int, start_idx: int, end_idx: int, alpha: float) -> Tuple[np.ndarray, np.ndarray]:
    sentence = np.arange(n_tokens)
    answer = sentence[start_idx : end_idx + 1]

    start_labels = np.zeros(n_tokens, dtype=np.float32)
    for i in range(end_idx + 1):
        jac = jaccard_array(answer, sentence[i : end_idx + 1])
        start_labels[i] = jac + jac**2
    start_labels = (1.0 - alpha) * start_labels / (start_labels.sum() + 1e-8)
    start_labels[start_idx] += alpha

    end_labels = np.zeros(n_tokens, dtype=np.float32)
    for i in range(start_idx, n_tokens):
        jac = jaccard_array(answer, sentence[start_idx : i + 1])
        end_labels[i] = jac + jac**2
    end_labels = (1.0 - alpha) * end_labels / (end_labels.sum() + 1e-8)
    end_labels[end_idx] += alpha

    return start_labels, end_labels


def find_char_span(tweet: str, selected_text: str) -> Tuple[int, int]:
    # Closely follows the original roberta_base dataset behavior:
    # tweet = " " + normalized_tweet, selected_text = " " + normalized_selected_text
    tweet = str(tweet)
    selected_text = str(selected_text)
    if len(selected_text) <= 1:
        return 0, max(1, len(tweet))

    len_sel_text = len(selected_text) - 1
    idx_0 = None
    idx_1 = None

    first_char = selected_text[1]
    for ind, ch in enumerate(tweet):
        if ch != first_char:
            continue
        if " " + tweet[ind : ind + len_sel_text] == selected_text:
            idx_0 = ind
            idx_1 = ind + len_sel_text - 1
            break

    if idx_0 is not None and idx_1 is not None:
        return idx_0, idx_1 + 1

    # Fallback for rare normalization mismatches.
    stripped = selected_text.strip()
    start = tweet.find(stripped)
    if start >= 0:
        return start, start + len(stripped)

    start = tweet.lower().find(stripped.lower())
    if start >= 0:
        return start, start + len(stripped)

    return 0, max(1, len(tweet))


def token_overlap(offset: Tuple[int, int], span: Tuple[int, int]) -> bool:
    o0, o1 = offset
    s0, s1 = span
    return max(o0, s0) < min(o1, s1)


class TweetExtractionCausalDataset(Dataset):
    def __init__(
        self,
        records: List[Dict[str, str]],
        tokenizer,
        prompt_text: str,
        max_len: int = 256,
        soft_alpha: float = 0.6,
    ):
        self.records = records
        self.tokenizer = tokenizer
        self.prompt_text = prompt_text
        self.max_len = max_len
        self.soft_alpha = soft_alpha

        self.token_ids = {
            "tweet_open": tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS.tweet_open),
            "tweet_close": tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS.tweet_close),
        }
        for name, token_id in self.token_ids.items():
            if token_id == tokenizer.unk_token_id:
                raise ValueError(
                    f"Special token for {name} unresolved. Call register_special_tokens first."
                )

    def __len__(self) -> int:
        return len(self.records)

    def _build_prompt(self, prompt: str, tweet: str, sentiment: str) -> str:
        return (
            f"{SPECIAL_TOKENS.prompt_open} {prompt} {SPECIAL_TOKENS.prompt_close} "
            f"{SPECIAL_TOKENS.tweet_open}{tweet} {SPECIAL_TOKENS.tweet_close} "
            f"{SPECIAL_TOKENS.sentiment_open} {sentiment} {SPECIAL_TOKENS.sentiment_close} "
            f"{SPECIAL_TOKENS.answer_open}"
        )

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.records[idx]
        prompt = normalize_text(row.get("prompt", self.prompt_text))
        tweet = normalize_kaggle_span_text(row["text"])
        sentiment = normalize_text(row["sentiment"])
        selected_text = normalize_kaggle_span_text(row["selected_text"])

        prefix_text = self._build_prompt(prompt, tweet, sentiment)
        full_text = f"{prefix_text}{selected_text} {SPECIAL_TOKENS.answer_close}"

        prefix_ids = self.tokenizer(prefix_text, add_special_tokens=False)["input_ids"]
        input_ids = self.tokenizer(full_text, add_special_tokens=False)["input_ids"]

        bos_id = self.tokenizer.bos_token_id
        eos_id = self.tokenizer.eos_token_id
        if bos_id is not None:
            prefix_ids = [bos_id] + prefix_ids
            input_ids = [bos_id] + input_ids
        if eos_id is not None:
            input_ids = input_ids + [eos_id]

        if len(input_ids) > self.max_len:
            raise ValueError(
                f"Sample {idx} length {len(input_ids)} exceeds max_len={self.max_len}."
            )

        labels = np.full(len(input_ids), -100, dtype=np.int64)
        answer_start = len(prefix_ids)
        labels[answer_start:] = np.array(input_ids[answer_start:], dtype=np.int64)

        attention_mask = np.ones(len(input_ids), dtype=np.int64)
        source_mask = np.zeros(len(input_ids), dtype=np.float32)
        start_soft = np.zeros(len(input_ids), dtype=np.float32)
        end_soft = np.zeros(len(input_ids), dtype=np.float32)
        select_targets = np.zeros(len(input_ids), dtype=np.float32)
        offsets = np.zeros((len(input_ids), 2), dtype=np.int64)

        tweet_open_id = self.token_ids["tweet_open"]
        tweet_close_id = self.token_ids["tweet_close"]
        tweet_open_idx = input_ids.index(tweet_open_id)
        tweet_close_idx = input_ids.index(tweet_close_id, tweet_open_idx + 1)
        tweet_global_positions = list(range(tweet_open_idx + 1, tweet_close_idx))

        tweet_encoding = self.tokenizer(
            tweet,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
        tweet_offsets = tweet_encoding["offset_mapping"]

        effective_len = min(len(tweet_global_positions), len(tweet_offsets))
        tweet_global_positions = tweet_global_positions[:effective_len]
        tweet_offsets = tweet_offsets[:effective_len]
        for p in tweet_global_positions:
            source_mask[p] = 1.0

        char_span = find_char_span(tweet, selected_text)
        target_local_ids = [
            i for i, offset in enumerate(tweet_offsets) if token_overlap(offset, char_span)
        ]

        if not target_local_ids:
            target_local_ids = [0]

        local_start = target_local_ids[0]
        local_end = target_local_ids[-1]
        local_start_soft, local_end_soft = build_soft_span_labels(
            n_tokens=effective_len,
            start_idx=local_start,
            end_idx=local_end,
            alpha=self.soft_alpha,
        )

        selected_set = set(target_local_ids)
        for local_idx, global_pos in enumerate(tweet_global_positions):
            start_soft[global_pos] = local_start_soft[local_idx]
            end_soft[global_pos] = local_end_soft[local_idx]
            offsets[global_pos, 0] = int(tweet_offsets[local_idx][0])
            offsets[global_pos, 1] = int(tweet_offsets[local_idx][1])
            if local_idx in selected_set:
                select_targets[global_pos] = 1.0

        orig_start = tweet_global_positions[0] if tweet_global_positions else 0
        orig_end = tweet_global_positions[-1] if tweet_global_positions else 0

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "source_mask": torch.tensor(source_mask, dtype=torch.float),
            "start_soft": torch.tensor(start_soft, dtype=torch.float),
            "end_soft": torch.tensor(end_soft, dtype=torch.float),
            "select_targets": torch.tensor(select_targets, dtype=torch.float),
            "orig_start": orig_start,
            "orig_end": orig_end,
            "offsets": torch.tensor(offsets, dtype=torch.long),
            "prompt": prompt,
            "tweet": tweet,
            "sentiment": sentiment,
            "selected_text": selected_text,
        }


def load_records_from_csv(
    csv_path: str,
    prompt_text: str,
    prompt_col: str | None = None,
    drop_empty: bool = True,
) -> List[Dict[str, str]]:
    df = pd.read_csv(csv_path).fillna("")
    required = {"text", "sentiment", "selected_text"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {csv_path}: {sorted(missing)}")

    records: List[Dict[str, str]] = []
    for row in df.to_dict("records"):
        if drop_empty:
            if normalize_text(row["text"]) == "" or normalize_text(row["selected_text"]) == "":
                continue
        prompt = row.get(prompt_col, prompt_text) if prompt_col else prompt_text
        records.append(
            {
                "prompt": str(prompt),
                "text": str(row["text"]),
                "sentiment": str(row["sentiment"]),
                "selected_text": str(row["selected_text"]),
            }
        )
    return records


def causal_collate_fn(batch: List[Dict[str, torch.Tensor]], pad_token_id: int) -> Dict[str, torch.Tensor]:
    max_len = max(item["input_ids"].size(0) for item in batch)

    def pad_1d(x: torch.Tensor, pad_value: float) -> torch.Tensor:
        if x.size(0) == max_len:
            return x
        pad_len = max_len - x.size(0)
        pad = torch.full((pad_len,), pad_value, dtype=x.dtype)
        return torch.cat([x, pad], dim=0)

    def pad_offsets(x: torch.Tensor) -> torch.Tensor:
        if x.size(0) == max_len:
            return x
        pad_len = max_len - x.size(0)
        pad = torch.zeros((pad_len, 2), dtype=x.dtype)
        return torch.cat([x, pad], dim=0)

    out: Dict[str, List] = {
        "input_ids": [],
        "attention_mask": [],
        "labels": [],
        "source_mask": [],
        "start_soft": [],
        "end_soft": [],
        "select_targets": [],
        "offsets": [],
        "orig_start": [],
        "orig_end": [],
        "tweet": [],
        "selected_text": [],
        "sentiment": [],
        "prompt": [],
    }

    for item in batch:
        out["input_ids"].append(pad_1d(item["input_ids"], pad_token_id))
        out["attention_mask"].append(pad_1d(item["attention_mask"], 0))
        out["labels"].append(pad_1d(item["labels"], -100))
        out["source_mask"].append(pad_1d(item["source_mask"], 0.0))
        out["start_soft"].append(pad_1d(item["start_soft"], 0.0))
        out["end_soft"].append(pad_1d(item["end_soft"], 0.0))
        out["select_targets"].append(pad_1d(item["select_targets"], 0.0))
        out["offsets"].append(pad_offsets(item["offsets"]))
        out["orig_start"].append(item["orig_start"])
        out["orig_end"].append(item["orig_end"])
        out["tweet"].append(item["tweet"])
        out["selected_text"].append(item["selected_text"])
        out["sentiment"].append(item["sentiment"])
        out["prompt"].append(item["prompt"])

    return {
        "input_ids": torch.stack(out["input_ids"], dim=0),
        "attention_mask": torch.stack(out["attention_mask"], dim=0),
        "labels": torch.stack(out["labels"], dim=0),
        "source_mask": torch.stack(out["source_mask"], dim=0),
        "start_soft": torch.stack(out["start_soft"], dim=0),
        "end_soft": torch.stack(out["end_soft"], dim=0),
        "select_targets": torch.stack(out["select_targets"], dim=0),
        "offsets": torch.stack(out["offsets"], dim=0),
        "orig_start": torch.tensor(out["orig_start"], dtype=torch.long),
        "orig_end": torch.tensor(out["orig_end"], dtype=torch.long),
        "tweet": out["tweet"],
        "selected_text": out["selected_text"],
        "sentiment": out["sentiment"],
        "prompt": out["prompt"],
    }
