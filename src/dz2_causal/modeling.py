from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM


def _get_hidden_size(config) -> int:
    for key in ("hidden_size", "n_embd", "d_model"):
        if hasattr(config, key):
            return int(getattr(config, key))
    raise ValueError("Unable to infer hidden size from model config.")


class CausalExtractionModel(nn.Module):
    def __init__(self, model_name: str, trust_remote_code: bool = True):
        super().__init__()
        self.lm = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code,
        )
        hidden_size = _get_hidden_size(self.lm.config)
        self.start_head = nn.Linear(hidden_size, 1)
        self.end_head = nn.Linear(hidden_size, 1)
        self.select_head = nn.Linear(hidden_size, 1)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        compute_span_logits: bool = True,
    ) -> Dict[str, torch.Tensor]:
        outputs = self.lm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=compute_span_logits,
            use_cache=False,
            return_dict=True,
        )
        start_logits = None
        end_logits = None
        select_logits = None

        if compute_span_logits:
            hidden = outputs.hidden_states[-1]

            # Some base checkpoints run internal matmuls in bf16 while these heads
            # are initialized in fp32. Align activations to head dtype to avoid
            # matmul dtype mismatch errors in mixed/no-amp runs.
            head_dtype = self.start_head.weight.dtype
            if hidden.dtype != head_dtype:
                hidden = hidden.to(head_dtype)

            start_logits = self.start_head(hidden).squeeze(-1)
            end_logits = self.end_head(hidden).squeeze(-1)
            select_logits = self.select_head(hidden).squeeze(-1)

        return {
            "ce_loss": outputs.loss,
            "start_logits": start_logits,
            "end_logits": end_logits,
            "select_logits": select_logits,
        }
