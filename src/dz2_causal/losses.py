from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F


def span_kl_loss(
    start_logits: torch.Tensor,
    end_logits: torch.Tensor,
    start_soft: torch.Tensor,
    end_soft: torch.Tensor,
    source_mask: torch.Tensor,
) -> torch.Tensor:
    neg = torch.tensor(-1e4, dtype=start_logits.dtype, device=start_logits.device)
    masked_start = torch.where(source_mask > 0, start_logits, neg)
    masked_end = torch.where(source_mask > 0, end_logits, neg)

    start_log_probs = F.log_softmax(masked_start, dim=-1)
    end_log_probs = F.log_softmax(masked_end, dim=-1)

    l_start = F.kl_div(start_log_probs, start_soft, reduction="batchmean")
    l_end = F.kl_div(end_log_probs, end_soft, reduction="batchmean")
    return l_start + l_end


def bce_dice_select_loss(
    select_logits: torch.Tensor,
    select_targets: torch.Tensor,
    source_mask: torch.Tensor,
) -> torch.Tensor:
    m = source_mask.float()
    bce = F.binary_cross_entropy_with_logits(
        select_logits, select_targets, reduction="none"
    )
    bce = (bce * m).sum() / (m.sum() + 1e-8)

    probs = torch.sigmoid(select_logits) * m
    targets = select_targets * m
    inter = (probs * targets).sum(dim=1)
    denom = probs.sum(dim=1) + targets.sum(dim=1)
    dice = 1.0 - ((2.0 * inter + 1e-8) / (denom + 1e-8)).mean()
    return bce + dice


def compute_total_loss(
    outputs: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
    lambda_kl: float,
    lambda_select: float,
) -> Dict[str, torch.Tensor]:
    ce_loss = outputs["ce_loss"]
    total = ce_loss

    kl_loss = torch.tensor(0.0, device=ce_loss.device, dtype=ce_loss.dtype)
    select_loss = torch.tensor(0.0, device=ce_loss.device, dtype=ce_loss.dtype)

    if lambda_kl > 0.0:
        kl_loss = span_kl_loss(
            start_logits=outputs["start_logits"],
            end_logits=outputs["end_logits"],
            start_soft=batch["start_soft"],
            end_soft=batch["end_soft"],
            source_mask=batch["source_mask"],
        )
        total = total + lambda_kl * kl_loss

    if lambda_select > 0.0:
        select_loss = bce_dice_select_loss(
            select_logits=outputs["select_logits"],
            select_targets=batch["select_targets"],
            source_mask=batch["source_mask"],
        )
        total = total + lambda_select * select_loss

    return {
        "loss": total,
        "ce_loss": ce_loss.detach(),
        "kl_loss": kl_loss.detach(),
        "select_loss": select_loss.detach(),
    }

