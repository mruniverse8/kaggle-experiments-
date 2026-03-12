from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F


def _reduce_dp_scalar(x: torch.Tensor) -> torch.Tensor:
    """
    DataParallel gathers scalar outputs from each device into a 1D tensor.
    Reduce them to a true scalar for stable backward() and .item() calls.
    """
    if torch.is_tensor(x) and x.ndim > 0:
        return x.mean()
    return x


def span_kl_loss(
    start_logits: torch.Tensor,
    end_logits: torch.Tensor,
    start_soft: torch.Tensor,
    end_soft: torch.Tensor,
    source_mask: torch.Tensor,
) -> torch.Tensor:
    # Keep all span-loss tensors in the same compute dtype to avoid
    # inadvertent fp32/bf16 promotions during no-AMP training.
    start_soft = start_soft.to(start_logits.dtype)
    end_soft = end_soft.to(end_logits.dtype)
    source_mask = source_mask.to(start_logits.dtype)

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
    # Match dtype with logits to avoid explicit fp32 upcasts.
    m = source_mask.to(select_logits.dtype)
    select_targets = select_targets.to(select_logits.dtype)
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
    ce_loss = _reduce_dp_scalar(outputs["ce_loss"])
    total = ce_loss

    kl_loss = torch.tensor(0.0, device=ce_loss.device, dtype=ce_loss.dtype)
    select_loss = torch.tensor(0.0, device=ce_loss.device, dtype=ce_loss.dtype)

    if lambda_kl > 0.0:
        if outputs.get("start_logits") is None or outputs.get("end_logits") is None:
            raise ValueError("KL loss requested but span logits are missing from model outputs.")
        kl_loss = span_kl_loss(
            start_logits=outputs["start_logits"],
            end_logits=outputs["end_logits"],
            start_soft=batch["start_soft"],
            end_soft=batch["end_soft"],
            source_mask=batch["source_mask"],
        )
        kl_loss = _reduce_dp_scalar(kl_loss)
        total = total + lambda_kl * kl_loss

    if lambda_select > 0.0:
        if outputs.get("select_logits") is None:
            raise ValueError("Select loss requested but select logits are missing from model outputs.")
        select_loss = bce_dice_select_loss(
            select_logits=outputs["select_logits"],
            select_targets=batch["select_targets"],
            source_mask=batch["source_mask"],
        )
        select_loss = _reduce_dp_scalar(select_loss)
        total = total + lambda_select * select_loss

    return {
        "loss": total,
        "ce_loss": ce_loss.detach(),
        "kl_loss": kl_loss.detach(),
        "select_loss": select_loss.detach(),
    }
