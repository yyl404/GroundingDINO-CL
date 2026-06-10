"""
Loss functions for :class:`finetune.models.GroundingDINOWrapper`.

This module implements a DETR-style criterion:
- Hungarian matching between queries and GT boxes
- classification loss on all queries
- box L1 + GIoU losses on matched queries
"""

from typing import Dict, List, Tuple

from scipy.optimize import linear_sum_assignment

import torch
from torch import Tensor
import torch.nn.functional as F

from groundingdino.models.GroundingDINO.utils import sigmoid_focal_loss
from groundingdino.util.box_ops import (
    box_cxcywh_to_xyxy,
    generalized_box_iou,
    generalized_box_iou_pairwise,
)


def _batch_num_boxes(targets: List[Dict[str, Tensor]]) -> int:
    """
    Count the sum up GT boxes in the current batch.
    """
    return max(sum(int(t["labels"].numel()) for t in targets), 1)


def _ensure_prob(x: Tensor) -> Tensor:
    """Accept either logits or probabilities and return probabilities."""
    if x.numel() == 0:
        return x
    if float(x.min()) >= 0.0 and float(x.max()) <= 1.0:
        return x.clamp(min=1e-6, max=1.0 - 1e-6)
    return x.sigmoid()


def _ensure_logit(x: Tensor) -> Tensor:
    """Accept either logits or probabilities and return logits."""
    if x.numel() == 0:
        return x
    if float(x.min()) >= 0.0 and float(x.max()) <= 1.0:
        x = x.clamp(min=1e-6, max=1.0 - 1e-6)
        return torch.logit(x)
    return x


def _hungarian_cost_matrix(
    pred_cls: Tensor,
    pred_boxes: Tensor,
    tgt_labels: Tensor,
    tgt_boxes: Tensor,
    cost_class: float,
    cost_bbox: float,
    cost_giou: float,
) -> Tensor:
    """
    Args:
        pred_cls: [Q, C], classification predictions (logits or probabilities).
        pred_boxes: [Q, 4], cxcywh normalized.
        tgt_labels: [N]
        tgt_boxes: [N, 4], cxcywh normalized.
    Returns:
        Cost matrix [Q, N].
    """
    if tgt_labels.numel() == 0:
        raise ValueError("Empty targets should use the no-GT branch.")

    pred_prob = _ensure_prob(pred_cls)
    class_cost = -pred_prob[:, tgt_labels.long()]
    bbox_cost = torch.cdist(pred_boxes, tgt_boxes, p=1)

    pred_xyxy = box_cxcywh_to_xyxy(pred_boxes)
    tgt_xyxy = box_cxcywh_to_xyxy(tgt_boxes)
    giou = generalized_box_iou(pred_xyxy, tgt_xyxy)
    giou_cost = 1.0 - giou

    return cost_class * class_cost + cost_bbox * bbox_cost + cost_giou * giou_cost


def _hungarian_match_single(
    pred_cls: Tensor,
    pred_boxes: Tensor,
    tgt_labels: Tensor,
    tgt_boxes: Tensor,
    cost_class: float,
    cost_bbox: float,
    cost_giou: float,
) -> Tuple[Tensor, Tensor]:
    cost = _hungarian_cost_matrix(
        pred_cls,
        pred_boxes,
        tgt_labels,
        tgt_boxes,
        cost_class,
        cost_bbox,
        cost_giou,
    )
    row_ind, col_ind = linear_sum_assignment(cost.detach().cpu().numpy())
    row_ind = torch.as_tensor(row_ind, dtype=torch.long, device=pred_cls.device)
    col_ind = torch.as_tensor(col_ind, dtype=torch.long, device=pred_cls.device)
    return row_ind, col_ind


def wrapper_loss(
    outputs: Dict[str, Tensor],
    targets: List[Dict[str, Tensor]],
    cost_class: float = 2.0,
    cost_bbox: float = 5.0,
    cost_giou: float = 2.0,
    loss_cls_weight: float = 1.0,
    loss_bbox_weight: float = 5.0,
    loss_giou_weight: float = 2.0,
    focal_alpha: float = 0.25,
    focal_gamma: float = 2.0,
) -> Dict[str, Tensor]:
    """
    DETR-style one-stage loss for wrapper outputs.

    Required keys in ``outputs``:
    - ``pred_boxes``: [B, Q, 4], cxcywh
    - ``pred_class_logits`` (preferred) or ``pred_logits``: [B, Q, C]
    """
    pred_cls = outputs["pred_class_logits"]
    pred_boxes = outputs["pred_boxes"]

    if pred_cls.dim() != 3 or pred_boxes.dim() != 3:
        raise ValueError(
            f"Expected 3D tensors, got pred_cls={tuple(pred_cls.shape)}, "
            f"pred_boxes={tuple(pred_boxes.shape)}"
        )
    if pred_cls.shape[:2] != pred_boxes.shape[:2]:
        raise ValueError(
            f"Shape mismatch: pred_cls={tuple(pred_cls.shape)}, "
            f"pred_boxes={tuple(pred_boxes.shape)}"
        )

    device = pred_cls.device
    B, Q, C = pred_cls.shape
    if len(targets) != B:
        raise ValueError(f"Batch mismatch: len(targets)={len(targets)} but batch={B}.")

    num_boxes = _batch_num_boxes(targets)
    tgt_cls = torch.zeros(B, Q, C, device=device, dtype=pred_cls.dtype)
    loss_bbox_list: List[Tensor] = []
    loss_giou_list: List[Tensor] = []

    for b in range(B):
        pred_cls_b = pred_cls[b]
        pred_box_b = pred_boxes[b]
        labels = targets[b]["labels"].long()
        boxes = targets[b]["boxes"]
        n = int(labels.numel())

        if n > Q:
            raise ValueError(
                f"batch {b}: num_gt ({n}) cannot exceed num_queries ({Q}) for Hungarian matching."
            )
        if n > 0:
            max_label = int(labels.max().item())
            if max_label >= C or int(labels.min().item()) < 0:
                raise ValueError(
                    f"batch {b}: label out of range [0, {C - 1}], got min={int(labels.min().item())}, "
                    f"max={max_label}."
                )

        if n == 0:
            loss_bbox_list.append(pred_box_b.sum() * 0.0)
            loss_giou_list.append(pred_box_b.sum() * 0.0)
            continue

        matched_q, matched_gt = _hungarian_match_single(
            pred_cls_b,
            pred_box_b,
            labels,
            boxes,
            cost_class=cost_class,
            cost_bbox=cost_bbox,
            cost_giou=cost_giou,
        )

        tgt_cls[b, matched_q, labels[matched_gt]] = 1.0

        pred_m = pred_box_b[matched_q]
        tgt_m = boxes[matched_gt]
        loss_bbox_list.append(F.l1_loss(pred_m, tgt_m, reduction="sum"))

        pred_xy = box_cxcywh_to_xyxy(pred_m)
        tgt_xy = box_cxcywh_to_xyxy(tgt_m)
        giou = generalized_box_iou_pairwise(pred_xy, tgt_xy)
        loss_giou_list.append((1.0 - giou).sum())

    pred_cls = _ensure_logit(pred_cls)
    loss_cls = loss_cls_weight * sigmoid_focal_loss(
        pred_cls.reshape(-1, C),
        tgt_cls.reshape(-1, C),
        num_boxes,
        alpha=focal_alpha,
        gamma=focal_gamma,
    )
    loss_bbox = loss_bbox_weight * torch.stack(loss_bbox_list).sum() / float(num_boxes)
    loss_giou = loss_giou_weight * torch.stack(loss_giou_list).sum() / float(num_boxes)
    loss_total = loss_cls + loss_bbox + loss_giou

    return {
        "loss_cls": loss_cls,
        "loss_bbox": loss_bbox,
        "loss_giou": loss_giou,
        "loss_total": loss_total,
    }