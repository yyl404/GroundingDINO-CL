from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from groundingdino.util import box_ops


def _hungarian_match(
    pred_boxes_cxcywh: torch.Tensor, gt_boxes_cxcywh: torch.Tensor
) -> List[Tuple[int, int]]:
    if pred_boxes_cxcywh.numel() == 0 or gt_boxes_cxcywh.numel() == 0:
        return []

    pred_xyxy = box_ops.box_cxcywh_to_xyxy(pred_boxes_cxcywh)
    gt_xyxy = box_ops.box_cxcywh_to_xyxy(gt_boxes_cxcywh)

    # Cost matrix: L1 box distance + (1 - GIoU)
    # Shape: [num_queries, num_gt]
    cost_bbox = torch.cdist(pred_boxes_cxcywh, gt_boxes_cxcywh, p=1)
    cost_giou = 1.0 - box_ops.generalized_box_iou(pred_xyxy, gt_xyxy)
    cost = cost_bbox + cost_giou

    row_ind, col_ind = linear_sum_assignment(cost.detach().cpu().numpy())
    return list(zip(row_ind.tolist(), col_ind.tolist()))


def compute_prompt_tuning_loss(
    outputs: Dict[str, torch.Tensor],
    targets: List[Dict[str, torch.Tensor]],
    class_token_map: Dict[int, List[int]],
    bg_loss_weight: float = 0.25,
    bbox_loss_weight: float = 5.0,
    giou_loss_weight: float = 2.0,
) -> Dict[str, torch.Tensor]:
    pred_logits = outputs["pred_logits"]  # [B, Q, T]
    pred_boxes = outputs["pred_boxes"]  # [B, Q, 4]
    pred_logits = torch.nan_to_num(pred_logits, nan=0.0, posinf=50.0, neginf=-50.0).clamp(-50.0, 50.0)

    batch_size, num_queries, num_tokens = pred_logits.shape
    device = pred_logits.device

    cls_loss = torch.zeros((), device=device)
    bbox_loss = torch.zeros((), device=device)
    giou_loss = torch.zeros((), device=device)
    bg_loss = torch.zeros((), device=device)

    matched_pairs = 0
    bg_queries = 0

    for b in range(batch_size):
        gt_boxes = targets[b]["boxes"].to(device)
        gt_labels = targets[b]["labels"].to(device)
        matches = _hungarian_match(pred_boxes[b], gt_boxes)

        matched_query_indices = set()
        for pred_idx, gt_idx in matches:
            matched_query_indices.add(pred_idx)
            matched_pairs += 1

            token_target = torch.zeros((num_tokens,), device=device)
            class_id = int(gt_labels[gt_idx].item())
            for token_idx in class_token_map.get(class_id, []):
                if token_idx < num_tokens:
                    token_target[token_idx] = 1.0

            cls_term = F.binary_cross_entropy_with_logits(pred_logits[b, pred_idx], token_target)
            if torch.isfinite(cls_term):
                cls_loss = cls_loss + cls_term
            bbox_loss = bbox_loss + F.l1_loss(pred_boxes[b, pred_idx], gt_boxes[gt_idx], reduction="mean")

            pred_box_xyxy = box_ops.box_cxcywh_to_xyxy(pred_boxes[b, pred_idx : pred_idx + 1])
            gt_box_xyxy = box_ops.box_cxcywh_to_xyxy(gt_boxes[gt_idx : gt_idx + 1])
            giou = box_ops.generalized_box_iou(pred_box_xyxy, gt_box_xyxy)[0, 0]
            giou_loss = giou_loss + (1.0 - giou)

        neg_indices = [q for q in range(num_queries) if q not in matched_query_indices]
        if neg_indices:
            neg_logits = pred_logits[b, neg_indices]
            bg_term = F.binary_cross_entropy_with_logits(neg_logits, torch.zeros_like(neg_logits))
            if torch.isfinite(bg_term):
                bg_loss = bg_loss + bg_term
                bg_queries += len(neg_indices)

    matched_pairs = max(matched_pairs, 1)
    bg_queries = max(bg_queries, 1)
    cls_loss = cls_loss / matched_pairs
    bbox_loss = bbox_loss / matched_pairs
    giou_loss = giou_loss / matched_pairs
    bg_loss = bg_loss / bg_queries

    total_loss = cls_loss + (bg_loss_weight * bg_loss) + (bbox_loss_weight * bbox_loss) + (
        giou_loss_weight * giou_loss
    )
    return {
        "loss": total_loss,
        "loss_cls": cls_loss.detach(),
        "loss_bg": bg_loss.detach(),
        "loss_bbox": bbox_loss.detach(),
        "loss_giou": giou_loss.detach(),
    }
