from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import torch
from torch import Tensor
from tqdm import tqdm

from groundingdino.util import box_ops
from groundingdino.util.misc import NestedTensor


@dataclass
class Detection:
    image_id: int
    cls_id: int
    score: float
    box_xyxy: Tensor


def _to_xyxy(boxes_cxcywh: Tensor) -> Tensor:
    if boxes_cxcywh.numel() == 0:
        return boxes_cxcywh.new_zeros((0, 4))
    return box_ops.box_cxcywh_to_xyxy(boxes_cxcywh)


def _iou_single_to_many(box: Tensor, boxes: Tensor) -> Tensor:
    if boxes.numel() == 0:
        return boxes.new_zeros((0,))
    ious, _ = box_ops.box_iou(box.unsqueeze(0), boxes)
    return ious.squeeze(0)


def _collect_predictions(
    model,
    images: Sequence[Tensor] | NestedTensor | Tensor,
    classes: Sequence[str],
    *,
    device: torch.device,
    ap_score_threshold: float,
    zero_shot: bool = False,
) -> List[List[Detection]]:
    if isinstance(images, (NestedTensor, Tensor)):
        images = images.to(device)
    else:
        images = [img.to(device) for img in images]
    if zero_shot:
        outputs = model.forward_zeroshot(images, classes=classes, aggregation_method="max")
    else:
        outputs = model(images, classes=classes, aggregation_method="max")

    pred_boxes_batch = outputs["pred_boxes"].detach().cpu()  # [B,Q,4]
    pred_cls_logits_batch = outputs["pred_class_logits"].detach().cpu()
    # Wrapper keeps backward compatibility for B=1 and may return [Q,C].
    if pred_cls_logits_batch.dim() == 2:
        pred_cls_logits_batch = pred_cls_logits_batch.unsqueeze(0)

    batch_preds: List[List[Detection]] = []
    for bidx in range(pred_boxes_batch.shape[0]):
        pred_boxes = pred_boxes_batch[bidx]
        pred_cls_logits = pred_cls_logits_batch[bidx]  # [Q, C]
        scores, cls_ids = pred_cls_logits.max(dim=1)

        keep = scores >= ap_score_threshold
        kept_boxes = _to_xyxy(pred_boxes[keep])
        kept_scores = scores[keep]
        kept_cls_ids = cls_ids[keep]

        preds: List[Detection] = []
        for i in range(kept_scores.numel()):
            preds.append(
                Detection(
                    image_id=bidx,
                    cls_id=int(kept_cls_ids[i].item()),
                    score=float(kept_scores[i].item()),
                    box_xyxy=kept_boxes[i],
                )
            )
        batch_preds.append(preds)
    return batch_preds


def _average_precision(recalls: Tensor, precisions: Tensor) -> float:
    if recalls.numel() == 0:
        return 0.0
    mrec = torch.cat([torch.tensor([0.0]), recalls, torch.tensor([1.0])])
    mpre = torch.cat([torch.tensor([0.0]), precisions, torch.tensor([0.0])])
    for i in range(mpre.numel() - 1, 0, -1):
        mpre[i - 1] = torch.maximum(mpre[i - 1], mpre[i])
    idx = torch.where(mrec[1:] != mrec[:-1])[0]
    ap = torch.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1])
    return float(ap.item())


@torch.no_grad()
def evaluate_detection(
    model,
    data_loader,
    classes: Sequence[str],
    *,
    device: torch.device,
    iou_threshold: float = 0.5,
    ap_score_threshold: float = 1e-3,
    pr_score_threshold: float = 0.5,
    progress_desc: str = "Eval",
    zero_shot: bool = False,
) -> Dict[str, float]:
    model.eval()
    num_classes = len(classes)
    all_preds: List[List[Detection]] = [[] for _ in range(num_classes)]
    gt_by_image_class: Dict[Tuple[int, int], Tensor] = {}
    gt_count_by_class = torch.zeros(num_classes, dtype=torch.long)
    image_offset = 0

    pbar = tqdm(data_loader, desc=progress_desc, leave=False)
    for images, targets in pbar:
        preds_per_img = _collect_predictions(
            model,
            images,
            classes,
            device=device,
            ap_score_threshold=ap_score_threshold,
            zero_shot=zero_shot,
        )

        for local_img_id, target in enumerate(targets):
            global_img_id = image_offset + local_img_id
            gt_boxes = _to_xyxy(target["boxes"].detach().cpu())
            gt_labels = target["labels"].detach().cpu()
            for cls_id in range(num_classes):
                cls_mask = gt_labels == cls_id
                cls_boxes = gt_boxes[cls_mask]
                gt_by_image_class[(global_img_id, cls_id)] = cls_boxes
                gt_count_by_class[cls_id] += int(cls_boxes.shape[0])

            for det in preds_per_img[local_img_id]:
                det.image_id = global_img_id
                all_preds[det.cls_id].append(det)

        image_offset += len(targets)

    aps: List[float] = []
    tp_thr = 0
    fp_thr = 0
    fn_thr = 0

    for cls_id in range(num_classes):
        preds = sorted(all_preds[cls_id], key=lambda x: x.score, reverse=True)
        n_gt = int(gt_count_by_class[cls_id].item())
        if n_gt == 0:
            continue

        matched: Dict[Tuple[int, int], Tensor] = {}
        tp = torch.zeros(len(preds), dtype=torch.float32)
        fp = torch.zeros(len(preds), dtype=torch.float32)

        for i, det in enumerate(preds):
            key = (det.image_id, cls_id)
            gt_boxes = gt_by_image_class.get(key, torch.zeros((0, 4)))
            if gt_boxes.numel() == 0:
                fp[i] = 1.0
                continue

            if key not in matched:
                matched[key] = torch.zeros(gt_boxes.shape[0], dtype=torch.bool)

            ious = _iou_single_to_many(det.box_xyxy, gt_boxes)
            if ious.numel() == 0:
                fp[i] = 1.0
                continue
            best_iou, best_idx = torch.max(ious, dim=0)
            best_idx_int = int(best_idx.item())
            if (
                float(best_iou.item()) >= iou_threshold
                and not matched[key][best_idx_int]
            ):
                tp[i] = 1.0
                matched[key][best_idx_int] = True
            else:
                fp[i] = 1.0

        if tp.numel() > 0:
            tp_cum = torch.cumsum(tp, dim=0)
            fp_cum = torch.cumsum(fp, dim=0)
            recalls = tp_cum / max(n_gt, 1)
            precisions = tp_cum / torch.clamp(tp_cum + fp_cum, min=1e-8)
            aps.append(_average_precision(recalls, precisions))

            score_mask = torch.tensor(
                [det.score >= pr_score_threshold for det in preds], dtype=torch.bool
            )
            tp_sel = int(tp[score_mask].sum().item())
            fp_sel = int(fp[score_mask].sum().item())
            fn_sel = n_gt - tp_sel
            tp_thr += tp_sel
            fp_thr += fp_sel
            fn_thr += fn_sel
        else:
            fn_thr += n_gt

    mAP50 = float(sum(aps) / len(aps)) if aps else 0.0
    precision50 = float(tp_thr / max(tp_thr + fp_thr, 1))
    recall50 = float(tp_thr / max(tp_thr + fn_thr, 1))

    return {
        "mAP50": mAP50,
        "precision50": precision50,
        "recall50": recall50,
    }
