from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import torch
from torch import Tensor
from tqdm import tqdm

from groundingdino.util import box_ops
from groundingdino.util.misc import NestedTensor

EVAL_METRIC_CHOICES = ("mAP50", "mAP75", "mAP50-95")
COCO_IOU_THRESHOLDS = tuple(round(0.5 + 0.05 * i, 2) for i in range(10))


@dataclass
class Detection:
    image_id: int
    cls_id: int
    score: float
    box_xyxy: Tensor


@dataclass
class _ClassSample:
    pred_boxes: Tensor
    pred_scores: Tensor
    gt_boxes: Tensor


def _to_xyxy(boxes_cxcywh: Tensor) -> Tensor:
    if boxes_cxcywh.numel() == 0:
        return boxes_cxcywh.new_zeros((0, 4))
    return box_ops.box_cxcywh_to_xyxy(boxes_cxcywh)


def _collect_predictions(
    model,
    images: Sequence[Tensor] | NestedTensor | Tensor,
    classes: Sequence[str],
    *,
    device: torch.device,
    ap_score_threshold: float,
) -> List[List[Detection]]:
    if isinstance(images, (NestedTensor, Tensor)):
        images = images.to(device)
    else:
        images = [img.to(device) for img in images]
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


def _match_class_predictions(
    pred_boxes: Tensor,
    pred_scores: Tensor,
    gt_boxes: Tensor,
    *,
    iou_threshold: float,
) -> Tuple[Tensor, Tensor]:
    tp_cls = torch.zeros(pred_scores.shape[0], dtype=torch.float32)
    fp_cls = torch.ones(pred_scores.shape[0], dtype=torch.float32)

    n_gt = int(gt_boxes.shape[0])
    if n_gt == 0 or pred_scores.numel() == 0:
        return tp_cls, fp_cls

    iou_mat, _ = box_ops.box_iou(pred_boxes, gt_boxes)
    matched_gt = torch.zeros(n_gt, dtype=torch.bool)
    best_ious, best_gt_idx = iou_mat.max(dim=1)
    for pi in range(pred_scores.shape[0]):
        gi = int(best_gt_idx[pi].item())
        if float(best_ious[pi].item()) >= iou_threshold and not matched_gt[gi]:
            tp_cls[pi] = 1.0
            fp_cls[pi] = 0.0
            matched_gt[gi] = True
    return tp_cls, fp_cls


def _compute_map_at_iou(
    samples_by_class: List[List[_ClassSample]],
    gt_count_by_class: Tensor,
    num_classes: int,
    *,
    iou_threshold: float,
    pr_score_threshold: float,
) -> Tuple[float, int, int, int]:
    aps: List[float] = []
    tp_thr = 0
    fp_thr = 0
    fn_thr = 0

    for cls_id in range(num_classes):
        n_gt = int(gt_count_by_class[cls_id].item())
        if n_gt == 0:
            continue

        if not samples_by_class[cls_id]:
            fn_thr += n_gt
            continue

        scores_list: List[Tensor] = []
        tp_list: List[Tensor] = []
        fp_list: List[Tensor] = []

        for sample in samples_by_class[cls_id]:
            tp_cls, fp_cls = _match_class_predictions(
                sample.pred_boxes,
                sample.pred_scores,
                sample.gt_boxes,
                iou_threshold=iou_threshold,
            )
            if sample.pred_scores.numel() == 0:
                continue
            scores_list.append(sample.pred_scores)
            tp_list.append(tp_cls)
            fp_list.append(fp_cls)

        if not scores_list:
            fn_thr += n_gt
            continue

        scores = torch.cat(scores_list, dim=0)
        tp = torch.cat(tp_list, dim=0)
        fp = torch.cat(fp_list, dim=0)
        order = torch.argsort(scores, descending=True)
        scores = scores[order]
        tp = tp[order]
        fp = fp[order]

        tp_cum = torch.cumsum(tp, dim=0)
        fp_cum = torch.cumsum(fp, dim=0)
        recalls = tp_cum / max(n_gt, 1)
        precisions = tp_cum / torch.clamp(tp_cum + fp_cum, min=1e-8)
        aps.append(_average_precision(recalls, precisions))

        score_mask = scores >= pr_score_threshold
        tp_sel = int(tp[score_mask].sum().item())
        fp_sel = int(fp[score_mask].sum().item())
        fn_sel = n_gt - tp_sel
        tp_thr += tp_sel
        fp_thr += fp_sel
        fn_thr += fn_sel

    mean_ap = float(sum(aps) / len(aps)) if aps else 0.0
    return mean_ap, tp_thr, fp_thr, fn_thr


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
) -> Dict[str, float]:
    model.eval()
    num_classes = len(classes)
    samples_by_class: List[List[_ClassSample]] = [[] for _ in range(num_classes)]
    gt_count_by_class = torch.zeros(num_classes, dtype=torch.long)

    pbar = tqdm(data_loader, desc=progress_desc, leave=False)
    for images, targets in pbar:
        preds_per_img = _collect_predictions(
            model,
            images,
            classes,
            device=device,
            ap_score_threshold=ap_score_threshold,
        )

        for local_img_id, target in enumerate(targets):
            gt_boxes = _to_xyxy(target["boxes"].detach().cpu())
            gt_labels = target["labels"].detach().cpu()
            preds = preds_per_img[local_img_id]

            pred_boxes_all = (
                torch.stack([det.box_xyxy for det in preds], dim=0)
                if preds
                else torch.zeros((0, 4), dtype=torch.float32)
            )
            pred_scores_all = (
                torch.tensor([det.score for det in preds], dtype=torch.float32)
                if preds
                else torch.zeros((0,), dtype=torch.float32)
            )
            pred_cls_all = (
                torch.tensor([det.cls_id for det in preds], dtype=torch.long)
                if preds
                else torch.zeros((0,), dtype=torch.long)
            )

            for cls_id in range(num_classes):
                gt_mask = gt_labels == cls_id
                gt_boxes_cls = gt_boxes[gt_mask]
                n_gt = int(gt_boxes_cls.shape[0])
                gt_count_by_class[cls_id] += n_gt

                pred_mask = pred_cls_all == cls_id
                pred_boxes_cls = pred_boxes_all[pred_mask]
                pred_scores_cls = pred_scores_all[pred_mask]
                if pred_scores_cls.numel() > 0:
                    order = torch.argsort(pred_scores_cls, descending=True)
                    pred_boxes_cls = pred_boxes_cls[order]
                    pred_scores_cls = pred_scores_cls[order]

                samples_by_class[cls_id].append(
                    _ClassSample(
                        pred_boxes=pred_boxes_cls,
                        pred_scores=pred_scores_cls,
                        gt_boxes=gt_boxes_cls,
                    )
                )

    iou_thresholds = COCO_IOU_THRESHOLDS
    if iou_threshold not in iou_thresholds:
        iou_thresholds = tuple(sorted(set(iou_thresholds + (iou_threshold,))))

    map_by_iou: Dict[float, float] = {}
    for thr in iou_thresholds:
        mean_ap, _, _, _ = _compute_map_at_iou(
            samples_by_class,
            gt_count_by_class,
            num_classes,
            iou_threshold=thr,
            pr_score_threshold=pr_score_threshold,
        )
        map_by_iou[thr] = mean_ap

    _, tp_thr, fp_thr, fn_thr = _compute_map_at_iou(
        samples_by_class,
        gt_count_by_class,
        num_classes,
        iou_threshold=0.5,
        pr_score_threshold=pr_score_threshold,
    )

    mAP50 = map_by_iou[0.5]
    mAP75 = map_by_iou[0.75]
    mAP50_95 = float(sum(map_by_iou[thr] for thr in COCO_IOU_THRESHOLDS) / len(COCO_IOU_THRESHOLDS))

    precision50 = float(tp_thr / max(tp_thr + fp_thr, 1))
    recall50 = float(tp_thr / max(tp_thr + fn_thr, 1))

    return {
        "mAP50": mAP50,
        "mAP75": mAP75,
        "mAP50-95": mAP50_95,
        "precision50": precision50,
        "recall50": recall50,
    }
