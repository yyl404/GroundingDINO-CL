import json
import os
from typing import Dict, List, Tuple

import numpy as np
import torch
from torchvision.ops import box_convert

from groundingdino.models import build_model
from groundingdino.util.misc import clean_state_dict
from groundingdino.util.slconfig import SLConfig

from .voc import VOC_CLASSES


def load_groundingdino_model(config_path: str, checkpoint_path: str, device: str = "cuda"):
    args = SLConfig.fromfile(config_path)
    args.device = device
    model = build_model(args)

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    model.to(device)
    model.eval()
    return model


def _token_scores_to_class_scores(
    token_scores: torch.Tensor, class_token_map: Dict[int, List[int]], num_classes: int
) -> torch.Tensor:
    class_scores = torch.zeros((token_scores.shape[0], num_classes), dtype=token_scores.dtype)
    for class_idx in range(num_classes):
        token_ids = class_token_map[class_idx]
        valid_token_ids = [token_id for token_id in token_ids if token_id < token_scores.shape[1]]
        if valid_token_ids:
            class_scores[:, class_idx] = token_scores[:, valid_token_ids].max(dim=1)[0]
    return class_scores


def decode_predictions(
    outputs: Dict[str, torch.Tensor],
    class_token_map: Dict[int, List[int]],
    box_threshold: float,
    text_threshold: float,
    num_classes: int = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    logits = outputs["pred_logits"].sigmoid()[0].detach().cpu()
    boxes_cxcywh = outputs["pred_boxes"][0].detach().cpu()

    query_scores = logits.max(dim=1)[0]
    keep = query_scores > box_threshold
    logits = logits[keep]
    boxes_cxcywh = boxes_cxcywh[keep]
    if logits.numel() == 0:
        return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.float32), np.zeros(
            (0,), dtype=np.int64
        )

    if num_classes is None:
        num_classes = max(class_token_map.keys()) + 1 if class_token_map else len(VOC_CLASSES)
    class_scores = _token_scores_to_class_scores(logits, class_token_map, num_classes)
    best_scores, best_class = class_scores.max(dim=1)
    keep_text = best_scores > text_threshold
    boxes_cxcywh = boxes_cxcywh[keep_text]
    best_scores = best_scores[keep_text]
    best_class = best_class[keep_text]

    return (
        boxes_cxcywh.numpy().astype(np.float32),
        best_scores.numpy().astype(np.float32),
        best_class.numpy().astype(np.int64),
    )


def save_detection_txt(
    txt_path: str, boxes_xyxy: np.ndarray, scores: np.ndarray, class_ids: np.ndarray, class_names: List[str]
):
    os.makedirs(os.path.dirname(txt_path), exist_ok=True)
    with open(txt_path, "w", encoding="utf-8") as f:
        for box, score, class_id in zip(boxes_xyxy, scores, class_ids):
            x0, y0, x1, y1 = box.tolist()
            f.write(f"{class_names[int(class_id)]} {float(score):.6f} {x0:.2f} {y0:.2f} {x1:.2f} {y1:.2f}\n")


def _voc_ap(rec: np.ndarray, prec: np.ndarray) -> float:
    mrec = np.concatenate(([0.0], rec, [1.0]))
    mpre = np.concatenate(([0.0], prec, [0.0]))
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def evaluate_voc_map(
    all_predictions: Dict[str, List[Tuple[float, int, np.ndarray]]],
    all_ground_truths: Dict[str, List[Tuple[int, np.ndarray, int]]],
    num_classes: int = 20,
    iou_thresh: float = 0.5,
    class_names: List[str] = None,
    eval_class_ids: List[int] = None,
) -> Dict[str, float]:
    ap_per_class = {}
    if class_names is None:
        class_names = VOC_CLASSES
    if eval_class_ids is None:
        eval_class_ids = list(range(num_classes))

    for class_id in eval_class_ids:
        preds = []
        npos = 0
        gt_by_image = {}

        for image_name, items in all_ground_truths.items():
            gt_cls = [item for item in items if item[0] == class_id]
            gt_by_image[image_name] = {
                "boxes": np.array([x[1] for x in gt_cls], dtype=np.float32) if gt_cls else np.zeros((0, 4)),
                "difficult": np.array([x[2] for x in gt_cls], dtype=np.int64) if gt_cls else np.zeros((0,), dtype=np.int64),
                "det": np.zeros((len(gt_cls),), dtype=np.int64),
            }
            npos += int((gt_by_image[image_name]["difficult"] == 0).sum())

        for image_name, items in all_predictions.items():
            for score, pred_class_id, pred_box in items:
                if pred_class_id == class_id:
                    preds.append((image_name, float(score), pred_box.astype(np.float32)))
        preds.sort(key=lambda x: x[1], reverse=True)

        tp = np.zeros((len(preds),), dtype=np.float32)
        fp = np.zeros((len(preds),), dtype=np.float32)
        for i, (image_name, _score, pred_box) in enumerate(preds):
            gts = gt_by_image.get(image_name)
            if gts is None or gts["boxes"].shape[0] == 0:
                fp[i] = 1
                continue

            gt_boxes = gts["boxes"]
            ixmin = np.maximum(gt_boxes[:, 0], pred_box[0])
            iymin = np.maximum(gt_boxes[:, 1], pred_box[1])
            ixmax = np.minimum(gt_boxes[:, 2], pred_box[2])
            iymax = np.minimum(gt_boxes[:, 3], pred_box[3])
            iw = np.maximum(ixmax - ixmin + 1.0, 0.0)
            ih = np.maximum(iymax - iymin + 1.0, 0.0)
            inter = iw * ih
            union = (
                (pred_box[2] - pred_box[0] + 1.0) * (pred_box[3] - pred_box[1] + 1.0)
                + (gt_boxes[:, 2] - gt_boxes[:, 0] + 1.0) * (gt_boxes[:, 3] - gt_boxes[:, 1] + 1.0)
                - inter
            )
            iou = inter / np.maximum(union, 1e-9)
            max_iou_idx = int(np.argmax(iou))
            max_iou = float(iou[max_iou_idx])

            if max_iou >= iou_thresh:
                if gts["difficult"][max_iou_idx] == 1:
                    continue
                if gts["det"][max_iou_idx] == 0:
                    tp[i] = 1
                    gts["det"][max_iou_idx] = 1
                else:
                    fp[i] = 1
            else:
                fp[i] = 1

        if npos == 0:
            ap_per_class[class_id] = 0.0
            continue
        fp = np.cumsum(fp)
        tp = np.cumsum(tp)
        rec = tp / float(npos)
        prec = tp / np.maximum(tp + fp, 1e-9)
        ap_per_class[class_id] = _voc_ap(rec, prec)

    mAP = float(np.mean(list(ap_per_class.values()))) if ap_per_class else 0.0
    metrics = {"mAP@0.5": mAP}
    for class_id, ap in ap_per_class.items():
        class_name = class_names[class_id] if class_id < len(class_names) else f"class_{class_id}"
        metrics[f"AP50_{class_name}"] = float(ap)
    return metrics


def save_metrics(metrics_path: str, metrics: Dict[str, float]):
    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
