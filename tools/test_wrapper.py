import argparse
import json
import os
import random
import warnings
from pathlib import Path
from typing import List

import numpy as np
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as TVT
from PIL import Image, ImageDraw, ImageFont

import groundingdino.datasets.transforms as T
from groundingdino.util import box_ops

from finetune import GroundingDINOWrapper
from finetune.datasets.yolo import (
    YoloDetectionDataset,
    YoloOBBDataset,
    _load_yolo_yaml,
    collate_fn,
    xyxyxyxy2xywhr,
)
from finetune.eval import evaluate_detection, evaluate_obb
from utils import load_model, load_wrapper_checkpoint, xywhr_to_corners_xyxyxyxy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Test GroundingDINOWrapper on YOLO dataset")
    parser.add_argument("--config_file", type=str, required=True, help="Path to GroundingDINO config file.")
    parser.add_argument(
        "--pretrained_checkpoint",
        type=str,
        required=True,
        help="Path to pretrained GroundingDINO checkpoint.",
    )
    parser.add_argument("--dataset_yaml", type=str, required=True, help="Path to YOLO dataset yaml.")
    parser.add_argument(
        "--classes",
        type=str,
        default=None,
        help='Comma-separated class names. If omitted, use "names" from dataset yaml.',
    )
    parser.add_argument("--test_split", type=str, default="test", help='Dataset split name for evaluation, e.g. "test".')
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for dataloader.")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of dataloader worker processes.")
    parser.add_argument("--device", type=str, default="cuda", help='Device string, e.g. "cuda", "cuda:0", or "cpu".')
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--prompt_len", type=int, default=4, help="Prompt token length per class.")
    parser.add_argument(
        "--inject_before_encoder",
        action="store_true",
        help="Use prompt embeddings before the text encoder (must match wrapper checkpoint setting).",
    )
    parser.add_argument("--eval_iou_threshold", type=float, default=0.5, help="IoU threshold for TP/FP matching.")
    parser.add_argument("--eval_ap_score_threshold", type=float, default=1e-3, help="Score threshold used when computing AP.")
    parser.add_argument("--eval_pr_score_threshold", type=float, default=0.25, help="Score threshold used for precision/recall.")
    parser.add_argument(
        "--vis-batch",
        "--vis_batch",
        dest="vis_batch",
        type=int,
        default=3,
        metavar="N",
        help=(
            "If N > 0, save side-by-side visualization (pred | GT) for the first N dataloader batches "
            "under output_dir/visualizations/. Uses --vis-score-threshold if set, else --eval_pr_score_threshold."
        ),
    )
    parser.add_argument(
        "--vis-score-threshold",
        "--vis_score_threshold",
        dest="vis_score_threshold",
        type=float,
        default=None,
        help="Score threshold for drawing predictions when visualizing (default: same as --eval_pr_score_threshold).",
    )
    parser.add_argument(
        "--aggregation_method",
        type=str,
        choices=["mean", "sum", "max", "min"],
        default="max",
        help="Token aggregation for wrapper forward; should match training/eval settings.",
    )
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save test logs.")
    parser.add_argument("--weight", type=str, default=None, help="Path to wrapper checkpoint file for evaluation.")
    parser.add_argument(
        "--output_decode_info",
        action="store_true",
        help="Whether to decode and output class embedding information.",
    )

    parser.add_argument("--zero-shot", action="store_true", help="Switch on to use zero-shot mode to infer")
    parser.add_argument("--use_obb", action="store_true", help="Enable OBB evaluation and visualization.")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_classes(classes_arg: str | None, dataset_yaml: str) -> List[str]:
    if classes_arg:
        classes = [c.strip() for c in classes_arg.split(",") if c.strip()]
        if not classes:
            raise ValueError("--classes must be non-empty when provided.")
        return classes
    cfg = _load_yolo_yaml(dataset_yaml)
    if not cfg["class_names"]:
        raise ValueError("No classes in dataset yaml. Please set --classes explicitly.")
    return cfg["class_names"]


def _to_pil_image_from_normalized_chw(img_chw: torch.Tensor) -> Image.Image:
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=img_chw.dtype, device=img_chw.device).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=img_chw.dtype, device=img_chw.device).view(3, 1, 1)
    img = (img_chw * std + mean).clamp(0, 1)
    img_hwc = (img.permute(1, 2, 0).cpu().numpy() * 255.0).astype("uint8")
    return Image.fromarray(img_hwc)


def _draw_boxes(
    image: Image.Image,
    boxes_xyxy: torch.Tensor,
    labels: torch.Tensor,
    classes: List[str],
    scores: torch.Tensor | None = None,
) -> Image.Image:
    vis = image.copy()
    draw = ImageDraw.Draw(vis)

    def _color_for_class(cls_id: int) -> tuple[int, int, int]:
        rng = np.random.default_rng(cls_id + 1)
        return tuple(int(x) for x in rng.integers(0, 255, size=3))

    for i in range(int(boxes_xyxy.shape[0])):
        x0, y0, x1, y1 = boxes_xyxy[i].tolist()
        cls_id = int(labels[i].item())
        cls_name = classes[cls_id] if 0 <= cls_id < len(classes) else str(cls_id)
        score_text = f"{float(scores[i].item()):.2f}" if scores is not None else "1.00"
        text = f"{cls_name}({score_text})"
        color = _color_for_class(cls_id)
        draw.rectangle([x0, y0, x1, y1], outline=color, width=6)

        font = ImageFont.load_default()
        if hasattr(font, "getbbox"):
            bbox = draw.textbbox((x0, y0), text, font)
        else:
            w, h = draw.textsize(text, font)
            bbox = (x0, y0, w + x0, y0 + h)
        draw.rectangle(bbox, fill=color)
        draw.text((x0, y0), text, fill="white")
    return vis


def _draw_obb_boxes(
    image: Image.Image,
    boxes_xywhr: torch.Tensor,
    labels: torch.Tensor,
    classes: List[str],
    image_size: tuple[int, int],
    scores: torch.Tensor | None = None,
) -> Image.Image:
    vis = image.copy()
    draw = ImageDraw.Draw(vis)
    width, height = image_size
    corners = xywhr_to_corners_xyxyxyxy(boxes_xywhr, float(width), float(height))

    def _color_for_class(cls_id: int) -> tuple[int, int, int]:
        rng = np.random.default_rng(cls_id + 1)
        return tuple(int(x) for x in rng.integers(0, 255, size=3))

    for i in range(int(corners.shape[0])):
        pts = corners[i].reshape(4, 2).tolist()
        cls_id = int(labels[i].item())
        cls_name = classes[cls_id] if 0 <= cls_id < len(classes) else str(cls_id)
        score_text = f"{float(scores[i].item()):.2f}" if scores is not None else "1.00"
        text = f"{cls_name}({score_text})"
        color = _color_for_class(cls_id)
        draw.polygon([tuple(p) for p in pts], outline=color, width=6)

        x0, y0 = pts[0]
        font = ImageFont.load_default()
        if hasattr(font, "getbbox"):
            bbox = draw.textbbox((x0, y0), text, font)
        else:
            w, h = draw.textsize(text, font)
            bbox = (x0, y0, w + x0, y0 + h)
        draw.rectangle(bbox, fill=color)
        draw.text((x0, y0), text, fill="white")
    return vis


@torch.no_grad()
def visualize_batches(
    wrapper: GroundingDINOWrapper,
    data_loader: DataLoader,
    classes: List[str],
    device: torch.device,
    vis_batch: int,
    score_threshold: float,
    vis_dir: Path,
    zero_shot: bool = False,
    use_obb: bool = False,
    aggregation_method: str = "max",
) -> int:
    """Save visualization images for the first ``vis_batch`` batches. Returns number of images written."""
    if vis_batch <= 0:
        return 0
    wrapper.eval()
    vis_dir.mkdir(parents=True, exist_ok=True)
    num_saved = 0
    for batch_idx, (images, targets) in enumerate(data_loader):
        if batch_idx >= vis_batch:
            break
        if zero_shot:
            outputs = wrapper.forward_zeroshot(
                images.to(device), classes=classes, aggregation_method=aggregation_method
            )
        else:
            outputs = wrapper(
                images.to(device), classes=classes, aggregation_method=aggregation_method
            )
        pred_boxes_batch = outputs["pred_boxes"].detach().cpu()
        pred_cls_logits_batch = outputs["pred_class_logits"].detach().cpu()
        if pred_cls_logits_batch.dim() == 2:
            pred_cls_logits_batch = pred_cls_logits_batch.unsqueeze(0)

        for local_idx, target in enumerate(targets):
            w, h = target["info"]["ori_size"]
            img_chw = images.tensors[local_idx, :, :h, :w]
            base_img = _to_pil_image_from_normalized_chw(img_chw)

            pred_scores, pred_labels = pred_cls_logits_batch[local_idx].max(dim=1)
            keep = pred_scores >= score_threshold
            if use_obb:
                pred_boxes_xywhr = pred_boxes_batch[local_idx]
                if pred_boxes_xywhr.shape[-1] != 5:
                    raise ValueError(
                        f"Expected OBB predictions with 5 channels, got shape {tuple(pred_boxes_xywhr.shape)}."
                    )
                pred_img = _draw_obb_boxes(
                    base_img,
                    pred_boxes_xywhr[keep],
                    pred_labels[keep],
                    classes,
                    image_size=(w, h),
                    scores=pred_scores[keep],
                )

                gt_boxes = target["boxes"].detach().cpu().clone()
                if gt_boxes.numel() == 0:
                    gt_boxes_xywhr = gt_boxes.new_zeros((0, 5))
                elif gt_boxes.shape[-1] == 8:
                    gt_boxes_xywhr = xyxyxyxy2xywhr(gt_boxes)
                elif gt_boxes.shape[-1] == 5:
                    gt_boxes_xywhr = gt_boxes
                else:
                    raise ValueError(f"Unsupported OBB GT shape {tuple(gt_boxes.shape)}.")
                gt_labels = target["labels"].detach().cpu()
                gt_img = _draw_obb_boxes(base_img, gt_boxes_xywhr, gt_labels, classes, image_size=(w, h), scores=None)
            else:
                pred_boxes_xyxy = box_ops.box_cxcywh_to_xyxy(pred_boxes_batch[local_idx])
                pred_boxes_xyxy[:, [0, 2]] *= float(w)
                pred_boxes_xyxy[:, [1, 3]] *= float(h)
                pred_img = _draw_boxes(
                    base_img,
                    pred_boxes_xyxy[keep],
                    pred_labels[keep],
                    classes,
                    pred_scores[keep],
                )

                gt_boxes_xyxy = box_ops.box_cxcywh_to_xyxy(target["boxes"].detach().cpu().clone())
                gt_boxes_xyxy[:, [0, 2]] *= float(w)
                gt_boxes_xyxy[:, [1, 3]] *= float(h)
                gt_labels = target["labels"].detach().cpu()
                gt_img = _draw_boxes(base_img, gt_boxes_xyxy, gt_labels, classes, None)

            paired = Image.new("RGB", (pred_img.width * 2, pred_img.height))
            paired.paste(pred_img, (0, 0))
            paired.paste(gt_img, (pred_img.width, 0))
            save_path = vis_dir / f"batch{batch_idx:03d}_img{local_idx:03d}.jpg"
            paired.save(save_path)
            num_saved += 1
    return num_saved


def main() -> None:
    args = parse_args()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    classes = parse_classes(args.classes, args.dataset_yaml)
    (log_dir / "train_args.json").write_text(
        json.dumps({**vars(args), "resolved_classes": classes}, indent=2),
        encoding="utf-8",
    )

    device = args.device
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        warnings.warn(
            f"Requested device '{args.device}' is not available. Falling back to 'cpu'."
        )
        device = "cpu"
    device = torch.device(device)

    test_transform = TVT.Compose(
        [
            TVT.ToTensor(),
            TVT.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    if args.use_obb:
        test_dataset = YoloOBBDataset(
            args.dataset_yaml,
            split=args.test_split,
            transform=test_transform,
        )
    else:
        test_dataset = YoloDetectionDataset(
            args.dataset_yaml,
            split=args.test_split,
            transform=test_transform,
        )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_fn,
    )

    base_model = load_model(
        args.config_file,
        args.pretrained_checkpoint,
        device=str(device),
    ).to(device)
    wrapper = GroundingDINOWrapper(
        model=base_model,
        classes=classes,
        prompt_len=args.prompt_len,
        inject_before_encoder=args.inject_before_encoder,
        use_obb=args.use_obb,
    ).to(device)
    if args.weight is not None:
        weight_ckpt = torch.load(args.weight, map_location="cpu")
        load_wrapper_checkpoint(wrapper, weight_ckpt, device=device)

    log_file = log_dir / "test_log.jsonl"
    if args.use_obb:
        test_metrics = evaluate_obb(
            wrapper,
            test_loader,
            classes,
            device=device,
            iou_threshold=args.eval_iou_threshold,
            ap_score_threshold=args.eval_ap_score_threshold,
            pr_score_threshold=args.eval_pr_score_threshold,
            progress_desc="Testing",
            zero_shot=args.zero_shot,
        )
    else:
        test_metrics = evaluate_detection(
            wrapper,
            test_loader,
            classes,
            device=device,
            iou_threshold=args.eval_iou_threshold,
            ap_score_threshold=args.eval_ap_score_threshold,
            pr_score_threshold=args.eval_pr_score_threshold,
            progress_desc="Testing",
            zero_shot=args.zero_shot,
        )
    vis_info = None
    if args.vis_batch > 0:
        vis_score_thr = (
            args.vis_score_threshold
            if args.vis_score_threshold is not None
            else args.eval_pr_score_threshold
        )
        vis_dir = output_dir / "visualizations"
        num_saved = visualize_batches(
            wrapper=wrapper,
            data_loader=test_loader,
            classes=classes,
            device=device,
            vis_batch=args.vis_batch,
            score_threshold=vis_score_thr,
            vis_dir=vis_dir,
            zero_shot=args.zero_shot,
            use_obb=args.use_obb,
            aggregation_method=args.aggregation_method,
        )
        vis_info = {
            "enabled": True,
            "vis_batch": args.vis_batch,
            "score_threshold": vis_score_thr,
            "aggregation_method": args.aggregation_method,
            "output_dir": str(vis_dir.resolve()),
            "images_saved": num_saved,
        }
        print(
            f"Saved {num_saved} visualization image(s) under {vis_dir} "
            f"(first {args.vis_batch} batch(es), score_threshold={vis_score_thr})."
        )
    print(
        f"mAP50={test_metrics['mAP50']:.4f} "
        f"precision50={test_metrics['precision50']:.4f} "
        f"recall50={test_metrics['recall50']:.4f}"
    )

    test_log = {"test_metrics": test_metrics}
    if vis_info is not None:
        test_log["visualizations"] = vis_info
    if args.output_decode_info:
        decoded_embeddings = wrapper.decode_embeddings(classes)
        test_log["decoded_class_embeddings"] = decoded_embeddings

        print("Decoded class embeddings:")
        for cls_name in classes:
            decoded_info = decoded_embeddings.get(cls_name, {})
            decoded_text = decoded_info.get("decoded_text", "")
            decoded_tokens = decoded_info.get("decoded_tokens", [])
            cosine_similarities = decoded_info.get("cosine_similarities", [])
            print(f"- {cls_name}: {decoded_text}")
            print(f"  tokens={decoded_tokens}")
            print(f"  cosine_similarities={cosine_similarities}")

    with log_file.open("w", encoding="utf-8") as f:
        f.write(json.dumps(test_log, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
