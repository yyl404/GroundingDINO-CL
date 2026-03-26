import argparse
import os
import sys
from typing import Dict, List, Tuple

import numpy as np
import torch
from PIL import Image
from torchvision.ops import box_convert

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import groundingdino.datasets.transforms as T
from groundingdino.prompt_tuning.predictor import (
    decode_predictions,
    evaluate_voc_map,
    load_groundingdino_model,
    save_detection_txt,
    save_metrics,
)
from groundingdino.prompt_tuning.voc import (
    VOC_CLASSES,
    VOCDataset,
    build_caption,
    build_class_token_map,
)


def parse_args():
    parser = argparse.ArgumentParser("Prompt tuning inference/evaluation for GroundingDINO")
    parser.add_argument("--config_file", type=str, required=True, help="Model config path.")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Base model checkpoint path.")
    parser.add_argument("--prompt_path", type=str, default="", help="Trained prompt weight path.")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Image path, image directory, or VOC root directory when --test is enabled.",
    )
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save outputs.")
    parser.add_argument(
        "--save_mode",
        type=str,
        default="both",
        choices=["vis", "txt", "both"],
        help="Save visualization, txt, or both.",
    )
    parser.add_argument("--device", type=str, default="cuda", help="Device, e.g., cuda or cpu.")
    parser.add_argument("--box_threshold", type=float, default=0.3, help="Box filtering threshold.")
    parser.add_argument("--text_threshold", type=float, default=0.25, help="Class token score threshold.")
    parser.add_argument("--test", action="store_true", help="Enable VOC dataset test mode and compute mAP.")
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test", "trainval"],
        help="VOC split for test mode.",
    )
    parser.add_argument("--prompt_length", type=int, default=16, help="Prompt length (must match training).")
    parser.add_argument(
        "--image_exts",
        nargs="+",
        default=[".jpg", ".jpeg", ".png", ".bmp"],
        help="Valid image extensions when input is a directory.",
    )
    return parser.parse_args()


def build_infer_transform():
    return T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )


def read_image_for_model(image_path: str, transform):
    image_pil = Image.open(image_path).convert("RGB")
    image_np = np.asarray(image_pil)
    image_tensor, _ = transform(image_pil, None)
    return image_np, image_tensor


def to_absolute_xyxy(boxes_cxcywh: np.ndarray, width: int, height: int) -> np.ndarray:
    if boxes_cxcywh.shape[0] == 0:
        return np.zeros((0, 4), dtype=np.float32)
    boxes = torch.from_numpy(boxes_cxcywh) * torch.tensor([width, height, width, height], dtype=torch.float32)
    return box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy().astype(np.float32)


def draw_detections(image_bgr: np.ndarray, boxes_xyxy: np.ndarray, scores: np.ndarray, class_ids: np.ndarray):
    import cv2

    vis = image_bgr.copy()
    for box, score, class_id in zip(boxes_xyxy, scores, class_ids):
        x0, y0, x1, y1 = [int(v) for v in box.tolist()]
        color = (0, 255, 0)
        cv2.rectangle(vis, (x0, y0), (x1, y1), color, 2)
        label = f"{VOC_CLASSES[int(class_id)]} {float(score):.2f}"
        cv2.putText(vis, label, (x0, max(y0 - 5, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return vis


def list_images(input_path: str, image_exts: List[str]) -> List[str]:
    if os.path.isfile(input_path):
        return [input_path]
    if os.path.isdir(input_path):
        ext_set = {x.lower() for x in image_exts}
        names = sorted(os.listdir(input_path))
        return [
            os.path.join(input_path, name)
            for name in names
            if os.path.splitext(name)[1].lower() in ext_set
        ]
    raise FileNotFoundError(f"Input path does not exist: {input_path}")


def run_inference_mode(args, model, class_token_map):
    os.makedirs(args.output_dir, exist_ok=True)
    vis_dir = os.path.join(args.output_dir, "vis")
    txt_dir = os.path.join(args.output_dir, "txt")
    if args.save_mode in {"vis", "both"}:
        os.makedirs(vis_dir, exist_ok=True)
    if args.save_mode in {"txt", "both"}:
        os.makedirs(txt_dir, exist_ok=True)

    transform = build_infer_transform()
    caption = build_caption(VOC_CLASSES)
    image_paths = list_images(args.input, args.image_exts)
    if not image_paths:
        print("No images found for inference.")
        return

    model.eval()
    for image_path in image_paths:
        image_rgb, image_tensor = read_image_for_model(image_path, transform)
        image_tensor = image_tensor.to(args.device)
        with torch.no_grad():
            outputs = model(image_tensor[None], captions=[caption])

        h, w = image_rgb.shape[:2]
        boxes_cxcywh, scores, class_ids = decode_predictions(
            outputs=outputs,
            class_token_map=class_token_map,
            box_threshold=args.box_threshold,
            text_threshold=args.text_threshold,
        )
        boxes_xyxy = to_absolute_xyxy(boxes_cxcywh, w, h)

        base = os.path.splitext(os.path.basename(image_path))[0]
        if args.save_mode in {"txt", "both"}:
            save_detection_txt(os.path.join(txt_dir, f"{base}.txt"), boxes_xyxy, scores, class_ids, VOC_CLASSES)
        if args.save_mode in {"vis", "both"}:
            import cv2

            image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
            vis = draw_detections(image_bgr, boxes_xyxy, scores, class_ids)
            cv2.imwrite(os.path.join(vis_dir, f"{base}.jpg"), vis)
        print(f"Inference done: {image_path}, detections={len(scores)}")


def run_test_mode(args, model, class_token_map):
    os.makedirs(args.output_dir, exist_ok=True)
    caption = build_caption(VOC_CLASSES)
    transform = build_infer_transform()
    dataset = VOCDataset(
        voc_root=args.input,
        split=args.split,
        transforms=transform,
        classes=VOC_CLASSES,
        keep_difficult=True,
    )

    all_predictions: Dict[str, List[Tuple[float, int, np.ndarray]]] = {}
    all_ground_truths: Dict[str, List[Tuple[int, np.ndarray, int]]] = {}
    txt_dir = os.path.join(args.output_dir, "txt")
    if args.save_mode in {"txt", "both"}:
        os.makedirs(txt_dir, exist_ok=True)

    model.eval()
    for i in range(len(dataset)):
        image_tensor, target = dataset[i]
        image_name = target["image_name"]
        image_tensor = image_tensor.to(args.device)

        with torch.no_grad():
            outputs = model(image_tensor[None], captions=[caption])

        orig_h = int(target["orig_size"][0].item())
        orig_w = int(target["orig_size"][1].item())
        boxes_cxcywh, scores, class_ids = decode_predictions(
            outputs=outputs,
            class_token_map=class_token_map,
            box_threshold=args.box_threshold,
            text_threshold=args.text_threshold,
        )
        boxes_xyxy = to_absolute_xyxy(boxes_cxcywh, orig_w, orig_h)

        all_predictions[image_name] = [
            (float(score), int(class_id), box.astype(np.float32))
            for score, class_id, box in zip(scores, class_ids, boxes_xyxy)
        ]
        if args.save_mode in {"txt", "both"}:
            save_detection_txt(os.path.join(txt_dir, f"{image_name}.txt"), boxes_xyxy, scores, class_ids, VOC_CLASSES)

        gt_boxes = target["boxes_abs"].numpy().astype(np.float32)
        gt_labels = target["labels"].numpy().astype(np.int64)
        gt_difficult = target["difficult"].numpy().astype(np.int64)
        all_ground_truths[image_name] = [
            (int(label), box.astype(np.float32), int(diff))
            for label, box, diff in zip(gt_labels, gt_boxes, gt_difficult)
        ]

        if (i + 1) % 100 == 0:
            print(f"Processed {i + 1}/{len(dataset)} samples for evaluation.")

    metrics = evaluate_voc_map(all_predictions, all_ground_truths, num_classes=len(VOC_CLASSES))
    metrics_path = os.path.join(args.output_dir, f"metrics_{args.split}.json")
    save_metrics(metrics_path, metrics)
    print(f"Evaluation done on split={args.split}. mAP@0.5={metrics['mAP@0.5']:.4f}")
    print(f"Metrics saved to: {metrics_path}")


def main():
    args = parse_args()
    model = load_groundingdino_model(args.config_file, args.checkpoint_path, device=args.device)
    model.init_prompt_tuning(prompt_length=args.prompt_length)
    if args.prompt_path:
        prompt_state = torch.load(args.prompt_path, map_location="cpu")
        model.load_prompt_state_dict(prompt_state)
        print(f"Loaded prompt weights from: {args.prompt_path}")
    model.to(args.device)

    class_token_map = build_class_token_map(model.tokenizer, VOC_CLASSES)
    if args.test:
        run_test_mode(args, model, class_token_map)
    else:
        run_inference_mode(args, model, class_token_map)


if __name__ == "__main__":
    main()
