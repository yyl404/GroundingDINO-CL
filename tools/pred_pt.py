import argparse
import os
import sys
from typing import Dict, List, Optional, Set, Tuple

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
    build_aux_caption,
    build_caption,
    build_class_token_map,
    build_domain_category_name,
    get_split_present_class_names,
    normalize_pose_text,
)


def parse_args():
    parser = argparse.ArgumentParser("Prompt tuning inference/evaluation for GroundingDINO")
    parser.add_argument("--config_file", type=str, required=True, help="Model config path.")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Base model checkpoint path.")
    parser.add_argument("--prompt_path", type=str, default="", help="Trained prompt weight path.")
    parser.add_argument(
        "--domain_id",
        type=str,
        default="voc2007",
        help="Domain id for reconstructing class-independent mapping when metadata is incomplete.",
    )
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
        "--dual_prompt",
        action="store_true",
        help="Enable dual-prompt text encoding with auxiliary prompts.",
    )
    parser.add_argument(
        "--aux_prompts",
        nargs="*",
        default=[],
        help="Auxiliary prompt candidates for inference mode (e.g. left right frontal).",
    )
    parser.add_argument(
        "--image_exts",
        nargs="+",
        default=[".jpg", ".jpeg", ".png", ".bmp"],
        help="Valid image extensions when input is a directory.",
    )
    return parser.parse_args()


def _load_prompt_checkpoint(ckpt_path: str):
    payload = torch.load(ckpt_path, map_location="cpu")
    if isinstance(payload, dict) and "prompt_state_dict" in payload:
        return payload["prompt_state_dict"], payload.get("metadata", {})
    if isinstance(payload, dict) and "prompt_embeddings" in payload:
        return payload, {}
    raise ValueError(f"Unsupported prompt checkpoint format: {ckpt_path}")


def _infer_prompt_mode(prompt_state: Dict, metadata: Dict) -> str:
    mode = metadata.get("prompt_mode")
    if mode in {"shared", "class_independent"}:
        return mode
    if isinstance(prompt_state, dict) and prompt_state.get("prompt_token_map") is not None:
        return "class_independent"
    return "shared"


def _extract_learned_class_ids(metadata: Dict) -> Optional[Set[int]]:
    cat_to_idx = metadata.get("category_to_prompt_embedding_idx", {}) or {}
    if not cat_to_idx:
        return None
    learned = set()
    for category_name in cat_to_idx.keys():
        class_name = category_name.split(":")[-1]
        if class_name in VOC_CLASSES:
            learned.add(VOC_CLASSES.index(class_name))
    return learned


def _build_token_prompt_map_from_metadata(metadata: Dict, class_token_map: Dict[int, List[int]]) -> Dict[int, int]:
    cat_to_idx = metadata.get("category_to_prompt_embedding_idx", {}) or {}
    token_to_prompt_idx = {}
    for category_name, prompt_idx in cat_to_idx.items():
        class_name = category_name.split(":")[-1]
        if class_name not in VOC_CLASSES:
            continue
        class_id = VOC_CLASSES.index(class_name)
        for token_id in class_token_map[class_id]:
            token_to_prompt_idx[token_id] = int(prompt_idx)
    return token_to_prompt_idx


def _build_phrase_spans(tokenizer, caption: str, phrases: List[str]) -> List[List[int]]:
    caption_ids = tokenizer(caption, add_special_tokens=True)["input_ids"]
    spans = []
    cursor = 0
    for phrase in phrases:
        phrase_ids = tokenizer(phrase, add_special_tokens=False)["input_ids"]
        if not phrase_ids:
            spans.append([])
            continue
        found = []
        for start in range(cursor, len(caption_ids) - len(phrase_ids) + 1):
            if caption_ids[start : start + len(phrase_ids)] == phrase_ids:
                found = list(range(start, start + len(phrase_ids)))
                cursor = start + len(phrase_ids)
                break
        spans.append(found)
    return spans


def _build_dual_combo_prompts(tokenizer, classes: List[str], aux_candidates: List[str]):
    normalized_aux = [normalize_pose_text(x) for x in aux_candidates if str(x).strip()]
    unique_aux = []
    for pose in normalized_aux:
        if pose not in unique_aux:
            unique_aux.append(pose)
    if "none" not in unique_aux:
        unique_aux.append("none")

    combo_class_ids = []
    main_terms = []
    aux_terms = []
    for class_id, class_name in enumerate(classes):
        for aux in unique_aux:
            combo_class_ids.append(class_id)
            main_terms.append(class_name)
            aux_terms.append(aux)

    main_caption = build_caption(main_terms)
    aux_caption = build_aux_caption(aux_terms)
    spans = _build_phrase_spans(tokenizer, main_caption, main_terms)
    class_token_map = {class_id: [] for class_id in range(len(classes))}
    for class_id, token_span in zip(combo_class_ids, spans):
        class_token_map[class_id].extend(token_span)
    return main_caption, aux_caption, class_token_map, unique_aux


def _build_aux_token_map(tokenizer, aux_caption: str, pose_to_idx: Dict[str, int], max_text_len: int) -> torch.Tensor:
    token_map = torch.full((max_text_len,), -1, dtype=torch.long)
    caption_ids = tokenizer(aux_caption, add_special_tokens=True)["input_ids"][:max_text_len]
    for pose, pose_idx in pose_to_idx.items():
        pose_ids = tokenizer(pose, add_special_tokens=False)["input_ids"]
        if not pose_ids:
            continue
        for start in range(len(caption_ids) - len(pose_ids) + 1):
            if caption_ids[start : start + len(pose_ids)] == pose_ids:
                for token_pos in range(start, start + len(pose_ids)):
                    if token_pos < max_text_len:
                        token_map[token_pos] = int(pose_idx)
                break
    return token_map


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


def run_inference_mode(args, model, class_token_map, aux_pose_to_idx: Dict[str, int]):
    os.makedirs(args.output_dir, exist_ok=True)
    vis_dir = os.path.join(args.output_dir, "vis")
    txt_dir = os.path.join(args.output_dir, "txt")
    if args.save_mode in {"vis", "both"}:
        os.makedirs(vis_dir, exist_ok=True)
    if args.save_mode in {"txt", "both"}:
        os.makedirs(txt_dir, exist_ok=True)

    transform = build_infer_transform()
    caption = build_caption(VOC_CLASSES)
    aux_caption = ""
    class_token_map_infer = class_token_map
    aux_prompt_token_map_override = None
    if args.dual_prompt:
        caption, aux_caption, class_token_map_infer, used_aux = _build_dual_combo_prompts(
            tokenizer=model.tokenizer,
            classes=VOC_CLASSES,
            aux_candidates=args.aux_prompts,
        )
        if aux_pose_to_idx:
            aux_prompt_token_map_override = _build_aux_token_map(
                tokenizer=model.tokenizer,
                aux_caption=aux_caption,
                pose_to_idx=aux_pose_to_idx,
                max_text_len=model.max_text_len,
            )
        print(
            f"Dual-prompt inference enabled: classes={len(VOC_CLASSES)}, "
            f"aux_candidates_with_none={len(used_aux)}, total_combinations={len(VOC_CLASSES) * len(used_aux)}"
        )
    image_paths = list_images(args.input, args.image_exts)
    if not image_paths:
        print("No images found for inference.")
        return

    model.eval()
    for image_path in image_paths:
        image_rgb, image_tensor = read_image_for_model(image_path, transform)
        image_tensor = image_tensor.to(args.device)
        with torch.no_grad():
            if args.dual_prompt:
                model_kwargs = {"captions": [caption], "aux_captions": [aux_caption]}
                if aux_prompt_token_map_override is not None:
                    model_kwargs["aux_prompt_token_map_override"] = aux_prompt_token_map_override
                outputs = model(image_tensor[None], **model_kwargs)
            else:
                outputs = model(image_tensor[None], captions=[caption])

        h, w = image_rgb.shape[:2]
        boxes_cxcywh, scores, class_ids = decode_predictions(
            outputs=outputs,
            class_token_map=class_token_map_infer,
            box_threshold=args.box_threshold,
            text_threshold=args.text_threshold,
            num_classes=len(VOC_CLASSES),
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


def run_test_mode(
    args,
    model,
    class_token_map,
    eval_class_ids: List[int],
    aux_pose_to_idx: Dict[str, int],
):
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
            class_token_map_eval = class_token_map
            if args.dual_prompt:
                target_poses = [normalize_pose_text(x) for x in target.get("poses", [])]
                caption_eval, aux_caption_eval, class_token_map_eval, _ = _build_dual_combo_prompts(
                    tokenizer=model.tokenizer,
                    classes=VOC_CLASSES,
                    aux_candidates=target_poses,
                )
                model_kwargs = {"captions": [caption_eval], "aux_captions": [aux_caption_eval]}
                if aux_pose_to_idx:
                    model_kwargs["aux_prompt_token_map_override"] = _build_aux_token_map(
                        tokenizer=model.tokenizer,
                        aux_caption=aux_caption_eval,
                        pose_to_idx=aux_pose_to_idx,
                        max_text_len=model.max_text_len,
                    )
                outputs = model(image_tensor[None], **model_kwargs)
            else:
                outputs = model(image_tensor[None], captions=[caption])

        orig_h = int(target["orig_size"][0].item())
        orig_w = int(target["orig_size"][1].item())
        boxes_cxcywh, scores, class_ids = decode_predictions(
            outputs=outputs,
            class_token_map=class_token_map_eval,
            box_threshold=args.box_threshold,
            text_threshold=args.text_threshold,
            num_classes=len(VOC_CLASSES),
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

    metrics = evaluate_voc_map(
        all_predictions,
        all_ground_truths,
        num_classes=len(VOC_CLASSES),
        class_names=VOC_CLASSES,
        eval_class_ids=eval_class_ids,
    )
    metrics_path = os.path.join(args.output_dir, f"metrics_{args.split}.json")
    save_metrics(metrics_path, metrics)
    print(
        f"Evaluation done on split={args.split}. classes={len(eval_class_ids)} "
        f"mAP@0.5={metrics['mAP@0.5']:.4f}"
    )
    print(f"Metrics saved to: {metrics_path}")


def main():
    args = parse_args()
    model = load_groundingdino_model(args.config_file, args.checkpoint_path, device=args.device)
    class_token_map = build_class_token_map(model.tokenizer, VOC_CLASSES)

    metadata = {}
    learned_class_ids = set(range(len(VOC_CLASSES)))
    aux_pose_to_idx: Dict[str, int] = {}
    if args.prompt_path:
        prompt_state, metadata = _load_prompt_checkpoint(args.prompt_path)
        prompt_mode = _infer_prompt_mode(prompt_state, metadata)
        prompt_emb = prompt_state.get("prompt_embeddings")
        if prompt_emb is None:
            raise KeyError("prompt_embeddings was not found in prompt checkpoint.")
        prompt_len = int(prompt_emb.shape[0])
        model.init_prompt_tuning(
            prompt_length=prompt_len,
            num_embeddings=prompt_len,
            mode=prompt_mode,
        )
        model.load_prompt_state_dict(prompt_state)
        if prompt_mode == "class_independent" and model.prompt_token_map is None:
            if not metadata:
                print("[Warning] Missing metadata for class-independent prompt; fallback to shared decode behavior.")
            else:
                token_to_prompt_idx = _build_token_prompt_map_from_metadata(metadata, class_token_map)
                model.set_prompt_token_map(token_to_prompt_idx)
        extracted = _extract_learned_class_ids(metadata)
        if extracted is not None:
            learned_class_ids = extracted
        pose_vocab = metadata.get("pose_vocabulary", []) or []
        aux_pose_to_idx = {normalize_pose_text(x): idx for idx, x in enumerate(pose_vocab)}
        print(f"Loaded prompt weights from: {args.prompt_path}")
        print(f"Prompt mode inferred from checkpoint: {_infer_prompt_mode(prompt_state, metadata)}")
    else:
        model.init_prompt_tuning(prompt_length=args.prompt_length, mode="shared")
        if args.dual_prompt:
            # Fallback initialization for dual inference without prompt checkpoint.
            model.init_aux_prompt_tuning(prompt_length=args.prompt_length, mode="shared")

    if args.dual_prompt and model.aux_prompt_embeddings is None:
        if aux_pose_to_idx:
            model.init_aux_prompt_tuning(num_embeddings=len(aux_pose_to_idx), mode="class_independent")
        else:
            model.init_aux_prompt_tuning(prompt_length=args.prompt_length, mode="shared")

    model.to(args.device)

    if args.test:
        dataset_classes = get_split_present_class_names(args.input, args.split, classes=VOC_CLASSES)
        dataset_class_ids = {VOC_CLASSES.index(c) for c in dataset_classes}
        eval_class_ids = sorted(dataset_class_ids.intersection(learned_class_ids))
        missing_ids = sorted(dataset_class_ids - learned_class_ids)
        if missing_ids:
            missing_names = [VOC_CLASSES[i] for i in missing_ids]
            print(
                "[Warning] Dataset contains classes not covered by current prompt embedding: "
                + ", ".join(missing_names)
            )
        if not eval_class_ids:
            print("[Warning] No overlap between learned classes and dataset classes; skip mAP evaluation.")
            return
        run_test_mode(
            args,
            model,
            class_token_map,
            eval_class_ids=eval_class_ids,
            aux_pose_to_idx=aux_pose_to_idx,
        )
    else:
        run_inference_mode(args, model, class_token_map, aux_pose_to_idx=aux_pose_to_idx)


if __name__ == "__main__":
    main()
