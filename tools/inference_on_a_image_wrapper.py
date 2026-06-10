import argparse
import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import warnings
from quiet_warnings import silence_known_training_warnings

silence_known_training_warnings()

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

import groundingdino.datasets.transforms as T
from groundingdino.util import box_ops
from groundingdino.util.utils import get_phrases_from_posmap
from groundingdino.util.vl_utils import create_positive_map_from_span

from finetune import GroundingDINOWrapper
from utils import (
    configure_model_trainable_flags,
    load_model,
    load_wrapper_checkpoint,
    parse_lora_layers,
)


def plot_boxes_to_image(image_pil, tgt):
    H, W = tgt["size"]
    boxes = tgt["boxes"]
    labels = tgt["labels"]
    assert len(boxes) == len(labels), "boxes and labels must have same length"

    draw = ImageDraw.Draw(image_pil)
    mask = Image.new("L", image_pil.size, 0)
    mask_draw = ImageDraw.Draw(mask)

    for box, label in zip(boxes, labels):
        # from 0..1 to 0..W, 0..H
        box = box * torch.Tensor([W, H, W, H])
        # from xywh to xyxy
        box[:2] -= box[2:] / 2
        box[2:] += box[:2]
        # random color
        color = tuple(np.random.randint(0, 255, size=3).tolist())
        # draw
        x0, y0, x1, y1 = box
        x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)

        draw.rectangle([x0, y0, x1, y1], outline=color, width=6)

        font = ImageFont.load_default()
        if hasattr(font, "getbbox"):
            bbox = draw.textbbox((x0, y0), str(label), font)
        else:
            w, h = draw.textsize(str(label), font)
            bbox = (x0, y0, w + x0, y0 + h)
        draw.rectangle(bbox, fill=color)
        draw.text((x0, y0), str(label), fill="white")

        mask_draw.rectangle([x0, y0, x1, y1], fill=255, width=6)

    return image_pil, mask


def load_image(image_path):
    # load image
    image_pil = Image.open(image_path).convert("RGB")  # load image

    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image, _ = transform(image_pil, None)  # 3, h, w
    return image_pil, image


def get_grounding_output(model, image, classes, box_threshold, aggregation_method='mean', with_logits=True, cpu_only=False):
    device = "cuda" if not cpu_only else "cpu"
    model = model.to(device)
    image = image.to(device)
    if not classes:
        # if classes are not designated, use the built-in vocabulary of wrapper
        classes = model.classes
    with torch.no_grad():
        outputs = model(image[None], classes, aggregation_method=aggregation_method)
    boxes = outputs["pred_boxes"][0]  # (nq, 4)
    class_logits = outputs["pred_class_logits"][0] # (nq, n_classes)

    # filter output
    filt_mask = class_logits.max(dim=1).values > box_threshold
    logits_filt = class_logits[filt_mask]  # num_filt, n_classes
    boxes_filt = boxes[filt_mask]  # num_filt, 4

    # get phrase
    pred_phrases = []
    for logit in logits_filt:
        pred_class_id = int(torch.argmax(logit))
        pred_phrase = classes[pred_class_id]
        if with_logits:
            pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
        else:
            pred_phrases.append(pred_phrase)

    return boxes_filt.cpu(), pred_phrases


if __name__ == "__main__":

    parser = argparse.ArgumentParser("Grounding DINO example", add_help=True)
    parser.add_argument("--config_file", "-c", type=str, required=True, help="path to config file")
    parser.add_argument(
        "--checkpoint_path", "-p", type=str, required=True, help="path to checkpoint file"
    )
    parser.add_argument(
        "--wrapper_checkpoint", "-wp",
        type=str,
        default=None,
        help="path to local wrapper checkpoint (.pt). If provided, load wrapper embeddings/classes from it.",
    )
    parser.add_argument("--image_path", "-i", type=str, required=True, help="path to image file")
    parser.add_argument("--classes", "-cls", type=str, help='Comma-separated class names, e.g. "chair,person,dog".'
                        'If not provided, use the built-in vocabulary of model wrapper.')
    parser.add_argument(
        "--output_dir", "-o", type=str, default="outputs", required=True, help="output directory"
    )

    parser.add_argument("--box_threshold", type=float, default=0.3, help="box threshold")
    parser.add_argument("--aggregation_method", type=str, choices=['mean', 'sum', 'max', 'min'], default='mean')
    parser.add_argument(
        "--text_mode",
        type=str,
        choices=["prompt", "fixed"],
        default="prompt",
        help="Text input mode: 'prompt' for learnable prompt embeddings, 'fixed' for class-name captions.",
    )
    parser.add_argument(
        "--param_tune",
        type=str,
        choices=["full", "lora", "delta", "frozen"],
        default="delta",
        help="Parameter tuning mode: full non-text, LoRA, output delta heads, or fully frozen.",
    )
    parser.add_argument("--prompt_len", type=int, default=4, help="prompt token length per class")
    parser.add_argument("--inject_before_encoder",
        action="store_true",
        help="whether to inject learnable class embeddings before text encoder",
    )
    parser.add_argument("--lora_r", type=int, default=8, help="LoRA rank.")
    parser.add_argument("--lora_alpha", type=float, default=16.0, help="LoRA alpha scaling.")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout.")
    parser.add_argument(
        "--lora_targets",
        type=str,
        default="value_proj,output_proj,linear1,linear2",
        help="Comma-separated module-name keywords to select base nn.Linear layers for LoRA injection.",
    )
    parser.add_argument(
        "--lora_layers",
        type=str,
        default="all",
        help="Transformer layer indices for LoRA, e.g. '0,1,2'; use 'all' for all layers.",
    )
    parser.add_argument("--cpu-only", action="store_true", help="running on cpu only!, default=False")
    args = parser.parse_args()

    # cfg
    config_file = args.config_file  # change the path of the model config file
    checkpoint_path = args.checkpoint_path  # change the path of the model
    wrapper_checkpoint = args.wrapper_checkpoint
    image_path = args.image_path
    classes = [c.strip() for c in args.classes.split(",") if c.strip()] if args.classes else None
    output_dir = args.output_dir
    box_threshold = args.box_threshold
    aggregation_method = args.aggregation_method
    prompt_len = args.prompt_len
    inject_before_encoder = args.inject_before_encoder
    lora_targets = [x.strip() for x in args.lora_targets.split(",") if x.strip()]
    lora_layers = parse_lora_layers(args.lora_layers)
    if not lora_targets:
        raise ValueError("--lora_targets must include at least one non-empty target.")

    # make dir
    os.makedirs(output_dir, exist_ok=True)
    # load image
    image_pil, image = load_image(image_path)
    # load model
    model = load_model(
        config_file,
        checkpoint_path,
        device=("cpu" if args.cpu_only else "cuda"),
    )
    # wrap model
    model = GroundingDINOWrapper(classes=classes,
                                 model=model,
                                 prompt_len=prompt_len,
                                 text_mode=args.text_mode,
                                 inject_before_encoder=inject_before_encoder,
                                 use_lora=(args.param_tune == "lora"),
                                 lora_r=args.lora_r,
                                 lora_alpha=args.lora_alpha,
                                 lora_dropout=args.lora_dropout,
                                 lora_targets=lora_targets,
                                 lora_layers=lora_layers)
    configure_model_trainable_flags(
        model,
        text_mode=args.text_mode,
        param_tune=args.param_tune,
    )
    if wrapper_checkpoint is not None:
        load_wrapper_checkpoint(
            model,
            torch.load(wrapper_checkpoint, map_location="cpu"),
            device=("cpu" if args.cpu_only else "cuda"),
        )

    # visualize raw image
    image_pil.save(os.path.join(output_dir, "raw_image.jpg"))

    # run model
    boxes_filt, pred_phrases = get_grounding_output(
        model, image, classes, box_threshold, aggregation_method, cpu_only=args.cpu_only
    )

    # visualize pred
    size = image_pil.size
    pred_dict = {
        "boxes": boxes_filt,
        "size": [size[1], size[0]],  # H,W
        "labels": pred_phrases,
    }
    # import ipdb; ipdb.set_trace()
    image_with_box = plot_boxes_to_image(image_pil, pred_dict)[0]
    image_with_box.save(os.path.join(output_dir, "pred.jpg"))
