import argparse
import json
import os
import random
import warnings
from pathlib import Path
from typing import Dict, List, Sequence

import torch
import torchvision.transforms as TVT
from torch import Tensor
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from groundingdino.util.misc import NestedTensor
import groundingdino.datasets.transforms as T

from finetune import GroundingDINOWrapper
from finetune.datasets.yolo import YoloDetectionDataset, _load_yolo_yaml, collate_fn
from finetune.eval import evaluate_detection
from finetune.losses import wrapper_loss
from utils import load_model, load_wrapper_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Train GroundingDINOWrapper on YOLO dataset")
    parser.add_argument("--config_file", type=str, required=True, help="GroundingDINO config path")
    parser.add_argument("--pretrained_checkpoint", type=str, required=True, help="GroundingDINO checkpoint path")
    parser.add_argument("--dataset_yaml", type=str, required=True, help="YOLO dataset yaml path")
    parser.add_argument("--classes", type=str, default=None, help='Comma-separated classes. If omitted, use dataset yaml "names".')

    parser.add_argument("--train_split", type=str, default="train")
    parser.add_argument("--val_split", type=str, default="val")

    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--clip_grad_norm", type=float, default=0.0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--prompt_len", type=int, default=4)
    parser.add_argument("--inject_before_encoder", action="store_true")
    parser.add_argument("--aggregation_method", type=str, choices=["mean", "sum", "max", "min"], default="max")

    parser.add_argument("--eval_iou_threshold", type=float, default=0.5)
    parser.add_argument("--eval_ap_score_threshold", type=float, default=1e-3)
    parser.add_argument("--eval_pr_score_threshold", type=float, default=0.25)

    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--load_wrapper", type=str, default=None, help="Path to wrapper checkpoint")
    parser.add_argument("--resume", action="store_true", help="If set, also restore optimizer/epoch state from --load_wrapper.")
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


def move_targets_to_device(targets: Sequence[Dict[str, Tensor]], device: torch.device) -> List[Dict[str, Tensor]]:
    out: List[Dict[str, Tensor]] = []
    for target in targets:
        out.append({k: (v.to(device) if isinstance(v, (Tensor, NestedTensor)) else v) for k, v in target.items()})
    return out


def save_checkpoint(
    path: Path,
    wrapper: GroundingDINOWrapper,
    optimizer: AdamW=None,
    epoch: int=None,
    metrics: Dict[str, float]=None,
    best_map50: float=None,
    args: argparse.Namespace=None,
) -> None:
    ckpt = {
        "epoch": epoch,
        "classes": list(wrapper.classes),
        "wrapper_kwargs": {
            "prompt_len": args.prompt_len if args else None,
            "inject_before_encoder": args.inject_before_encoder if args else None,
        },
        "wrapper_state_dict": {k: v.detach().cpu() for k, v in wrapper.state_dict().items()},
        "optimizer": optimizer.state_dict() if optimizer else None,
        "metrics": metrics if metrics else None,
        "best_map50": best_map50 if best_map50 else None
    }
    torch.save(ckpt, path)


def train_one_epoch(
    wrapper: GroundingDINOWrapper,
    train_loader: DataLoader,
    optimizer: AdamW,
    classes: Sequence[str],
    device: torch.device,
    args: argparse.Namespace,
) -> Dict[str, float]:
    wrapper.train()
    running = {}
    pbar = tqdm(train_loader, desc="Train")

    for step, (images, targets) in enumerate(pbar, start=1):
        images = images.to(device)
        targets = move_targets_to_device(targets, device=device)

        outputs = wrapper(
            images,
            classes=list(classes),
            aggregation_method=args.aggregation_method,
        )

        loss_dict = wrapper_loss(outputs, targets)
        
        optimizer.zero_grad()
        loss_dict["loss_total"].backward()
        if args.clip_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(wrapper.parameters(), args.clip_grad_norm)
        optimizer.step()

        for key, value in loss_dict.items():
            running[key] = running.get(key, 0.0) + float(value.detach().item())
        averages = {k: v / step for k, v in running.items()}
        pbar.set_postfix({k: f"{v:.4f}" for k, v in averages.items()})

    return {k: v / max(len(train_loader), 1) for k, v in running.items()}


def main() -> None:
    args = parse_args()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    log_dir = output_dir / "logs"
    ckpt_dir = output_dir / "checkpoints"
    log_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

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
    
    train_transform = TVT.Compose(
        [
            TVT.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
            TVT.RandomGrayscale(p=0.05),
            TVT.ToTensor(),
            TVT.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    val_transform = TVT.Compose(
        [
            TVT.ToTensor(),
            TVT.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    train_dataset = YoloDetectionDataset(
        args.dataset_yaml,
        split=args.train_split,
        transform=train_transform,
    )
    val_dataset = YoloDetectionDataset(
        args.dataset_yaml,
        split=args.val_split,
        transform=val_transform,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
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
    ).to(device)
    checkpoint = None
    if args.load_wrapper:
        checkpoint = torch.load(args.load_wrapper, map_location="cpu")
        load_wrapper_checkpoint(wrapper, checkpoint, device=device)

    if args.resume and checkpoint is None:
        raise ValueError("--resume requires --load_wrapper to provide the checkpoint path.")

    trainable_params = [
        p
        for name, p in wrapper.named_parameters()
        if p.requires_grad and not name.startswith("model.")
    ]
    optimizer = AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)

    start_epoch = 1
    best_map50 = -1.0
    if args.resume:
        start_epoch = int(checkpoint["epoch"]) + 1
        best_map50 = float(checkpoint.get("best_map50", -1.0))
        try:
            optimizer.load_state_dict(checkpoint["optimizer"])
        except ValueError as e:
            warnings.warn(
                "Failed to load optimizer state from resume checkpoint; "
                "falling back to a fresh optimizer state. "
                f"Reason: {e}",
                UserWarning,
            )

    log_file = log_dir / "train_log.jsonl"
    for epoch in range(start_epoch, args.epochs + 1):
        train_losses = train_one_epoch(
            wrapper=wrapper,
            train_loader=train_loader,
            optimizer=optimizer,
            classes=classes,
            device=device,
            args=args,
        )
        train_log = {"epoch": epoch, "phase": "train", **train_losses}
        with log_file.open("a", encoding="utf-8") as f:
            f.write(json.dumps(train_log, ensure_ascii=False) + "\n")
        epoch_ckpt = ckpt_dir / f"epoch_{epoch:03d}.pt"
        save_checkpoint(
            path=epoch_ckpt,
            wrapper=wrapper
        )

        val_metrics = evaluate_detection(
            wrapper,
            val_loader,
            classes,
            device=device,
            iou_threshold=args.eval_iou_threshold,
            ap_score_threshold=args.eval_ap_score_threshold,
            pr_score_threshold=args.eval_pr_score_threshold,
            progress_desc=f"Eval Epoch {epoch}",
        )
        val_log = {"epoch": epoch, "phase": "val", **val_metrics}
        with log_file.open("a", encoding="utf-8") as f:
            f.write(json.dumps(val_log, ensure_ascii=False) + "\n")
        
        save_checkpoint(
            path=epoch_ckpt,
            wrapper=wrapper,
            optimizer=optimizer,
            epoch=epoch,
            metrics=val_metrics,
            best_map50=best_map50,
            args=args,
        )

        if val_metrics["mAP50"] > best_map50:
            best_map50 = val_metrics["mAP50"]
            best_ckpt = ckpt_dir / "best_map50.pt"
            save_checkpoint(
                path=best_ckpt,
                wrapper=wrapper,
                optimizer=optimizer,
                epoch=epoch,
                metrics=val_metrics,
                best_map50=best_map50,
                args=args,
            )

        print(
            f"[Epoch {epoch}/{args.epochs}] "
            f"loss_total={train_losses.get('loss_total', 0.0):.4f} "
            f"mAP50={val_metrics['mAP50']:.4f} "
            f"precision50={val_metrics['precision50']:.4f} "
            f"recall50={val_metrics['recall50']:.4f}"
        )


if __name__ == "__main__":
    main()
