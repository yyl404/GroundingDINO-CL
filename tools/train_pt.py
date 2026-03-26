import argparse
import json
import os
import sys
import time
from typing import List, Tuple

import torch
from torch.utils.data import DataLoader

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from groundingdino.prompt_tuning.losses import compute_prompt_tuning_loss
from groundingdino.prompt_tuning.predictor import load_groundingdino_model
from groundingdino.prompt_tuning.voc import (
    VOC_CLASSES,
    VOCDataset,
    build_caption,
    build_class_token_map,
    build_train_transform,
)
from groundingdino.util.misc import nested_tensor_from_tensor_list


def _build_progress_bar(current: int, total: int, width: int = 28) -> str:
    if total <= 0:
        return "[" + ("-" * width) + "] 0.0%"
    ratio = min(max(current / total, 0.0), 1.0)
    filled = int(ratio * width)
    bar = "#" * filled + "-" * (width - filled)
    return f"[{bar}] {ratio * 100:6.2f}%"


def parse_args():
    parser = argparse.ArgumentParser("Prompt tuning training for GroundingDINO (VOC format)")
    parser.add_argument("--config_file", type=str, required=True, help="Model config path.")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Base model checkpoint path.")
    parser.add_argument("--voc_root", type=str, required=True, help="VOC2007 root path.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save logs and prompt weights.")
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "val", "test", "trainval"],
        help="Training split from ImageSets/Main.",
    )
    parser.add_argument("--device", type=str, default="cuda", help="Device, e.g., cuda or cpu.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size.")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for prompt params.")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay.")
    parser.add_argument("--prompt_length", type=int, default=16, help="Soft prompt token length.")
    parser.add_argument("--prompt_init_std", type=float, default=0.02, help="Prompt init std.")
    parser.add_argument("--save_every", type=int, default=1, help="Save checkpoint every N epochs.")
    parser.add_argument("--resume_prompt", type=str, default="", help="Optional existing prompt weight path.")
    parser.add_argument("--disable_tqdm", action="store_true", help="Disable tqdm progress bar.")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Gradient clipping norm.")
    return parser.parse_args()


def collate_fn(batch: List[Tuple[torch.Tensor, dict]]):
    images, targets = zip(*batch)
    return list(images), list(targets)


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    ckpt_dir = os.path.join(args.output_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    log_file = os.path.join(args.output_dir, "train_log.jsonl")

    dataset = VOCDataset(
        voc_root=args.voc_root,
        split=args.split,
        transforms=build_train_transform(),
        classes=VOC_CLASSES,
        keep_difficult=False,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )

    model = load_groundingdino_model(args.config_file, args.checkpoint_path, device=args.device)
    model.init_prompt_tuning(prompt_length=args.prompt_length, init_std=args.prompt_init_std)
    if args.resume_prompt:
        prompt_state = torch.load(args.resume_prompt, map_location="cpu")
        model.load_prompt_state_dict(prompt_state)
    model.freeze_except_prompt()
    model.to(args.device)
    model.eval()

    caption = build_caption(VOC_CLASSES)
    class_token_map = build_class_token_map(model.tokenizer, VOC_CLASSES)

    optimizer = torch.optim.AdamW(
        [model.prompt_embeddings], lr=args.lr, weight_decay=args.weight_decay
    )

    global_step = 0
    print(
        f"Start training: samples={len(dataset)}, split={args.split}, epochs={args.epochs}, "
        f"batch_size={args.batch_size}, device={args.device}"
    )
    for epoch in range(args.epochs):
        epoch_loss = 0.0
        epoch_cls = 0.0
        epoch_bg = 0.0
        epoch_bbox = 0.0
        epoch_giou = 0.0
        valid_steps = 0
        epoch_start = time.time()
        total_steps = len(dataloader)
        header_printed = False
        if not args.disable_tqdm:
            header = (
                "step/total   avg_loss   avg_cls    avg_bg     avg_bbox   avg_giou   elapsed(s)   progress"
            )

        for step_in_epoch, (images, targets) in enumerate(dataloader, start=1):
            images = [img.to(args.device) for img in images]
            samples = nested_tensor_from_tensor_list(images)
            for target in targets:
                target["boxes"] = target["boxes"].to(args.device)
                target["labels"] = target["labels"].to(args.device)

            outputs = model(samples, captions=[caption] * len(images))
            losses = compute_prompt_tuning_loss(outputs, targets, class_token_map)
            loss = losses["loss"]

            if not torch.isfinite(loss):
                print(
                    f"[Warn] Non-finite loss at epoch={epoch + 1}, step={step_in_epoch}. "
                    "Skipping optimizer step."
                )
                continue

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if args.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_([model.prompt_embeddings], max_norm=args.max_grad_norm)
            optimizer.step()

            global_step += 1
            valid_steps += 1
            epoch_loss += float(loss.item())
            epoch_cls += float(losses["loss_cls"].item())
            epoch_bg += float(losses["loss_bg"].item())
            epoch_bbox += float(losses["loss_bbox"].item())
            epoch_giou += float(losses["loss_giou"].item())

            avg_loss = epoch_loss / valid_steps
            avg_cls = epoch_cls / valid_steps
            avg_bg = epoch_bg / valid_steps
            avg_bbox = epoch_bbox / valid_steps
            avg_giou = epoch_giou / valid_steps

            if not args.disable_tqdm:
                elapsed = time.time() - epoch_start
                progress = _build_progress_bar(step_in_epoch, total_steps)
                status_line = (
                    f"{step_in_epoch:4d}/{total_steps:<7d} "
                    f"{avg_loss:9.4f} "
                    f"{avg_cls:9.4f} "
                    f"{avg_bg:9.4f} "
                    f"{avg_bbox:10.4f} "
                    f"{avg_giou:10.4f} "
                    f"{elapsed:10.1f}   {progress}"
                )
                if not header_printed:
                    print(f"\nEpoch {epoch + 1}/{args.epochs}")
                    print(header)
                    sys.stdout.write(status_line)
                    sys.stdout.flush()
                    header_printed = True
                else:
                    sys.stdout.write("\r" + status_line)
                    sys.stdout.flush()

            log_obj = {
                "step": global_step,
                "epoch": epoch + 1,
                "step_in_epoch": step_in_epoch,
                "loss": float(loss.item()),
                "loss_cls": float(losses["loss_cls"].item()),
                "loss_bg": float(losses["loss_bg"].item()),
                "loss_bbox": float(losses["loss_bbox"].item()),
                "loss_giou": float(losses["loss_giou"].item()),
            }
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_obj, ensure_ascii=False) + "\n")

        if not args.disable_tqdm and header_printed:
            sys.stdout.write("\n")
            sys.stdout.flush()

        avg_epoch_loss = epoch_loss / max(valid_steps, 1)
        epoch_seconds = time.time() - epoch_start
        print(
            f"[Epoch {epoch + 1}/{args.epochs}] "
            f"avg_loss={avg_epoch_loss:.6f} time={epoch_seconds:.1f}s"
        )

        if (epoch + 1) % args.save_every == 0:
            save_path = os.path.join(ckpt_dir, f"prompt_epoch_{epoch + 1}.pth")
            torch.save(model.get_prompt_state_dict(), save_path)
            print(f"Saved prompt checkpoint: {save_path}")

    final_path = os.path.join(args.output_dir, "prompt_final.pth")
    torch.save(model.get_prompt_state_dict(), final_path)
    print(f"Training finished. Final prompt saved to: {final_path}")


if __name__ == "__main__":
    main()
