"""Core-set selection for YOLO datasets using GroundingDINO backbone features.

Usage:
python coreset_selection/select_coreset.py \
    --src_data /path/to/data.yaml \
    --out_dir /path/to/output_dir \
    --num_sample 200

`--num_sample` can be:
1) an integer: distributed to existing splits by original split size ratio;
2) a list string: e.g. "[120, 40, 40]" or "120,40,40", mapped to train/val/test
   in the order they exist in source yaml.
"""

from __future__ import annotations

import argparse
import ast
import json
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn.functional as F
import torchvision.transforms as TVT
import yaml
from PIL import Image
from tqdm import tqdm

from finetune.datasets.yolo import _collect_images_from_dir, _label_path_for_image, _load_yolo_yaml
from groundingdino.models import build_model
from groundingdino.util.misc import nested_tensor_from_tensor_list
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_model(model_config_path: str, model_checkpoint_path: str, device: str = "cuda"):
    args = SLConfig.fromfile(model_config_path)
    args.device = str(device)
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu", weights_only=False)
    load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    print(load_res)
    model.eval()
    return model


@dataclass
class SampleInfo:
    split: str
    image_path: Path
    label_path: Path
    image_dir: Path
    source_dir_index: int
    labels: torch.Tensor  # [N]
    boxes: torch.Tensor  # [N, 4] (cx, cy, w, h), normalized


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Core-set selection for YOLO dataset splits")
    parser.add_argument("--src_data", type=str, required=True, help="Source YOLO data.yaml path")
    parser.add_argument("--out_dir", type=str, required=True, help="Output directory for selected core-set")
    parser.add_argument(
        "--num_sample",
        type=str,
        required=True,
        help='Total sample count (e.g. "300") or split counts (e.g. "[200,80,20]" / "200,80,20").',
    )
    parser.add_argument(
        "--config_file",
        type=str,
        default="groundingdino/config/GroundingDINO_SwinT_OGC.py",
        help="GroundingDINO config path",
    )
    parser.add_argument(
        "--pretrained_checkpoint",
        type=str,
        default="weights/groundingdino_swint_ogc.pth",
        help="GroundingDINO checkpoint path",
    )
    parser.add_argument("--lambda_weight", type=float, default=1.0, help="Lambda in score = lambda*s1 - s2")
    parser.add_argument(
        "--global_feature_selection",
        action="store_true",
        help="Enable class-agnostic coreset selection based on image-level global features.",
    )
    parser.add_argument("--device", type=str, default="cuda", help="cuda / cuda:0 / cpu")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--save_meta", action="store_true", help="Save feature/selection metadata to json")
    return parser.parse_args()


def parse_num_sample(raw: str, split_order: Sequence[str], split_sizes: Dict[str, int]) -> Dict[str, int]:
    value = raw.strip()
    if value.isdigit():
        total = int(value)
        total = max(total, 0)
        if total == 0:
            return {s: 0 for s in split_order}
        total_size = sum(split_sizes[s] for s in split_order)
        if total_size == 0:
            return {s: 0 for s in split_order}
        raw_alloc = {s: total * split_sizes[s] / total_size for s in split_order}
        alloc = {s: int(raw_alloc[s]) for s in split_order}
        left = total - sum(alloc.values())
        ranked = sorted(split_order, key=lambda s: (raw_alloc[s] - alloc[s]), reverse=True)
        for i in range(left):
            alloc[ranked[i % len(ranked)]] += 1
        return alloc

    parsed = None
    try:
        parsed = ast.literal_eval(value)
    except Exception:
        pass
    if parsed is None:
        parsed = [x.strip() for x in value.split(",") if x.strip()]
    if not isinstance(parsed, (list, tuple)):
        raise ValueError(f"Unsupported --num_sample format: {raw}")

    nums = [int(x) for x in parsed]
    if len(nums) != len(split_order):
        raise ValueError(
            f"--num_sample list length must match split count ({len(split_order)}), got {len(nums)}"
        )
    return {s: max(nums[i], 0) for i, s in enumerate(split_order)}


def load_split_samples(dataset_yaml: str) -> Tuple[Dict[str, List[SampleInfo]], Dict]:
    cfg = _load_yolo_yaml(dataset_yaml)
    split_to_samples: Dict[str, List[SampleInfo]] = {}
    for split, image_dirs in cfg["splits"].items():
        samples: List[SampleInfo] = []
        for dir_idx, image_dir in enumerate(image_dirs):
            image_dir_path = Path(image_dir).expanduser().resolve()
            labels_dir_path = Path(str(image_dir).replace("images", "labels")).expanduser().resolve()
            for image_path_str in _collect_images_from_dir(image_dir_path):
                image_path = Path(image_path_str).resolve()
                label_path = Path(_label_path_for_image(image_path, image_dir_path, labels_dir_path)).resolve()
                labels, boxes = read_yolo_label_file(label_path)
                samples.append(
                    SampleInfo(
                        split=split,
                        image_path=image_path,
                        label_path=label_path,
                        image_dir=image_dir_path,
                        source_dir_index=dir_idx,
                        labels=labels,
                        boxes=boxes,
                    )
                )
        split_to_samples[split] = samples
    return split_to_samples, cfg


def read_yolo_label_file(label_path: Path) -> Tuple[torch.Tensor, torch.Tensor]:
    if not label_path.is_file():
        return torch.zeros(0, dtype=torch.long), torch.zeros((0, 4), dtype=torch.float32)
    labels: List[int] = []
    boxes: List[List[float]] = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        labels.append(int(float(parts[0])))
        boxes.append([float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])])
    if not labels:
        return torch.zeros(0, dtype=torch.long), torch.zeros((0, 4), dtype=torch.float32)
    return torch.tensor(labels, dtype=torch.long), torch.tensor(boxes, dtype=torch.float32)


def build_image_transform() -> TVT.Compose:
    return TVT.Compose(
        [
            TVT.ToTensor(),
            TVT.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )


def clamp_roi_bounds(x0: int, y0: int, x1: int, y1: int, w: int, h: int) -> Tuple[int, int, int, int]:
    x0 = max(0, min(x0, w - 1))
    y0 = max(0, min(y0, h - 1))
    x1 = max(x0 + 1, min(x1, w))
    y1 = max(y0 + 1, min(y1, h))
    return x0, y0, x1, y1


@torch.no_grad()
def extract_per_image_multiscale_class_features(
    model,
    image_path: Path,
    labels: torch.Tensor,
    boxes_cxcywh: torch.Tensor,
    num_classes: int,
    transform: TVT.Compose,
    device: torch.device,
) -> List[torch.Tensor]:
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).to(device)
    nested = nested_tensor_from_tensor_list([image_tensor]).to(device)

    model.set_image_tensor(nested)
    features = model.features
    model.unset_image_tensor()

    per_level: List[torch.Tensor] = []
    if boxes_cxcywh.numel() == 0:
        for feat in features:
            fmap = feat.tensors[0]
            per_level.append(torch.zeros((num_classes, fmap.shape[0]), dtype=torch.float32))
        return per_level

    boxes = boxes_cxcywh.to(device=device, dtype=torch.float32).clone()
    cx, cy, bw, bh = boxes.unbind(dim=1)
    xmin = (cx - bw * 0.5).clamp(0.0, 1.0)
    ymin = (cy - bh * 0.5).clamp(0.0, 1.0)
    xmax = (cx + bw * 0.5).clamp(0.0, 1.0)
    ymax = (cy + bh * 0.5).clamp(0.0, 1.0)

    labels_local = labels.to(device=device, dtype=torch.long)

    for feat in features:
        fmap = feat.tensors[0]
        channels, h, w = fmap.shape
        class_sum = torch.zeros((num_classes, channels), device=device, dtype=fmap.dtype)
        class_cnt = torch.zeros((num_classes,), device=device, dtype=fmap.dtype)

        for idx in range(labels_local.numel()):
            c = int(labels_local[idx].item())
            if c < 0 or c >= num_classes:
                continue
            x0 = int(torch.floor(xmin[idx] * w).item())
            y0 = int(torch.floor(ymin[idx] * h).item())
            x1 = int(torch.ceil(xmax[idx] * w).item())
            y1 = int(torch.ceil(ymax[idx] * h).item())
            x0, y0, x1, y1 = clamp_roi_bounds(x0, y0, x1, y1, w, h)
            roi_feat = fmap[:, y0:y1, x0:x1].mean(dim=(1, 2))
            class_sum[c] += roi_feat
            class_cnt[c] += 1.0

        denom = class_cnt.clamp_min(1.0).unsqueeze(1)
        class_avg = class_sum / denom
        zero_mask = class_cnt <= 0
        class_avg[zero_mask] = 0.0
        per_level.append(class_avg.detach().cpu().to(torch.float32))

    return per_level


@torch.no_grad()
def extract_per_image_multiscale_global_features(
    model,
    image_path: Path,
    transform: TVT.Compose,
    device: torch.device,
) -> List[torch.Tensor]:
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).to(device)
    nested = nested_tensor_from_tensor_list([image_tensor]).to(device)

    model.set_image_tensor(nested)
    features = model.features
    model.unset_image_tensor()

    per_level: List[torch.Tensor] = []
    for feat in features:
        fmap = feat.tensors[0]
        global_feat = fmap.mean(dim=(1, 2))
        per_level.append(global_feat.detach().cpu().to(torch.float32))
    return per_level


def class_instance_count(samples: Sequence[SampleInfo], num_classes: int) -> torch.Tensor:
    counts = torch.zeros((num_classes,), dtype=torch.long)
    for sample in samples:
        if sample.labels.numel() == 0:
            continue
        for c in sample.labels.tolist():
            if 0 <= c < num_classes:
                counts[c] += 1
    return counts


def allocate_per_class(total_k: int, class_counts: torch.Tensor) -> torch.Tensor:
    num_classes = int(class_counts.numel())
    out = torch.zeros((num_classes,), dtype=torch.long)
    if total_k <= 0:
        return out
    count_sum = int(class_counts.sum().item())
    if count_sum <= 0:
        return out
    raw = class_counts.float() * (float(total_k) / float(count_sum))
    out = torch.floor(raw).to(torch.long)
    remain = total_k - int(out.sum().item())
    if remain > 0:
        frac = raw - out.float()
        order = torch.argsort(frac, descending=True)
        for i in range(remain):
            out[int(order[i % len(order)].item())] += 1
    return out


def compute_similarity_matrix_for_class(per_image_feats: List[List[torch.Tensor]], cls_idx: int) -> torch.Tensor:
    if not per_image_feats:
        return torch.zeros((0, 0), dtype=torch.float32)
    num_images = len(per_image_feats)
    num_levels = len(per_image_feats[0])
    sim_sum = torch.zeros((num_images, num_images), dtype=torch.float32)
    for lvl in range(num_levels):
        feats_lvl = torch.stack([per_image_feats[i][lvl][cls_idx] for i in range(num_images)], dim=0)  # [N, D]
        feats_lvl = F.normalize(feats_lvl, p=2, dim=1, eps=1e-12)
        sim_sum += feats_lvl @ feats_lvl.t()
    sim_avg = sim_sum / float(max(num_levels, 1))
    return sim_avg


def compute_similarity_matrix_for_global(per_image_feats: List[List[torch.Tensor]]) -> torch.Tensor:
    if not per_image_feats:
        return torch.zeros((0, 0), dtype=torch.float32)
    num_images = len(per_image_feats)
    num_levels = len(per_image_feats[0])
    sim_sum = torch.zeros((num_images, num_images), dtype=torch.float32)
    for lvl in range(num_levels):
        feats_lvl = torch.stack([per_image_feats[i][lvl] for i in range(num_images)], dim=0)  # [N, D]
        feats_lvl = F.normalize(feats_lvl, p=2, dim=1, eps=1e-12)
        sim_sum += feats_lvl @ feats_lvl.t()
    sim_avg = sim_sum / float(max(num_levels, 1))
    return sim_avg


def greedy_select_with_similarity(sim: torch.Tensor, k: int, lambda_weight: float) -> List[int]:
    n = int(sim.shape[0])
    if k <= 0 or n == 0:
        return []
    if k >= n:
        return list(range(n))

    remaining = torch.ones((n,), dtype=torch.bool)
    s1 = sim.sum(dim=1).clone()
    s2 = torch.zeros((n,), dtype=sim.dtype)

    selected: List[int] = []
    for _ in range(k):
        rem_idx = torch.where(remaining)[0]
        if rem_idx.numel() == 0:
            break
        score = lambda_weight * s1[rem_idx] - s2[rem_idx]
        best_local = int(torch.argmax(score).item())
        best_idx = int(rem_idx[best_local].item())
        selected.append(best_idx)
        remaining[best_idx] = False

        rem_idx = torch.where(remaining)[0]
        if rem_idx.numel() == 0:
            break
        contrib = sim[rem_idx, best_idx]
        s1[rem_idx] -= contrib
        s2[rem_idx] += contrib

    return selected


def dedup_and_fill(selected_by_class: Dict[int, List[int]], total_target: int, n_samples: int) -> List[int]:
    selected = set()
    for ids in selected_by_class.values():
        selected.update(ids)
    selected_list = sorted(selected)

    if len(selected_list) < total_target:
        remaining = [i for i in range(n_samples) if i not in selected]
        need = total_target - len(selected_list)
        selected_list.extend(random.sample(remaining, k=min(need, len(remaining))))
    if len(selected_list) > total_target:
        random.shuffle(selected_list)
        selected_list = selected_list[:total_target]
    selected_list.sort()
    return selected_list


def copy_selected_split(
    selected_samples: Sequence[SampleInfo],
    split: str,
    out_dir: Path,
) -> str:
    split_img_root = out_dir / "images" / split
    split_lbl_root = out_dir / "labels" / split
    split_img_root.mkdir(parents=True, exist_ok=True)
    split_lbl_root.mkdir(parents=True, exist_ok=True)

    for sample in selected_samples:
        try:
            rel = sample.image_path.relative_to(sample.image_dir)
        except ValueError:
            rel = Path(sample.image_path.name)
        rel = Path(f"src{sample.source_dir_index}") / rel

        dst_img = split_img_root / rel
        dst_lbl = split_lbl_root / rel.with_suffix(".txt")
        dst_img.parent.mkdir(parents=True, exist_ok=True)
        dst_lbl.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(sample.image_path, dst_img)
        if sample.label_path.is_file():
            shutil.copy2(sample.label_path, dst_lbl)
        else:
            dst_lbl.write_text("", encoding="utf-8")

    return str((out_dir / "images" / split).resolve())


def save_output_yaml(
    src_cfg: Dict,
    split_paths: Dict[str, str],
    out_dir: Path,
) -> Path:
    raw = dict(src_cfg["raw"])
    raw["path"] = str(out_dir.resolve())
    for split in ("train", "val", "test"):
        if split in split_paths:
            raw[split] = str(Path("images") / split)
        elif split in raw:
            del raw[split]
    out_yaml = out_dir / "data.yaml"
    out_yaml.write_text(yaml.safe_dump(raw, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return out_yaml


def ensure_device(device_arg: str) -> torch.device:
    if device_arg.startswith("cuda") and not torch.cuda.is_available():
        print(f"[warn] Requested device '{device_arg}' unavailable, fallback to cpu.")
        return torch.device("cpu")
    return torch.device(device_arg)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    random.seed(args.seed)

    src_data = Path(args.src_data).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    split_to_samples, cfg = load_split_samples(str(src_data))
    split_order = [s for s in ("train", "val", "test") if s in split_to_samples]
    if not split_order:
        raise ValueError("No valid split found in source yaml (train/val/test).")

    split_sizes = {s: len(split_to_samples[s]) for s in split_order}
    split_targets = parse_num_sample(args.num_sample, split_order, split_sizes)
    num_classes = len(cfg["class_names"])
    print(f"[info] splits={split_order}, split_sizes={split_sizes}, split_targets={split_targets}")
    print(f"[info] num_classes={num_classes}")
    print(f"[info] global_feature_selection={args.global_feature_selection}")

    device = ensure_device(args.device)
    model = load_model(
        model_config_path=args.config_file,
        model_checkpoint_path=args.pretrained_checkpoint,
        device=str(device),
    ).to(device)
    model.eval()
    transform = build_image_transform()

    split_output_paths: Dict[str, str] = {}
    all_meta: Dict[str, Dict] = {}

    for split in split_order:
        samples = split_to_samples[split]
        n = len(samples)
        target_n = min(split_targets.get(split, 0), n)
        print(f"[info] split={split}, total={n}, target={target_n}")
        if target_n <= 0 or n == 0:
            split_output_paths[split] = copy_selected_split([], split, out_dir)
            all_meta[split] = {
                "total_samples": n,
                "target_samples": target_n,
                "selection_mode": "global_feature" if args.global_feature_selection else "class_aware",
                "selected_indices": [],
            }
            continue

        if args.global_feature_selection:
            per_image_feats_global: List[List[torch.Tensor]] = []
            for sample in tqdm(samples, desc=f"extract_global:{split}"):
                feats = extract_per_image_multiscale_global_features(
                    model=model,
                    image_path=sample.image_path,
                    transform=transform,
                    device=device,
                )
                per_image_feats_global.append(feats)
            sim_global = compute_similarity_matrix_for_global(per_image_feats_global)
            selected_indices = greedy_select_with_similarity(
                sim=sim_global,
                k=target_n,
                lambda_weight=args.lambda_weight,
            )
            class_counts = class_instance_count(samples, num_classes)
            class_targets = None
        else:
            per_image_feats: List[List[torch.Tensor]] = []
            for sample in tqdm(samples, desc=f"extract:{split}"):
                feats = extract_per_image_multiscale_class_features(
                    model=model,
                    image_path=sample.image_path,
                    labels=sample.labels,
                    boxes_cxcywh=sample.boxes,
                    num_classes=num_classes,
                    transform=transform,
                    device=device,
                )
                per_image_feats.append(feats)

            class_counts = class_instance_count(samples, num_classes)
            class_targets = allocate_per_class(target_n, class_counts)
            selected_by_class: Dict[int, List[int]] = {}

            for cls_idx in range(num_classes):
                k_cls = int(class_targets[cls_idx].item())
                if k_cls <= 0:
                    selected_by_class[cls_idx] = []
                    continue
                sim = compute_similarity_matrix_for_class(per_image_feats, cls_idx)
                selected_by_class[cls_idx] = greedy_select_with_similarity(
                    sim=sim,
                    k=min(k_cls, n),
                    lambda_weight=args.lambda_weight,
                )

            selected_indices = dedup_and_fill(
                selected_by_class=selected_by_class,
                total_target=target_n,
                n_samples=n,
            )
        selected_samples = [samples[i] for i in selected_indices]
        split_output_paths[split] = copy_selected_split(selected_samples, split, out_dir)

        split_meta = {
            "total_samples": n,
            "target_samples": target_n,
            "selection_mode": "global_feature" if args.global_feature_selection else "class_aware",
            "class_instance_counts": class_counts.tolist(),
            "selected_indices": selected_indices,
            "selected_files": [str(s.image_path) for s in selected_samples],
        }
        if class_targets is not None:
            split_meta["class_targets"] = class_targets.tolist()
        all_meta[split] = split_meta
        print(f"[info] split={split}, selected={len(selected_indices)}")

    out_yaml = save_output_yaml(src_cfg=cfg, split_paths=split_output_paths, out_dir=out_dir)
    print(f"[done] saved selected dataset yaml to: {out_yaml}")

    if args.save_meta:
        meta_path = out_dir / "selection_meta.json"
        meta_path.write_text(json.dumps(all_meta, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[done] saved selection metadata to: {meta_path}")


if __name__ == "__main__":
    main()
