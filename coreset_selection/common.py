"""Shared utilities for core-set selection pipelines."""

from __future__ import annotations

import ast
import json
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch
import yaml

from finetune.datasets.yolo import _collect_images_from_dir, _label_path_for_image, _load_yolo_yaml


@dataclass
class SampleInfo:
    split: str
    image_path: Path
    label_path: Path
    image_dir: Path
    source_dir_index: int
    labels: torch.Tensor  # [N]
    boxes: torch.Tensor  # [N, 4] (cx, cy, w, h), normalized


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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

    if value.startswith("[") or value.startswith("("):
        parsed = ast.literal_eval(value)
    else:
        parsed = [x.strip() for x in value.split(",") if x.strip()]
    if not isinstance(parsed, (list, tuple)):
        raise ValueError(f"Unsupported --num_sample format: {raw}")

    nums = [int(x) for x in parsed]
    if len(nums) != len(split_order):
        raise ValueError(
            f"--num_sample list length must match split count ({len(split_order)}), got {len(nums)}"
        )
    return {s: max(nums[i], 0) for i, s in enumerate(split_order)}


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
        rel = sample.image_path.relative_to(sample.image_dir)
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


def save_meta(meta_path: Path, meta: Dict) -> None:
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
