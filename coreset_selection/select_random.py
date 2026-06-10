"""Random subset selection for YOLO datasets (control group baseline).

Usage:
python coreset_selection/select_random.py \
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

import yaml

from finetune.datasets.yolo import _collect_images_from_dir, _label_path_for_image, _load_yolo_yaml


@dataclass
class SampleInfo:
    split: str
    image_path: Path
    label_path: Path
    image_dir: Path
    source_dir_index: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Random subset selection for YOLO dataset splits")
    parser.add_argument("--src_data", type=str, required=True, help="Source YOLO data.yaml path")
    parser.add_argument("--out_dir", type=str, required=True, help="Output directory for random subset")
    parser.add_argument(
        "--num_sample",
        type=str,
        required=True,
        help='Total sample count (e.g. "300") or split counts (e.g. "[200,80,20]" / "200,80,20").',
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--save_meta", action="store_true", help="Save selection metadata to json")
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
                samples.append(
                    SampleInfo(
                        split=split,
                        image_path=image_path,
                        label_path=label_path,
                        image_dir=image_dir_path,
                        source_dir_index=dir_idx,
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


def main() -> None:
    args = parse_args()
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
    print(f"[info] splits={split_order}, split_sizes={split_sizes}, split_targets={split_targets}")

    split_output_paths: Dict[str, str] = {}
    all_meta: Dict[str, Dict] = {}

    for split in split_order:
        samples = split_to_samples[split]
        n = len(samples)
        target_n = min(split_targets.get(split, 0), n)
        print(f"[info] split={split}, total={n}, target={target_n}")

        selected_indices = sorted(random.sample(range(n), k=target_n)) if target_n > 0 else []
        selected_samples = [samples[i] for i in selected_indices]
        split_output_paths[split] = copy_selected_split(selected_samples, split, out_dir)

        all_meta[split] = {
            "total_samples": n,
            "target_samples": target_n,
            "selected_indices": selected_indices,
            "selected_files": [str(s.image_path) for s in selected_samples],
        }
        print(f"[info] split={split}, selected={len(selected_indices)}")

    out_yaml = save_output_yaml(src_cfg=cfg, split_paths=split_output_paths, out_dir=out_dir)
    print(f"[done] saved random-selected dataset yaml to: {out_yaml}")

    if args.save_meta:
        meta_path = out_dir / "selection_meta_random.json"
        meta_path.write_text(json.dumps(all_meta, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[done] saved selection metadata to: {meta_path}")


if __name__ == "__main__":
    main()
