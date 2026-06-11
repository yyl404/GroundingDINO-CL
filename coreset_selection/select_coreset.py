"""Core-set selection for YOLO datasets using feature-space density filtering and diversity sampling.

Pipeline:
1. Feature extraction (global or classwise GroundingDINO backbone features)
2. Optional density-based outlier filtering
3. Core-set selection (kcenter / facility_location / kmeans)

Usage:
python coreset_selection/select_coreset.py \
    --src_data /path/to/data.yaml \
    --out_dir /path/to/output_dir \
    --num_sample 200 \
    --feature_mode global \
    --selection_algo kcenter
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from typing import Dict, List

import torch
from tqdm import tqdm

from coreset_selection.algorithms import run_selection
from coreset_selection.common import (
    SampleInfo,
    copy_selected_split,
    load_split_samples,
    parse_num_sample,
    save_meta,
    save_output_yaml,
    set_seed,
)
from coreset_selection.density_filter import filter_outlier_images
from coreset_selection.features import (
    build_feature_matrix,
    build_image_transform,
    extract_classwise_prototypes,
    extract_global_feature,
    load_model,
)


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
        "--feature_mode",
        type=str,
        default="classwise",
        choices=["global", "classwise"],
        help="Feature extraction mode: global image pooling or classwise RoI prototypes.",
    )
    parser.add_argument(
        "--selection_algo",
        type=str,
        default="kcenter",
        choices=["kcenter", "facility_location", "kmeans"],
        help="Core-set selection algorithm.",
    )
    parser.add_argument(
        "--enable_density_filter",
        action="store_true",
        help="Enable KNN density-based dirty sample filtering before core-set selection.",
    )
    parser.add_argument(
        "--density_k",
        type=int,
        default=20,
        help="K for KNN mean distance in density filtering.",
    )
    parser.add_argument(
        "--density_outlier_percentile",
        type=float,
        default=95.0,
        help="Outlier threshold percentile on KNN mean distance (e.g. 95 removes top 5%%).",
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
    parser.add_argument("--device", type=str, default="cuda", help="cuda / cuda:0 / cpu")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--save_meta", action="store_true", help="Save feature/selection metadata to json")
    return parser.parse_args()


def ensure_device(device_arg: str) -> torch.device:
    if device_arg.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"Requested device '{device_arg}' is unavailable.")
    return torch.device(device_arg)


def extract_split_features(
    model,
    samples: List[SampleInfo],
    feature_mode: str,
    num_classes: int,
    transform,
    device: torch.device,
) -> tuple[List, List]:
    global_features: List = []
    classwise_features: List = []
    for sample in tqdm(samples, desc=f"extract:{feature_mode}"):
        if feature_mode == "global":
            global_features.append(
                extract_global_feature(
                    model=model,
                    image_path=sample.image_path,
                    transform=transform,
                    device=device,
                )
            )
        else:
            classwise_features.append(
                extract_classwise_prototypes(
                    model=model,
                    image_path=sample.image_path,
                    labels=sample.labels,
                    boxes_cxcywh=sample.boxes,
                    num_classes=num_classes,
                    transform=transform,
                    device=device,
                )
            )
    return global_features, classwise_features


def select_for_split(
    samples: List[SampleInfo],
    global_features: List,
    classwise_features: List,
    feature_mode: str,
    target_n: int,
    enable_density_filter: bool,
    density_k: int,
    density_outlier_percentile: float,
    selection_algo: str,
    seed: int,
) -> tuple[List[int], Dict]:
    n = len(samples)
    all_indices = list(range(n))
    meta: Dict = {
        "total_samples": n,
        "target_samples": target_n,
        "feature_mode": feature_mode,
        "selection_algo": selection_algo,
        "enable_density_filter": enable_density_filter,
    }

    if target_n <= 0 or n == 0:
        meta["selected_indices"] = []
        return [], meta

    candidate_indices = all_indices
    density_stats = None
    if enable_density_filter:
        feats_all, image_idx_map = build_feature_matrix(
            feature_mode=feature_mode,
            global_features=global_features,
            classwise_features=classwise_features,
            image_indices=all_indices,
        )
        candidate_indices, density_stats = filter_outlier_images(
            feats=feats_all,
            image_idx_map=image_idx_map,
            candidate_image_indices=all_indices,
            k=density_k,
            outlier_percentile=density_outlier_percentile,
        )
        meta["density_filter"] = density_stats
        meta["num_after_density_filter"] = len(candidate_indices)

    if not candidate_indices:
        raise RuntimeError("All samples were removed by density filtering.")

    feats_clean, image_idx_map_clean = build_feature_matrix(
        feature_mode=feature_mode,
        global_features=global_features,
        classwise_features=classwise_features,
        image_indices=candidate_indices,
    )
    if feats_clean.shape[0] == 0:
        raise RuntimeError("Empty feature matrix after density filtering.")

    selected_indices = run_selection(
        algorithm=selection_algo,
        feats=feats_clean,
        image_idx_map=image_idx_map_clean,
        budget=min(target_n, len(candidate_indices)),
        seed=seed,
    )
    meta["selected_indices"] = selected_indices
    meta["selected_files"] = [str(samples[i].image_path) for i in selected_indices]
    if density_stats is not None:
        meta["removed_by_density_filter"] = sorted(set(all_indices) - set(candidate_indices))
    return selected_indices, meta


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

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
    print(f"[info] feature_mode={args.feature_mode}")
    print(f"[info] selection_algo={args.selection_algo}")
    print(f"[info] enable_density_filter={args.enable_density_filter}")

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
                "feature_mode": args.feature_mode,
                "selection_algo": args.selection_algo,
                "enable_density_filter": args.enable_density_filter,
                "selected_indices": [],
            }
            continue

        global_features, classwise_features = extract_split_features(
            model=model,
            samples=samples,
            feature_mode=args.feature_mode,
            num_classes=num_classes,
            transform=transform,
            device=device,
        )
        selected_indices, split_meta = select_for_split(
            samples=samples,
            global_features=global_features,
            classwise_features=classwise_features,
            feature_mode=args.feature_mode,
            target_n=target_n,
            enable_density_filter=args.enable_density_filter,
            density_k=args.density_k,
            density_outlier_percentile=args.density_outlier_percentile,
            selection_algo=args.selection_algo,
            seed=args.seed,
        )
        selected_samples = [samples[i] for i in selected_indices]
        split_output_paths[split] = copy_selected_split(selected_samples, split, out_dir)
        all_meta[split] = split_meta
        print(f"[info] split={split}, selected={len(selected_indices)}")

    out_yaml = save_output_yaml(src_cfg=cfg, split_paths=split_output_paths, out_dir=out_dir)
    print(f"[done] saved selected dataset yaml to: {out_yaml}")

    if args.save_meta:
        meta_path = out_dir / "selection_meta.json"
        save_meta(meta_path, all_meta)
        print(f"[done] saved selection metadata to: {meta_path}")


if __name__ == "__main__":
    main()
