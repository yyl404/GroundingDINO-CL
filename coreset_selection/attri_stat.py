"""
统计对比两个不同的数据集图像属性的分布情况

使用方式：
    python coreset_selection/attri_stat.py \
        --src_data /path/to/source_data/yaml_file.yaml \
        --core_data /path/to/core_data/yaml_file.yaml \
        --output_dir /path/to/output_dir \
        --split train

利用函数接口get_attri(img_path)，获取每一张图像样本的属性组，每个属性都是一个实数。

1. 分别统计 source/core 指定 split 的样本集各个属性分布（max/min/median/mean/var），
   并自动估计合适的 num_bins 绘制直方图。相同属性的 source/core 直方图以左右子图形式保存。

2. 在每个样本集上将属性归一化为 0 均值、1 方差，然后对属性向量做 PCA，
   输出主成分方向构成：w1*attr1 + w2*attr2 + ...
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from finetune.datasets.yolo import _collect_images_from_dir, _load_yolo_yaml

matplotlib.use("Agg")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Compare attribute distributions between source/core datasets")
    parser.add_argument("--src_data", type=str, required=True, help="Source dataset yaml path")
    parser.add_argument("--core_data", type=str, required=True, help="Core-set dataset yaml path")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--split", type=str, default="train", choices=["train", "val", "test"], help="Dataset split")
    parser.add_argument(
        "--pca_components",
        type=int,
        default=3,
        help="Number of principal components to output for each dataset",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=0,
        help="Optional cap on number of images to process per dataset (0 means all)",
    )
    return parser.parse_args()


def list_split_images(data_yaml: str, split: str) -> List[Path]:
    cfg = _load_yolo_yaml(data_yaml)
    if split not in cfg["splits"]:
        raise ValueError(f"Split '{split}' not found in dataset yaml: {data_yaml}")
    image_paths: List[Path] = []
    for image_dir in cfg["splits"][split]:
        image_paths.extend(Path(p).resolve() for p in _collect_images_from_dir(image_dir))
    image_paths = sorted(set(image_paths))
    return image_paths


def get_attri(img_path: Path) -> Dict[str, float]:
    """
    图像属性接口：
    - average_gray: 平均灰度值
    - average_gradient: 平均梯度值（基于灰度图梯度幅值）
    - global_contrast: 全局对比度（灰度标准差）
    - fg_bg_contrast: 前景背景对比度（以前景/背景灰度均值差定义）
    - information_entropy: 全局像素色彩分布信息熵
    """
    with Image.open(img_path).convert("RGB") as image:
        rgb_u8 = np.asarray(image).astype(np.uint8)
    rgb = rgb_u8.astype(np.float32)
    gray = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]

    # 平均灰度值
    average_gray = float(gray.mean())

    # 平均梯度值（灰度梯度幅值的均值）
    grad_y, grad_x = np.gradient(gray)
    grad_mag = np.sqrt(grad_x * grad_x + grad_y * grad_y)
    average_gradient = float(grad_mag.mean())

    # 全局对比度（RMS contrast）
    global_contrast = float(gray.std())

    # 前景背景对比度：以前景/背景灰度均值差值定义
    # 这里使用图像平均灰度作为阈值分前景和背景，避免引入额外依赖。
    threshold = average_gray
    fg_mask = gray >= threshold
    bg_mask = ~fg_mask
    if np.any(fg_mask) and np.any(bg_mask):
        fg_mean = float(gray[fg_mask].mean())
        bg_mean = float(gray[bg_mask].mean())
        fg_bg_contrast = abs(fg_mean - bg_mean)
    else:
        fg_bg_contrast = 0.0

    # 信息熵：将像素视作色彩空间采样，统计全局颜色分布熵
    # 对每个通道做 16 级量化，得到 16^3 = 4096 个颜色桶。
    bins_per_channel = 16
    quant = (rgb_u8 // (256 // bins_per_channel)).astype(np.int32)
    color_index = (
        quant[:, :, 0] * (bins_per_channel * bins_per_channel)
        + quant[:, :, 1] * bins_per_channel
        + quant[:, :, 2]
    )
    hist = np.bincount(color_index.reshape(-1), minlength=bins_per_channel**3).astype(np.float64)
    prob = hist / max(float(hist.sum()), 1.0)
    prob = prob[prob > 0.0]
    information_entropy = float(-np.sum(prob * np.log2(prob)))

    return {
        "average_gray": average_gray,
        "average_gradient": average_gradient,
        "global_contrast": global_contrast,
        "fg_bg_contrast": fg_bg_contrast,
        "information_entropy": information_entropy,
    }


def collect_attr_matrix(image_paths: List[Path], max_samples: int = 0) -> Tuple[List[str], np.ndarray]:
    if max_samples > 0:
        image_paths = image_paths[:max_samples]
    if not image_paths:
        raise ValueError("No images found for attribute statistics.")

    first = get_attri(image_paths[0])
    attr_names = sorted(first.keys())
    rows: List[List[float]] = [[float(first[name]) for name in attr_names]]

    for img_path in image_paths[1:]:
        attrs = get_attri(img_path)
        rows.append([float(attrs[name]) for name in attr_names])
    return attr_names, np.asarray(rows, dtype=np.float64)


def save_attr_matrix_csv(csv_path: Path, attr_names: List[str], x: np.ndarray) -> None:
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(attr_names)
        for row in x:
            writer.writerow([f"{float(v):.10g}" for v in row.tolist()])


def compute_stats(values: np.ndarray) -> Dict[str, float]:
    return {
        "max": float(np.max(values)),
        "min": float(np.min(values)),
        "median": float(np.median(values)),
        "mean": float(np.mean(values)),
        "var": float(np.var(values)),
    }


def choose_num_bins(values_a: np.ndarray, values_b: np.ndarray) -> int:
    x = np.concatenate([values_a, values_b], axis=0)
    n = int(x.size)
    if n <= 1:
        return 1
    q75, q25 = np.percentile(x, [75.0, 25.0])
    iqr = float(q75 - q25)
    if iqr > 0:
        bin_width = 2.0 * iqr / (n ** (1.0 / 3.0))
        if bin_width > 0:
            bins = int(math.ceil((float(np.max(x)) - float(np.min(x))) / bin_width))
            return max(10, min(bins, 120))
    # fallback to Sturges
    bins = int(math.ceil(math.log2(n) + 1))
    return max(10, min(bins, 120))


def stats_text(stats_dict: Dict[str, float]) -> str:
    return (
        f"max={stats_dict['max']:.4f}\n"
        f"min={stats_dict['min']:.4f}\n"
        f"median={stats_dict['median']:.4f}\n"
        f"mean={stats_dict['mean']:.4f}\n"
        f"var={stats_dict['var']:.4f}"
    )


def plot_attr_hist(
    output_dir: Path,
    attr_name: str,
    src_values: np.ndarray,
    core_values: np.ndarray,
    src_stats: Dict[str, float],
    core_stats: Dict[str, float],
) -> None:
    bins = choose_num_bins(src_values, core_values)
    global_min = float(min(np.min(src_values), np.min(core_values)))
    global_max = float(max(np.max(src_values), np.max(core_values)))
    if global_min == global_max:
        global_max = global_min + 1e-6

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True)
    src_weights = np.ones_like(src_values, dtype=np.float64) / max(float(src_values.size), 1.0)
    core_weights = np.ones_like(core_values, dtype=np.float64) / max(float(core_values.size), 1.0)

    axes[0].hist(
        src_values,
        bins=bins,
        range=(global_min, global_max),
        weights=src_weights,
        color="#4e79a7",
        alpha=0.85,
    )
    axes[0].set_title(f"Source - {attr_name}")
    axes[0].set_xlabel(attr_name)
    axes[0].set_ylabel("Normalized frequency")
    axes[0].grid(alpha=0.2)
    axes[0].text(
        0.98,
        0.98,
        stats_text(src_stats),
        transform=axes[0].transAxes,
        va="top",
        ha="right",
        fontsize=9,
        bbox={"facecolor": "white", "alpha": 0.7, "edgecolor": "none"},
    )

    axes[1].hist(
        core_values,
        bins=bins,
        range=(global_min, global_max),
        weights=core_weights,
        color="#e15759",
        alpha=0.85,
    )
    axes[1].set_title(f"Core - {attr_name}")
    axes[1].set_xlabel(attr_name)
    axes[1].grid(alpha=0.2)
    axes[1].text(
        0.98,
        0.98,
        stats_text(core_stats),
        transform=axes[1].transAxes,
        va="top",
        ha="right",
        fontsize=9,
        bbox={"facecolor": "white", "alpha": 0.7, "edgecolor": "none"},
    )

    fig.tight_layout()
    out_path = output_dir / f"hist_{attr_name}.png"
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def standardize_matrix(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std_safe = np.where(std < 1e-12, 1.0, std)
    z = (x - mean) / std_safe
    return z, mean, std_safe


def pca_from_standardized(z: np.ndarray, n_components: int) -> Tuple[np.ndarray, np.ndarray]:
    n_samples, n_features = z.shape
    if n_samples < 2:
        k = min(max(n_components, 1), n_features)
        return np.eye(n_features, dtype=np.float64)[:k], np.zeros((k,), dtype=np.float64)
    cov = (z.T @ z) / float(max(n_samples - 1, 1))
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    k = min(max(n_components, 1), n_features)
    comps = eigvecs[:, :k].T
    exp_ratio = eigvals[:k] / max(float(np.sum(eigvals)), 1e-12)
    return comps, exp_ratio


def component_expression(component: np.ndarray, attr_names: List[str]) -> str:
    terms = [f"{w:+.4f}*{name}" for w, name in zip(component.tolist(), attr_names)]
    return " ".join(terms)


def run_pca(
    data_name: str,
    x: np.ndarray,
    attr_names: List[str],
    n_components: int,
) -> Dict:
    z, mean, std = standardize_matrix(x)
    comps, exp_ratio = pca_from_standardized(z, n_components=n_components)
    result = {
        "dataset": data_name,
        "num_samples": int(x.shape[0]),
        "num_attributes": int(x.shape[1]),
        "attribute_names": list(attr_names),
        "standardize_mean": mean.tolist(),
        "standardize_std": std.tolist(),
        "explained_variance_ratio": exp_ratio.tolist(),
        "components": [],
    }
    print(f"\n[PCA] {data_name}")
    for idx, comp in enumerate(comps, start=1):
        expr = component_expression(comp, attr_names)
        ratio = float(exp_ratio[idx - 1]) if idx - 1 < len(exp_ratio) else 0.0
        print(f"  PC{idx} (ratio={ratio:.4f}): {expr}")
        result["components"].append(
            {
                "component_index": idx,
                "explained_variance_ratio": ratio,
                "expression": expr,
                "weights": comp.tolist(),
            }
        )
    return result


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    src_images = list_split_images(args.src_data, args.split)
    core_images = list_split_images(args.core_data, args.split)
    print(f"[info] split={args.split}, src_images={len(src_images)}, core_images={len(core_images)}")

    src_attr_names, src_x = collect_attr_matrix(src_images, max_samples=args.max_samples)
    core_attr_names, core_x = collect_attr_matrix(core_images, max_samples=args.max_samples)
    if src_attr_names != core_attr_names:
        raise RuntimeError("Attribute names mismatch between source and core dataset extraction.")
    attr_names = src_attr_names

    src_attr_csv = out_dir / "source_attr_values.csv"
    core_attr_csv = out_dir / "core_attr_values.csv"
    save_attr_matrix_csv(src_attr_csv, attr_names, src_x)
    save_attr_matrix_csv(core_attr_csv, attr_names, core_x)
    print(f"[info] Saved per-sample source attributes: {src_attr_csv}")
    print(f"[info] Saved per-sample core attributes:   {core_attr_csv}")

    summary = {
        "split": args.split,
        "src_data": str(Path(args.src_data).expanduser().resolve()),
        "core_data": str(Path(args.core_data).expanduser().resolve()),
        "num_src_samples": int(src_x.shape[0]),
        "num_core_samples": int(core_x.shape[0]),
        "attributes": {},
    }

    for idx, attr_name in enumerate(attr_names):
        src_values = src_x[:, idx]
        core_values = core_x[:, idx]
        src_stats = compute_stats(src_values)
        core_stats = compute_stats(core_values)
        summary["attributes"][attr_name] = {"source": src_stats, "core": core_stats}
        plot_attr_hist(out_dir, attr_name, src_values, core_values, src_stats, core_stats)

    pca_src = run_pca("source", src_x, attr_names, n_components=args.pca_components)
    pca_core = run_pca("core", core_x, attr_names, n_components=args.pca_components)
    summary["pca"] = {"source": pca_src, "core": pca_core}

    stats_path = out_dir / "attr_stats_summary.json"
    pca_path = out_dir / "attr_pca_summary.json"
    stats_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    pca_path.write_text(json.dumps(summary["pca"], ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n[done] Histogram and summaries saved to: {out_dir}")
    print(f"[done] Source attr csv: {src_attr_csv}")
    print(f"[done] Core attr csv:   {core_attr_csv}")
    print(f"[done] Stats json: {stats_path}")
    print(f"[done] PCA json:   {pca_path}")


if __name__ == "__main__":
    main()