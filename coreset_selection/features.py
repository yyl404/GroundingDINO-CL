"""GroundingDINO feature extraction for core-set selection."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as TVT
from PIL import Image

from groundingdino.models import build_model
from groundingdino.util.misc import nested_tensor_from_tensor_list
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict


def load_model(model_config_path: str, model_checkpoint_path: str, device: str = "cuda"):
    args = SLConfig.fromfile(model_config_path)
    args.device = str(device)
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu", weights_only=False)
    load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    print(load_res)
    model.eval()
    return model


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


def _normalize_vector(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm <= 1e-12:
        return vec.astype(np.float32)
    return (vec / norm).astype(np.float32)


def aggregate_multiscale_vector(per_level: Sequence[torch.Tensor]) -> np.ndarray:
    parts: List[np.ndarray] = []
    for feat in per_level:
        parts.append(_normalize_vector(feat.detach().cpu().numpy().astype(np.float32)))
    return np.concatenate(parts, axis=0)


@torch.no_grad()
def extract_global_feature(
    model,
    image_path: Path,
    transform: TVT.Compose,
    device: torch.device,
) -> np.ndarray:
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
    return aggregate_multiscale_vector(per_level)


@torch.no_grad()
def extract_classwise_prototypes(
    model,
    image_path: Path,
    labels: torch.Tensor,
    boxes_cxcywh: torch.Tensor,
    num_classes: int,
    transform: TVT.Compose,
    device: torch.device,
) -> Dict[int, np.ndarray]:
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).to(device)
    nested = nested_tensor_from_tensor_list([image_tensor]).to(device)

    model.set_image_tensor(nested)
    features = model.features
    model.unset_image_tensor()

    per_level_class_sum: List[torch.Tensor] = []
    per_level_class_cnt: List[torch.Tensor] = []

    if boxes_cxcywh.numel() == 0:
        return {}

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
            cls_idx = int(labels_local[idx].item())
            if cls_idx < 0 or cls_idx >= num_classes:
                continue
            x0 = int(torch.floor(xmin[idx] * w).item())
            y0 = int(torch.floor(ymin[idx] * h).item())
            x1 = int(torch.ceil(xmax[idx] * w).item())
            y1 = int(torch.ceil(ymax[idx] * h).item())
            x0, y0, x1, y1 = clamp_roi_bounds(x0, y0, x1, y1, w, h)
            roi_feat = fmap[:, y0:y1, x0:x1].mean(dim=(1, 2))
            class_sum[cls_idx] += roi_feat
            class_cnt[cls_idx] += 1.0

        per_level_class_sum.append(class_sum.detach().cpu())
        per_level_class_cnt.append(class_cnt.detach().cpu())

    prototypes: Dict[int, np.ndarray] = {}
    present_classes = torch.unique(labels_local).tolist()
    for cls_idx in present_classes:
        cls_idx = int(cls_idx)
        if cls_idx < 0 or cls_idx >= num_classes:
            continue
        per_level_vecs: List[torch.Tensor] = []
        for lvl in range(len(per_level_class_sum)):
            cnt = float(per_level_class_cnt[lvl][cls_idx].item())
            if cnt <= 0:
                continue
            per_level_vecs.append(per_level_class_sum[lvl][cls_idx] / cnt)
        if not per_level_vecs:
            continue
        prototypes[cls_idx] = aggregate_multiscale_vector(per_level_vecs)
    return prototypes


def build_feature_matrix(
    feature_mode: str,
    global_features: Sequence[np.ndarray],
    classwise_features: Sequence[Dict[int, np.ndarray]],
    image_indices: Sequence[int],
) -> Tuple[np.ndarray, np.ndarray]:
    rows: List[np.ndarray] = []
    image_idx_map: List[int] = []

    if feature_mode == "global":
        for local_idx, feat in zip(image_indices, global_features):
            rows.append(feat)
            image_idx_map.append(int(local_idx))
    elif feature_mode == "classwise":
        for local_idx, proto_dict in zip(image_indices, classwise_features):
            for _cls, feat in sorted(proto_dict.items()):
                rows.append(feat)
                image_idx_map.append(int(local_idx))
    else:
        raise ValueError(f"Unsupported feature_mode: {feature_mode}")

    if not rows:
        return np.zeros((0, 0), dtype=np.float32), np.zeros((0,), dtype=np.int64)
    return np.stack(rows, axis=0).astype(np.float32), np.asarray(image_idx_map, dtype=np.int64)
