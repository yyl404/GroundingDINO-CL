import warnings
from typing import Dict, List

import torch
from torch import nn

from groundingdino.models import build_model
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict

from finetune import GroundingDINOWrapper


def xywhr_to_corners_xyxyxyxy(boxes_xywhr: torch.Tensor, width: float, height: float) -> torch.Tensor:
    """Convert normalized xywhr boxes to pixel-space corners [N, 8]."""
    if boxes_xywhr.numel() == 0:
        return boxes_xywhr.new_zeros((0, 8))
    cx = boxes_xywhr[:, 0] * width
    cy = boxes_xywhr[:, 1] * height
    bw = boxes_xywhr[:, 2] * width
    bh = boxes_xywhr[:, 3] * height
    theta = boxes_xywhr[:, 4]

    dx = bw * 0.5
    dy = bh * 0.5
    local = torch.tensor(
        [[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]],
        dtype=boxes_xywhr.dtype,
        device=boxes_xywhr.device,
    ).unsqueeze(0)
    local[..., 0] *= dx.unsqueeze(1)
    local[..., 1] *= dy.unsqueeze(1)

    cos_t = torch.cos(theta).unsqueeze(1)
    sin_t = torch.sin(theta).unsqueeze(1)
    rot_x = local[..., 0] * cos_t - local[..., 1] * sin_t
    rot_y = local[..., 0] * sin_t + local[..., 1] * cos_t
    x = rot_x + cx.unsqueeze(1)
    y = rot_y + cy.unsqueeze(1)
    return torch.stack((x, y), dim=-1).reshape(-1, 8)


def load_model(model_config_path, model_checkpoint_path, device="cuda"):
    args = SLConfig.fromfile(model_config_path)
    args.device = str(device)
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    print(load_res)
    _ = model.eval()
    return model


def load_wrapper_checkpoint(wrapper: GroundingDINOWrapper, checkpoint: Dict, device="cuda"):
    ckpt_kwargs = checkpoint["wrapper_kwargs"]
    ckpt_prompt_len = ckpt_kwargs["prompt_len"]
    ckpt_inject_before_encoder = ckpt_kwargs["inject_before_encoder"]

    if int(ckpt_prompt_len) != int(wrapper.prompt_len):
        raise ValueError(
            f"prompt_len mismatch: checkpoint={ckpt_prompt_len}, wrapper={wrapper.prompt_len}"
        )
    if bool(ckpt_inject_before_encoder) != bool(wrapper.inject_before_encoder):
        raise ValueError(
            "inject_before_encoder mismatch: "
            f"checkpoint={ckpt_inject_before_encoder}, wrapper={wrapper.inject_before_encoder}"
        )

    ckpt_classes = list(checkpoint["classes"])
    if not ckpt_classes:
        warnings.warn(
            "Empty 'classes' in checkpoint; skip wrapper vocabulary/embedding update.",
            UserWarning,
            stacklevel=2,
        )
        return wrapper

    ckpt_state_dict = checkpoint["wrapper_state_dict"]
    ckpt_embeddings = ckpt_state_dict["embeddings"]
    ckpt_cls_head_last_layer_weight = ckpt_state_dict["cls_head.0.weight"]

    emb_dtype = wrapper.embeddings.dtype
    wrapper_embeddings = wrapper.embeddings.detach().to(device=device, dtype=emb_dtype).clone()
    ckpt_embeddings = ckpt_embeddings.to(device=device, dtype=emb_dtype)

    cls_head_dtype = wrapper.cls_head[-1].weight.dtype
    cls_head_device = wrapper.cls_head[-1].weight.device
    wrapper_cls_head_last_layer_weight = wrapper.cls_head[-1].weight.detach().to(
        device=cls_head_device, dtype=cls_head_dtype
    ).clone()
    ckpt_cls_head_last_layer_weight = ckpt_cls_head_last_layer_weight.to(device=cls_head_device, dtype=cls_head_dtype)

    wrapper_class_to_idx = {name: idx for idx, name in enumerate(wrapper.classes)}
    updated_classes = list(wrapper.classes)
    emb_rows_to_append = []
    cls_rows_to_append = []
    mapping_logs: List[str] = []

    for ckpt_idx, cls_name in enumerate(ckpt_classes):
        if cls_name in wrapper_class_to_idx:
            dst_idx = wrapper_class_to_idx[cls_name]
            wrapper_embeddings[dst_idx] = ckpt_embeddings[ckpt_idx]
            wrapper_cls_head_last_layer_weight[dst_idx] = ckpt_cls_head_last_layer_weight[ckpt_idx]
            mapping_logs.append(
                f"  - ckpt[{ckpt_idx}] '{cls_name}' -> wrapper[{dst_idx}] (replace)"
            )
        else:
            dst_idx = len(updated_classes)
            updated_classes.append(cls_name)
            emb_rows_to_append.append(ckpt_embeddings[ckpt_idx : ckpt_idx + 1])
            cls_rows_to_append.append(ckpt_cls_head_last_layer_weight[ckpt_idx : ckpt_idx + 1])
            mapping_logs.append(
                f"  - ckpt[{ckpt_idx}] '{cls_name}' -> wrapper[{dst_idx}] (append)"
            )

    wrapper_only_classes = [name for name in wrapper.classes if name not in ckpt_classes]
    if mapping_logs:
        print("[load_wrapper_checkpoint] class mapping:")
        print("\n".join(mapping_logs))
    if wrapper_only_classes:
        print(
            "[load_wrapper_checkpoint] wrapper-only classes unchanged: "
            + ", ".join(wrapper_only_classes)
        )

    if emb_rows_to_append:
        wrapper_embeddings = torch.cat([wrapper_embeddings] + emb_rows_to_append, dim=0)
        wrapper_cls_head_last_layer_weight = torch.cat([wrapper_cls_head_last_layer_weight] + cls_rows_to_append, dim=0)

    wrapper.classes = updated_classes
    wrapper.embeddings = nn.Parameter(wrapper_embeddings)
    wrapper.cls_head = wrapper._build_cls_head(
        len(updated_classes), device=cls_head_device, dtype=cls_head_dtype
    )
    wrapper.cls_head[-1].weight.data = wrapper_cls_head_last_layer_weight.detach().to(device=cls_head_device, dtype=cls_head_dtype)

    # Load wrapper parameters except embeddings/cls_head when keys and shapes match.
    current_state = wrapper.state_dict()
    loadable_state = {}
    skipped_keys = []
    for key, value in ckpt_state_dict.items():
        if key == "embeddings" or key.startswith("cls_head."):
            continue
        if key not in current_state:
            skipped_keys.append(f"{key} (missing in current wrapper)")
            continue
        target = current_state[key]
        if tuple(value.shape) != tuple(target.shape):
            skipped_keys.append(
                f"{key} (shape mismatch: ckpt={tuple(value.shape)} vs current={tuple(target.shape)})"
            )
            continue
        loadable_state[key] = value.to(device=target.device, dtype=target.dtype)

    if loadable_state:
        wrapper.load_state_dict(loadable_state, strict=False)
        # print(
        #     "[load_wrapper_checkpoint] loaded non-embedding params: "
        #     + ", ".join(sorted(loadable_state.keys()))
        # )
    # if skipped_keys:
        # print("[load_wrapper_checkpoint] skipped params:")
        # print("\n".join(f"  - {msg}" for msg in skipped_keys))

    return wrapper