import warnings
from typing import Dict, List

import torch
from torch import nn

from groundingdino.models import build_model
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict

from finetune import GroundingDINOWrapper
from finetune.datasets.coco import CocoDetectionDataset, _load_coco_json
from finetune.datasets.yolo import YoloDetectionDataset, _load_yolo_yaml


def set_seed(seed: int) -> None:
    import random

    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_classes(
    classes_arg: str | None,
    dataset_path: str,
    dataset_format: str = "yolo",
) -> List[str]:
    if classes_arg:
        classes = [c.strip() for c in classes_arg.split(",") if c.strip()]
        if not classes:
            raise ValueError("--classes must be non-empty when provided.")
        return classes
    if dataset_format == "yolo":
        cfg = _load_yolo_yaml(dataset_path)
    elif dataset_format == "coco":
        cfg = _load_coco_json(dataset_path)
    else:
        raise ValueError(f"Unsupported dataset_format: {dataset_format!r}")
    if not cfg["class_names"]:
        raise ValueError("No classes in dataset config. Please set --classes explicitly.")
    return cfg["class_names"]


def build_detection_dataset(
    dataset_format: str,
    path: str,
    split: str = None,
    image_dir: str | None = None,
    transform=None,
) -> YoloDetectionDataset | CocoDetectionDataset:
    if dataset_format == "yolo":
        return YoloDetectionDataset(path, split=split, transform=transform)
    if dataset_format == "coco":
        return CocoDetectionDataset(path, image_dir=image_dir, transform=transform)
    raise ValueError(f"Unsupported dataset_format: {dataset_format!r}")


def parse_lora_targets(raw: str) -> List[str]:
    targets = [x.strip() for x in raw.split(",") if x.strip()]
    if not targets:
        raise ValueError("--lora_targets must include at least one non-empty target.")
    return targets


def parse_lora_layers(raw: str) -> list[int] | None:
    value = raw.strip().lower()
    if value in {"", "all"}:
        return None
    layers = sorted({int(x.strip()) for x in raw.split(",") if x.strip()})
    if not layers:
        raise ValueError("--lora_layers must be 'all' or a comma-separated list of integers.")
    if any(x < 0 for x in layers):
        raise ValueError("--lora_layers cannot contain negative indices.")
    return layers


_TEXT_BRANCH_PARAM_MARKERS = (
    "bert.",
    "feat_map.",
    ".text_layers.",
    ".fusion_layers.",
    ".ca_text.",
    ".catext_",
)


def is_text_branch_parameter(name: str) -> bool:
    return any(marker in name for marker in _TEXT_BRANCH_PARAM_MARKERS)


TEXT_MODES = ("prompt", "fixed")
PARAM_TUNES = ("full", "lora", "delta", "frozen")


def configure_model_trainable_flags(
    wrapper: GroundingDINOWrapper,
    *,
    text_mode: str,
    param_tune: str,
) -> None:
    if text_mode not in TEXT_MODES:
        raise ValueError(f"text_mode must be one of {TEXT_MODES}, got {text_mode!r}.")
    if param_tune not in PARAM_TUNES:
        raise ValueError(f"param_tune must be one of {PARAM_TUNES}, got {param_tune!r}.")

    for p in wrapper.parameters():
        p.requires_grad = False

    if text_mode == "prompt":
        wrapper.embeddings.requires_grad = True

    if param_tune == "full":
        for name, p in wrapper.model.named_parameters():
            if not is_text_branch_parameter(name):
                p.requires_grad = True
    elif param_tune == "lora":
        for name, p in wrapper.model.named_parameters():
            p.requires_grad = "lora_A" in name or "lora_B" in name
    elif param_tune == "delta":
        for p in wrapper.bbox_embed_last_layer.parameters():
            p.requires_grad = True
        for p in wrapper.cls_head.parameters():
            p.requires_grad = True
    elif param_tune == "frozen":
        pass


def load_model(model_config_path, model_checkpoint_path, device="cuda"):
    args = SLConfig.fromfile(model_config_path)
    args.device = str(device)
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu", weights_only=False)
    load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    print(load_res)
    _ = model.eval()
    return model


def load_wrapper_checkpoint(wrapper: GroundingDINOWrapper, checkpoint: Dict, device="cuda"):
    ckpt_kwargs = checkpoint["wrapper_kwargs"]
    ckpt_prompt_len = ckpt_kwargs["prompt_len"]
    ckpt_inject_before_encoder = ckpt_kwargs["inject_before_encoder"]
    ckpt_text_mode = ckpt_kwargs.get("text_mode", "prompt")

    if int(ckpt_prompt_len) != int(wrapper.prompt_len):
        raise ValueError(
            f"prompt_len mismatch: checkpoint={ckpt_prompt_len}, wrapper={wrapper.prompt_len}"
        )
    if bool(ckpt_inject_before_encoder) != bool(wrapper.inject_before_encoder):
        raise ValueError(
            "inject_before_encoder mismatch: "
            f"checkpoint={ckpt_inject_before_encoder}, wrapper={wrapper.inject_before_encoder}"
        )
    if ckpt_text_mode != wrapper.text_mode:
        raise ValueError(
            f"text_mode mismatch: checkpoint={ckpt_text_mode}, wrapper={wrapper.text_mode}"
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
    ckpt_cls_head_last_layer_weight = ckpt_state_dict["cls_head.0.weight"]

    cls_head_dtype = wrapper.cls_head[-1].weight.dtype
    cls_head_device = wrapper.cls_head[-1].weight.device
    wrapper_cls_head_last_layer_weight = wrapper.cls_head[-1].weight.detach().to(
        device=cls_head_device, dtype=cls_head_dtype
    ).clone()
    ckpt_cls_head_last_layer_weight = ckpt_cls_head_last_layer_weight.to(device=cls_head_device, dtype=cls_head_dtype)

    wrapper_class_to_idx = {name: idx for idx, name in enumerate(wrapper.classes)}
    updated_classes = list(wrapper.classes)
    cls_rows_to_append = []
    mapping_logs: List[str] = []

    if wrapper.text_mode == "prompt":
        ckpt_embeddings = ckpt_state_dict["embeddings"]
        emb_dtype = wrapper.embeddings.dtype
        wrapper_embeddings = wrapper.embeddings.detach().to(device=device, dtype=emb_dtype).clone()
        ckpt_embeddings = ckpt_embeddings.to(device=device, dtype=emb_dtype)
        emb_rows_to_append = []

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

        if emb_rows_to_append:
            wrapper_embeddings = torch.cat([wrapper_embeddings] + emb_rows_to_append, dim=0)
            wrapper_cls_head_last_layer_weight = torch.cat(
                [wrapper_cls_head_last_layer_weight] + cls_rows_to_append, dim=0
            )

        wrapper.classes = updated_classes
        wrapper.embeddings = nn.Parameter(wrapper_embeddings)
    else:
        for ckpt_idx, cls_name in enumerate(ckpt_classes):
            if cls_name in wrapper_class_to_idx:
                dst_idx = wrapper_class_to_idx[cls_name]
                wrapper_cls_head_last_layer_weight[dst_idx] = ckpt_cls_head_last_layer_weight[ckpt_idx]
                mapping_logs.append(
                    f"  - ckpt[{ckpt_idx}] '{cls_name}' -> wrapper[{dst_idx}] (replace)"
                )
            else:
                dst_idx = len(updated_classes)
                updated_classes.append(cls_name)
                cls_rows_to_append.append(ckpt_cls_head_last_layer_weight[ckpt_idx : ckpt_idx + 1])
                mapping_logs.append(
                    f"  - ckpt[{ckpt_idx}] '{cls_name}' -> wrapper[{dst_idx}] (append)"
                )

        if cls_rows_to_append:
            wrapper_cls_head_last_layer_weight = torch.cat(
                [wrapper_cls_head_last_layer_weight] + cls_rows_to_append, dim=0
            )

        wrapper.classes = updated_classes

    wrapper.cls_head = wrapper._build_cls_head(
        len(updated_classes), device=cls_head_device, dtype=cls_head_dtype
    )
    wrapper.cls_head[-1].weight.data = wrapper_cls_head_last_layer_weight.detach().to(device=cls_head_device, dtype=cls_head_dtype)

    wrapper_only_classes = [name for name in wrapper.classes if name not in ckpt_classes]
    if mapping_logs:
        print("[load_wrapper_checkpoint] class mapping:")
        print("\n".join(mapping_logs))
    if wrapper_only_classes:
        print(
            "[load_wrapper_checkpoint] wrapper-only classes unchanged: "
            + ", ".join(wrapper_only_classes)
        )

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


from quiet_warnings import silence_known_training_warnings  # noqa: E402  # re-export