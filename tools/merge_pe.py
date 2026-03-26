import argparse
import os
from typing import Dict, Tuple

import torch


def parse_args():
    parser = argparse.ArgumentParser("Merge prompt embedding checkpoints")
    parser.add_argument("--prompt_a", type=str, required=True, help="Checkpoint A path.")
    parser.add_argument("--prompt_b", type=str, required=True, help="Checkpoint B path.")
    parser.add_argument("--output_path", type=str, required=True, help="Merged checkpoint output path.")
    parser.add_argument(
        "--mode",
        type=str,
        default="fine",
        choices=["fine", "coarse"],
        help="fine: class-independent merge, coarse: shared/global weighted merge.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Weight for prompt_a in weighted average. merged = alpha * A + (1-alpha) * B",
    )
    return parser.parse_args()


def _load_prompt_checkpoint(path: str) -> Tuple[Dict, Dict]:
    payload = torch.load(path, map_location="cpu")
    if isinstance(payload, dict) and "prompt_state_dict" in payload:
        return payload["prompt_state_dict"], payload.get("metadata", {})
    if isinstance(payload, dict) and "prompt_embeddings" in payload:
        return payload, {}
    raise ValueError(f"Unsupported prompt checkpoint format: {path}")


def _infer_mode(prompt_state: Dict, metadata: Dict) -> str:
    mode = metadata.get("prompt_mode")
    if mode in {"shared", "class_independent"}:
        return mode
    if prompt_state.get("prompt_token_map") is not None:
        return "class_independent"
    return "shared"


def _merge_coarse(prompt_a: Dict, prompt_b: Dict, alpha: float):
    emb_a = prompt_a["prompt_embeddings"].float()
    emb_b = prompt_b["prompt_embeddings"].float()
    if emb_a.shape != emb_b.shape:
        raise ValueError(
            f"Coarse merge requires same shape. Got A={tuple(emb_a.shape)}, B={tuple(emb_b.shape)}"
        )
    merged = alpha * emb_a + (1.0 - alpha) * emb_b
    merged_state = {
        "prompt_embeddings": merged,
        "prompt_mode": "shared",
    }
    merged_meta = {
        "format_version": 2,
        "prompt_mode": "shared",
        "merge_mode": "coarse",
        "alpha": alpha,
        "category_to_prompt_embedding_idx": {},
        "prompt_embedding_idx_to_category": [],
    }
    return merged_state, merged_meta


def _merge_fine(prompt_a: Dict, meta_a: Dict, prompt_b: Dict, meta_b: Dict, alpha: float):
    map_a = meta_a.get("category_to_prompt_embedding_idx", {}) or {}
    map_b = meta_b.get("category_to_prompt_embedding_idx", {}) or {}
    if not map_a or not map_b:
        raise ValueError("Fine merge requires metadata.category_to_prompt_embedding_idx in both checkpoints.")

    emb_a = prompt_a["prompt_embeddings"].float()
    emb_b = prompt_b["prompt_embeddings"].float()
    if emb_a.shape[1] != emb_b.shape[1]:
        raise ValueError(
            f"Embedding dim mismatch for fine merge. Got A={emb_a.shape[1]}, B={emb_b.shape[1]}"
        )

    union_categories = list(map_a.keys())
    for name in map_b.keys():
        if name not in map_a:
            union_categories.append(name)

    new_map = {name: idx for idx, name in enumerate(union_categories)}
    merged_emb = torch.zeros((len(union_categories), emb_a.shape[1]), dtype=torch.float32)

    for category_name, new_idx in new_map.items():
        in_a = category_name in map_a and int(map_a[category_name]) < emb_a.shape[0]
        in_b = category_name in map_b and int(map_b[category_name]) < emb_b.shape[0]
        if in_a and in_b:
            merged_emb[new_idx] = alpha * emb_a[int(map_a[category_name])] + (1.0 - alpha) * emb_b[
                int(map_b[category_name])
            ]
        elif in_a:
            merged_emb[new_idx] = emb_a[int(map_a[category_name])]
        elif in_b:
            merged_emb[new_idx] = emb_b[int(map_b[category_name])]
        else:
            raise RuntimeError(f"Unexpected category mapping state: {category_name}")

    merged_state = {
        "prompt_embeddings": merged_emb,
        "prompt_mode": "class_independent",
    }
    merged_meta = {
        "format_version": 2,
        "prompt_mode": "class_independent",
        "merge_mode": "fine",
        "alpha": alpha,
        "category_to_prompt_embedding_idx": new_map,
        "prompt_embedding_idx_to_category": [
            {"prompt_embedding_idx": idx, "category_name": name} for name, idx in new_map.items()
        ],
    }
    return merged_state, merged_meta


def main():
    args = parse_args()
    if not (0.0 <= args.alpha <= 1.0):
        raise ValueError("--alpha must be in [0, 1].")

    prompt_a, meta_a = _load_prompt_checkpoint(args.prompt_a)
    prompt_b, meta_b = _load_prompt_checkpoint(args.prompt_b)
    mode_a = _infer_mode(prompt_a, meta_a)
    mode_b = _infer_mode(prompt_b, meta_b)

    if args.mode == "coarse":
        merged_state, merged_meta = _merge_coarse(prompt_a, prompt_b, args.alpha)
    else:
        if mode_a != "class_independent" or mode_b != "class_independent":
            raise ValueError(
                f"Fine merge expects class-independent checkpoints. Got A={mode_a}, B={mode_b}."
            )
        merged_state, merged_meta = _merge_fine(prompt_a, meta_a, prompt_b, meta_b, args.alpha)

    out_dir = os.path.dirname(args.output_path) or "."
    os.makedirs(out_dir, exist_ok=True)
    payload = {
        "prompt_state_dict": merged_state,
        "metadata": merged_meta,
    }
    torch.save(payload, args.output_path)
    print(f"Merged checkpoint saved to: {args.output_path}")


if __name__ == "__main__":
    main()
