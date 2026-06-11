"""Core-set selection algorithms in feature space."""

from __future__ import annotations

import heapq
from typing import List, Set

import numpy as np
from sklearn.cluster import MiniBatchKMeans


def _normalize_rows(feats: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(feats, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return feats / norms


def _fill_to_budget(
    selected_images: List[int],
    budget: int,
    image_idx_map: np.ndarray,
    feats: np.ndarray,
    rng: np.random.Generator,
) -> List[int]:
    selected_set = set(selected_images)
    if len(selected_set) >= budget:
        return sorted(selected_set)[:budget]

    remaining_images = sorted(set(int(i) for i in image_idx_map) - selected_set)
    if not remaining_images:
        return sorted(selected_set)

    selected_feature_indices = [i for i, img in enumerate(image_idx_map) if int(img) in selected_set]
    if not selected_feature_indices:
        first = int(rng.choice(remaining_images))
        selected_set.add(first)
        selected_feature_indices = [i for i, img in enumerate(image_idx_map) if int(img) == first]

    anchor = feats[selected_feature_indices].mean(axis=0, keepdims=True)
    min_dist = np.linalg.norm(feats - anchor, axis=1)

    while len(selected_set) < budget and remaining_images:
        masked_dist = min_dist.copy()
        for i, img in enumerate(image_idx_map):
            if int(img) in selected_set:
                masked_dist[i] = -1.0
        next_feat_idx = int(np.argmax(masked_dist))
        next_image = int(image_idx_map[next_feat_idx])
        if next_image in selected_set:
            break
        selected_set.add(next_image)
        remaining_images.remove(next_image)
        new_dist = np.linalg.norm(feats - feats[next_feat_idx : next_feat_idx + 1], axis=1)
        min_dist = np.minimum(min_dist, new_dist)

    return sorted(selected_set)[:budget]


def select_kcenter(
    feats: np.ndarray,
    image_idx_map: np.ndarray,
    budget: int,
    seed: int,
) -> List[int]:
    if budget <= 0:
        return []
    n = int(feats.shape[0])
    if n == 0:
        return []

    rng = np.random.default_rng(seed)
    unique_images = sorted(set(int(i) for i in image_idx_map))
    budget = min(budget, len(unique_images))
    if budget == len(unique_images):
        return unique_images

    first_idx = int(rng.integers(0, n))
    selected_images: Set[int] = {int(image_idx_map[first_idx])}
    min_dist = np.linalg.norm(feats - feats[first_idx : first_idx + 1], axis=1)

    while len(selected_images) < budget:
        masked_dist = min_dist.copy()
        for i, img in enumerate(image_idx_map):
            if int(img) in selected_images:
                masked_dist[i] = -1.0
        next_idx = int(np.argmax(masked_dist))
        if masked_dist[next_idx] < 0:
            break
        selected_images.add(int(image_idx_map[next_idx]))
        new_dist = np.linalg.norm(feats - feats[next_idx : next_idx + 1], axis=1)
        min_dist = np.minimum(min_dist, new_dist)

    return _fill_to_budget(sorted(selected_images), budget, image_idx_map, feats, rng)


def _marginal_gain(sim_row: np.ndarray, current_max: np.ndarray) -> float:
    return float((np.maximum(current_max, sim_row) - current_max).sum())


def select_facility_location(
    feats: np.ndarray,
    image_idx_map: np.ndarray,
    budget: int,
    seed: int,
) -> List[int]:
    if budget <= 0:
        return []
    n = int(feats.shape[0])
    if n == 0:
        return []

    rng = np.random.default_rng(seed)
    unique_images = sorted(set(int(i) for i in image_idx_map))
    budget = min(budget, len(unique_images))
    if budget == len(unique_images):
        return unique_images

    norm_feats = _normalize_rows(feats)
    sim = norm_feats @ norm_feats.T
    current_max = np.full((n,), -np.inf, dtype=np.float32)
    selected_images: Set[int] = set()
    pq: List[tuple[float, int, int]] = []
    for cand in range(n):
        gain = _marginal_gain(sim[cand], current_max)
        heapq.heappush(pq, (-gain, 0, cand))

    step = 0
    while pq and len(selected_images) < budget:
        neg_gain, last_updated, cand = heapq.heappop(pq)
        if int(image_idx_map[cand]) in selected_images:
            continue
        true_gain = _marginal_gain(sim[cand], current_max)
        if true_gain < -neg_gain - 1e-8:
            heapq.heappush(pq, (-true_gain, step, cand))
            continue
        selected_images.add(int(image_idx_map[cand]))
        current_max = np.maximum(current_max, sim[cand])
        step += 1

    return _fill_to_budget(sorted(selected_images), budget, image_idx_map, feats, rng)


def select_kmeans(
    feats: np.ndarray,
    image_idx_map: np.ndarray,
    budget: int,
    seed: int,
) -> List[int]:
    if budget <= 0:
        return []
    n = int(feats.shape[0])
    if n == 0:
        return []

    rng = np.random.default_rng(seed)
    unique_images = sorted(set(int(i) for i in image_idx_map))
    budget = min(budget, len(unique_images))
    if budget == len(unique_images):
        return unique_images

    kmeans = MiniBatchKMeans(
        n_clusters=budget,
        random_state=seed,
        batch_size=min(4096, n),
        n_init="auto",
    )
    kmeans.fit(feats)
    centers = kmeans.cluster_centers_

    selected_images: Set[int] = set()
    used_feature_indices: Set[int] = set()
    for center in centers:
        dist = np.linalg.norm(feats - center.reshape(1, -1), axis=1)
        order = np.argsort(dist)
        for feat_idx in order:
            feat_idx = int(feat_idx)
            if feat_idx in used_feature_indices:
                continue
            image_idx = int(image_idx_map[feat_idx])
            if image_idx in selected_images:
                continue
            selected_images.add(image_idx)
            used_feature_indices.add(feat_idx)
            break

    return _fill_to_budget(sorted(selected_images), budget, image_idx_map, feats, rng)


def run_selection(
    algorithm: str,
    feats: np.ndarray,
    image_idx_map: np.ndarray,
    budget: int,
    seed: int,
) -> List[int]:
    if algorithm == "kcenter":
        return select_kcenter(feats, image_idx_map, budget, seed)
    if algorithm == "facility_location":
        return select_facility_location(feats, image_idx_map, budget, seed)
    if algorithm == "kmeans":
        return select_kmeans(feats, image_idx_map, budget, seed)
    raise ValueError(f"Unsupported selection algorithm: {algorithm}")
