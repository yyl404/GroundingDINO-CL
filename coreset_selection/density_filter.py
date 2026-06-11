"""Density-based outlier filtering in feature space."""

from __future__ import annotations

from typing import Sequence, Set

import numpy as np
from sklearn.neighbors import NearestNeighbors


def filter_outlier_images(
    feats: np.ndarray,
    image_idx_map: np.ndarray,
    candidate_image_indices: Sequence[int],
    k: int,
    outlier_percentile: float,
) -> tuple[list[int], dict]:
    n_feats = int(feats.shape[0])
    if n_feats == 0:
        return list(candidate_image_indices), {
            "num_features": 0,
            "num_outlier_features": 0,
            "num_outlier_images": 0,
            "threshold": None,
        }

    k_eff = min(max(int(k), 1), n_feats - 1)
    nn = NearestNeighbors(n_neighbors=k_eff + 1, metric="euclidean")
    nn.fit(feats)
    distances, _ = nn.kneighbors(feats)
    mean_knn_dist = distances[:, 1 : k_eff + 1].mean(axis=1)

    threshold = float(np.percentile(mean_knn_dist, float(outlier_percentile)))
    outlier_feature_mask = mean_knn_dist >= threshold
    outlier_images: Set[int] = set(int(image_idx_map[i]) for i in np.where(outlier_feature_mask)[0])

    clean = [int(i) for i in candidate_image_indices if int(i) not in outlier_images]
    stats = {
        "num_features": n_feats,
        "num_outlier_features": int(outlier_feature_mask.sum()),
        "num_outlier_images": len(outlier_images),
        "threshold": threshold,
        "k": k_eff,
        "outlier_percentile": float(outlier_percentile),
    }
    return clean, stats
