from __future__ import annotations

from typing import Tuple

import torch
from torch import Tensor


def get_covariance_matrix(boxes: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Convert oriented boxes to independent covariance matrix elements.

    Args:
        boxes: [..., 5] in (cx, cy, w, h, theta) format.
    Returns:
        c11, c22, c12 for the corresponding 2x2 covariance matrix.
    """
    w = boxes[..., 2]
    h = boxes[..., 3]
    theta = boxes[..., 4]

    # Use variance of a uniform distribution along box axes.
    a = (w ** 2) / 12.0
    b = (h ** 2) / 12.0

    cos_t = torch.cos(theta)
    sin_t = torch.sin(theta)
    cos2 = cos_t ** 2
    sin2 = sin_t ** 2
    cos_sin = cos_t * sin_t

    c11 = a * cos2 + b * sin2
    c22 = a * sin2 + b * cos2
    c12 = (a - b) * cos_sin
    return c11, c22, c12


def probiou_pairwise(boxes1: Tensor, boxes2: Tensor, eps: float = 1e-7) -> Tensor:
    """
    Compute pairwise probabilistic IoU for OBB boxes in xywhr format.

    Args:
        boxes1: [N, 5]
        boxes2: [M, 5]
        eps: Numerical stability epsilon.
    Returns:
        Tensor [N, M] with values in [0, 1].
    """
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return boxes1.new_zeros((boxes1.shape[0], boxes2.shape[0]))

    cx1 = boxes1[:, 0].unsqueeze(1)
    cy1 = boxes1[:, 1].unsqueeze(1)
    cx2 = boxes2[:, 0].unsqueeze(0)
    cy2 = boxes2[:, 1].unsqueeze(0)

    a1, b1, c1 = get_covariance_matrix(boxes1)
    a2, b2, c2 = get_covariance_matrix(boxes2)
    a1, b1, c1 = a1.unsqueeze(1), b1.unsqueeze(1), c1.unsqueeze(1)
    a2, b2, c2 = a2.unsqueeze(0), b2.unsqueeze(0), c2.unsqueeze(0)

    num = (a1 + a2) * (b1 + b2) - (c1 + c2).pow(2)
    den = num + eps
    t1 = (((a1 + a2) * (cy1 - cy2).pow(2) + (b1 + b2) * (cx1 - cx2).pow(2)) / den) * 0.25
    t2 = (((c1 + c2) * (cx2 - cx1) * (cy1 - cy2)) / den) * 0.5
    t3 = (
        num / (4 * ((a1 * b1 - c1.pow(2)).clamp(min=0) * (a2 * b2 - c2.pow(2)).clamp(min=0)).sqrt() + eps) + eps
    ).log() * 0.5

    bd = (t1 + t2 + t3).clamp(min=eps, max=100.0)
    hd = (1.0 - (-bd).exp() + eps).sqrt()
    return 1.0 - hd


def probiou(boxes1: Tensor, boxes2: Tensor, eps: float = 1e-7) -> Tensor:
    """
    Compute probabilistic IoU for one-to-one matched OBB pairs.

    Args:
        boxes1: [N, 5]
        boxes2: [N, 5]
        eps: Numerical stability epsilon.
    Returns:
        Tensor [N] with values in [0, 1].
    """
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return boxes1.new_zeros((boxes1.shape[0],))
    return probiou_pairwise(boxes1, boxes2, eps=eps).diagonal()
