# -*- coding: utf-8 -*-
"""
Module Fold: Mô phỏng nếp gấp giấy học bạ (1-3 nếp gấp ngang/dọc/chéo).
Bao gồm: vệt rãnh tối mảnh, gờ sáng phản quang, vùng bóng lượn phập phồng và vi dịch chuyển pixel.
Tối ưu hiệu năng cao bằng broadcasting 1D và xử lý uint8.
"""

import math
import cv2
import numpy as np


def apply_folds(
    image: np.ndarray,
    intensity: float = 0.5,
    num_folds: int = None,
    seed: int = None
) -> np.ndarray:
    """
    Tạo các nếp gấp giấy tự nhiên như học bạ thực tế bị gấp làm đôi/ba khi cất giữ.
    """
    if intensity <= 0.01:
        return image

    rng = np.random.RandomState(seed)
    h, w = image.shape[:2]

    if num_folds is None:
        num_folds = rng.choice([1, 2, 3], p=[0.4, 0.4, 0.2])

    # Tích luỹ shading của tất cả các nếp gấp (mặc định bằng 1.0)
    total_darken = np.zeros((h, w), dtype=np.float32)
    total_lighten = np.zeros((h, w), dtype=np.float32)
    total_bulge = np.zeros((h, w), dtype=np.float32)

    has_displacement = intensity > 0.35
    accum_dx = np.zeros((h, w), dtype=np.float32) if has_displacement else None
    accum_dy = np.zeros((h, w), dtype=np.float32) if has_displacement else None

    xs = np.arange(w, dtype=np.float32)[None, :]
    ys = np.arange(h, dtype=np.float32)[:, None]

    for _ in range(num_folds):
        fold_type = rng.choice(["horizontal", "horizontal", "vertical", "diagonal"])

        if fold_type == "horizontal":
            y0 = h * rng.uniform(0.25, 0.75)
            x0 = w * 0.5
            angle = rng.uniform(-0.06, 0.06)
        elif fold_type == "vertical":
            y0 = h * 0.5
            x0 = w * rng.uniform(0.3, 0.7)
            angle = math.pi / 2.0 + rng.uniform(-0.06, 0.06)
        else:
            y0 = h * rng.uniform(0.2, 0.8)
            x0 = w * rng.uniform(0.2, 0.8)
            angle = rng.uniform(-math.pi / 4, math.pi / 4)

        cos_a = math.cos(angle)
        sin_a = math.sin(angle)

        # Khoảng cách dist từ mỗi pixel tới đường nếp gấp tính qua broadcasting
        dist = (xs - x0) * (-sin_a) + (ys - y0) * cos_a

        # 1. Rãnh nếp gấp (Crease core)
        crease_width = 2.5 + 2.0 * intensity
        crease_dark = np.exp(-((dist + 1.0) ** 2) / (2.0 * (crease_width * 0.7) ** 2))
        crease_light = np.exp(-((dist - 1.5) ** 2) / (2.0 * (crease_width * 0.8) ** 2))

        # 2. Vùng bóng lượn thân giấy (Paper bulge curvature)
        bulge_width = 45.0 + 60.0 * intensity
        bulge_shadow = np.exp(-np.abs(dist) / bulge_width) * np.sin(np.clip(dist / bulge_width, -math.pi, math.pi))

        total_darken += crease_dark * (0.35 * intensity)
        total_lighten += crease_light * (0.22 * intensity)
        total_bulge += bulge_shadow * (0.12 * intensity)

        if has_displacement:
            shift = (1.5 * intensity) * np.sign(dist) * np.exp(-np.abs(dist) / 6.0)
            accum_dx += (shift * cos_a).astype(np.float32)
            accum_dy += (shift * sin_a).astype(np.float32)

    # Tổng hợp hệ số sáng tối
    total_shading = np.clip(1.0 - total_darken + total_lighten + total_bulge, 0.5, 1.4)
    shading_u8 = np.clip(total_shading * 255.0, 0, 255).astype(np.uint8)
    shading_3ch = cv2.merge([shading_u8, shading_u8, shading_u8])

    out = cv2.multiply(image, shading_3ch, scale=1.0 / 255.0)

    # Áp dụng displacement 1 lần duy nhất trên toàn bộ nếp gấp
    if has_displacement:
        y_grid, x_grid = np.indices((h, w), dtype=np.float32)
        map_x = x_grid + accum_dx
        map_y = y_grid + accum_dy
        out = cv2.remap(out, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

    return out
