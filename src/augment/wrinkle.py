# -*- coding: utf-8 -*-
"""
Module Wrinkle: Nhăn giấy học bạ 3D sử dụng Perlin / Fractal Noise.
Mô phỏng cả độ gồ ghề bề mặt (Normal Shading) và độ uốn lượn hình học (Displacement Mapping).
Tối ưu hiệu năng cao bằng multi-scale và phép tính mảng trực tiếp.
"""

import math
import cv2
import numpy as np


def generate_fractal_height_map(h: int, w: int, octaves: int = 4, seed: int = None) -> np.ndarray:
    """
    Sinh bản đồ độ cao Perlin/Fractal đa tần số cho bề mặt giấy nhăn.
    """
    rng = np.random.RandomState(seed)
    height_map = np.zeros((h, w), dtype=np.float32)

    scales = [160, 80, 40, 20]
    weights = [1.0, 0.5, 0.25, 0.12]

    for scale, weight in zip(scales[:octaves], weights[:octaves]):
        grid_h = max(2, int(math.ceil(h / scale)))
        grid_w = max(2, int(math.ceil(w / scale)))
        low_res = rng.normal(0, 1.0, (grid_h, grid_w)).astype(np.float32)
        upscaled = cv2.resize(low_res, (w, h), interpolation=cv2.INTER_LINEAR)
        height_map += weight * upscaled

    denom = np.max(np.abs(height_map)) + 1e-6
    return height_map / denom


def apply_wrinkles(
    image: np.ndarray,
    intensity: float = 0.5,
    seed: int = None
) -> np.ndarray:
    """
    Tạo hiệu ứng nhăn giấy học bạ 3D với Shading và Displacement.
    """
    if intensity <= 0.01:
        return image

    rng = np.random.RandomState(seed)
    h, w = image.shape[:2]

    # Tính toán shading ở độ phân giải 1/2 để tăng tốc gấp 4 lần
    sh, sw = max(2, h // 2), max(2, w // 2)

    height_map = generate_fractal_height_map(sh, sw, octaves=4, seed=seed)

    gx = cv2.Sobel(height_map, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(height_map, cv2.CV_32F, 0, 1, ksize=3)

    light_angle = rng.uniform(-math.pi * 0.7, -math.pi * 0.3)
    lx = math.cos(light_angle) * 0.6
    ly = math.sin(light_angle) * 0.6
    lz = 1.0

    len_l = math.sqrt(lx ** 2 + ly ** 2 + lz ** 2)
    lx /= len_l
    ly /= len_l
    lz /= len_l

    norm_len = np.sqrt(gx ** 2 + gy ** 2 + 1.0)
    shading = (-gx * lx - gy * ly + lz) / norm_len

    shading_factor = 1.0 + (shading - 1.0) * (0.35 * intensity)
    shading_factor = np.clip(shading_factor, 0.65, 1.35)

    shading_full = cv2.resize(shading_factor, (w, h), interpolation=cv2.INTER_LINEAR)
    shading_u8 = np.clip(shading_full * 255.0 / 1.35, 0, 255).astype(np.uint8)
    shading_3ch = cv2.merge([shading_u8, shading_u8, shading_u8])

    out = cv2.multiply(image, shading_3ch, scale=1.35 / 255.0)

    # Displacement mapping
    if intensity > 0.35:
        disp_scale = 1.8 * intensity
        gx_full = cv2.resize(gx * disp_scale, (w, h), interpolation=cv2.INTER_LINEAR)
        gy_full = cv2.resize(gy * disp_scale, (w, h), interpolation=cv2.INTER_LINEAR)

        y_grid, x_grid = np.indices((h, w), dtype=np.float32)
        map_x = x_grid + gx_full
        map_y = y_grid + gy_full
        out = cv2.remap(out, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

    return out
