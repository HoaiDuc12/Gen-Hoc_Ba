# -*- coding: utf-8 -*-
"""
Module Stain: Vết ố vàng, vệt nước loang đọng giọt (Coffee ring effect) và vết bẩn thời gian trên giấy.
"""

import math
import cv2
import numpy as np


def generate_organic_stain_mask(h: int, w: int, cx: int, cy: int, radius: float, rng: np.random.RandomState) -> tuple:
    """
    Tạo mặt nạ vết ố hữu cơ có hình dạng méo tự nhiên và hiệu ứng viền giọt nước đọng.
    """
    mask = np.zeros((h, w), dtype=np.float32)

    # Sinh chuỗi bán kính r(theta) quanh tâm với nhiễu hữu cơ
    num_pts = 60
    angles = np.linspace(0, 2 * math.pi, num_pts, endpoint=False)
    # Nhiễu dạng sóng làm méo hình tròn
    r_variation = radius * (1.0 + rng.uniform(-0.35, 0.35, num_pts))
    # Làm mượt đa giác bằng cách lọc lặp lại
    r_smooth = np.convolve(np.tile(r_variation, 3), np.ones(5) / 5.0, mode="same")[num_pts:2 * num_pts]

    poly_pts = []
    for a, r in zip(angles, r_smooth):
        px = int(cx + r * math.cos(a))
        py = int(cy + r * math.sin(a))
        poly_pts.append([px, py])

    poly_np = np.array(poly_pts, dtype=np.int32)
    # Vẽ thân vết ố
    cv2.fillPoly(mask, [poly_np], 1.0)

    # Hiệu ứng viền đọng giọt nước (Coffee-ring effect: mép đậm hơn tâm)
    edge_mask = np.zeros((h, w), dtype=np.float32)
    cv2.polylines(edge_mask, [poly_np], isClosed=True, color=1.0, thickness=max(2, int(radius * 0.08)))

    # Làm mờ mềm hoá
    blur_k = max(5, int(radius * 0.2))
    if blur_k % 2 == 0:
        blur_k += 1
    body_blurred = cv2.GaussianBlur(mask, (blur_k, blur_k), blur_k * 0.3)
    edge_blurred = cv2.GaussianBlur(edge_mask, (blur_k, blur_k), blur_k * 0.25)

    return body_blurred, edge_blurred


def apply_stains(
    image: np.ndarray,
    intensity: float = 0.5,
    num_stains: int = None,
    seed: int = None
) -> np.ndarray:
    """
    Thêm các vết ố vàng nâu, vệt nước loang hoặc vết bẩn thời gian lên giấy học bạ.
    """
    if intensity <= 0.01:
        return image

    rng = np.random.RandomState(seed)
    h, w = image.shape[:2]
    out = image.astype(np.float32)

    if num_stains is None:
        num_stains = rng.randint(1, 4)

    for _ in range(num_stains):
        stain_type = rng.choice(["coffee_ring", "yellow_patch", "dirt_smudge"])

        # Vị trí ngẫu nhiên
        cx = rng.randint(int(w * 0.1), int(w * 0.9))
        cy = rng.randint(int(h * 0.1), int(h * 0.9))
        radius = rng.uniform(min(w, h) * 0.05, min(w, h) * 0.18)

        body_mask, edge_mask = generate_organic_stain_mask(h, w, cx, cy, radius, rng)

        if stain_type == "coffee_ring":
            # Vết đọng cốc nước / cà phê: viền vàng nâu đậm, lòng trong nhạt
            # Màu BGR: Vàng nâu nhạt bên trong, nâu cánh gián viền ngoài
            stain_body_color = np.array([55, 110, 160], dtype=np.float32)
            stain_edge_color = np.array([30, 75, 120], dtype=np.float32)

            alpha_body = body_mask[:, :, None] * (0.15 * intensity)
            alpha_edge = edge_mask[:, :, None] * (0.35 * intensity)

            # Hoà trộn nhân (multiply blend) để giữ nguyên chữ bên dưới
            out = out * (1.0 - alpha_body * (1.0 - stain_body_color / 255.0))
            out = out * (1.0 - alpha_edge * (1.0 - stain_edge_color / 255.0))

        elif stain_type == "yellow_patch":
            # Vết ố vàng ẩm mốc loang rộng
            stain_color = np.array([40, 100, 150], dtype=np.float32)
            alpha = body_mask[:, :, None] * (0.22 * intensity)
            out = out * (1.0 - alpha * (1.0 - stain_color / 255.0))

        else:
            # Vệt bụi bẩn xám mờ (dirt smudge)
            dirt_color = np.array([80, 80, 85], dtype=np.float32)
            alpha = body_mask[:, :, None] * (0.18 * intensity)
            out = out * (1.0 - alpha * (1.0 - dirt_color / 255.0))

    return np.clip(out, 0, 255).astype(np.uint8)

