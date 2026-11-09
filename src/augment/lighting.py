# -*- coding: utf-8 -*-
"""
Module Lighting: Hiệu ứng ánh sáng thực tế bao gồm chói đèn flash (flash glare),
bóng người/vật thể che khuất (cast shadow gradient), và tối 4 góc ống kính (vignette).
Tối ưu hoá hiệu năng cao bằng OpenCV uint8 và xử lý đa tỉ lệ (Multi-scale).
"""

import math
import cv2
import numpy as np


def apply_flash_glare(
    image: np.ndarray,
    intensity: float = 0.5,
    center: tuple = None,
    seed: int = None
) -> np.ndarray:
    """
    Tạo điểm chói sáng đèn flash điện thoại (Specular highlight / Glare).
    Ánh sáng tập trung ở một điểm tâm và toả dần ra xung quanh, làm lóa một phần tài liệu.
    """
    if intensity <= 0.01:
        return image

    rng = np.random.RandomState(seed)
    h, w = image.shape[:2]

    # Vị trí điểm sáng flash
    if center is None:
        cx = int(w * rng.uniform(0.3, 0.7))
        cy = int(h * rng.uniform(0.25, 0.65))
    else:
        cx, cy = center

    # Bán kính vùng chói sáng
    radius = int(min(w, h) * (0.35 + 0.35 * intensity))

    # Tối ưu: Tính toán bản đồ ánh sáng ở độ phân giải 1/4 rồi phóng to nội suy tuyến tính
    sh, sw = max(2, h // 4), max(2, w // 4)
    scx, scy = cx / 4.0, cy / 4.0
    sradius = radius / 4.0

    y_grid, x_grid = np.ogrid[:sh, :sw]
    dist_sq = (x_grid - scx) ** 2 + (y_grid - scy) ** 2
    sigma_sq = (sradius * 0.55) ** 2

    # Điểm sáng tâm + vầng sáng phụ
    glare_map = np.exp(-dist_sq / (2.0 * sigma_sq))
    halo_map = np.exp(-dist_sq / (2.0 * (sradius * 1.5) ** 2)) * 0.35
    total_glare = np.clip(glare_map + halo_map, 0, 1.0)

    # Cường độ sáng tăng thêm (tối đa +160..+240 cho vùng trung tâm đèn)
    boost = 180.0 * intensity
    glare_small = (total_glare * boost).astype(np.float32)

    # Phóng to về kích thước gốc
    glare_full = cv2.resize(glare_small, (w, h), interpolation=cv2.INTER_LINEAR)

    # Ánh sáng flash trắng xanh nhẹ tự nhiên (BGR)
    b_ch = np.clip(glare_full * 1.05, 0, 255).astype(np.uint8)
    g_ch = np.clip(glare_full * 1.02, 0, 255).astype(np.uint8)
    r_ch = np.clip(glare_full * 0.98, 0, 255).astype(np.uint8)

    glare_bgr = cv2.merge([b_ch, g_ch, r_ch])
    return cv2.add(image, glare_bgr)


def apply_shadow_gradient(
    image: np.ndarray,
    intensity: float = 0.5,
    seed: int = None
) -> np.ndarray:
    """
    Mô phỏng bóng đổ của người chụp, bàn tay hoặc đồ vật chắn ánh sáng đèn phòng.
    Tạo dải bóng mờ có viền mềm và chuyển độ mượt mà qua tài liệu.
    """
    if intensity <= 0.01:
        return image

    rng = np.random.RandomState(seed)
    h, w = image.shape[:2]

    # Góc của dải bóng đổ
    angle = rng.uniform(0, 2 * math.pi)
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)

    # Tính toán ở độ phân giải 1/4 để tăng tốc gấp 25 lần
    sh, sw = max(2, h // 4), max(2, w // 4)
    scx, scy = sw / 2.0, sh / 2.0

    xs = ((np.arange(sw, dtype=np.float32) - scx) * cos_a)[None, :]
    ys = ((np.arange(sh, dtype=np.float32) - scy) * sin_a)[:, None]
    max_extent = math.sqrt(scx ** 2 + scy ** 2)
    norm_proj = (xs + ys) / max_extent

    shadow_offset = rng.uniform(-0.3, 0.3)
    sharpness = rng.uniform(2.5, 5.0)
    shadow_mask = 1.0 / (1.0 + np.exp(-sharpness * (norm_proj - shadow_offset)))

    # Mức độ tối cực đại (tối đi khoảng 25% - 45%)
    max_darken = 0.45 * intensity
    darkening = 1.0 - shadow_mask * max_darken

    # Mềm hoá và phóng to về kích thước gốc
    darkening = cv2.GaussianBlur(darkening.astype(np.float32), (9, 9), 4.0)
    dark_full = cv2.resize(darkening, (w, h), interpolation=cv2.INTER_LINEAR)

    dark_u8 = np.clip(dark_full * 255.0, 0, 255).astype(np.uint8)
    dark_3ch = cv2.merge([dark_u8, dark_u8, dark_u8])

    return cv2.multiply(image, dark_3ch, scale=1.0 / 255.0)


def apply_vignette(image: np.ndarray, intensity: float = 0.5) -> np.ndarray:
    """
    Hiệu ứng vignette: 4 góc bức ảnh bị tối dần do đặc tính quang học của ống kính camera.
    """
    if intensity <= 0.01:
        return image

    h, w = image.shape[:2]
    sh, sw = max(2, h // 4), max(2, w // 4)
    scx, scy = sw / 2.0, sh / 2.0
    max_dist = math.sqrt(scx ** 2 + scy ** 2)

    xs = ((np.arange(sw, dtype=np.float32) - scx) / max_dist) ** 2
    ys = ((np.arange(sh, dtype=np.float32) - scy) / max_dist) ** 2
    dist_sq = xs[None, :] + ys[:, None]

    falloff = np.sqrt(dist_sq)
    vignette = 1.0 - np.clip(falloff ** 2.2 * (0.55 * intensity), 0, 1.0)
    vig_full = cv2.resize(vignette, (w, h), interpolation=cv2.INTER_LINEAR)

    vig_u8 = np.clip(vig_full * 255.0, 0, 255).astype(np.uint8)
    vig_3ch = cv2.merge([vig_u8, vig_u8, vig_u8])

    return cv2.multiply(image, vig_3ch, scale=1.0 / 255.0)
