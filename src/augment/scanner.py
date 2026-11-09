# -*- coding: utf-8 -*-
"""
Module Scanner: Giả lập hiệu ứng quét ứng dụng CamScanner (nền trắng xóa, tương phản cực mạnh,
giữ màu mực thần kỳ / Magic Color) và giả lập bản photocopy đen trắng có vệt gáy kính quét.
"""

import math
import cv2
import numpy as np


def apply_camscanner(
    image: np.ndarray,
    intensity: float = 0.8,
    mode: str = "magic_color",
    seed: int = None
) -> np.ndarray:
    """
    Giả lập thuật toán xử lý ảnh của CamScanner:
    - Loại bỏ hoàn toàn bóng đổ không đều, đưa nền giấy về trắng tinh (245-255).
    - Làm sắc nét mép chữ (Unsharp masking).
    - Mode 'magic_color': Giữ lại màu con dấu đỏ và mực bút bi xanh nhưng tăng tương phản.
    - Mode 'bw': Chuyển thành scan đen trắng đơn sắc có đường nét sắc lẹm.
    """
    if intensity <= 0.01:
        return image

    rng = np.random.RandomState(seed)
    h, w = image.shape[:2]

    # Chuyển sang ảnh xám để ước lượng nền chiếu sáng (Background Illumination Map)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Ước lượng nền bằng bộ lọc làm mịn diện rộng
    kernel_size = int(max(w, h) * 0.08)
    if kernel_size % 2 == 0:
        kernel_size += 1
    bg_illum = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0).astype(np.float32) + 1.0

    # Chuẩn hoá loại bỏ bóng (Division normalization)
    # Chia ảnh cho bản đồ nền để san phẳng ánh sáng
    norm_channels = []
    for c in range(3):
        ch = image[:, :, c].astype(np.float32)
        norm_ch = (ch / bg_illum) * 230.0
        norm_channels.append(norm_ch)

    flat_img = np.stack(norm_channels, axis=-1)

    # Kéo giãn tương phản (Contrast Stretch):
    # Đẩy nền trắng > 220 lên 255, kéo chữ tối < 120 xuống sâu hơn
    white_thresh = 220.0 - 25.0 * intensity
    dark_thresh = 110.0 + 20.0 * intensity

    stretched = np.clip((flat_img - dark_thresh) / (white_thresh - dark_thresh) * 255.0, 0, 255.0)

    # Làm sắc nét viền chữ (Unsharp Masking)
    blurred = cv2.GaussianBlur(stretched, (0, 0), sigmaX=1.5)
    sharpened = cv2.addWeighted(stretched, 1.4, blurred, -0.4, 0)
    sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)

    if mode == "bw":
        # Chế độ đen trắng
        gray_sharp = cv2.cvtColor(sharpened, cv2.COLOR_BGR2GRAY)
        # Ngưỡng thích nghi nhẹ
        _, binarized = cv2.threshold(gray_sharp, 205, 255, cv2.THRESH_BINARY)
        # Ghép thành 3 kênh BGR
        sharpened = cv2.cvtColor(binarized, cv2.COLOR_GRAY2BGR)

    # Hoà trộn theo intensity
    out = (image.astype(np.float32) * (1.0 - intensity) + sharpened.astype(np.float32) * intensity)
    return np.clip(out, 0, 255).astype(np.uint8)


def apply_photocopy(
    image: np.ndarray,
    intensity: float = 0.7,
    has_spine_shadow: bool = True,
    seed: int = None
) -> np.ndarray:
    """
    Mô phỏng bản photocopy đen trắng:
    - Mất hoàn toàn màu sắc, chuyển về đơn sắc có độ tương phản gắt.
    - Hạt bụi mực toner (Toner speckles).
    - Vệt đen mép gáy giấy (Spine shadow do gáy sổ đè lên mặt kính máy photo).
    """
    if intensity <= 0.01:
        return image

    rng = np.random.RandomState(seed)
    h, w = image.shape[:2]

    # 1. Chuyển sang ảnh xám
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 2. Tăng tương phản gắt (S-curve)
    norm = gray.astype(np.float32) / 255.0
    gamma = 1.3
    contrast = np.clip((norm ** gamma - 0.25) / 0.55 * 255.0, 0, 255.0)

    # 3. Vệt đen gáy mép sổ photocopy (Spine Shadow)
    if has_spine_shadow:
        spine_side = rng.choice(["left", "right"])
        spine_width = int(w * rng.uniform(0.04, 0.08))
        if spine_side == "left":
            x_spine = np.linspace(1.0, 0.0, spine_width)[None, :]
            spine_shadow = np.zeros((h, w), dtype=np.float32)
            spine_shadow[:, :spine_width] = x_spine ** 1.8
        else:
            x_spine = np.linspace(0.0, 1.0, spine_width)[None, :]
            spine_shadow = np.zeros((h, w), dtype=np.float32)
            spine_shadow[:, -spine_width:] = x_spine ** 1.8

        contrast = contrast * (1.0 - spine_shadow * 0.75)

    # 4. Hạt bụi mực photocopy (Toner specks)
    num_specks = int(w * h * 0.0003 * intensity)
    if num_specks > 0:
        speck_y = rng.randint(0, h, num_specks)
        speck_x = rng.randint(0, w, num_specks)
        contrast[speck_y, speck_x] = rng.uniform(30, 90, num_specks)

    out_gray = np.clip(contrast, 0, 255).astype(np.uint8)
    # Chuyển về 3 kênh BGR
    return cv2.cvtColor(out_gray, cv2.COLOR_GRAY2BGR)

