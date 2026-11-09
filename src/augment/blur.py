# -*- coding: utf-8 -*-
"""
Module Blur: Nhòe cục bộ (Local Blur) theo vùng ô điểm/chữ ký và Nhòe mực/loang mực bút bi (Ink Bleed).
"""

import cv2
import numpy as np


def apply_local_blur(
    image: np.ndarray,
    annotations: list = None,
    intensity: float = 0.5,
    num_regions: int = None,
    seed: int = None
) -> np.ndarray:
    """
    Làm mờ cục bộ theo vùng (ví dụ 1 vài ô điểm, 1 chữ ký hoặc 1 dòng chữ bị mất nét/mờ tay cọ).
    Sử dụng mặt nạ biên mềm (feathered mask) để chuyển tiếp mượt mà.
    """
    if intensity <= 0.01:
        return image

    rng = np.random.RandomState(seed)
    h, w = image.shape[:2]
    out = image.copy()

    # Tạo ảnh làm mờ sẵn
    ksize = int(round(9 + 12 * intensity))
    if ksize % 2 == 0:
        ksize += 1
    sigma = 2.0 + 4.0 * intensity
    blurred = cv2.GaussianBlur(image, (ksize, ksize), sigma)

    # Mặt nạ tích luỹ các vùng bị mờ
    blur_mask = np.zeros((h, w), dtype=np.float32)

    if num_regions is None:
        num_regions = rng.randint(1, 4)

    # 1. Nếu có danh sách annotation, ưu tiên chọn ngẫu nhiên các ô điểm/chữ ký
    target_boxes = []
    if annotations:
        candidates = [ann for ann in annotations if ann.get("kind") in ("value", "signature")]
        if candidates:
            chosen = rng.choice(candidates, size=min(num_regions, len(candidates)), replace=False)
            for item in chosen:
                pts = np.array(item["bbox"], dtype=np.int32)
                target_boxes.append(pts)

    # Nếu không có hoặc ít hơn mong muốn, chọn vùng ngẫu nhiên
    while len(target_boxes) < num_regions:
        bw = rng.randint(int(w * 0.08), int(w * 0.25))
        bh = rng.randint(int(h * 0.03), int(h * 0.08))
        bx = rng.randint(int(w * 0.1), int(w * 0.8))
        by = rng.randint(int(h * 0.1), int(h * 0.85))
        pts = np.array([[bx, by], [bx + bw, by], [bx + bw, by + bh], [bx, by + bh]], dtype=np.int32)
        target_boxes.append(pts)

    for pts in target_boxes:
        # Mở rộng nhẹ vùng chọn để bao trùm
        pad = int(10 * intensity)
        expanded_pts = pts.copy()
        cx = np.mean(pts[:, 0])
        cy = np.mean(pts[:, 1])
        for p in expanded_pts:
            p[0] = int(p[0] + np.sign(p[0] - cx) * pad)
            p[1] = int(p[1] + np.sign(p[1] - cy) * pad)

        cv2.fillPoly(blur_mask, [expanded_pts], 1.0)

    # Làm mềm biên của mặt nạ mờ (Feathering)
    blur_mask = cv2.GaussianBlur(blur_mask, (35, 35), 12.0)
    blur_mask = np.clip(blur_mask * (0.85 * intensity), 0, 1.0)[:, :, None]

    out = (out.astype(np.float32) * (1.0 - blur_mask) + blurred.astype(np.float32) * blur_mask)
    return np.clip(out, 0, 255).astype(np.uint8)


def apply_ink_bleed(
    image: np.ndarray,
    intensity: float = 0.5,
    seed: int = None
) -> np.ndarray:
    """
    Mô phỏng hiện tượng mực bút bi / mực dấu bị loang nhẹ vào thớ giấy (Ink Bleed)
    và độ đậm nhạt không đều do lực tì của tay người viết.
    """
    if intensity <= 0.01:
        return image

    rng = np.random.RandomState(seed)
    h, w = image.shape[:2]

    # Chuyển sang ảnh xám để nhận diện nét mực đậm
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Mặt nạ nét mực (chữ, đường kẻ bảng, con dấu)
    # Mực càng đậm thì giá trị càng cao
    ink_mask = np.clip((220.0 - gray.astype(np.float32)) / 180.0, 0, 1.0)

    # 1. Mực loang (Bleed): Làm dày nhẹ nét mực bằng phép giãn nở (Dilation) mềm
    kernel_size = 3 if intensity < 0.6 else 5
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    dilated_mask = cv2.dilate(ink_mask, kernel, iterations=1)

    # Vùng viền nét mực bị thấm nhẹ vào giấy
    edge_bleed = np.clip(dilated_mask - ink_mask, 0, 1.0)

    # Thêm độ rỗ sợi giấy ngẫu nhiên vào viền loang
    bleed_noise = rng.uniform(0.3, 1.0, (h, w)).astype(np.float32)
    edge_bleed = edge_bleed * bleed_noise * (0.35 * intensity)

    # 2. Đậm nhạt không đều (Pressure variation):
    # Dao động độ đậm nhạt của nét viết theo tần số mịn
    pressure_lowres = rng.uniform(0.85, 1.15, (h // 60 + 2, w // 60 + 2)).astype(np.float32)
    pressure_map = cv2.resize(pressure_lowres, (w, h), interpolation=cv2.INTER_LINEAR)
    pressure_mod = 1.0 + (pressure_map - 1.0) * intensity

    # Tổng hợp lên ảnh
    out = image.astype(np.float32)

    # Làm tối vùng viền thấm mực (lấy màu trung bình của nét mực xung quanh)
    blurred_img = cv2.GaussianBlur(image, (5, 5), 1.2).astype(np.float32)
    out = out * (1.0 - edge_bleed[:, :, None]) + blurred_img * edge_bleed[:, :, None]

    # Áp dụng lực tì đậm nhạt lên các pixel mực
    ink_weight = (ink_mask * (0.25 * intensity))[:, :, None]
    out = out * (1.0 - ink_weight) + (out * pressure_mod[:, :, None]) * ink_weight

    return np.clip(out, 0, 255).astype(np.uint8)

