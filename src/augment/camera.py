# -*- coding: utf-8 -*-
"""
Module Camera: Xử lý góc chụp nghiêng (Perspective Transform), nền bàn, mờ quang học và nhiễu cảm biến.
Cập nhật toạ độ Bounding Box 4 điểm (Polygon) chính xác tuyệt đối sau khi biến đổi hình học.
Tối ưu hiệu năng xử lý cực nhanh bằng multi-scale và phép tính mảng trực tiếp.
"""

import math
import cv2
import numpy as np


def generate_desk_background(width: int, height: int, style: str = "wood", seed: int = None) -> np.ndarray:
    """
    Tạo ảnh nền mặt bàn (gỗ văn phòng hoặc mặt bàn màu trung tính) bằng procedural đa tỉ lệ.
    """
    rng = np.random.RandomState(seed)

    if style == "wood":
        # Sinh ở độ phân giải 1/2 rồi phóng to để tăng tốc gấp 4 lần
        sh, sw = max(2, height // 2), max(2, width // 2)

        y_coords = np.linspace(0, 10, sh)[:, None]
        x_coords = np.linspace(0, 4, sw)[None, :]

        grain = np.sin(x_coords * 8 + np.sin(y_coords * 2) * 2.0)
        grain += 0.5 * np.sin(x_coords * 25 + y_coords * 0.5)
        grain = (grain - grain.min()) / (grain.max() - grain.min() + 1e-6)

        c_dark = np.array([35, 60, 105], dtype=np.float32)   # Nâu sẫm BGR
        c_light = np.array([65, 110, 175], dtype=np.float32) # Nâu sáng BGR

        desk_small = c_dark[None, None, :] + grain[:, :, None] * (c_light - c_dark)[None, None, :]
        desk_small = np.clip(desk_small, 0, 255).astype(np.uint8)

        desk = cv2.resize(desk_small, (width, height), interpolation=cv2.INTER_LINEAR)

        # Thêm vi hạt gỗ nhanh
        noise = rng.randint(-6, 7, (height, width, 1), dtype=np.int16)
        desk_int = desk.astype(np.int16) + noise
        return np.clip(desk_int, 0, 255).astype(np.uint8)
    else:
        # Mặt bàn màu trung tính văn phòng
        base_color = rng.uniform([175, 175, 180], [210, 215, 220])
        desk = np.full((height, width, 3), base_color, dtype=np.uint8)
        noise = rng.randint(-5, 6, (height, width, 1), dtype=np.int16)
        desk_int = desk.astype(np.int16) + noise
        return np.clip(desk_int, 0, 255).astype(np.uint8)


def apply_perspective_transform(
    image: np.ndarray,
    annotations: list = None,
    intensity: float = 0.5,
    bg_style: str = "wood",
    seed: int = None
):
    """
    Biến đổi phối cảnh nghiêng (chụp điện thoại từ góc xiên) và đặt trên nền bàn có bóng đổ.
    Đồng thời cập nhật toàn bộ bounding box polygon 4 điểm sang toạ độ mới.
    """
    if intensity <= 0.001:
        return image.copy(), [dict(ann) for ann in (annotations or [])], np.eye(3, dtype=np.float32)

    rng = np.random.RandomState(seed)
    h, w = image.shape[:2]

    # Kích thước canvas mở rộng
    pad_x = int(w * (0.08 + 0.12 * intensity))
    pad_y = int(h * (0.06 + 0.10 * intensity))
    canvas_w = w + 2 * pad_x
    canvas_h = h + 2 * pad_y

    src_pts = np.array([
        [0.0, 0.0],
        [float(w), 0.0],
        [float(w), float(h)],
        [0.0, float(h)]
    ], dtype=np.float32)

    max_skew_x = pad_x * 1.2 * intensity
    max_skew_y = pad_y * 1.2 * intensity

    tilt_type = rng.choice(["top_far", "slight_tilt", "corner_tilt"])

    if tilt_type == "top_far":
        dx_tl = rng.uniform(0.3, 0.8) * max_skew_x
        dy_tl = rng.uniform(0.1, 0.5) * max_skew_y
        dx_tr = -rng.uniform(0.3, 0.8) * max_skew_x
        dy_tr = rng.uniform(0.1, 0.5) * max_skew_y
        dx_br = rng.uniform(-0.2, 0.4) * max_skew_x
        dy_br = -rng.uniform(0.1, 0.4) * max_skew_y
        dx_bl = rng.uniform(-0.4, 0.2) * max_skew_x
        dy_bl = -rng.uniform(0.1, 0.4) * max_skew_y
    elif tilt_type == "corner_tilt":
        dx_tl = rng.uniform(-0.5, 0.8) * max_skew_x
        dy_tl = rng.uniform(-0.5, 0.8) * max_skew_y
        dx_tr = rng.uniform(-0.8, 0.5) * max_skew_x
        dy_tr = rng.uniform(-0.5, 0.8) * max_skew_y
        dx_br = rng.uniform(-0.8, 0.5) * max_skew_x
        dy_br = rng.uniform(-0.8, 0.5) * max_skew_y
        dx_bl = rng.uniform(-0.5, 0.8) * max_skew_x
        dy_bl = rng.uniform(-0.8, 0.5) * max_skew_y
    else:
        angle = rng.uniform(-3.0, 3.0) * intensity
        dx_tl = -pad_x * 0.2
        dy_tl = -pad_y * 0.2
        dx_tr = pad_x * 0.2
        dy_tr = -pad_y * 0.2
        dx_br = pad_x * 0.2
        dy_br = pad_y * 0.2
        dx_bl = -pad_x * 0.2
        dy_bl = pad_y * 0.2

    dst_pts = np.array([
        [pad_x + dx_tl, pad_y + dy_tl],
        [pad_x + w + dx_tr, pad_y + dy_tr],
        [pad_x + w + dx_br, pad_y + h + dy_br],
        [pad_x + dx_bl, pad_y + h + dy_bl]
    ], dtype=np.float32)

    H = cv2.getPerspectiveTransform(src_pts, dst_pts)

    # 1. Tạo nền bàn nhanh
    desk_bg = generate_desk_background(canvas_w, canvas_h, style=bg_style, seed=seed)

    # 2. Tạo bóng đổ giấy trên bàn (tính trên độ phân giải 1/4)
    sh_w, sh_h = max(2, canvas_w // 4), max(2, canvas_h // 4)
    shadow_mask_small = np.zeros((sh_h, sh_w), dtype=np.uint8)
    shadow_offset = np.array([12 * intensity / 4.0, 16 * intensity / 4.0], dtype=np.float32)
    shadow_pts_small = (dst_pts / 4.0) + shadow_offset
    cv2.fillPoly(shadow_mask_small, [shadow_pts_small.astype(np.int32)], 255)
    shadow_blur_small = cv2.GaussianBlur(shadow_mask_small, (15, 15), 5.0)
    shadow_blur = cv2.resize(shadow_blur_small, (canvas_w, canvas_h), interpolation=cv2.INTER_LINEAR)

    # Làm tối nền bàn tại vùng bóng đổ
    shadow_factor = 255 - (shadow_blur.astype(np.float32) * (0.45 * intensity)).astype(np.uint8)
    shadow_factor_3ch = cv2.merge([shadow_factor, shadow_factor, shadow_factor])
    desk_bg = cv2.multiply(desk_bg, shadow_factor_3ch, scale=1.0 / 255.0)

    # 3. Biến đổi ảnh học bạ vào canvas
    paper_warped = cv2.warpPerspective(
        image, H, (canvas_w, canvas_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0)
    )

    paper_mask = np.full((h, w), 255, dtype=np.uint8)
    warped_mask = cv2.warpPerspective(
        paper_mask, H, (canvas_w, canvas_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )

    # Khử răng cưa mép giấy
    alpha = cv2.GaussianBlur(warped_mask, (3, 3), 0.8)
    alpha_3ch = cv2.merge([alpha, alpha, alpha])
    inv_alpha_3ch = cv2.bitwise_not(alpha_3ch)

    part_paper = cv2.multiply(paper_warped, alpha_3ch, scale=1.0 / 255.0)
    part_desk = cv2.multiply(desk_bg, inv_alpha_3ch, scale=1.0 / 255.0)
    result_image = cv2.add(part_paper, part_desk)

    # 4. Biến đổi đồng bộ toàn bộ bounding box polygon 4 điểm
    updated_annotations = []
    if annotations:
        all_polys = [ann["bbox"] for ann in annotations]
        polys_np = np.array(all_polys, dtype=np.float32)
        N = len(all_polys)
        flat_pts = polys_np.reshape(-1, 1, 2)
        trans_pts = cv2.perspectiveTransform(flat_pts, H).reshape(N, 4, 2)

        for i, ann in enumerate(annotations):
            new_ann = dict(ann)
            new_ann["bbox"] = trans_pts[i].round().astype(int).tolist()
            updated_annotations.append(new_ann)

    return result_image, updated_annotations, H


def apply_lens_blur(
    image: np.ndarray,
    intensity: float = 0.5,
    gradient: bool = True,
    seed: int = None
) -> np.ndarray:
    """
    Hiệu ứng out-of-focus mờ nhẹ do tiêu cự ống kính camera.
    """
    if intensity <= 0.01:
        return image

    rng = np.random.RandomState(seed)
    h, w = image.shape[:2]

    ksize = int(round(5 + 8 * intensity))
    if ksize % 2 == 0:
        ksize += 1
    sigma = 1.0 + 2.5 * intensity

    blurred = cv2.GaussianBlur(image, (ksize, ksize), sigma)

    if not gradient:
        return blurred

    axis = rng.choice(["vertical", "radial", "corner"])
    sh, sw = max(2, h // 4), max(2, w // 4)

    if axis == "vertical":
        weight_s = np.linspace(1.0, 0.0, sh)[:, None]
        weight_s = np.tile(weight_s, (1, sw))
    elif axis == "corner":
        yy, xx = np.mgrid[0:sh, 0:sw]
        dist = np.sqrt((xx / sw) ** 2 + (yy / sh) ** 2)
        weight_s = 1.0 - np.clip(dist / 1.2, 0, 1)
    else:
        yy, xx = np.mgrid[0:sh, 0:sw]
        cx, cy = sw / 2.0, sh / 2.0
        dist = np.sqrt(((xx - cx) / sw) ** 2 + ((yy - cy) / sh) ** 2)
        weight_s = np.clip(dist * 2.0, 0, 1)

    weight_s = (weight_s * intensity * 255.0).astype(np.uint8)
    weight_full = cv2.resize(weight_s, (w, h), interpolation=cv2.INTER_LINEAR)
    weight_3ch = cv2.merge([weight_full, weight_full, weight_full])
    inv_weight_3ch = cv2.bitwise_not(weight_3ch)

    part_blur = cv2.multiply(blurred, weight_3ch, scale=1.0 / 255.0)
    part_orig = cv2.multiply(image, inv_weight_3ch, scale=1.0 / 255.0)
    return cv2.add(part_blur, part_orig)


def apply_sensor_noise_and_jpeg(
    image: np.ndarray,
    noise_intensity: float = 0.3,
    jpeg_quality: int = 75,
    seed: int = None
) -> np.ndarray:
    """
    Mô phỏng nhiễu ISO cảm biến camera điện thoại và suy hao nén JPEG.
    """
    rng = np.random.RandomState(seed)
    h, w, c = image.shape
    out = image

    # 1. Nhiễu hạt cảm biến (tính toán nhanh bằng int16)
    if noise_intensity > 0.01:
        bound_lum = int(round(12.0 * noise_intensity))
        noise = rng.randint(-bound_lum, bound_lum + 1, (h, w, 1), dtype=np.int16)
        if c == 3:
            bound_chroma = int(round(6.0 * noise_intensity))
            color_noise = rng.randint(-bound_chroma, bound_chroma + 1, (h, w, 3), dtype=np.int16)
            total_noise = noise + color_noise
        else:
            total_noise = noise

        img_int = image.astype(np.int16) + total_noise
        out = np.clip(img_int, 0, 255).astype(np.uint8)

    # 2. Nén JPEG
    if jpeg_quality < 98:
        quality = int(np.clip(jpeg_quality, 25, 95))
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        success, enc = cv2.imencode(".jpg", out, encode_param)
        if success:
            out = cv2.imdecode(enc, cv2.IMREAD_COLOR)

    return out
