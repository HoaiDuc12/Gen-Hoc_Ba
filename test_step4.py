# -*- coding: utf-8 -*-
"""
Script kiểm tra Bước 4: Augmentation (Biến dị dữ liệu)
- Kiểm tra các preset: 'phone_photo', 'scan_clean', 'aged_document', 'photocopy'
- Kiểm tra việc biến đổi toạ độ polygon bounding box đồng bộ khi chụp nghiêng/xoay
- Vẽ ảnh bbox check để kiểm tra trực quan độ khít của nhãn sau khi bị biến dị
- Đo thời gian xử lý
"""

import os
import sys
import json
import time
from pathlib import Path
import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent / "src"))

from generator import TranscriptDataGenerator
from renderer import TranscriptRenderer
from augment import AugmentationPipeline, get_preset_config


def draw_bounding_boxes(image: np.ndarray, annotations: list) -> np.ndarray:
    """
    Vẽ các polygon bounding box lên ảnh để kiểm tra độ khít của ground truth.
    - Màu xanh lá: Text giá trị (value)
    - Màu cam: Text in sẵn (printed)
    - Màu đỏ tím: Chữ ký (signature)
    - Màu đỏ tươi: Con dấu (stamp)
    - Màu xám: Watermark
    """
    vis = image.copy()
    color_map = {
        "value": (0, 220, 0),        # Xanh lá
        "printed": (220, 120, 0),     # Cam
        "signature": (200, 0, 200),   # Đỏ tím
        "stamp": (0, 0, 230),         # Đỏ
        "watermark": (120, 120, 120)  # Xám
    }

    for ann in annotations:
        pts = np.array(ann["bbox"], dtype=np.int32)
        kind = ann.get("kind", "value")
        color = color_map.get(kind, (0, 255, 0))
        cv2.polylines(vis, [pts], isClosed=True, color=color, thickness=2)

    return vis


def main():
    out_dir = Path("output_step4")
    out_dir.mkdir(exist_ok=True)
    print("=" * 65)
    print("BƯỚC 4: KIỂM TRA AUGMENTATION & ĐỒNG BỘ BOUNDING BOX")
    print("=" * 65)

    # 1. Khởi tạo generator & renderer
    gen = TranscriptDataGenerator(seed=2024)
    renderer_a = TranscriptRenderer("A")
    renderer_b = TranscriptRenderer("B")

    # Sinh 2 học sinh mẫu đầy đủ chữ ký, con dấu
    data_a = gen.generate("A", options={"class_name": "11A6", "grade": 11, "ban": "Cơ bản"})
    data_b = gen.generate("B", options={"class_name": "10A4", "grade": 10})

    img_a_clean, anns_a = renderer_a.render(data_a, options={"enable_signatures": True, "enable_stamp": True})
    img_b_clean, anns_b = renderer_b.render(data_b, options={"enable_signatures": True, "enable_stamp": True})

    # Chuyển PIL Image sang BGR numpy array
    img_a_bgr = cv2.cvtColor(np.array(img_a_clean), cv2.COLOR_RGB2BGR)
    img_b_bgr = cv2.cvtColor(np.array(img_b_clean), cv2.COLOR_RGB2BGR)

    # 2. Danh sách các bài test kiểm tra các trường hợp thực tế
    test_cases = [
        {
            "name": "case1_templateA_phone_photo",
            "desc": "Mẫu A: Chụp điện thoại nghiêng trên bàn gỗ, đổ bóng, flash nhẹ (Giống ảnh mẫu A thực tế)",
            "src_img": img_a_bgr,
            "src_ann": anns_a,
            "preset": "phone_photo",
            "seed": 42
        },
        {
            "name": "case2_templateB_scan_clean",
            "desc": "Mẫu B: Scan CamScanner trắng bóc, tương phản cao, phẳng (Giống ảnh mẫu B thực tế)",
            "src_img": img_b_bgr,
            "src_ann": anns_b,
            "preset": "scan_clean",
            "seed": 43
        },
        {
            "name": "case3_templateA_aged_document",
            "desc": "Mẫu A: Học bạ cũ nát (Nhăn 3D, 2 nếp gấp, vết ố vàng cà phê, mờ vài ô điểm)",
            "src_img": img_a_bgr,
            "src_ann": anns_a,
            "preset": "aged_document",
            "seed": 44
        },
        {
            "name": "case4_templateB_photocopy",
            "desc": "Mẫu B: Bản photocopy đen trắng (Đơn sắc, gáy đen, hạt bụi mực toner)",
            "src_img": img_b_bgr,
            "src_ann": anns_b,
            "preset": "photocopy",
            "seed": 45
        }
    ]

    for tc in test_cases:
        print(f"\n--- Đang thực hiện: {tc['name']} ---")
        print(f"Mô tả: {tc['desc']}")

        config = get_preset_config(tc["preset"])
        pipeline = AugmentationPipeline(config)

        t0 = time.perf_counter()
        aug_img, aug_ann = pipeline.process(
            tc["src_img"],
            annotations=tc["src_ann"],
            seed=tc["seed"]
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        print(f"-> Thời gian xử lý: {elapsed_ms:.1f} ms | Kích thước ảnh: {aug_img.shape[1]}x{aug_img.shape[0]}")

        # Lưu ảnh kết quả
        out_img_path = out_dir / f"{tc['name']}.jpg"
        cv2.imwrite(str(out_img_path), aug_img)

        # Lưu file nhãn json tương ứng
        out_json_path = out_dir / f"{tc['name']}_labels.json"
        labels_export = {
            "name": tc["name"],
            "preset": tc["preset"],
            "seed": tc["seed"],
            "image_size": [aug_img.shape[1], aug_img.shape[0]],
            "num_annotations": len(aug_ann),
            "annotations": aug_ann
        }
        with open(out_json_path, "w", encoding="utf-8") as f:
            json.dump(labels_export, f, ensure_ascii=False, indent=2)

        # Vẽ ảnh kiểm tra độ khít của bounding box sau khi biến dị
        vis_bbox = draw_bounding_boxes(aug_img, aug_ann)
        out_vis_path = out_dir / f"{tc['name']}_bbox_check.jpg"
        cv2.imwrite(str(out_vis_path), vis_bbox)

        print(f"-> Đã lưu ảnh: {out_img_path}")
        print(f"-> Đã lưu ảnh bbox: {out_vis_path}")
        print(f"-> Đã lưu nhãn: {out_json_path}")

    # 3. Benchmark tốc độ sinh hàng loạt
    print("\n--- BENCHMARK HIỆU NĂNG AUGMENTATION ---")
    benchmark_config = get_preset_config("phone_photo") # Preset phức tạp nhất
    bench_pipeline = AugmentationPipeline(benchmark_config)
    num_samples = 20

    t_start = time.perf_counter()
    for i in range(num_samples):
        bench_pipeline.process(img_a_bgr, anns_a, seed=1000 + i)
    t_total = time.perf_counter() - t_start
    avg_ms = (t_total / num_samples) * 1000.0

    print(f"Xử lý {num_samples} ảnh mất: {t_total:.2f}s (Trung bình: {avg_ms:.1f} ms/ảnh)")
    print(f"Ước tính sinh 1.000 ảnh biến dị hoàn chỉnh mất: {(avg_ms * 1000 / 1000):.1f} giây (~{avg_ms * 1000 / 60000:.1f} phút)")
    print("\n=== HOÀN TẤT BƯỚC 4! ===")


if __name__ == "__main__":
    main()
