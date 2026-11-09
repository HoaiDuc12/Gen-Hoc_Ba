"""
test_step3.py - Kiểm tra Bước 3:
1. Sinh chữ ký Bézier mực xanh/đen tự nhiên cho từng GV bộ môn, GVCN và Hiệu trưởng.
2. Sinh con dấu tròn đỏ (vòng tròn kép, tên trường hư cấu, rỗ mực, xoay nhẹ, đè 1/3 lên chữ ký).
3. Thêm dấu chức danh chữ nhật ở Mẫu A ("PHÓ HIỆU TRƯỞNG / [TÊN]").
4. Thêm watermark CamScanner ở góc dưới Mẫu B.
5. Xuất ảnh và ground truth labels.json kèm phân loại kind: value, printed, signature, stamp, watermark.
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from generator import TranscriptDataGenerator  # noqa: E402
from renderer import TranscriptRenderer, draw_debug_bboxes  # noqa: E402


def run_step3_test():
    out_dir = Path("output_step3")
    out_dir.mkdir(exist_ok=True)

    print("=" * 65)
    print("BƯỚC 3: KIỂM TRA CHỮ KÝ BÉZIER, CON DẤU ĐỎ VÀ WATERMARK")
    print("=" * 65)

    renderer_a = TranscriptRenderer("A")
    renderer_b = TranscriptRenderer("B")

    # 1. Render Mẫu A có chữ ký + dấu tròn đỏ + dấu chức danh
    gen_a = TranscriptDataGenerator(seed=2023)
    data_a = gen_a.generate("A", options={"class_name": "11A6", "grade": 11, "ban": "Cơ bản"})

    t0 = time.perf_counter()
    img_a, anns_a = renderer_a.render(data_a, options={"enable_signatures": True, "enable_stamp": True})
    t_render_a = (time.perf_counter() - t0) * 1000

    img_a.save(out_dir / "templateA_signed_stamped.jpg", quality=95)
    draw_debug_bboxes(img_a, anns_a).save(out_dir / "templateA_bbox_check.jpg", quality=92)
    with open(out_dir / "templateA_labels.json", "w", encoding="utf-8") as f:
        json.dump({"data": data_a, "annotations": anns_a}, f, ensure_ascii=False, indent=2)

    counts_a = {}
    for a in anns_a:
        k = a.get("kind", "other")
        counts_a[k] = counts_a.get(k, 0) + 1

    print(f"\n✓ MẪU A (CHỮ KÝ + DẤU TRÒN + DẤU CHỨC DANH) THÀNH CÔNG:")
    print(f"  - Học sinh: {data_a['student_name']} | Lớp: {data_a['class_name']}")
    print(f"  - Tổng số bbox: {len(anns_a)} chi tiết:")
    for k, v in sorted(counts_a.items()):
        print(f"    + {k}: {v} nhãn")
    print(f"  - Thời gian render: {t_render_a:.1f} ms")

    # 2. Render Mẫu B có chữ ký + dấu tròn đỏ + watermark CamScanner
    gen_b = TranscriptDataGenerator(seed=2024)
    data_b = gen_b.generate("B", options={"class_name": "11A4", "grade": 11})

    t0 = time.perf_counter()
    img_b, anns_b = renderer_b.render(data_b, options={"enable_signatures": True, "enable_stamp": True, "enable_watermark": True})
    t_render_b = (time.perf_counter() - t0) * 1000

    img_b.save(out_dir / "templateB_signed_stamped.jpg", quality=95)
    draw_debug_bboxes(img_b, anns_b).save(out_dir / "templateB_bbox_check.jpg", quality=92)
    with open(out_dir / "templateB_labels.json", "w", encoding="utf-8") as f:
        json.dump({"data": data_b, "annotations": anns_b}, f, ensure_ascii=False, indent=2)

    counts_b = {}
    for a in anns_b:
        k = a.get("kind", "other")
        counts_b[k] = counts_b.get(k, 0) + 1

    print(f"\n✓ MẪU B (CHỮ KÝ + DẤU TRÒN + CAMSCANNER WATERMARK) THÀNH CÔNG:")
    print(f"  - Học sinh: {data_b['student_name']} | Lớp: {data_b['class_name']}")
    print(f"  - Tổng số bbox: {len(anns_b)} chi tiết:")
    for k, v in sorted(counts_b.items()):
        print(f"    + {k}: {v} nhãn")
    print(f"  - Thời gian render: {t_render_b:.1f} ms")

    print(f"\nĐã lưu toàn bộ ảnh và labels.json vào thư mục '{out_dir}/'!")


if __name__ == "__main__":
    run_step3_test()

