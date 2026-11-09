"""
test_step2.py - Kiểm tra Bước 2:
1. Sinh dữ liệu học sinh hư cấu bằng generator.py (họ tên, lớp, điểm, GV, ĐTB tự động, có seed).
2. Render Mẫu B sạch (bản scan giấy trắng, ô GDCD gộp 2 hàng, footer ngoài khung).
3. Render Mẫu A với dữ liệu sinh tự động.
4. Xuất file labels.json chuẩn kèm toạ độ polygon 4 điểm bbox cho từng ô.
5. Kiểm tra tính tái lập tuyệt đối của seed (deterministic).
"""
import json
import sys
import time
from pathlib import Path

# Đảm bảo import được từ src/
sys.path.insert(0, str(Path(__file__).parent / "src"))

from generator import TranscriptDataGenerator, generate_batch  # noqa: E402
from renderer import TranscriptRenderer, draw_debug_bboxes  # noqa: E402


def run_step2_test():
    out_dir = Path("output_step2")
    out_dir.mkdir(exist_ok=True)

    print("=" * 60)
    print("BƯỚC 2: KIỂM TRA GENERATOR VÀ RENDER MẪU B + MẪU A")
    print("=" * 60)

    # 1. Kiểm tra tính tái lập của seed
    gen1 = TranscriptDataGenerator(seed=12345)
    rec1 = gen1.generate("A")

    gen2 = TranscriptDataGenerator(seed=12345)
    rec2 = gen2.generate("A")

    assert json.dumps(rec1, sort_keys=True) == json.dumps(rec2, sort_keys=True), "Lỗi: Seed không tái lập được kết quả!"
    print("✓ Đã xác minh: Tham số seed tái lập kết quả chính xác 100%.")

    # 2. Khởi tạo 2 renderer
    renderer_a = TranscriptRenderer("A")
    renderer_b = TranscriptRenderer("B")

    # 3. Sinh và render Mẫu B (lớp 10A4, scan CamScanner)
    gen_b = TranscriptDataGenerator(seed=2024)
    data_b = gen_b.generate("B", options={"class_name": "11A4", "grade": 11})

    t0 = time.perf_counter()
    img_b, anns_b = renderer_b.render(data_b)
    t_render_b = (time.perf_counter() - t0) * 1000

    img_b.save(out_dir / "templateB_clean.jpg", quality=95)
    draw_debug_bboxes(img_b, anns_b).save(out_dir / "templateB_bbox_check.jpg", quality=92)

    with open(out_dir / "templateB_labels.json", "w", encoding="utf-8") as f:
        json.dump({"data": data_b, "annotations": anns_b}, f, ensure_ascii=False, indent=2)

    n_val_b = sum(a["kind"] == "value" for a in anns_b)
    n_print_b = len(anns_b) - n_val_b
    print(f"\n✓ MẪU B RENDER THÀNH CÔNG:")
    print(f"  - Học sinh: {data_b['student_name']} | Lớp: {data_b['class_name']} | Năm học: {data_b['school_year']}")
    print(f"  - Địa điểm & ngày: {data_b['date_line']}")
    print(f"  - Điểm TB các môn: {[s['cn'] for s in data_b['scores'] if 'Điểm TB' in s['subject']][0]}")
    print(f"  - Tổng số bbox: {len(anns_b)} (Giá trị điền: {n_val_b}, Tiêu đề in sẵn: {n_print_b})")
    print(f"  - Thời gian render: {t_render_b:.1f} ms")

    # 4. Sinh và render Mẫu A (lớp 11A6, nền trống đồng)
    gen_a = TranscriptDataGenerator(seed=2023)
    data_a = gen_a.generate("A", options={"class_name": "11A6", "grade": 11, "ban": "Cơ bản"})

    t0 = time.perf_counter()
    img_a, anns_a = renderer_a.render(data_a)
    t_render_a = (time.perf_counter() - t0) * 1000

    img_a.save(out_dir / "templateA_clean.jpg", quality=95)
    draw_debug_bboxes(img_a, anns_a).save(out_dir / "templateA_bbox_check.jpg", quality=92)

    with open(out_dir / "templateA_labels.json", "w", encoding="utf-8") as f:
        json.dump({"data": data_a, "annotations": anns_a}, f, ensure_ascii=False, indent=2)

    n_val_a = sum(a["kind"] == "value" for a in anns_a)
    n_print_a = len(anns_a) - n_val_a
    print(f"\n✓ MẪU A RENDER THÀNH CÔNG:")
    print(f"  - Học sinh: {data_a['student_name']} | Lớp: {data_a['class_name']} | Ban: {data_a['ban']}")
    print(f"  - ĐTB các môn học: {[s['cn'] for s in data_a['scores'] if 'ĐTB' in s['subject']][0]}")
    print(f"  - Tổng số bbox: {len(anns_a)} (Giá trị điền: {n_val_a}, Tiêu đề in sẵn: {n_print_a})")
    print(f"  - Thời gian render: {t_render_a:.1f} ms")

    # 5. Benchmark tốc độ sinh hàng loạt
    print(f"\n✓ BENCHMARK TỐC ĐỘ:")
    t_bench_start = time.perf_counter()
    batch_records = generate_batch(count=100, template_id="B", base_seed=1000)
    t_gen_batch = (time.perf_counter() - t_bench_start) * 1000
    print(f"  - Sinh dữ liệu 100 học sinh: {t_gen_batch:.1f} ms ({t_gen_batch / 100:.2f} ms/record)")

    t_render_start = time.perf_counter()
    for rec in batch_records[:20]:
        renderer_b.render(rec)
    t_render_batch = (time.perf_counter() - t_render_start) * 1000
    print(f"  - Render 20 trang học bạ: {t_render_batch:.1f} ms ({t_render_batch / 20:.1f} ms/ảnh)")
    print(f"  -> Ước tính 1.000 ảnh mất ~{((t_render_batch / 20) * 1000) / 1000:.1f} giây!")

    print(f"\nĐã lưu toàn bộ ảnh và labels.json vào thư mục '{out_dir}/'!")


if __name__ == "__main__":
    run_step2_test()

