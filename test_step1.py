"""
test_step1.py - Chạy thử Bước 1: render Mẫu A sạch + ảnh kiểm tra bbox.

    python test_step1.py

Dữ liệu ở đây là dữ liệu mẫu cố định, 100% hư cấu. Bộ sinh ngẫu nhiên đầy đủ
(generator.py, có seed) sẽ được làm ở Bước 2.
"""
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))
from renderer import TranscriptRenderer, draw_debug_bboxes  # noqa: E402

# Tên GV hư cấu (không lấy từ ảnh mẫu)
TEACHERS = {
    "Toán": "Lê Hoàng Nam", "Vật lí": "Trịnh Thị Bích Ngọc", "Hóa học": "Nguyễn Minh Tuấn",
    "Sinh học": "Phạm Thu Hương", "Tin học": "Vũ Quốc Khánh", "Ngữ Văn": "Đặng Thị Mai Lan",
    "Lịch sử": "Bùi Văn Hùng", "Địa lí": "Hoàng Thị Thanh", "Tiếng Anh": "Ngô Phương Thảo",
    "GDCD": "Kiều Văn Sơn", "Công Nghệ": "Lý Thị Kim Ngân", "Thể dục": "Đỗ Mạnh Cường",
    "GDQP AN": "Phan Văn Đức", "Nghề phổ thông": "Trịnh Thị Bích Ngọc",
}


def sample_data(seed: int = 42) -> dict:
    rng = random.Random(seed)
    scores, hk1s, hk2s, cns = [], [], [], []
    for sub, gv in TEACHERS.items():
        if sub == "Thể dục":
            scores.append({"subject": sub, "hk1": "Đ", "hk2": "Đ", "cn": "Đ", "teacher": gv})
            continue
        hk1 = round(rng.uniform(5.0, 9.5), 1)
        hk2 = round(rng.uniform(5.0, 9.5), 1)
        cn = round((hk1 + 2 * hk2) / 3, 1)
        hk1s.append(hk1); hk2s.append(hk2); cns.append(cn)
        scores.append({"subject": sub, "hk1": f"{hk1:.1f}", "hk2": f"{hk2:.1f}", "cn": f"{cn:.1f}", "teacher": gv})
    scores.append({"subject": "Tự Chọn"})
    avg = lambda xs: f"{sum(xs) / len(xs):.1f}"
    scores.append({"subject": "ĐTB các môn học", "hk1": avg(hk1s), "hk2": avg(hk2s), "cn": avg(cns),
                   "teacher": "Lê Hoàng Nam"})
    return {
        "student_name": "Nguyễn Hoàng Minh Khôi", "class_name": "11A6", "school_year": "2023 - 2024",
        "ban": "Cơ bản", "specialized_subject": "",
        "scores": scores, "gvcn_name": "Lê Hoàng Nam", "principal_name": "Trần Quốc Bảo",
    }


if __name__ == "__main__":
    out = Path("output_step1")
    out.mkdir(exist_ok=True)

    renderer = TranscriptRenderer("A")
    data = sample_data(42)

    t0 = time.perf_counter()
    img, anns = renderer.render(data)          # lần đầu: có vẽ nền
    t1 = time.perf_counter()
    for _ in range(20):                         # các lần sau: nền đã cache
        renderer.render(data)
    t2 = time.perf_counter()

    img.save(out / "templateA_clean.jpg", quality=95)
    draw_debug_bboxes(img, anns).save(out / "templateA_bbox_check.jpg", quality=90)
    with open(out / "templateA_labels.json", "w", encoding="utf-8") as f:
        json.dump({"data": data, "annotations": anns}, f, ensure_ascii=False, indent=2)

    n_val = sum(a["kind"] == "value" for a in anns)
    print(f"Đã lưu vào {out}/  |  bbox: {len(anns)} (giá trị: {n_val}, chữ in sẵn: {len(anns) - n_val})")
    print(f"Thời gian render: lần đầu {1000 * (t1 - t0):.0f} ms, trung bình sau đó {1000 * (t2 - t1) / 20:.0f} ms/ảnh")

