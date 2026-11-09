"""
generator.py - Bộ sinh dữ liệu học bạ giả lập (100% hư cấu) có hỗ trợ seed.

Chức năng:
- Sinh ngẫu nhiên tên học sinh, lớp, năm học, ban / môn nâng cao.
- Sinh bảng điểm thực tế: HK1, HK2 (5.0 - 9.5), CN = (HK1 + 2*HK2)/3 làm tròn 1 chữ số.
- Thể dục: xếp loại "Đ".
- Tính ĐTB các môn học tự động từ các môn có điểm số.
- Tên Giáo viên bộ môn, GVCN, Hiệu trưởng ghép từ danh sách Họ + Đệm + Tên tiếng Việt phổ biến.
- Hỗ trợ tùy chọn: ô trống (Nghề PT), điểm sau KT lại, sửa chữa điểm.
- Tham số `seed` đảm bảo tái lập kết quả hoàn toàn khi cần.
"""
from __future__ import annotations

import random
from typing import Any, Dict, List, Optional

# ----------------------------------------------------------------------------
# Ngân hàng từ vựng tiếng Việt (100% hư cấu)
# ----------------------------------------------------------------------------
LAST_NAMES = [
    "Nguyễn", "Trần", "Lê", "Phạm", "Hoàng", "Huỳnh", "Phan", "Vũ", "Võ",
    "Đặng", "Bùi", "Đỗ", "Hồ", "Ngô", "Dương", "Lý", "Đinh", "Đoàn", "Trịnh",
    "Mai", "Lương", "Thái", "Hà", "Cao", "Nghiêm", "Tô", "Kiều", "Tạ"
]

MIDDLE_NAMES_MALE = [
    "Văn", "Đức", "Hoàng", "Minh", "Quốc", "Hữu", "Đình", "Thành", "Gia",
    "Quang", "Xuân", "Mạnh", "Tiến", "Bảo", "Tuấn", "Hải", "Khánh", "Việt"
]

MIDDLE_NAMES_FEMALE = [
    "Thị", "Ngọc", "Thu", "Bích", "Phương", "Thanh", "Mai", "Kim", "Hồng",
    "Thùy", "Ánh", "Mỹ", "Diệu", "Tuyết", "Hương", "Hà", "Khánh", "Quỳnh"
]

FIRST_NAMES_MALE = [
    "Nam", "Tuấn", "Hùng", "Dũng", "Khánh", "Huy", "Hoàng", "Long", "Bách",
    "Khoa", "Phong", "Thắng", "Quân", "Sơn", "Trung", "Cường", "Đạt", "Phúc",
    "Tùng", "Vinh", "Khôi", "Hiếu", "Hải", "Bình", "Duy", "Thịnh", "Thành"
]

FIRST_NAMES_FEMALE = [
    "Lan", "Hương", "Trang", "Ngọc", "Thảo", "Linh", "Mai", "Hà", "Huyền",
    "Hạnh", "Nga", "Ngân", "Hoa", "Uyên", "Châu", "Vy", "Yến", "Hiền",
    "Loan", "Trâm", "Chi", "Tú", "Ly", "Quyên", "An", "Phượng", "Tuyết"
]

DISTRICTS = [
    "Diễn Châu", "Cẩm Giàng", "Nam Sách", "Yên Phong", "Tiên Du", "Gia Lộc",
    "Thanh Miện", "Quỳnh Phụ", "Đông Hưng", "Hải Hậu", "Ý Yên", "Kim Bảng",
    "Duy Tiên", "Khoái Châu", "Văn Lâm", "Nghi Lộc", "Đô Lương", "Can Lộc"
]


def random_full_name(rng: random.Random, gender: Optional[str] = None) -> str:
    """Sinh họ tên tiếng Việt 3 hoặc 4 từ."""
    if gender is None:
        gender = rng.choice(["male", "female"])

    last = rng.choice(LAST_NAMES)
    if gender == "male":
        mid = rng.choice(MIDDLE_NAMES_MALE)
        first = rng.choice(FIRST_NAMES_MALE)
        # 30% có 4 từ: Họ + Đệm 1 + Đệm 2 + Tên
        if rng.random() < 0.3:
            mid = f"{mid} {rng.choice(['Minh', 'Đức', 'Hoàng', 'Quốc', 'Văn'])}"
    else:
        mid = rng.choice(MIDDLE_NAMES_FEMALE)
        first = rng.choice(FIRST_NAMES_FEMALE)
        if rng.random() < 0.35:
            mid = f"{mid} {rng.choice(['Ngọc', 'Thu', 'Phương', 'Mai', 'Bích'])}"

    return f"{last} {mid} {first}"


# ----------------------------------------------------------------------------
# Bộ sinh dữ liệu chính
# ----------------------------------------------------------------------------
class TranscriptDataGenerator:
    """Sinh bản ghi dữ liệu học bạ chuẩn theo Template A hoặc B."""

    def __init__(self, seed: Optional[int] = None):
        self.seed = seed
        self.rng = random.Random(seed)

    def generate(self, template_id: str = "A", options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Sinh dữ liệu học bạ hoàn chỉnh.

        Tham số options có thể chứa:
            - grade: 10, 11 hoặc 12 (mặc định ngẫu nhiên)
            - class_name: tên lớp cụ thể (mặc định ngẫu nhiên 10A1 - 12A10)
            - school_year: năm học (mặc định ví dụ 2022 - 2023 hoặc 2023 - 2024)
            - retest_prob: xác suất có môn phải kiểm tra lại (0.0 -> 1.0)
            - correction_prob: xác suất có chỗ sửa điểm (0.0 -> 1.0)
        """
        opts = options or {}
        rng = self.rng

        template_id = template_id.upper()
        if template_id not in ("A", "B"):
            template_id = "A"

        # 1. Thông tin học sinh & năm học
        student_gender = rng.choice(["male", "female"])
        student_name = opts.get("student_name") or random_full_name(rng, student_gender)

        grade = opts.get("grade") or rng.choice([10, 11, 12])
        class_num = rng.randint(1, 8)
        class_name = opts.get("class_name") or f"{grade}A{class_num}"

        start_year = opts.get("start_year") or rng.choice([2021, 2022, 2023])
        school_year = opts.get("school_year") or f"{start_year} - {start_year + 1}"

        # 2. Danh sách giáo viên bộ môn
        # Tạo sẵn 1 pool giáo viên để gán vào các môn học
        teacher_pool = [random_full_name(rng) for _ in range(15)]
        gvcn_name = teacher_pool[0]  # GVCN thường kiêm 1 môn học
        principal_name = random_full_name(rng, gender=rng.choice(["male", "female"]))

        # 3. Danh sách môn học theo từng template
        if template_id == "A":
            subjects = [
                "Toán", "Vật lí", "Hóa học", "Sinh học", "Tin học", "Ngữ Văn",
                "Lịch sử", "Địa lí", "Tiếng Anh", "GDCD", "Công Nghệ", "Thể dục",
                "GDQP AN", "Nghề phổ thông", "Tự Chọn"
            ]
            ban = opts.get("ban") or rng.choice(["Cơ bản", "KHTN", "KHXH", "Cơ bản A", "Cơ bản D"])
            specialized = opts.get("specialized_subject") or ""
            date_line = ""
            advanced_subjects = ""
        else:  # Template B
            subjects = [
                "Toán học", "Vật lí", "Hóa học", "Sinh học", "Tin học", "Ngữ văn",
                "Lịch sử", "Địa lí", "Tiếng Anh (10 năm)", "GDCD", "Công nghệ",
                "Thể dục", "GDQP", "Nghề PT"
            ]
            ban = ""
            specialized = ""
            district = rng.choice(DISTRICTS)
            day = rng.randint(20, 28)
            month = 5
            year = start_year + 1
            date_line = f"Huyện {district}, ngày {day} tháng {month} năm {year}"
            advanced_subjects = opts.get("advanced_subjects") or "......................................................"

        # 4. Sinh bảng điểm cho từng môn
        scores_data = []
        hk1_numeric: List[float] = []
        hk2_numeric: List[float] = []
        cn_numeric: List[float] = []

        retest_prob = opts.get("retest_prob", 0.08)
        correction_prob = opts.get("correction_prob", 0.05)

        for idx, sub in enumerate(subjects):
            row_dict: Dict[str, Any] = {"subject": sub}

            # Trường hợp môn Thể dục: Đạt ("Đ")
            if "Thể dục" in sub:
                row_dict.update({
                    "hk1": "Đ", "hk2": "Đ", "cn": "Đ", "retest": "",
                    "teacher": teacher_pool[idx % len(teacher_pool)]
                })
                scores_data.append(row_dict)
                continue

            # Trường hợp "Tự Chọn" (ở Mẫu A): thường để trống hoặc có tên chuyên đề
            if sub == "Tự Chọn":
                elective_name = opts.get("elective_name", "")
                row_dict.update({
                    "elective_name": elective_name,
                    "hk1": "", "hk2": "", "cn": "", "retest": "",
                    "teacher": ""
                })
                scores_data.append(row_dict)
                continue

            # Trường hợp "Nghề phổ thông" / "Nghề PT":
            # Thông thường ở THPT chỉ thi nghề ở HK2 (lớp 11) hoặc để trống nếu lớp 10
            if "Nghề" in sub:
                has_nghe = (grade >= 11) or (rng.random() < 0.6)
                if has_nghe:
                    hk2_val = round(rng.uniform(7.0, 9.8), 1)
                    cn_val = hk2_val
                    row_dict.update({
                        "hk1": "",  # HK1 để trống đúng thực tế học bạ
                        "hk2": f"{hk2_val:.1f}",
                        "cn": f"{cn_val:.1f}",
                        "retest": "",
                        "teacher": teacher_pool[idx % len(teacher_pool)]
                    })
                    hk2_numeric.append(hk2_val)
                    cn_numeric.append(cn_val)
                else:
                    row_dict.update({
                        "hk1": "", "hk2": "", "cn": "", "retest": "",
                        "teacher": ""
                    })
                scores_data.append(row_dict)
                continue

            # Các môn tính điểm thông thường (Toán, Lí, Hóa, Sinh, Tin, Văn, Sử, Địa, Anh, GDCD, Công nghệ, GDQP)
            # Sinh điểm thông thường phân phối học sinh THPT: 5.0 - 9.6
            # Đôi khi có điểm kém (<5.0) để phát sinh trường hợp kiểm tra lại
            has_retest = (rng.random() < retest_prob)

            if has_retest:
                hk1 = round(rng.uniform(4.0, 5.2), 1)
                hk2 = round(rng.uniform(4.0, 4.9), 1)
            else:
                hk1 = round(rng.uniform(5.5, 9.4), 1)
                # Điểm HK2 thường nhỉnh hơn hoặc dao động quanh HK1
                hk2 = round(min(9.8, max(5.0, hk1 + rng.uniform(-0.8, 1.2))), 1)

            cn = round((hk1 + 2.0 * hk2) / 3.0, 1)

            hk1_numeric.append(hk1)
            hk2_numeric.append(hk2)
            cn_numeric.append(cn)

            retest_val = ""
            if has_retest and cn < 5.0:
                retest_val = f"{round(rng.uniform(5.0, 6.5), 1):.1f}"

            # Tùy chọn sửa chữa (gạch ngang điểm cũ, ghi điểm mới)
            is_corrected = (rng.random() < correction_prob)

            # Gán giáo viên bộ môn (Toán hoặc Văn lấy GVCN)
            if idx == 0:
                teacher = gvcn_name
            else:
                teacher = teacher_pool[idx % len(teacher_pool)]

            row_dict.update({
                "hk1": f"{hk1:.1f}",
                "hk2": f"{hk2:.1f}",
                "cn": f"{cn:.1f}",
                "retest": retest_val,
                "teacher": teacher,
                "corrected": is_corrected
            })
            scores_data.append(row_dict)

        # 5. Dòng tổng kết: Điểm TB các môn học (tính tự động)
        avg_sub_name = "ĐTB các môn học" if template_id == "A" else "Điểm TB các môn học"
        if hk1_numeric:
            avg_hk1 = sum(hk1_numeric) / len(hk1_numeric)
            avg_hk2 = sum(hk2_numeric) / len(hk2_numeric)
            avg_cn = sum(cn_numeric) / len(cn_numeric)
            scores_data.append({
                "subject": avg_sub_name,
                "hk1": f"{avg_hk1:.1f}",
                "hk2": f"{avg_hk2:.1f}",
                "cn": f"{avg_cn:.1f}",
                "retest": "",
                "teacher": gvcn_name  # Hàng ĐTB do GVCN ký và ghi tên
            })

        return {
            "template_id": template_id,
            "seed": self.seed,
            "student_name": student_name,
            "class_name": class_name,
            "school_year": school_year,
            "ban": ban,
            "specialized_subject": specialized,
            "advanced_subjects": advanced_subjects,
            "date_line": date_line,
            "gvcn_name": gvcn_name,
            "principal_name": principal_name,
            "scores": scores_data
        }


def generate_batch(count: int, template_id: str = "A", base_seed: int = 42) -> List[Dict[str, Any]]:
    """Sinh danh sách hàng loạt bản ghi học bạ đảm bảo tính tái lập theo seed."""
    records = []
    for i in range(count):
        gen = TranscriptDataGenerator(seed=base_seed + i)
        records.append(gen.generate(template_id=template_id))
    return records

