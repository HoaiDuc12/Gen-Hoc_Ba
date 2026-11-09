"""
signature_stamp.py - Tạo chữ ký giả lập (Bézier curve) và con dấu tròn đỏ thực tế.

Chức năng:
1. SignatureGenerator:
   - Sinh chữ ký giả lập đa dạng phong cách bằng đường cong Bézier bậc 3.
   - Thêm nét vòng lặp (loops), vạch gạch chân (flourish/underline), nét vút đuôi.
   - Giả lập áp lực bút bi (độ dày nét mềm, màu mực xanh/đen tự nhiên).
   - Mỗi giáo viên / tên có kiểu chữ ký cố định và nhất quán dựa trên seed.
2. StampGenerator:
   - Sinh con dấu tròn đỏ chuẩn quy thức trường học Việt Nam (100% hư cấu).
   - Vòng tròn kép, chữ uốn cong theo viền trên ("SỞ GIÁO DỤC VÀ ĐÀO TẠO"),
     ngôi sao 5 cánh ở đáy, tên trường in hoa ở tâm.
   - Giả lập hiệu ứng mực thực tế: bề mặt cao su rỗ hạt, áp lực đậm nhạt không đều,
     xoay lệch góc nhẹ, đè tự nhiên lên 1/3 chữ ký của Hiệu trưởng.
   - Hỗ trợ thêm dấu chức danh chữ nhật đỏ (như ở Mẫu A: "PHÓ HIỆU TRƯỞNG / [TÊN]").
3. WatermarkGenerator:
   - Logo nhỏ ở góc ảnh Mẫu B: icon CS + chữ "Được quét bằng CamScanner".
"""
from __future__ import annotations

import math
import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

FICTIONAL_SCHOOLS = [
    "NGUYỄN HUỆ", "LÊ QUÝ ĐÔN", "TRẦN HƯNG ĐẠO", "NGUYỄN TRÃI", "QUANG TRUNG",
    "HOÀNG VĂN THỤ", "NGÔ QUYỀN", "CHU VĂN AN", "LÝ THƯỜNG KIỆT", "PHAN CHÂU TRINH",
    "HUỲNH THÚC KHÁNG", "NGUYỄN KHUYẾN", "LÊ HỒNG PHONG", "TRẦN PHÚ", "VÕ NGUYÊN GIÁP"
]

FICTIONAL_PROVINCES = [
    "NAM ĐỊNH", "NGHỆ AN", "HẢI DƯƠNG", "BẮC NINH",
    "THÁI BÌNH", "HÀ NAM", "HƯNG YÊN", "THANH HÓA"
]


# ----------------------------------------------------------------------------
# 1. BÉZIER CURVES SIGNATURE GENERATOR
# ----------------------------------------------------------------------------
def _cubic_bezier(p0: Tuple[float, float], p1: Tuple[float, float],
                  p2: Tuple[float, float], p3: Tuple[float, float],
                  num_pts: int = 35) -> List[Tuple[float, float]]:
    """Tính các điểm trên đường cong Bézier bậc 3."""
    pts = []
    for i in range(num_pts + 1):
        t = i / num_pts
        t2 = t * t
        t3 = t2 * t
        mt = 1.0 - t
        mt2 = mt * mt
        mt3 = mt2 * mt
        x = mt3 * p0[0] + 3 * mt2 * t * p1[0] + 3 * mt * t2 * p2[0] + t3 * p3[0]
        y = mt3 * p0[1] + 3 * mt2 * t * p1[1] + 3 * mt * t2 * p2[1] + t3 * p3[1]
        pts.append((x, y))
    return pts


class SignatureGenerator:
    """Sinh ảnh chữ ký giả lập tự nhiên bằng đường cong Bézier."""

    INK_COLORS = [
        (22, 55, 145),   # Xanh bút bi chuẩn
        (18, 65, 160),   # Xanh tươi bút nước
        (15, 42, 115),   # Xanh mực Cửu Long truyền thống
        (32, 58, 140),   # Xanh tím
        (28, 28, 34),    # Mực đen bút bi
        (22, 24, 28),    # Mực đen đậm
    ]

    @classmethod
    def generate(cls, seed: int, width: int = 160, height: int = 42,
                 ink_type: str = "blue") -> Tuple[Image.Image, Tuple[int, int, int, int]]:
        """Tạo 1 chữ ký trong suốt (RGBA) kích thước width x height."""
        rng = random.Random(seed)

        if ink_type == "black":
            base_col = (28, 28, 34)
        elif ink_type == "blue":
            base_col = rng.choice(cls.INK_COLORS[:4])
        else:
            base_col = rng.choice(cls.INK_COLORS)

        sig_img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(sig_img)

        pad_x = width * 0.08
        pad_y = height * 0.12
        usable_w = width - 2 * pad_x
        usable_h = height - 2 * pad_y

        all_strokes: List[List[Tuple[float, float]]] = []

        # 1. Nét chính (Thân chữ ký): 2 - 4 đoạn Bézier liên tục
        num_segments = rng.randint(2, 4)
        cur_pt = (pad_x + rng.uniform(0, usable_w * 0.15), pad_y + usable_h * rng.uniform(0.3, 0.7))
        main_stroke = []

        for seg_idx in range(num_segments):
            prog = (seg_idx + 1) / num_segments
            target_x = pad_x + usable_w * (prog * 0.72 + rng.uniform(-0.06, 0.06))
            target_y = pad_y + usable_h * rng.uniform(0.15, 0.85)

            cp1 = (cur_pt[0] + rng.uniform(usable_w * 0.05, usable_w * 0.22),
                   pad_y + usable_h * rng.uniform(-0.15, 1.15))
            cp2 = (target_x + rng.uniform(-usable_w * 0.18, usable_w * 0.08),
                   pad_y + usable_h * rng.uniform(-0.1, 1.1))

            curve = _cubic_bezier(cur_pt, cp1, cp2, (target_x, target_y), num_pts=25)
            main_stroke.extend(curve)
            cur_pt = (target_x, target_y)

        all_strokes.append(main_stroke)

        # 2. Vòng xoắn / khuyên (Loop)
        if rng.random() < 0.75:
            loop_stroke = []
            loop_cx = cur_pt[0] + rng.uniform(-usable_w * 0.1, usable_w * 0.1)
            loop_cy = cur_pt[1] + rng.uniform(-usable_h * 0.2, usable_h * 0.2)
            rx = rng.uniform(usable_w * 0.08, usable_w * 0.16)
            ry = rng.uniform(usable_h * 0.22, usable_h * 0.42)
            for deg in range(0, 380, 15):
                rad = math.radians(deg)
                loop_stroke.append((loop_cx + rx * math.cos(rad), loop_cy + ry * math.sin(rad)))
            all_strokes.append(loop_stroke)

        # 3. Nét gạch chân hoặc vút đuôi (Underline / Slash)
        if rng.random() < 0.85:
            und_stroke = []
            und_start_x = pad_x + usable_w * rng.uniform(0.08, 0.28)
            und_start_y = pad_y + usable_h * rng.uniform(0.70, 0.95)
            und_end_x = width - pad_x * 0.4
            und_end_y = und_start_y + rng.uniform(-usable_h * 0.2, usable_h * 0.1)

            cp1 = (und_start_x + usable_w * 0.3, und_start_y + rng.uniform(-usable_h * 0.2, usable_h * 0.2))
            cp2 = (und_start_x + usable_w * 0.65, und_end_y + rng.uniform(-usable_h * 0.2, usable_h * 0.2))
            und_curve = _cubic_bezier((und_start_x, und_start_y), cp1, cp2, (und_end_x, und_end_y), num_pts=25)
            und_stroke.extend(und_curve)
            all_strokes.append(und_stroke)

        # 4. Vẽ các nét lên canvas
        base_alpha = rng.randint(230, 255)
        col_rgba = (*base_col, base_alpha)
        all_pts_flat = []

        for stroke in all_strokes:
            all_pts_flat.extend(stroke)
            for i in range(len(stroke) - 1):
                p_a = stroke[i]
                p_b = stroke[i + 1]
                w = rng.uniform(1.8, 2.6)
                draw.line([p_a, p_b], fill=col_rgba, width=int(round(w)))

        xs = [p[0] for p in all_pts_flat]
        ys = [p[1] for p in all_pts_flat]
        bbox_actual = (
            int(max(0, min(xs) - 2)),
            int(max(0, min(ys) - 2)),
            int(min(width, max(xs) + 2)),
            int(min(height, max(ys) + 2))
        )
        return sig_img, bbox_actual


# ----------------------------------------------------------------------------
# 2. RED CIRCULAR STAMP GENERATOR (CON DẤU TRÒN ĐỎ)
# ----------------------------------------------------------------------------
class StampGenerator:
    """Sinh con dấu đỏ tròn chuẩn quy thức trường học Việt Nam."""

    STAMP_RED_COLORS = [
        (220, 36, 36),   # Đỏ son tươi
        (205, 30, 30),   # Đỏ son đậm
        (215, 42, 38),   # Đỏ truyền thống
        (195, 26, 26),   # Đỏ thẫm
    ]

    @classmethod
    def generate_circular_stamp(
        cls,
        seed: int,
        school_name: Optional[str] = None,
        province: Optional[str] = None,
        diameter: int = 175,
        font_path: str = "C:/Windows/Fonts/timesbd.ttf"
    ) -> Tuple[Image.Image, Tuple[int, int, int, int]]:
        """Tạo con dấu tròn đỏ kích thước diameter x diameter với hiệu ứng mực thực tế."""
        rng = random.Random(seed)
        r = diameter // 2
        pad = 22
        canvas_size = diameter + 2 * pad
        cx, cy = r + pad, r + pad

        school_name = school_name or rng.choice(FICTIONAL_SCHOOLS)
        province = province or rng.choice(FICTIONAL_PROVINCES)
        stamp_color = rng.choice(cls.STAMP_RED_COLORS)

        stamp_img = Image.new("RGBA", (canvas_size, canvas_size), (0, 0, 0, 0))
        draw = ImageDraw.Draw(stamp_img)

        # Nạp font chữ
        f_top = ImageFont.truetype(font_path, size=int(diameter * 0.080))
        f_center = ImageFont.truetype(font_path, size=int(diameter * 0.082))
        f_school = ImageFont.truetype(font_path, size=int(diameter * 0.088))

        # 1. Vẽ 2 vòng tròn đồng tâm
        r_outer = r - 4
        r_inner = int(r * 0.68)
        draw.ellipse([cx - r_outer, cy - r_outer, cx + r_outer, cy + r_outer],
                     outline=(*stamp_color, 240), width=3)
        draw.ellipse([cx - r_inner, cy - r_inner, cx + r_inner, cy + r_inner],
                     outline=(*stamp_color, 240), width=2)

        # 2. Dòng chữ uốn vòng cung phía trên: SỞ GIÁO DỤC VÀ ĐÀO TẠO [TỈNH]
        top_text = f"SỞ GIÁO DỤC VÀ ĐÀO TẠO {province}".upper()
        # Chạy từ góc -72 độ sang +72 độ (đỉnh là 0 độ)
        cls._draw_arc_text(stamp_img, (cx, cy), r_outer * 0.84, top_text,
                           f_top, (*stamp_color, 245), start_deg=-72, end_deg=72, outward=True)

        # 3. Ngôi sao 5 cánh ở đáy vòng tròn
        cls._draw_star(draw, (cx, cy + int(r_outer * 0.84)), 8, 3.5, fill=(*stamp_color, 245))

        # 4. Phần trung tâm con dấu: TRƯỜNG / TRUNG HỌC / PHỔ THÔNG / [TÊN TRƯỜNG]
        f_center = ImageFont.truetype(font_path, size=int(diameter * 0.080))
        school_size = int(diameter * 0.086)
        f_school = ImageFont.truetype(font_path, size=school_size)
        while school_size > 10 and f_school.getlength(school_name.upper()) > (2 * r_inner - 16):
            school_size -= 1
            f_school = ImageFont.truetype(font_path, size=school_size)

        lines_center = ["TRƯỜNG", "TRUNG HỌC", "PHỔ THÔNG", school_name.upper()]
        fonts_center = [f_center, f_center, f_center, f_school]

        total_center_h = sum(f.getbbox(ln)[3] - f.getbbox(ln)[1] for f, ln in zip(fonts_center, lines_center)) + 12
        cur_y = cy - total_center_h // 2 + 3

        for f, ln in zip(fonts_center, lines_center):
            bb = f.getbbox(ln)
            lh = bb[3] - bb[1]
            draw.text((cx, cur_y + lh // 2), ln, font=f, fill=(*stamp_color, 245), anchor="mm")
            cur_y += lh + 4

        # 5. Hiệu ứng thực tế: Đốm rỗ bề mặt cao su + áp lực không đều (Gradient)
        stamp_arr = np.array(stamp_img)
        angle_rad = math.radians(rng.uniform(0, 360))
        gx, gy = math.cos(angle_rad), math.sin(angle_rad)
        y_grid, x_grid = np.ogrid[:canvas_size, :canvas_size]
        grad = (x_grid - cx) * gx + (y_grid - cy) * gy
        grad = (grad - grad.min()) / (grad.max() - grad.min() + 1e-5)
        pressure = 0.72 + 0.32 * grad

        noise = np.random.RandomState(seed).normal(0.92, 0.14, (canvas_size, canvas_size))
        noise = np.clip(noise, 0.55, 1.05)

        alpha = stamp_arr[:, :, 3].astype(np.float32) / 255.0 * pressure * noise
        stamp_arr[:, :, 3] = np.clip(alpha * 255.0, 0, 255).astype(np.uint8)

        # 6. Xoay góc ngẫu nhiên (-15° đến +15°)
        rot_deg = rng.uniform(-15.0, 15.0)
        stamp_processed = Image.fromarray(stamp_arr)
        stamp_rotated = stamp_processed.rotate(rot_deg, resample=Image.BICUBIC, expand=True)

        return stamp_rotated, (0, 0, stamp_rotated.width, stamp_rotated.height)

    @classmethod
    def generate_rectangular_title_stamp(
        cls,
        title: str,
        name: str,
        seed: int,
        font_path: str = "C:/Windows/Fonts/timesbd.ttf"
    ) -> Image.Image:
        """Tạo con dấu chức danh chữ nhật đỏ (như ở Mẫu A)."""
        rng = random.Random(seed)
        stamp_color = rng.choice(cls.STAMP_RED_COLORS)

        f_title = ImageFont.truetype(font_path, size=21)
        f_name = ImageFont.truetype(font_path, size=24)

        w1 = f_title.getlength(title)
        w2 = f_name.getlength(name)
        w = int(max(w1, w2) + 24)
        h = 58

        img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)

        draw.text((w // 2, 14), title, font=f_title, fill=(*stamp_color, 240), anchor="mm")
        draw.text((w // 2, 42), name, font=f_name, fill=(*stamp_color, 240), anchor="mm")

        arr = np.array(img)
        noise = np.random.RandomState(seed + 1).normal(0.9, 0.15, (h, w))
        alpha = arr[:, :, 3].astype(np.float32) / 255.0 * np.clip(noise, 0.6, 1.0)
        arr[:, :, 3] = np.clip(alpha * 255, 0, 255).astype(np.uint8)

        rot_deg = rng.uniform(-4.0, 4.0)
        return Image.fromarray(arr).rotate(rot_deg, resample=Image.BICUBIC, expand=True)

    @staticmethod
    def _draw_star(draw: ImageDraw.ImageDraw, center: Tuple[int, int],
                   r_outer: float, r_inner: float, fill: Tuple[int, int, int, int]) -> None:
        """Vẽ ngôi sao 5 cánh bằng polygon."""
        cx, cy = center
        pts = []
        for i in range(10):
            r = r_outer if i % 2 == 0 else r_inner
            ang = math.radians(i * 36 - 90)
            pts.append((cx + r * math.cos(ang), cy + r * math.sin(ang)))
        draw.polygon(pts, fill=fill)

    @staticmethod
    def _draw_arc_text(img: Image.Image, center: Tuple[int, int], radius: float,
                       text: str, font: ImageFont.FreeTypeFont, fill: Tuple[int, int, int, int],
                       start_deg: float, end_deg: float, outward: bool = True) -> None:
        """Viết chữ uốn theo cung tròn."""
        n = len(text)
        if n == 0:
            return
        step = (end_deg - start_deg) / max(n - 1, 1)
        cx, cy = center
        for i, ch in enumerate(text):
            if ch == " ":
                continue
            ang = start_deg + i * step
            rad = math.radians(ang)
            px = cx + radius * math.sin(rad)
            py = cy - radius * math.cos(rad)
            bb = font.getbbox(ch)
            w, h = bb[2] - bb[0] + 6, bb[3] - bb[1] + 6
            tile = Image.new("RGBA", (w, h), (0, 0, 0, 0))
            ImageDraw.Draw(tile).text((w / 2, h / 2), ch, font=font, fill=fill, anchor="mm")
            rot = -ang if outward else 180 - ang
            tile = tile.rotate(rot, resample=Image.BICUBIC, expand=True)
            img.paste(tile, (int(px - tile.width / 2), int(py - tile.height / 2)), tile)


# ----------------------------------------------------------------------------
# 3. CAMSCANNER WATERMARK GENERATOR
# ----------------------------------------------------------------------------
class WatermarkGenerator:
    """Tạo logo CamScanner giả lập ở góc dưới ảnh Mẫu B."""

    @classmethod
    def draw_camscanner_watermark(
        cls,
        img: Image.Image,
        xy: Tuple[int, int],
        font_path: str = "C:/Windows/Fonts/times.ttf"
    ) -> Tuple[int, int, int, int]:
        """Vẽ watermark 'Được quét bằng CamScanner' và trả về bounding box."""
        draw = ImageDraw.Draw(img)
        f_cs = ImageFont.truetype(font_path, size=24)
        f_icon = ImageFont.truetype(font_path, size=18)
        text = "Được quét bằng CamScanner"
        col_text = (150, 150, 150)

        x, y = xy
        # Icon tròn bo góc [CS]
        draw.rounded_rectangle([x, y - 2, x + 34, y + 26], radius=4, outline=col_text, width=2)
        draw.text((x + 17, y + 12), "CS", font=f_icon, fill=col_text, anchor="mm")

        # Chữ CamScanner
        draw.text((x + 44, y + 12), text, font=f_cs, fill=col_text, anchor="lm")
        w_text = draw.textlength(text, font=f_cs)

        bbox = (x, y - 4, int(x + 48 + w_text), y + 28)
        return bbox
