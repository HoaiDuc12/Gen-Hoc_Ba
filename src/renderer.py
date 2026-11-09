"""
renderer.py - Vẽ trang học bạ từ template JSON (Mẫu A & Mẫu B) + dữ liệu học sinh + chữ ký, con dấu.

Kết quả trả về:
    img          : ảnh PIL RGB (khổ A4 1240 x 1754, 150 DPI)
    annotations  : danh sách nhãn + toạ độ polygon 4 điểm [[x1,y1],[x2,y1],[x2,y2],[x1,y2]]:
        {
          "kind":    "value" | "printed" | "signature" | "stamp" | "watermark",
          "field":   tên trường (student_name, hk1, signature_teacher, stamp_school, ...),
          "subject": tên môn (nếu thuộc hàng môn học),
          "row":     chỉ số hàng môn học (-1 nếu ngoài bảng),
          "col":     chỉ số cột (0: môn, 1: hk1, 2: hk2, 3: cn, 4: retest, 5: gv),
          "text":    nội dung văn bản (hoặc mô tả đối tượng),
          "bbox":    polygon 4 điểm [[x1,y1],[x2,y1],[x2,y2],[x1,y2]]
        }
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image, ImageDraw, ImageFont

from signature_stamp import SignatureGenerator, StampGenerator, WatermarkGenerator

TEMPLATE_DIR = Path(__file__).parent / "templates"


def load_template(name_or_path: str | Path) -> dict:
    """Đọc template: truyền 'A' / 'B' hoặc đường dẫn file JSON."""
    p = Path(name_or_path)
    if not p.suffix:
        p = TEMPLATE_DIR / f"template{name_or_path}.json"
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


class FontBank:
    """Nạp font theo danh sách đường dẫn trong template, có cache để chạy nhanh."""

    def __init__(self, font_cfg: dict):
        self.paths: dict[str, str] = {}
        for style, candidates in font_cfg.items():
            found = next((c for c in candidates if os.path.exists(c)), None)
            if found is None:
                raise FileNotFoundError(
                    f"Không tìm thấy font '{style}'. Hãy cài Times New Roman hoặc Liberation Serif. "
                    f"Đã thử: {candidates}"
                )
            self.paths[style] = found
        self._cache: dict[tuple[str, int], ImageFont.FreeTypeFont] = {}

    def get(self, style: str, size: int) -> ImageFont.FreeTypeFont:
        key = (style, size)
        if key not in self._cache:
            self._cache[key] = ImageFont.truetype(self.paths[style], size=size)
        return self._cache[key]


def rect_to_poly(x1: float, y1: float, x2: float, y2: float, pad: int = 2) -> list[list[int]]:
    """Chuyển đổi (x1, y1, x2, y2) -> polygon 4 điểm theo chiều kim đồng hồ."""
    x1, y1, x2, y2 = int(x1 - pad), int(y1 - pad), int(x2 + pad), int(y2 + pad)
    return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]


def draw_text_on_arc(img: Image.Image, center: tuple[int, int], radius: float, text: str,
                     font: ImageFont.FreeTypeFont, fill, start_deg: float, end_deg: float,
                     outward: bool = True) -> None:
    """Viết chữ uốn theo cung tròn (dùng cho huy hiệu)."""
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


class TranscriptRenderer:
    """Renderer học bạ: hỗ trợ Mẫu A, Mẫu B, kèm chữ ký giáo viên và con dấu đỏ."""

    def __init__(self, template: str | Path | dict):
        self.tpl = template if isinstance(template, dict) else load_template(template)
        self.W, self.H = self.tpl["page_size"]
        c = self.tpl["colors"]
        self.c_paper = tuple(c["paper"])
        self.c_line = tuple(c["line"])
        self.c_text = tuple(c["text"])
        self.fonts = FontBank(self.tpl["fonts"])
        self.fs = self.tpl["font_sizes"]
        self.layout = self._compute_layout()
        self._background: Optional[Image.Image] = None

    def _compute_layout(self) -> dict:
        t, m = self.tpl["table"], self.tpl["margins"]
        x_left, x_right = m["left"], self.W - m["right"]
        width = x_right - x_left

        col_xs = [x_left]
        for r in t["col_ratios"][:-1]:
            col_xs.append(col_xs[-1] + round(width * r))
        col_xs.append(x_right)

        info = self.tpl["header_info"]
        y_info1 = m["top"]
        y_info2 = y_info1 + info["line_gap"]
        y_table = y_info2 + info["line_gap"] + t["gap_after_info"]
        y_split = y_table + round(t["header_height"] * t["header_split_ratio"])
        y_rows = y_table + t["header_height"]

        row_intervals: list[tuple[int, int]] = []
        cur_y = y_rows
        subjects = self.tpl["subjects"]
        merged_cfg = t.get("merged_rows", {})

        for i, sub in enumerate(subjects):
            is_last = (i == len(subjects) - 1)
            h = t["last_row_height"] if is_last else t["row_height"]
            span = merged_cfg.get(sub, {}).get("span", 1)

            if span == 1:
                row_intervals.append((cur_y, cur_y + h))
                cur_y += h
            else:
                total_h = h * span
                sub_h = total_h // span
                start_y = cur_y
                for s in range(span):
                    end_y = cur_y + sub_h if s < span - 1 else start_y + total_h
                    row_intervals.append((cur_y, end_y))
                    cur_y = end_y

        y_frame_bottom = (
            self.H - self.tpl["footer"]["frame_bottom_from_page"]
            if self.tpl["footer"].get("has_outer_frame", True) and "frame_bottom_from_page" in self.tpl["footer"]
            else cur_y
        )

        return dict(
            x_left=x_left, x_right=x_right, width=width, col_xs=col_xs,
            y_info1=y_info1, y_info2=y_info2, y_table=y_table,
            y_split=y_split, y_rows=y_rows, row_intervals=row_intervals,
            y_table_bottom=cur_y, y_frame_bottom=y_frame_bottom
        )

    def _text(self, draw: ImageDraw.ImageDraw, anns: list, xy, text: str, font, anchor: str,
              kind: str, field: str, row: int = -1, col: int = -1, subject: Optional[str] = None,
              fill=None) -> None:
        if text is None or str(text).strip() == "":
            return
        text = str(text)
        draw.text(xy, text, font=font, fill=fill or self.c_text, anchor=anchor)
        x1, y1, x2, y2 = draw.textbbox(xy, text, font=font, anchor=anchor)
        ann = {
            "kind": kind, "field": field, "row": row, "col": col,
            "text": text, "bbox": rect_to_poly(x1, y1, x2, y2, pad=3)
        }
        if subject is not None:
            ann["subject"] = subject
        anns.append(ann)

    def _multiline(self, draw, anns, box, lines, style, size, field, gap=4, pad=6) -> None:
        x1, y1, x2, y2 = box
        font = self.fonts.get(style, size)
        while size > 10 and max(font.getlength(ln) for ln in lines) > (x2 - x1) - 2 * pad:
            size -= 1
            font = self.fonts.get(style, size)
        asc, desc = font.getmetrics()
        lh = asc + desc
        total = lh * len(lines) + gap * (len(lines) - 1)
        y = (y1 + y2 - total) / 2
        for ln in lines:
            self._text(draw, anns, ((x1 + x2) / 2, y), ln, font, "ma", "printed", field)
            y += lh + gap

    def _get_background(self) -> Image.Image:
        if self._background is None:
            img = Image.new("RGB", (self.W, self.H), self.c_paper)
            wm = self.tpl.get("watermark", {})
            if wm.get("enabled"):
                self._draw_trong_dong(img)
                if wm.get("emblem_enabled"):
                    L = self.layout
                    cx = L["col_xs"][3]
                    cy = L["row_intervals"][-3][0]
                    self._draw_emblem(img, (cx, cy), wm["emblem_radius"])
            self._background = img
        return self._background.copy()

    def _draw_trong_dong(self, img: Image.Image) -> None:
        draw = ImageDraw.Draw(img)
        col = tuple(self.tpl["colors"]["watermark"])
        cxr, cyr = self.tpl["watermark"]["center_ratio"]
        cx, cy = self.W * cxr, self.H * cyr

        def ring(r, w=2):
            draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline=col, width=w)

        def ticks(r_in, r_out, step_deg, w=1):
            a = 0.0
            while a < 360:
                rad = math.radians(a)
                draw.line([cx + r_in * math.cos(rad), cy + r_in * math.sin(rad),
                           cx + r_out * math.cos(rad), cy + r_out * math.sin(rad)], fill=col, width=w)
                a += step_deg

        def dots(r, step_deg, size=4):
            a = 0.0
            while a < 360:
                rad = math.radians(a)
                px, py = cx + r * math.cos(rad), cy + r * math.sin(rad)
                draw.ellipse([px - size, py - size, px + size, py + size], outline=col, width=1)
                a += step_deg

        def birds(r, count, size):
            for i in range(count):
                a = math.radians(i * 360 / count)
                tx, ty = -math.sin(a), math.cos(a)
                nx, ny = math.cos(a), math.sin(a)
                px, py = cx + r * nx, cy + r * ny
                pts = [
                    (px - tx * size, py - ty * size),
                    (px + nx * size * 0.35, py + ny * size * 0.35),
                    (px + tx * size * 1.6, py + ty * size * 1.6),
                    (px - nx * size * 0.25, py - ny * size * 0.25),
                ]
                draw.polygon(pts, outline=col, width=2)
                draw.line([px, py, px + nx * size * 0.9 - tx * size * 0.4,
                           py + ny * size * 0.9 - ty * size * 0.4], fill=col, width=2)

        pts = []
        for i in range(28):
            r = 120 if i % 2 == 0 else 40
            a = math.radians(i * 360 / 28)
            pts.append((cx + r * math.cos(a), cy + r * math.sin(a)))
        draw.polygon(pts, outline=col, width=2)

        ring(135); dots(150, 6, 5); ring(165); ticks(165, 190, 2.5); ring(190); ring(205)
        dots(225, 4.5, 6); ring(245); ticks(245, 270, 2); ring(270, 3); birds(320, 18, 34)
        ring(370, 3); ticks(370, 395, 1.6); ring(395); dots(415, 3, 7); ring(435); ring(450)
        birds(510, 24, 40); ring(570, 3); ticks(570, 600, 1.2); ring(600); dots(622, 2.2, 8)
        ring(645); ring(665, 3); birds(730, 30, 46); ring(800, 3); ticks(800, 830, 0.9)
        ring(830); dots(855, 1.7, 9); ring(880); ring(900, 3); birds(970, 36, 50)
        ring(1040, 3); ticks(1040, 1070, 0.7); ring(1070)

    def _draw_emblem(self, img: Image.Image, center: tuple[int, int], r: int) -> None:
        draw = ImageDraw.Draw(img)
        col = tuple(self.tpl["colors"]["emblem"])
        cx, cy = center
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline=col, width=5)
        ri = int(r * 0.72)
        draw.ellipse([cx - ri, cy - ri, cx + ri, cy + ri], outline=col, width=3)
        bw, bh = ri * 0.75, ri * 0.32
        by = cy + ri * 0.22
        draw.polygon([(cx, by), (cx - bw, by - bh * 0.4), (cx - bw, by + bh * 0.6), (cx, by + bh)], outline=col, width=3)
        draw.polygon([(cx, by), (cx + bw, by - bh * 0.4), (cx + bw, by + bh * 0.6), (cx, by + bh)], outline=col, width=3)
        draw.line([cx, by - 4, cx, cy - ri * 0.65], fill=col, width=5)
        draw.ellipse([cx - 12, cy - ri * 0.72 - 12, cx + 12, cy - ri * 0.72 + 12], outline=col, width=3)
        wm = self.tpl["watermark"]
        f = self.fonts.get("bold", 15)
        draw_text_on_arc(img, center, r * 0.86, wm["emblem_text"], f, col, 250, 110, outward=False)
        draw.text((cx, cy + r * 0.58), wm["emblem_year"], font=self.fonts.get("bold", 18), fill=col, anchor="mm")

    # -------------------------------------------------------------- Render Dispatch
    def render(self, data: dict, options: Optional[dict] = None) -> Tuple[Image.Image, List[Dict[str, Any]]]:
        """Render bản ghi học bạ ra ảnh kèm chữ ký, con dấu và nhãn ground truth."""
        opts = options or {}
        layout = self.tpl.get("layout", "A")
        if layout == "A":
            return self._render_a(data, opts)
        elif layout == "B":
            return self._render_b(data, opts)
        raise ValueError(f"Không hỗ trợ layout '{layout}'")

    # -------------------------------------------------------------- MẪU A
    def _render_a(self, data: dict, opts: dict) -> Tuple[Image.Image, List[Dict[str, Any]]]:
        img = self._get_background()
        draw = ImageDraw.Draw(img)
        anns: list[dict] = []
        L, t, ftr = self.layout, self.tpl["table"], self.tpl["footer"]
        cx_ = L["col_xs"]
        lw = t["line_width"]
        base_seed = data.get("seed") or 42

        enable_sigs = opts.get("enable_signatures", True)
        enable_stamp = opts.get("enable_stamp", True)

        f_info = self.fonts.get("bold_italic", self.fs["header_info"])
        f_cell = self.fonts.get("regular", self.fs["cell"])

        # 1. Header thông tin học sinh
        info = self.tpl["header_info"]
        for y, items in ((L["y_info1"], info["line1"]), (L["y_info2"], info["line2"])):
            for (label, key), xr in zip(items, info["col_x_ratios"]):
                x = L["x_left"] + L["width"] * xr
                self._text(draw, anns, (x, y), label.strip(), f_info, "ls", "printed", f"label_{key}")
                x_val = x + draw.textlength(label, font=f_info)
                self._text(draw, anns, (x_val, y), data.get(key, ""), f_info, "ls", "value", key)

        # 2. Header bảng
        y0, ys, yr = L["y_table"], L["y_split"], L["y_rows"]
        H = t["header_texts"]
        draw.line([cx_[1], ys, cx_[4], ys], fill=self.c_line, width=lw)
        hs = self.fs["table_header"]
        self._multiline(draw, anns, (cx_[0], y0, cx_[1], yr), H["subject"], "bold", hs, "hdr_subject")
        self._multiline(draw, anns, (cx_[1], y0, cx_[4], ys), H["score_group"], "bold", hs, "hdr_score_group")
        self._multiline(draw, anns, (cx_[1], ys, cx_[2], yr), H["hk1"], "regular", hs, "hdr_hk1", pad=3)
        self._multiline(draw, anns, (cx_[2], ys, cx_[3], yr), H["hk2"], "regular", hs, "hdr_hk2", pad=3)
        self._multiline(draw, anns, (cx_[3], ys, cx_[4], yr), H["cn"], "regular", hs, "hdr_cn", pad=3)
        self._multiline(draw, anns, (cx_[4], y0, cx_[5], yr), H["retest"], "bold", hs, "hdr_retest", gap=0)
        self._multiline(draw, anns, (cx_[5], y0, cx_[6], yr), H["teacher"], "regular", hs, "hdr_teacher", gap=2)

        # 3. Các hàng môn học
        scores = {s["subject"]: s for s in data.get("scores", [])}
        intervals = L["row_intervals"]

        for i, sub in enumerate(self.tpl["subjects"]):
            ya, yb = intervals[i]
            ym = (ya + yb) / 2
            draw.line([L["x_left"], ya, L["x_right"], ya], fill=self.c_line, width=lw)
            rd = scores.get(sub, {})

            if sub == t.get("elective_row"):
                xs = cx_[0] + (cx_[1] - cx_[0]) * t.get("elective_split_ratio", 0.58)
                draw.line([xs, ya, xs, yb], fill=self.c_line, width=lw)
                self._text(draw, anns, (cx_[0] + 8, ym), sub, f_cell, "lm", "printed", "subject_name", i, 0, sub)
                self._text(draw, anns, ((xs + cx_[1]) / 2, ym), rd.get("elective_name", ""), f_cell,
                           "mm", "value", "elective_name", i, 0, sub)
            else:
                self._text(draw, anns, ((cx_[0] + cx_[1]) / 2, ym), sub, f_cell, "mm", "printed",
                           "subject_name", i, 0, sub)

            for c, key in enumerate(("hk1", "hk2", "cn", "retest"), start=1):
                self._text(draw, anns, ((cx_[c] + cx_[c + 1]) / 2, ym), rd.get(key, ""), f_cell,
                           "mm", "value", key, i, c, sub)

            # Tên GV bộ môn căn phải
            teacher_name = rd.get("teacher", "")
            if teacher_name:
                self._text(draw, anns, (cx_[6] - 10, ym), teacher_name, f_cell, "rm", "value",
                           "teacher_name", i, 5, sub)

                # Vẽ chữ ký giáo viên ở bên trái ô Cột 5
                if enable_sigs:
                    t_seed = abs(hash(teacher_name)) % 100000 + i * 37
                    sig_w, sig_h = 135, (yb - ya) - 4
                    sig_img, (sx1, sy1, sx2, sy2) = SignatureGenerator.generate(
                        seed=t_seed, width=sig_w, height=sig_h, ink_type="blue"
                    )
                    sig_x = cx_[5] + 12
                    sig_y = ya + 2
                    img.paste(sig_img, (sig_x, sig_y), sig_img)
                    anns.append({
                        "kind": "signature", "field": "teacher_signature", "subject": sub,
                        "row": i, "col": 5, "text": f"Ký: {teacher_name}",
                        "bbox": rect_to_poly(sig_x + sx1, sig_y + sy1, sig_x + sx2, sig_y + sy2)
                    })

        y_end = intervals[-1][1]
        draw.line([L["x_left"], y_end, L["x_right"], y_end], fill=self.c_line, width=lw)
        for x in cx_[1:-1]:
            y_start = ys if x in (cx_[2], cx_[3]) else y0
            draw.line([x, y_start, x, y_end], fill=self.c_line, width=lw)

        # 4. Khung lớn bao bảng + footer
        draw.rectangle([L["x_left"], y0, L["x_right"], L["y_frame_bottom"]], outline=self.c_line, width=lw)

        # 5. Footer Mẫu A
        self._text(draw, anns, (L["x_left"] + 6, y_end + ftr["note_offset"]), ftr["note"],
                   f_cell, "lm", "printed", "footer_note")
        f_title = self.fonts.get("bold", self.fs["footer_title"])
        f_sub = self.fonts.get("italic", self.fs["footer_sub"])
        f_name = self.fonts.get("bold", self.fs["footer_name"])
        y_title = y_end + ftr["title_offset"]
        y_name = L["y_frame_bottom"] - ftr["name_from_bottom"]

        x_gvcn = L["x_left"] + int(L["width"] * ftr["left_x_ratio"])
        x_prin = L["x_left"] + int(L["width"] * ftr["right_x_ratio"])

        # Tiêu đề GVCN và Hiệu trưởng
        self._text(draw, anns, (x_gvcn, y_title), ftr["gvcn_title"], f_title, "mm", "printed", "gvcn_title")
        self._text(draw, anns, (x_gvcn, y_title + ftr["sub_offset"]), ftr["gvcn_sub"], f_sub, "mm", "printed", "gvcn_sub")
        self._text(draw, anns, (x_gvcn, y_name), data.get("gvcn_name", ""), f_name, "mm", "value", "gvcn_name")

        self._text(draw, anns, (x_prin, y_title), ftr["principal_title"], f_title, "mm", "printed", "principal_title")
        self._text(draw, anns, (x_prin, y_title + ftr["sub_offset"]), ftr["principal_sub"], f_sub, "mm", "printed", "principal_sub")
        self._text(draw, anns, (x_prin, y_name), data.get("principal_name", ""), f_name, "mm", "value", "principal_name")

        # 6. Chữ ký GVCN ở cuối trang
        if enable_sigs and data.get("gvcn_name"):
            g_seed = abs(hash(data["gvcn_name"])) % 100000 + 101
            sig_gvcn, (gx1, gy1, gx2, gy2) = SignatureGenerator.generate(
                seed=g_seed, width=180, height=65, ink_type="blue"
            )
            sig_g_x = x_gvcn - 90
            sig_g_y = y_title + ftr["sub_offset"] + 14
            img.paste(sig_gvcn, (sig_g_x, sig_g_y), sig_gvcn)
            anns.append({
                "kind": "signature", "field": "gvcn_signature", "row": -1, "col": -1,
                "text": f"Ký: {data['gvcn_name']}",
                "bbox": rect_to_poly(sig_g_x + gx1, sig_g_y + gy1, sig_g_x + gx2, sig_g_y + gy2)
            })

        # 7. Chữ ký Hiệu trưởng + Con dấu đỏ đè lên
        if enable_sigs and data.get("principal_name"):
            p_seed = abs(hash(data["principal_name"])) % 100000 + 202
            sig_prin, (px1, py1, px2, py2) = SignatureGenerator.generate(
                seed=p_seed, width=190, height=65, ink_type="blue"
            )
            sig_p_x = x_prin - 85
            sig_p_y = y_title + ftr["sub_offset"] + 14
            img.paste(sig_prin, (sig_p_x, sig_p_y), sig_prin)
            anns.append({
                "kind": "signature", "field": "principal_signature", "row": -1, "col": -1,
                "text": f"Ký: {data['principal_name']}",
                "bbox": rect_to_poly(sig_p_x + px1, sig_p_y + py1, sig_p_x + px2, sig_p_y + py2)
            })

        if enable_stamp:
            stamp_seed = base_seed + 999
            stamp_img, _ = StampGenerator.generate_circular_stamp(seed=stamp_seed, diameter=175)
            # Đóng dấu trùm lên ~1/3 chữ ký về phía bên trái
            st_x = x_prin - 145
            st_y = y_title + 18
            img.paste(stamp_img, (st_x, st_y), stamp_img)
            anns.append({
                "kind": "stamp", "field": "circular_stamp", "row": -1, "col": -1,
                "text": "Con dấu tròn đỏ THPT",
                "bbox": rect_to_poly(st_x + 10, st_y + 10, st_x + stamp_img.width - 10, st_y + stamp_img.height - 10)
            })

            # Dấu chức danh chữ nhật đỏ ở Mẫu A: "PHÓ HIỆU TRƯỞNG / [TÊN]"
            title_stamp = StampGenerator.generate_rectangular_title_stamp(
                "PHÓ HIỆU TRƯỞNG", data.get("principal_name", "TRẦN QUỐC BẢO"), seed=stamp_seed
            )
            ts_x = x_prin - 45
            ts_y = y_name - 24
            img.paste(title_stamp, (ts_x, ts_y), title_stamp)
            anns.append({
                "kind": "stamp", "field": "title_stamp", "row": -1, "col": -1,
                "text": "Dấu chức danh: PHÓ HIỆU TRƯỞNG",
                "bbox": rect_to_poly(ts_x, ts_y, ts_x + title_stamp.width, ts_y + title_stamp.height)
            })

        return img, anns

    # -------------------------------------------------------------- MẪU B
    def _render_b(self, data: dict, opts: dict) -> Tuple[Image.Image, List[Dict[str, Any]]]:
        img = self._get_background()
        draw = ImageDraw.Draw(img)
        anns: list[dict] = []
        L, t, ftr = self.layout, self.tpl["table"], self.tpl["footer"]
        cx_ = L["col_xs"]
        lw = t["line_width"]
        base_seed = data.get("seed") or 42

        enable_sigs = opts.get("enable_signatures", True)
        enable_stamp = opts.get("enable_stamp", True)
        enable_wm = opts.get("enable_watermark", True)

        f_info_b = self.fonts.get("bold_italic", self.fs["header_info"])
        f_info_r = self.fonts.get("italic", self.fs["header_info"])
        f_cell = self.fonts.get("regular", self.fs["cell"])
        f_cell_sub = self.fonts.get("regular", self.fs.get("cell_sub", 20))

        # 1. Header Mẫu B
        info = self.tpl["header_info"]
        y1 = L["y_info1"]
        for (label, key), xr in zip(info["line1"], info["col_x_ratios"]):
            x = L["x_left"] + L["width"] * xr
            self._text(draw, anns, (x, y1), label.strip(), f_info_b, "ls", "printed", f"label_{key}")
            x_val = x + draw.textlength(label, font=f_info_b)
            self._text(draw, anns, (x_val, y1), data.get(key, ""), f_info_r, "ls", "value", key)

        y2 = L["y_info2"]
        lbl_adv = info.get("line2_label", "Các môn học nâng cao: ")
        x_adv = L["x_left"] + L["width"] * info["col_x_ratios"][1]
        self._text(draw, anns, (x_adv, y2), lbl_adv.strip(), f_info_b, "ls", "printed", "label_advanced_subjects")
        w_adv = draw.textlength(lbl_adv, font=f_info_b)
        adv_val = data.get("advanced_subjects", "......................................................")
        self._text(draw, anns, (x_adv + w_adv, y2), adv_val, f_info_r, "ls", "value", "advanced_subjects")

        # 2. Header bảng
        y0, ys, yr = L["y_table"], L["y_split"], L["y_rows"]
        H = t["header_texts"]
        draw.line([cx_[1], ys, cx_[4], ys], fill=self.c_line, width=lw)
        hs = self.fs["table_header"]
        self._multiline(draw, anns, (cx_[0], y0, cx_[1], yr), H["subject"], "bold", hs, "hdr_subject")
        self._multiline(draw, anns, (cx_[1], y0, cx_[4], ys), H["score_group"], "bold", hs, "hdr_score_group")
        self._multiline(draw, anns, (cx_[1], ys, cx_[2], yr), H["hk1"], "regular", hs, "hdr_hk1", pad=3)
        self._multiline(draw, anns, (cx_[2], ys, cx_[3], yr), H["hk2"], "regular", hs, "hdr_hk2", pad=3)
        self._multiline(draw, anns, (cx_[3], ys, cx_[4], yr), H["cn"], "regular", hs, "hdr_cn", pad=3)
        self._multiline(draw, anns, (cx_[4], y0, cx_[5], yr), H["retest"], "bold", hs, "hdr_retest", gap=0)
        self._multiline(draw, anns, (cx_[5], y0, cx_[6], yr), H["teacher"], "regular", hs, "hdr_teacher", gap=2)

        # 3. Các hàng môn học Mẫu B
        scores = {s["subject"]: s for s in data.get("scores", [])}
        intervals = L["row_intervals"]
        merged_cfg = t.get("merged_rows", {})

        idx_interval = 0
        for i, sub in enumerate(self.tpl["subjects"]):
            rd = scores.get(sub, {})
            m_info = merged_cfg.get(sub)

            if m_info is None:
                ya, yb = intervals[idx_interval]
                ym = (ya + yb) / 2
                draw.line([L["x_left"], ya, L["x_right"], ya], fill=self.c_line, width=lw)

                if sub == "Tiếng Anh (10 năm)":
                    self._text(draw, anns, ((cx_[0] + cx_[1]) / 2, ym - 10), "Tiếng Anh", f_cell, "mm",
                               "printed", "subject_name", i, 0, sub)
                    self._text(draw, anns, ((cx_[0] + cx_[1]) / 2, ym + 11), "(10 năm)", f_cell_sub, "mm",
                               "printed", "subject_sub", i, 0, sub)
                elif sub == "Điểm TB các môn học":
                    self._text(draw, anns, ((cx_[0] + cx_[1]) / 2, ym - 10), "Điểm TB", f_cell, "mm",
                               "printed", "subject_name", i, 0, sub)
                    self._text(draw, anns, ((cx_[0] + cx_[1]) / 2, ym + 12), "các môn học", f_cell_sub, "mm",
                               "printed", "subject_sub", i, 0, sub)
                else:
                    self._text(draw, anns, ((cx_[0] + cx_[1]) / 2, ym), sub, f_cell, "mm",
                               "printed", "subject_name", i, 0, sub)

                for c, key in enumerate(("hk1", "hk2", "cn", "retest"), start=1):
                    self._text(draw, anns, ((cx_[c] + cx_[c + 1]) / 2, ym), rd.get(key, ""), f_cell,
                               "mm", "value", key, i, c, sub)

                teacher_name = rd.get("teacher", "")
                if teacher_name:
                    self._text(draw, anns, (cx_[6] - 10, ym), teacher_name, f_cell, "rm", "value",
                               "teacher_name", i, 5, sub)

                    if enable_sigs:
                        t_seed = abs(hash(teacher_name)) % 100000 + i * 43
                        sig_w, sig_h = 135, (yb - ya) - 4
                        sig_img, (sx1, sy1, sx2, sy2) = SignatureGenerator.generate(
                            seed=t_seed, width=sig_w, height=sig_h, ink_type="blue"
                        )
                        sig_x = cx_[5] + 10
                        sig_y = ya + 2
                        img.paste(sig_img, (sig_x, sig_y), sig_img)
                        anns.append({
                            "kind": "signature", "field": "teacher_signature", "subject": sub,
                            "row": i, "col": 5, "text": f"Ký: {teacher_name}",
                            "bbox": rect_to_poly(sig_x + sx1, sig_y + sy1, sig_x + sx2, sig_y + sy2)
                        })

                idx_interval += 1
            else:
                span = m_info.get("span", 2)
                ya_start = intervals[idx_interval][0]
                yb_end = intervals[idx_interval + span - 1][1]
                ym_block = (ya_start + yb_end) / 2
                y_mid_divider = intervals[idx_interval][1]
                g_style = m_info.get("style", "sample_b")

                draw.line([L["x_left"], ya_start, L["x_right"], ya_start], fill=self.c_line, width=lw)

                teacher_name = rd.get("teacher", "")

                if g_style == "merged_all":
                    self._text(draw, anns, ((cx_[0] + cx_[1]) / 2, ym_block), sub, f_cell, "mm",
                               "printed", "subject_name", i, 0, sub)
                    for c, key in enumerate(("hk1", "hk2", "cn", "retest"), start=1):
                        self._text(draw, anns, ((cx_[c] + cx_[c + 1]) / 2, ym_block), rd.get(key, ""), f_cell,
                                   "mm", "value", key, i, c, sub)
                    if teacher_name:
                        self._text(draw, anns, (cx_[6] - 10, ym_block), teacher_name, f_cell, "rm", "value",
                                   "teacher_name", i, 5, sub)
                else:
                    draw.line([cx_[1], y_mid_divider, cx_[5], y_mid_divider], fill=self.c_line, width=lw)
                    ym_row1 = (ya_start + y_mid_divider) / 2
                    ym_row2 = (y_mid_divider + yb_end) / 2

                    for c, key in enumerate(("hk1", "hk2", "cn", "retest"), start=1):
                        self._text(draw, anns, ((cx_[c] + cx_[c + 1]) / 2, ym_row1), rd.get(key, ""), f_cell,
                                   "mm", "value", key, i, c, sub)

                    self._text(draw, anns, ((cx_[0] + cx_[1]) / 2, ym_row2), sub, f_cell, "mm",
                               "printed", "subject_name", i, 0, sub)

                    if teacher_name:
                        self._text(draw, anns, (cx_[6] - 10, ym_row2), teacher_name, f_cell, "rm", "value",
                                   "teacher_name", i, 5, sub)

                # Chữ ký môn GDCD (uốn lượn cao trùm qua 2 hàng như Mẫu B)
                if enable_sigs and teacher_name:
                    t_seed = abs(hash(teacher_name)) % 100000 + i * 43
                    sig_w, sig_h = 145, (yb_end - ya_start) - 6
                    sig_img, (sx1, sy1, sx2, sy2) = SignatureGenerator.generate(
                        seed=t_seed, width=sig_w, height=sig_h, ink_type="blue"
                    )
                    sig_x = cx_[5] + 10
                    sig_y = ya_start + 3
                    img.paste(sig_img, (sig_x, sig_y), sig_img)
                    anns.append({
                        "kind": "signature", "field": "teacher_signature", "subject": sub,
                        "row": i, "col": 5, "text": f"Ký: {teacher_name}",
                        "bbox": rect_to_poly(sig_x + sx1, sig_y + sy1, sig_x + sx2, sig_y + sy2)
                    })

                idx_interval += span

        y_end = L["y_table_bottom"]
        draw.line([L["x_left"], y_end, L["x_right"], y_end], fill=self.c_line, width=lw)
        for x in cx_[1:-1]:
            y_start = ys if x in (cx_[2], cx_[3]) else y0
            draw.line([x, y_start, x, y_end], fill=self.c_line, width=lw)

        draw.rectangle([L["x_left"], y0, L["x_right"], y_end], outline=self.c_line, width=lw)

        # 4. Footer Mẫu B
        date_line = data.get("date_line") or "Huyện Diễn Châu, ngày 25 tháng 5 năm 2023"
        f_date = self.fonts.get("bold_italic", self.fs.get("date_line", 22))
        x_date = L["x_left"] + int(L["width"] * ftr.get("date_line_x_ratio", 0.73))
        y_date = y_end + ftr.get("gap_after_table", 40)
        self._text(draw, anns, (x_date, y_date), date_line, f_date, "mm", "value", "date_line")

        f_title = self.fonts.get("bold", self.fs["footer_title"])
        f_sub = self.fonts.get("regular", self.fs["footer_sub"])
        f_name = self.fonts.get("bold", self.fs["footer_name"])

        y_title = y_date + ftr.get("title_gap_after_date", 30)
        x_gvcn = L["x_left"] + int(L["width"] * ftr["left_x_ratio"])
        x_prin = L["x_left"] + int(L["width"] * ftr["right_x_ratio"])

        # Tiêu đề GVCN và Hiệu trưởng
        self._text(draw, anns, (x_gvcn, y_title), ftr["gvcn_title"], f_title, "mm", "printed", "gvcn_title")
        self._text(draw, anns, (x_gvcn, y_title + ftr["sub_offset"]), ftr["gvcn_sub"], f_sub, "mm", "printed", "gvcn_sub")
        self._text(draw, anns, (x_gvcn, y_title + ftr["name_offset"]), data.get("gvcn_name", ""), f_name, "mm", "value", "gvcn_name")

        self._text(draw, anns, (x_prin, y_title), ftr["principal_title"], f_title, "mm", "printed", "principal_title")
        self._text(draw, anns, (x_prin, y_title + ftr["sub_offset"]), ftr["principal_sub"], f_sub, "mm", "printed", "principal_sub")
        self._text(draw, anns, (x_prin, y_title + ftr["name_offset"]), data.get("principal_name", ""), f_name, "mm", "value", "principal_name")

        # 5. Chữ ký GVCN Mẫu B
        if enable_sigs and data.get("gvcn_name"):
            g_seed = abs(hash(data["gvcn_name"])) % 100000 + 303
            sig_gvcn, (gx1, gy1, gx2, gy2) = SignatureGenerator.generate(
                seed=g_seed, width=180, height=65, ink_type="blue"
            )
            sig_g_x = x_gvcn - 90
            sig_g_y = y_title + ftr["sub_offset"] + 20
            img.paste(sig_gvcn, (sig_g_x, sig_g_y), sig_gvcn)
            anns.append({
                "kind": "signature", "field": "gvcn_signature", "row": -1, "col": -1,
                "text": f"Ký: {data['gvcn_name']}",
                "bbox": rect_to_poly(sig_g_x + gx1, sig_g_y + gy1, sig_g_x + gx2, sig_g_y + gy2)
            })

        # 6. Chữ ký Hiệu trưởng + Con dấu đỏ đè lên ở Mẫu B
        if enable_sigs and data.get("principal_name"):
            p_seed = abs(hash(data["principal_name"])) % 100000 + 404
            sig_prin, (px1, py1, px2, py2) = SignatureGenerator.generate(
                seed=p_seed, width=190, height=65, ink_type="blue"
            )
            sig_p_x = x_prin - 85
            sig_p_y = y_title + ftr["sub_offset"] + 20
            img.paste(sig_prin, (sig_p_x, sig_p_y), sig_prin)
            anns.append({
                "kind": "signature", "field": "principal_signature", "row": -1, "col": -1,
                "text": f"Ký: {data['principal_name']}",
                "bbox": rect_to_poly(sig_p_x + px1, sig_p_y + py1, sig_p_x + px2, sig_p_y + py2)
            })

        if enable_stamp:
            stamp_seed = base_seed + 888
            stamp_img, _ = StampGenerator.generate_circular_stamp(seed=stamp_seed, diameter=170)
            st_x = x_prin - 140
            st_y = y_title + 22
            img.paste(stamp_img, (st_x, st_y), stamp_img)
            anns.append({
                "kind": "stamp", "field": "circular_stamp", "row": -1, "col": -1,
                "text": "Con dấu tròn đỏ THPT",
                "bbox": rect_to_poly(st_x + 10, st_y + 10, st_x + stamp_img.width - 10, st_y + stamp_img.height - 10)
            })

        # 7. Watermark CamScanner ở góc dưới mép trái
        if enable_wm:
            wm_x = L["x_left"]
            wm_y = self.H - 45
            wm_box = WatermarkGenerator.draw_camscanner_watermark(img, (wm_x, wm_y))
            anns.append({
                "kind": "watermark", "field": "camscanner_watermark", "row": -1, "col": -1,
                "text": "Được quét bằng CamScanner",
                "bbox": rect_to_poly(wm_box[0], wm_box[1], wm_box[2], wm_box[3])
            })

        return img, anns


def draw_debug_bboxes(img: Image.Image, anns: list[dict]) -> Image.Image:
    """Vẽ polygon bounding box đè lên ảnh để kiểm tra trực quan:
    - Màu đỏ: Dữ liệu biến thiên điền vào (tên, điểm, lớp, gv, ...)
    - Màu xanh dương: Tiêu đề in sẵn của biểu mẫu phôi.
    - Màu cam: Chữ ký giáo viên / hiệu trưởng.
    - Màu tím / cánh sen: Con dấu đỏ tròn / dấu chức danh.
    - Màu xám: Watermark CamScanner.
    """
    out = img.copy()
    d = ImageDraw.Draw(out)
    for a in anns:
        k = a.get("kind", "value")
        if k == "value":
            color = (220, 30, 30)       # Đỏ
        elif k == "printed":
            color = (30, 90, 220)       # Xanh dương
        elif k == "signature":
            color = (230, 130, 20)      # Cam
        elif k == "stamp":
            color = (190, 20, 190)      # Tím cánh sen
        elif k == "watermark":
            color = (120, 120, 120)     # Xám
        else:
            color = (0, 200, 0)         # Xanh lá

        d.polygon([tuple(p) for p in a["bbox"]], outline=color, width=2)
    return out
