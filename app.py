# -*- coding: utf-8 -*-
"""
Học Bạ Gen - WebUI Streamlit
Ứng dụng sinh ảnh học bạ THPT giả lập (Synthetic Data) phục vụ huấn luyện và đánh giá PaddleOCR.
"""

import io
import os
import sys
import json
import time
import zipfile
from pathlib import Path
from typing import Dict, Any, List

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

# Thêm thư mục gốc và src vào sys.path
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(BASE_DIR / "src"))

from src.generator import TranscriptDataGenerator
from src.renderer import TranscriptRenderer
from src.augment import (
    AugmentConfig,
    AugmentationPipeline,
    get_preset_config,
    EffectOption
)

# ----------------------------------------------------------------------------
# Cấu hình Trang Streamlit
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Học Bạ Gen - Synthetic Data Generator",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS cho giao diện hiện đại, sạch sẽ
st.markdown("""
<style>
    .main-header {
        font-size: 2.1rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.2rem;
    }
    .stButton>button {
        border-radius: 8px;
        font-weight: 600;
    }
    .badge-box {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 0.85rem;
        font-weight: 600;
        margin-right: 6px;
    }
    .badge-value { background-color: #DCFCE7; color: #166534; border: 1px solid #86EFAC; }
    .badge-printed { background-color: #DBEAFE; color: #1E40AF; border: 1px solid #93C5FD; }
    .badge-sig { background-color: #F3E8FF; color: #6B21A8; border: 1px solid #D8B4FE; }
    .badge-stamp { background-color: #FEE2E2; color: #991B1B; border: 1px solid #FCA5A5; }
</style>
""", unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# Cache Resource: Khởi tạo Renderer để tránh load lại font và file template
# ----------------------------------------------------------------------------
@st.cache_resource
def get_renderers():
    renderer_a = TranscriptRenderer("A")
    renderer_b = TranscriptRenderer("B")
    return renderer_a, renderer_b


def draw_bounding_boxes_vis(image_bgr: np.ndarray, annotations: List[Dict[str, Any]]) -> np.ndarray:
    """Vẽ polygon bounding box 4 điểm phân màu lên ảnh để kiểm tra Ground Truth."""
    vis = image_bgr.copy()
    color_map = {
        "value": (0, 220, 0),        # Xanh lá: Giá trị sinh
        "printed": (220, 120, 0),     # Xanh lam/Cam: Chữ in phôi
        "signature": (200, 0, 200),   # Đỏ tím: Chữ ký
        "stamp": (0, 0, 230),         # Đỏ: Con dấu
        "watermark": (120, 120, 120)  # Xám: Watermark
    }
    for ann in annotations:
        pts = np.array(ann["bbox"], dtype=np.int32)
        kind = ann.get("kind", "value")
        color = color_map.get(kind, (0, 255, 0))
        cv2.polylines(vis, [pts], isClosed=True, color=color, thickness=2)
    return vis


def convert_to_paddle_format(annotations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Chuyển đổi danh sách annotation sang định dạng PaddleOCR Text Detection."""
    paddle_items = []
    for ann in annotations:
        paddle_items.append({
            "transcription": ann.get("text", ""),
            "points": ann.get("bbox", []),
            "kind": ann.get("kind", "value"),
            "field": ann.get("field", "")
        })
    return paddle_items


# ----------------------------------------------------------------------------
# Khởi tạo Session State
# ----------------------------------------------------------------------------
if "seed" not in st.session_state:
    st.session_state["seed"] = 2024
if "current_image_bgr" not in st.session_state:
    st.session_state["current_image_bgr"] = None
if "current_annotations" not in st.session_state:
    st.session_state["current_annotations"] = None
if "current_data" not in st.session_state:
    st.session_state["current_data"] = None
if "show_bbox" not in st.session_state:
    st.session_state["show_bbox"] = False


# ----------------------------------------------------------------------------
# SIDEBAR: CẤU HÌNH THÔNG SỐ
# ----------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ 1. Cấu hình Mẫu Học Bạ")

    template_choice = st.radio(
        "Chọn mẫu học bạ:",
        ["Mẫu A (Lớp 11A6 - Nền trống đồng)", "Mẫu B (Lớp 10A4 - Scan CamScanner)"],
        index=0
    )
    template_id = "A" if "Mẫu A" in template_choice else "B"

    col_s1, col_s2 = st.columns([2, 1])
    with col_s1:
        seed_val = st.number_input("Random Seed:", value=int(st.session_state["seed"]), step=1)
        st.session_state["seed"] = seed_val
    with col_s2:
        st.write("")
        st.write("")
        if st.button("🎲 Mới"):
            st.session_state["seed"] = int(np.random.randint(1, 999999))
            st.rerun()

    # Tuỳ chọn chi tiết của văn bản
    with st.expander("📝 Chi tiết thông tin học sinh & phôi", expanded=False):
        grade_input = st.selectbox("Khối lớp:", [10, 11, 12], index=1 if template_id == "A" else 0)
        class_custom = st.text_input("Tên lớp (để trống để random):", value="11A6" if template_id == "A" else "10A4")
        year_custom = st.text_input("Năm học:", value="2022 - 2023")
        ban_custom = st.selectbox("Ban:", ["Cơ bản", "Nâng cao", "Tự nhiên", "Xã hội"], index=0) if template_id == "A" else ""

        retest_prob = st.slider("Tỉ lệ thi lại (%):", 0, 50, 0, step=5) / 100.0
        corr_prob = st.slider("Tỉ lệ sửa điểm gạch chéo (%):", 0, 30, 0, step=5) / 100.0

    st.header("🎨 2. Thành phần Giả lập")
    enable_sigs = st.checkbox("Chữ ký Bézier giáo viên", value=True)
    enable_stamp = st.checkbox("Con dấu tròn đỏ Hiệu trưởng", value=True)
    enable_watermark = st.checkbox("Watermark CamScanner (Mẫu B)", value=(template_id == "B"))

    st.header("📸 3. Hiệu ứng Augmentation")
    preset_choice = st.selectbox(
        "Bộ cấu hình sẵn (Preset):",
        ["Tuỳ chỉnh thủ công", "Chụp điện thoại (Phone Photo)", "Scan sạch (CamScanner)", "Cũ nát (Aged Document)", "Photocopy đen trắng"],
        index=1
    )

    preset_map = {
        "Chụp điện thoại (Phone Photo)": "phone_photo",
        "Scan sạch (CamScanner)": "scan_clean",
        "Cũ nát (Aged Document)": "aged_document",
        "Photocopy đen trắng": "photocopy"
    }

    # Lấy cấu hình gốc theo preset
    if preset_choice in preset_map:
        active_config = get_preset_config(preset_map[preset_choice])
    else:
        active_config = AugmentConfig(mode="fixed")

    # Các mục tuỳ chỉnh chi tiết từng nhóm hiệu ứng
    with st.expander("📐 Góc chụp & Nền bàn (Camera / Desk)", expanded=False):
        c_persp = st.checkbox("Góc chụp nghiêng (Perspective)", value=active_config.perspective.enabled)
        int_persp = st.slider("Độ nghiêng phối cảnh:", 0.05, 1.0, float(active_config.perspective.intensity), 0.05)
        bg_style = st.selectbox("Mặt bàn:", ["wood", "office"], index=0 if active_config.bg_style == "wood" else 1,
                                format_func=lambda x: "Mặt bàn gỗ ấm" if x == "wood" else "Mặt bàn văn phòng xám")
        c_lens = st.checkbox("Mờ ống kính (Lens Blur)", value=active_config.lens_blur.enabled)
        int_lens = st.slider("Độ mờ quang học:", 0.05, 1.0, float(active_config.lens_blur.intensity), 0.05)
        c_noise = st.checkbox("Nhiễu ISO cảm biến", value=active_config.sensor_noise.enabled)
        int_noise = st.slider("Cường độ nhiễu:", 0.05, 1.0, float(active_config.sensor_noise.intensity), 0.05)
        jpeg_q = st.slider("Chất lượng nén JPEG:", 30, 95, int(active_config.jpeg_quality), 5)

    with st.expander("💡 Ánh sáng & Bóng đổ (Lighting)", expanded=False):
        c_shadow = st.checkbox("Bóng đổ người chụp (Cast Shadow)", value=active_config.shadow.enabled)
        int_shadow = st.slider("Độ tối bóng đổ:", 0.05, 1.0, float(active_config.shadow.intensity), 0.05)
        c_glare = st.checkbox("Chói sáng đèn Flash", value=active_config.flash_glare.enabled)
        int_glare = st.slider("Độ chói Flash:", 0.05, 1.0, float(active_config.flash_glare.intensity), 0.05)
        c_vig = st.checkbox("Tối 4 góc (Vignette)", value=active_config.vignette.enabled)
        int_vig = st.slider("Cường độ Vignette:", 0.05, 1.0, float(active_config.vignette.intensity), 0.05)

    with st.expander("📄 Tình trạng giấy vật lý (Paper)", expanded=False):
        c_folds = st.checkbox("Nếp gấp giấy (Folds)", value=active_config.folds.enabled)
        int_folds = st.slider("Độ đậm nếp gấp:", 0.05, 1.0, float(active_config.folds.intensity), 0.05)
        c_wrink = st.checkbox("Nhăn giấy 3D (Wrinkles)", value=active_config.wrinkles.enabled)
        int_wrink = st.slider("Độ nhăn 3D:", 0.05, 1.0, float(active_config.wrinkles.intensity), 0.05)
        c_stains = st.checkbox("Vết ố vàng / Cà phê / Nước", value=active_config.stains.enabled)
        int_stains = st.slider("Mức độ vết ố:", 0.05, 1.0, float(active_config.stains.intensity), 0.05)

    with st.expander("✒️ Mực & Chế độ Quét (Ink / Scan)", expanded=False):
        c_lblur = st.checkbox("Nhòe cục bộ ô điểm / Chữ ký", value=active_config.local_blur.enabled)
        int_lblur = st.slider("Độ nhòe cục bộ:", 0.05, 1.0, float(active_config.local_blur.intensity), 0.05)
        c_bleed = st.checkbox("Mực loang thớ giấy (Ink Bleed)", value=active_config.ink_bleed.enabled)
        int_bleed = st.slider("Độ loang mực:", 0.05, 1.0, float(active_config.ink_bleed.intensity), 0.05)
        c_scan = st.checkbox("Chế độ CamScanner Magic Color", value=active_config.camscanner.enabled)
        int_scan = st.slider("Độ tương phản CamScanner:", 0.2, 1.0, float(active_config.camscanner.intensity), 0.05)
        c_pc = st.checkbox("Bản Photocopy đen trắng (Spine shadow)", value=active_config.photocopy.enabled)
        int_pc = st.slider("Tương phản photocopy:", 0.2, 1.0, float(active_config.photocopy.intensity), 0.05)

    # Cập nhật lại config theo input người dùng
    active_config.perspective = EffectOption(enabled=c_persp, intensity=int_persp, probability=1.0)
    active_config.bg_style = bg_style
    active_config.lens_blur = EffectOption(enabled=c_lens, intensity=int_lens, probability=1.0)
    active_config.sensor_noise = EffectOption(enabled=c_noise, intensity=int_noise, probability=1.0)
    active_config.jpeg_quality = jpeg_q

    active_config.shadow = EffectOption(enabled=c_shadow, intensity=int_shadow, probability=1.0)
    active_config.flash_glare = EffectOption(enabled=c_glare, intensity=int_glare, probability=1.0)
    active_config.vignette = EffectOption(enabled=c_vig, intensity=int_vig, probability=1.0)

    active_config.folds = EffectOption(enabled=c_folds, intensity=int_folds, probability=1.0)
    active_config.wrinkles = EffectOption(enabled=c_wrink, intensity=int_wrink, probability=1.0)
    active_config.stains = EffectOption(enabled=c_stains, intensity=int_stains, probability=1.0)

    active_config.local_blur = EffectOption(enabled=c_lblur, intensity=int_lblur, probability=1.0)
    active_config.ink_bleed = EffectOption(enabled=c_bleed, intensity=int_bleed, probability=1.0)
    active_config.camscanner = EffectOption(enabled=c_scan, intensity=int_scan, probability=1.0)
    active_config.photocopy = EffectOption(enabled=c_pc, intensity=int_pc, probability=1.0)


# ----------------------------------------------------------------------------
# HÀM SINH ẢNH ĐƠN LẺ
# ----------------------------------------------------------------------------
def generate_single_sample(seed: int, t_id: str, cfg: AugmentConfig):
    renderer_a, renderer_b = get_renderers()
    renderer = renderer_a if t_id == "A" else renderer_b

    # 1. Sinh dữ liệu học sinh
    gen = TranscriptDataGenerator(seed=seed)
    opts = {
        "grade": grade_input,
        "class_name": class_custom.strip() if class_custom.strip() else None,
        "school_year": year_custom.strip() if year_custom.strip() else None,
        "ban": ban_custom if t_id == "A" else None,
        "retest_rate": retest_prob,
        "correction_rate": corr_prob
    }
    data = gen.generate(t_id, options=opts)

    # 2. Render ảnh sạch kèm chữ ký, dấu
    render_opts = {
        "enable_signatures": enable_sigs,
        "enable_stamp": enable_stamp,
        "enable_watermark": enable_watermark if t_id == "B" else False
    }
    img_pil, anns = renderer.render(data, options=render_opts)
    img_bgr = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

    # 3. Áp dụng Augmentation & cập nhật Bounding Box
    pipeline = AugmentationPipeline(cfg)
    aug_bgr, updated_anns = pipeline.process(img_bgr, annotations=anns, seed=seed)

    return data, aug_bgr, updated_anns


# ----------------------------------------------------------------------------
# MAIN PANEL: TIÊU ĐỀ & THANH CÔNG CỤ
# ----------------------------------------------------------------------------
st.markdown('<div class="main-header">🎓 Học Bạ Gen — Synthetic Data Generator</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Hệ thống sinh ảnh học bạ THPT giả lập chất lượng cao kèm nhãn Polygon Ground Truth cho PaddleOCR</div>', unsafe_allow_html=True)

# Thanh nút điều khiển trên cùng
col_btn1, col_btn2, col_btn3, col_toggle = st.columns([1.5, 1.5, 1.5, 3])

with col_btn1:
    btn_generate = st.button("📸 Sinh ảnh Preview", type="primary", use_container_width=True)

with col_btn2:
    if st.button("🎲 Random All", use_container_width=True):
        st.session_state["seed"] = int(np.random.randint(1, 999999))
        btn_generate = True

with col_btn3:
    pass

with col_toggle:
    show_boxes = st.checkbox("👁️ Hiển thị Bounding Box Ground Truth", value=st.session_state["show_bbox"])
    st.session_state["show_bbox"] = show_boxes

# Tự động sinh lần đầu nếu chưa có ảnh
if st.session_state["current_image_bgr"] is None or btn_generate:
    with st.spinner("Đang kết xuất ảnh học bạ và áp dụng hiệu ứng..."):
        t0 = time.perf_counter()
        data, aug_img, aug_anns = generate_single_sample(
            seed=int(st.session_state["seed"]),
            t_id=template_id,
            cfg=active_config
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        st.session_state["current_data"] = data
        st.session_state["current_image_bgr"] = aug_img
        st.session_state["current_annotations"] = aug_anns
        st.session_state["last_elapsed_ms"] = elapsed_ms

# ----------------------------------------------------------------------------
# HIỂN THỊ KẾT QUẢ & GROUND TRUTH
# ----------------------------------------------------------------------------
main_col_img, main_col_info = st.columns([1.35, 1.0], gap="medium")

with main_col_img:
    if st.session_state["current_image_bgr"] is not None:
        display_bgr = st.session_state["current_image_bgr"]
        if st.session_state["show_bbox"]:
            display_bgr = draw_bounding_boxes_vis(display_bgr, st.session_state["current_annotations"])

        # Chuyển BGR sang RGB cho Pillow/Streamlit
        display_rgb = cv2.cvtColor(display_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(display_rgb)

        st.image(
            pil_img,
            caption=f"Ảnh học bạ ({display_bgr.shape[1]}x{display_bgr.shape[0]} px) | Thời gian sinh: {st.session_state.get('last_elapsed_ms', 0):.1f} ms",
            use_container_width=True
        )

        # Chú thích màu sắc Bounding Box khi bật chế độ xem nhãn
        if st.session_state["show_bbox"]:
            st.markdown("""
            <div>
                <span class="badge-box badge-value">■ Xanh lá: Giá trị (Điểm, Tên, Lớp)</span>
                <span class="badge-box badge-printed">■ Xanh lam: Chữ in sẵn phôi</span>
                <span class="badge-box badge-sig">■ Đỏ tím: Chữ ký Bézier</span>
                <span class="badge-box badge-stamp">■ Đỏ tươi: Con dấu tròn</span>
            </div>
            """, unsafe_allow_html=True)

        st.write("")
        # Nút tải ảnh và nhãn JSON đơn lẻ
        col_dl1, col_dl2 = st.columns(2)
        with col_dl1:
            buf = io.BytesIO()
            pil_clean = Image.fromarray(cv2.cvtColor(st.session_state["current_image_bgr"], cv2.COLOR_BGR2RGB))
            pil_clean.save(buf, format="JPEG", quality=90)
            st.download_button(
                label="⬇️ Tải ảnh JPEG",
                data=buf.getvalue(),
                file_name=f"hocba_{template_id}_{st.session_state['seed']}.jpg",
                mime="image/jpeg",
                use_container_width=True
            )
        with col_dl2:
            export_dict = {
                "template_id": template_id,
                "seed": int(st.session_state["seed"]),
                "image_size": [display_bgr.shape[1], display_bgr.shape[0]],
                "num_annotations": len(st.session_state["current_annotations"]),
                "annotations": st.session_state["current_annotations"]
            }
            json_str = json.dumps(export_dict, ensure_ascii=False, indent=2)
            st.download_button(
                label="⬇️ Tải nhãn Ground Truth (JSON)",
                data=json_str.encode("utf-8"),
                file_name=f"hocba_{template_id}_{st.session_state['seed']}.json",
                mime="application/json",
                use_container_width=True
            )

with main_col_info:
    data = st.session_state["current_data"]
    anns = st.session_state["current_annotations"]

    tab_table, tab_json, tab_paddle = st.tabs(["📊 Bảng Điểm Học Sinh", "🔍 Ground Truth JSON", "🤖 PaddleOCR Format"])

    with tab_table:
        if data:
            st.markdown(f"**Học sinh:** `{data['student_name']}` | **Lớp:** `{data['class_name']}` | **Năm:** `{data['school_year']}`")
            if data.get("ban"):
                st.markdown(f"**Ban:** `{data['ban']}`")
            st.markdown(f"**GVCN:** `{data['gvcn_name']}` | **Hiệu trưởng:** `{data['principal_name']}`")

            # Chuyển điểm thành bảng DataFrame
            score_rows = []
            for item in data.get("scores", []):
                score_rows.append({
                    "Môn học": item.get("subject", ""),
                    "HK I": item.get("hk1", ""),
                    "HK II": item.get("hk2", ""),
                    "Cả Năm": item.get("cn", ""),
                    "KT Lại": item.get("retest", ""),
                    "Giáo viên": item.get("teacher", ""),
                    "Sửa": "Có" if item.get("corrected") else ""
                })
            df = pd.DataFrame(score_rows)
            st.dataframe(df, use_container_width=True, height=450)

    with tab_json:
        if anns:
            st.caption(f"Tổng số vùng nhãn: **{len(anns)}** bboxes")
            st.json(anns[:25], expanded=False)

    with tab_paddle:
        if anns:
            paddle_data = convert_to_paddle_format(anns)
            st.caption("Định dạng tọa độ Polygon 4 điểm cho PaddleOCR Detection & Recognition:")
            sample_paddle_line = f"hocba_{template_id}_{st.session_state['seed']}.jpg\t{json.dumps(paddle_data[:3], ensure_ascii=False)}"
            st.code(sample_paddle_line, language="json")
            st.json(paddle_data[:10], expanded=False)


# ----------------------------------------------------------------------------
# PHẦN 4: BATCH GENERATION (SINH TẬP DỮ LIỆU HUẤN LUYỆN HÀNG LOẠT)
# ----------------------------------------------------------------------------
st.divider()
st.subheader("📦 Sinh Tập Dữ Liệu Hàng Loạt (Batch Dataset Generator)")
st.caption("Tự động sinh hàng chục đến hàng trăm ảnh học bạ với các biến dị ngẫu nhiên, đóng gói thành file .zip chuẩn cấu trúc dữ liệu huấn luyện PaddleOCR.")

col_b1, col_b2, col_b3 = st.columns([1.5, 1.5, 2])

with col_b1:
    batch_count = st.slider("Số lượng ảnh cần sinh:", min_value=5, max_value=200, value=20, step=5)

with col_b2:
    batch_template = st.selectbox("Tỉ lệ phôi mẫu:", ["50% Mẫu A + 50% Mẫu B", "100% Mẫu A", "100% Mẫu B"])

with col_b3:
    batch_preset = st.selectbox(
        "Hiệu ứng biến dị cho tập huấn luyện:",
        ["Hỗn hợp ngẫu nhiên (Realistic Mix)", "Chụp điện thoại (Phone Photo)", "Scan sạch (CamScanner)", "Cũ nát (Aged Document)", "Photocopy"]
    )

if st.button("🚀 Bắt đầu Sinh Batch Dataset", type="primary"):
    renderer_a, renderer_b = get_renderers()
    progress_bar = st.progress(0.0)
    status_text = st.empty()

    # Chuẩn bị file zip trong bộ nhớ RAM
    zip_buffer = io.BytesIO()

    # Danh sách các dòng nhãn cho Label.txt (chuẩn PaddleOCR)
    paddle_label_lines = []
    meta_records = []

    start_time = time.perf_counter()
    rng_batch = np.random.RandomState(int(st.session_state["seed"]))

    with zipfile.ZipFile(zip_buffer, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for i in range(batch_count):
            cur_seed = int(rng_batch.randint(1, 9999999))

            # Chọn template theo tỉ lệ
            if "50%" in batch_template:
                cur_tid = "A" if i % 2 == 0 else "B"
            elif "Mẫu A" in batch_template:
                cur_tid = "A"
            else:
                cur_tid = "B"

            renderer = renderer_a if cur_tid == "A" else renderer_b

            # Chọn preset biến dị
            if batch_preset == "Hỗn hợp ngẫu nhiên (Realistic Mix)":
                chosen_preset = rng_batch.choice(["phone_photo", "scan_clean", "aged_document", "photocopy"], p=[0.45, 0.25, 0.20, 0.10])
            else:
                chosen_preset = preset_map.get(batch_preset, "phone_photo")

            cur_cfg = get_preset_config(chosen_preset)

            # Sinh dữ liệu học sinh
            gen = TranscriptDataGenerator(seed=cur_seed)
            cur_data = gen.generate(cur_tid)

            # Render
            cur_img_pil, cur_anns = renderer.render(
                cur_data,
                options={"enable_signatures": True, "enable_stamp": True, "enable_watermark": (cur_tid == "B")}
            )
            cur_img_bgr = cv2.cvtColor(np.array(cur_img_pil), cv2.COLOR_RGB2BGR)

            # Augmentation
            pipeline = AugmentationPipeline(cur_cfg)
            cur_aug_bgr, cur_aug_anns = pipeline.process(cur_img_bgr, annotations=cur_anns, seed=cur_seed)

            # Mã hoá ảnh sang JPEG
            img_filename = f"hocba_{i+1:05d}_{cur_tid}_{chosen_preset}.jpg"
            img_rel_path = f"images/{img_filename}"
            success, enc_jpg = cv2.imencode(".jpg", cur_aug_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), cur_cfg.jpeg_quality])

            if success:
                zf.writestr(img_rel_path, enc_jpg.tobytes())

            # Chuẩn bị nhãn PaddleOCR
            paddle_items = convert_to_paddle_format(cur_aug_anns)
            label_entry = f"{img_rel_path}\t{json.dumps(paddle_items, ensure_ascii=False)}\n"
            paddle_label_lines.append(label_entry)

            # Metadata chi tiết
            meta_records.append({
                "filename": img_filename,
                "template": cur_tid,
                "preset": chosen_preset,
                "seed": cur_seed,
                "student_name": cur_data["student_name"],
                "class_name": cur_data["class_name"],
                "num_boxes": len(cur_aug_anns)
            })

            # Cập nhật tiến trình
            progress = (i + 1) / batch_count
            progress_bar.progress(progress)
            status_text.text(f"Đang sinh ảnh {i+1}/{batch_count}: {img_filename} ({chosen_preset})")

        # Ghi file Label.txt (PaddleOCR detection format)
        zf.writestr("Label.txt", "".join(paddle_label_lines).encode("utf-8"))

        # Ghi file metadata.json
        zf.writestr("metadata.json", json.dumps(meta_records, ensure_ascii=False, indent=2).encode("utf-8"))

    total_time = time.perf_counter() - start_time
    avg_speed = (total_time / batch_count) * 1000.0

    progress_bar.progress(1.0)
    status_text.success(f"🎉 Hoàn thành sinh {batch_count} ảnh trong {total_time:.1f} giây (Trung bình: {avg_speed:.1f} ms/ảnh)!")

    # Nút tải file Zip dataset
    st.download_button(
        label=f"⬇️ Tải Về Dataset ({batch_count} ảnh + Label.txt) [.ZIP]",
        data=zip_buffer.getvalue(),
        file_name=f"dataset_hocba_paddleocr_{batch_count}samples.zip",
        mime="application/zip",
        type="primary",
        use_container_width=True
    )
