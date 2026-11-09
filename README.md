# HocBaGen – Phần mềm sinh ảnh học bạ THPT giả lập cho OCR (PaddleOCR)

Toàn bộ dữ liệu (tên HS, GV, trường, chữ ký, con dấu) là **100% hư cấu**.

## Cài đặt

```bash
pip install -r requirements.txt
```

Cần font Times New Roman (mặc định sẵn trên Windows) hoặc Liberation Serif / DejaVu Serif (Linux).
Đường dẫn font khai báo trong mục `fonts` của `src/templates/template*.json`.

## Chạy thử từng bước

### Bước 1 – Render Mẫu A sạch
```bash
python test_step1.py
```
Kết quả trong `output_step1/`.

### Bước 2 – Mẫu B + Generator điểm + Xuất labels.json
```bash
python test_step2.py
```
Kết quả trong `output_step2/`:
- `templateB_clean.jpg` – Ảnh Mẫu B sạch (scan trắng đen, ô GDCD gộp 2 hàng, dòng Huyện... ngày... tháng... năm...).
- `templateB_bbox_check.jpg` – Ảnh overlay polygon 4 điểm bbox của Mẫu B (đỏ = dữ liệu, xanh = chữ in sẵn).
- `templateB_labels.json` – Ground truth Mẫu B.
- `templateA_clean.jpg`, `templateA_bbox_check.jpg`, `templateA_labels.json` – Mẫu A với dữ liệu sinh tự động.

## Tiến độ theo kế hoạch
- [x] **Bước 1**: Template A sạch, đúng form so với ảnh mẫu.
- [x] **Bước 2**: Template B + generator điểm + xuất labels.json.
- [ ] **Bước 3**: Chữ ký Bézier, con dấu đỏ tròn, watermark.
- [ ] **Bước 4**: Augmentation (fold → wrinkle → blur → stain → camera) kèm cập nhật bbox.
- [ ] **Bước 5**: Giao diện Streamlit + preview.
- [ ] **Bước 6**: Export định dạng PaddleOCR (det + rec) + script kiểm tra bbox.
