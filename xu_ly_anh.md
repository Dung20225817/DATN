# Quá trình xử lý ảnh OMR — Chi tiết từng bước

> Tài liệu này mô tả toàn bộ pipeline xử lý ảnh phiếu trắc nghiệm, từ ảnh chụp điện thoại thô đến kết quả chấm điểm cuối cùng. Mỗi bước đều trích dẫn tên hàm và module tương ứng trong code.

---

## Tổng quan luồng xử lý

```
Ảnh thô (JPEG/PNG, bất kỳ kích thước)
    │
    ▼ Bước 1
Tải ảnh (cv2.imread → BGR)
    │
    ▼ Bước 2
Phát hiện 4 góc phiếu (marker đen vuông)
    │  [nếu thất bại]
    ▼ Bước 3
Phát hiện trang bằng contour (phương án dự phòng)
    │
    ▼ Bước 4
Warp phối cảnh → chuẩn hóa về 1000×1400 px
    │
    ▼ Bước 5
Chuyển sang ảnh xám → nhị phân hóa (đen/trắng)
    │
    ▼ Bước 6
Trích xuất và phân loại marker trong ảnh chuẩn
    │
    ▼ Bước 7
Suy luận hình học lưới MCQ (khoảng cách dòng, số block)
    │
    ▼ Bước 8
Xác định điểm neo → xây dựng vùng quan tâm (ROI)
    │
    ▼ Bước 9
Tinh chỉnh ROI bằng template matching
    │
    ├─▶ Bước 10: Giải mã MSSV (mã số sinh viên)
    ├─▶ Bước 11: Giải mã mã đề
    └─▶ Bước 12: Giải mã câu MCQ
              │
              ▼ Bước 13 (nếu cần)
        MCQ Map Search Rescue (điều chỉnh lưới)
              │
              ▼ Bước 14 (nếu cần)
        Phát hiện drift → mở rộng ROI lên trên
              │
              ▼ Bước 15
        Chấm điểm & trả kết quả JSON
```

---

## Bước 1 — Tải ảnh

**Module:** `omr_service.py` → `process_omr_exam()`

Ảnh được đọc bằng `cv2.imread()` của thư viện OpenCV. OpenCV đọc ảnh dưới dạng **BGR** (Blue-Green-Red) — khác với thứ tự màu thông thường RGB. Mọi phép xử lý tiếp theo đều dùng định dạng này.

---

## Bước 2 — Chuẩn hóa ánh sáng và phát hiện marker góc

**Module:** `omr_marker_utils.py` → `_extract_black_square_markers_from_gray()`, `_detect_page_corners_from_black_square_markers()`

Mục tiêu của bước này: tìm 4 ô vuông đen in ở 4 góc phiếu (gọi là **corner marker** hoặc **fiducial marker** — điểm định vị). Đây là bước quan trọng nhất để warp ảnh về đúng góc nhìn.

### 2a. Chuẩn hóa ánh sáng (illumination normalization)

Ảnh chụp điện thoại thường có ánh sáng không đều: góc sáng, góc tối. Nếu dùng ngưỡng đen/trắng toàn cục thì vùng tối sẽ bị tràn. Hệ thống khử hiệu ứng này trước:

```
1. Morphological Close với kernel hình elip, kích thước k×k
   k = max(31, (min(chiều_cao, chiều_rộng) ÷ 10) | 1)  →  k lẻ, ≥ 31 px
   → Phép đóng hình thái học lấp đầy chi tiết nhỏ, giữ lại background (nền sáng)

2. Chia ảnh gốc cho background ước tính, scale 255
   norm = (gray ÷ background) × 255
   → Cân bằng độ sáng: vùng tối lên, vùng sáng xuống

3. Gaussian Blur 3×3
   → Làm mờ nhẹ, giảm nhiễu pixel lẻ

4. Otsu threshold (nhị phân hóa tự động)
   → Tìm ngưỡng tối ưu phân tách đen/trắng
   → Đảo ngược: marker đen → pixel = 255 (trắng trong binary_inv)
```

> **Morphological Close** (đóng hình thái học): là phép giãn (dilate) rồi co (erode). Dùng kernel lớn → xóa các chi tiết nhỏ hơn kernel, chỉ giữ lại vùng sáng nền lớn. Kết quả là ảnh ước tính background.
>
> **Otsu threshold**: thuật toán tự tính ngưỡng sao cho phương sai nội nhóm (intra-class variance) nhỏ nhất, tức là phân tách đen/trắng tốt nhất. Không cần đặt ngưỡng thủ công.

### 2b. Trích xuất marker (connected components)

Sau nhị phân hóa, tìm tất cả vùng pixel liên thông (**connected components**) trong ảnh và lọc theo 6 tiêu chí:

| Tiêu chí | Ngưỡng | Lý do |
|----------|--------|-------|
| Diện tích | 0.00002 – 0.02 × tổng ảnh | Lọc bụi nhỏ và vùng lớn không phải marker |
| Kích thước tối thiểu | rộng ≥ 4 px, cao ≥ 4 px | Loại pixel đơn lẻ |
| Tỷ lệ chiều (aspect ratio) | 0.62 – 1.55 | Chỉ nhận hình gần vuông |
| Fill ratio | ≥ 0.40 | Phải đặc ≥ 40% diện tích bounding box |
| Solidity | ≥ 0.82 | Diện tích contour / diện tích convex hull — lọc hình hõm |
| Circularity | < 0.90 | Loại hình tròn (bong bóng câu trả lời) |

> **Fill ratio** = số pixel đen ÷ diện tích bounding box. Marker đen in sẵn có fill cao (~0.85); bong bóng tô bằng bút có fill thấp hơn (~0.5–0.7).
>
> **Solidity** = diện tích contour ÷ diện tích convex hull. Hình chữ L hoặc hình chữ C có solidity thấp. Marker vuông có solidity gần 1.0.
>
> **Circularity** = 4π × diện_tích ÷ chu_vi². Hình tròn hoàn hảo = 1.0; hình vuông ≈ 0.785. Tiêu chí này loại bong bóng tô gần tròn.

Ngoài ra còn lọc qua contour approximation: `cv2.approxPolyDP()` với epsilon = 8% chu vi. Kết quả phải có 4–8 đỉnh (hình đa giác gần vuông).

### 2c. Chọn marker góc tốt nhất

Chia ảnh thành 4 vùng góc (mỗi vùng 30% × 30% từ cạnh). Trong mỗi vùng, chọn marker đạt điểm số cao nhất:

```
score = (area_norm × 1000) + (fill × 40) − (dist_norm × 230)

  area_norm = diện tích marker ÷ tổng diện tích ảnh
  fill      = fill ratio của marker
  dist_norm = khoảng cách bình phương đến góc ÷ tổng diện tích ảnh
```

→ Ưu tiên marker **to**, **đặc**, **gần góc**. Kết quả: 4 điểm góc (top-left, top-right, bottom-left, bottom-right).

---

## Bước 3 — Phát hiện trang bằng contour (phương án dự phòng)

**Module:** `omr_preprocess.py` → `_find_page_quad_by_contour()`

Nếu Bước 2 không tìm đủ 4 marker góc (ảnh mờ, góc bị che...), hệ thống thử phát hiện viền trang giấy:

```
1. Gaussian Blur 5×5
2. Canny Edge Detection với ngưỡng thấp=60, ngưỡng cao=180
   → Phát hiện biên (cạnh đổi gradient mạnh)
3. Dilate 3×3 (giãn biên để nối các đoạn gián đoạn)
4. findContours — tìm tất cả đường bao ngoài
5. Lọc: chỉ giữ contour có diện tích ≥ 18% tổng ảnh
6. Xấp xỉ đa giác: approxPolyDP với epsilon = 2% chu vi
   → Nếu được 4 đỉnh → dùng làm quad
   → Nếu không → dùng minAreaRect (hình chữ nhật tối thiểu)
```

> **Canny Edge Detection**: thuật toán phát hiện biên dựa trên gradient. Ngưỡng thấp (60) và cao (180) xác định điểm nào là biên yếu/mạnh. Biên yếu chỉ được giữ nếu liền kề biên mạnh.
>
> **approxPolyDP**: rút gọn contour thành đa giác ít đỉnh hơn (Douglas-Peucker algorithm). Epsilon càng lớn → đa giác càng đơn giản.

---

## Bước 4 — Warp phối cảnh và chuẩn hóa kích thước

**Module:** `omr_preprocess.py` → `_warp_to_standard_layout()`

Với 4 điểm góc đã tìm được, biến đổi ảnh về góc nhìn thẳng đứng:

### 4a. Perspective transform (biến đổi phối cảnh)

```python
M = cv2.getPerspectiveTransform(src_quad, dst_rect)
warped = cv2.warpPerspective(img_bgr, M, (2480, 3508))
```

- `src_quad`: 4 điểm góc phát hiện được trong ảnh gốc
- `dst_rect`: hình chữ nhật chuẩn `[[0,0], [2479,0], [2479,3507], [0,3507]]`
- Kích thước trung gian: **2480×3508 px** (tỷ lệ A4 @ 300 DPI)

> **getPerspectiveTransform**: tính ma trận biến đổi 3×3 (homography matrix) sao cho 4 điểm nguồn ánh xạ chính xác lên 4 điểm đích. Đây là biến đổi **projective** (phối cảnh) — tổng quát hơn biến đổi affine, bù được cả xoay, nghiêng, thay đổi góc nhìn.
>
> **Giới hạn**: biến đổi 4 điểm chỉ bù được tilt/skew phẳng. Nếu giấy cong (không phẳng tuyệt đối), sau warp vẫn còn biến dạng nhỏ theo chiều dọc.

Nếu không tìm được quad (cả Bước 2 và 3 đều thất bại) → chỉ resize ảnh gốc, không warp. Trường hợp này ghi cờ cảnh báo `COORD_GLOBAL_WARP_FALLBACK`.

### 4b. Resize về kích thước làm việc chuẩn

```python
resized = cv2.resize(warped, (1000, 1400),
    interpolation=cv2.INTER_AREA   # nếu thu nhỏ
                 cv2.INTER_CUBIC)  # nếu phóng to
```

Mọi xử lý tiếp theo đều trên ảnh **1000×1400 px**.

> **INTER_AREA**: lấy trung bình các pixel khi thu nhỏ — chất lượng tốt hơn INTER_LINEAR.
> **INTER_CUBIC**: nội suy bicubic khi phóng to — mượt hơn INTER_LINEAR.

---

## Bước 5 — Chuyển ảnh xám và nhị phân hóa

**Module:** `omr_preprocess.py` → `_binarize()`

### 5a. Chuyển sang ảnh xám (grayscale)

```
Y = 0.299×R + 0.587×G + 0.114×B
```

Công thức theo tiêu chuẩn ITU-R BT.601. Trọng số cao nhất cho Green vì mắt người nhạy cảm nhất với màu xanh lá.

### 5b. Chuẩn hóa background (lần 2)

Tương tự Bước 2a, nhưng trên ảnh 1000×1400 đã warp:

```
k = max(31, (min(1000, 1400) ÷ 9) | 1)  →  k = 111 (lẻ)
background = MorphologicalClose(gray, ellipse_kernel_111×111)
gray_norm   = (gray ÷ background) × 255
gray_norm   = GaussianBlur(gray_norm, 3×3)
```

### 5c. Nhị phân hóa

```
ngưỡng_tối_ưu, binary_inv = Otsu(gray_norm)
```

Tự tính ngưỡng T sao cho phương sai liên lớp giữa hai nhóm pixel (< T và ≥ T) đạt cực đại. Nhờ bước cân bằng nền ở 5b, histogram luôn có 2 đỉnh rõ → Otsu chọn T ổn định.

`THRESH_BINARY_INV`: pixel < T (mực tối) → 255, pixel ≥ T (nền sáng) → 0.

### 5d. Khử nhiễu sau nhị phân hóa

```
binary_inv = MorphologicalOpen(binary_inv, kernel 2×2, 1 lần)
```

> **Morphological Open** (mở hình thái học): là phép co (erode) rồi giãn (dilate). Xóa bỏ các vùng trắng nhỏ hơn kernel 2×2 px → loại nhiễu điểm lẻ.

Kết quả: `binary_inv` — ảnh nhị phân 1000×1400, **đảo ngược** (pixel đen của phiếu = 255, nền trắng = 0).

---

## Bước 6 — Trích xuất và phân loại marker trong ảnh chuẩn

**Module:** `omr_mcq.py`, `omr_layout.py`

Lặp lại quy trình phát hiện marker (tương tự Bước 2b) trên ảnh 1000×1400. Tất cả tọa độ từ đây trở đi đều trong hệ **1000×1400 px**.

Sau khi trích xuất, phân loại marker thành 2 loại:

| Loại | Fill | Size | Circularity | Vai trò |
|------|------|------|-------------|---------|
| **Fiducial marker** (neo định vị) | ≥ 0.86 | ≥ 11 px | ≤ 0.90 | Ô vuông đen in sẵn, dùng để căn lưới |
| **Bubble marker** (bong bóng) | ≥ 0.58 | ≥ 6 px | bất kỳ | Bong bóng học sinh tô |

---

## Bước 7 — Suy luận hình học lưới MCQ

**Module:** `omr_mcq.py` → `_infer_mcq_geometry_from_markers()`

Mục tiêu: xác định `line_h` (khoảng cách dòng), `top_center_y` (tọa độ Y dòng đầu tiên), vị trí các block MCQ.

```
1. Cluster marker theo tọa độ Y (tolerance ±6 px)
   → Nhóm tất cả marker có Y gần nhau thành cùng 1 hàng

2. Tìm hàng fiducial: ≥ 2 fiducial marker, trải rộng ≥ 34% chiều rộng ảnh
   → Hàng đầu tiên = fid_top_y, hàng cuối = fid_bottom_y

3. Tính khoảng cách dòng (line_h):
   line_h = median(khoảng cách giữa các hàng bubble liền kề)
   Chỉ nhận khoảng cách trong phạm vi [12.0, 65.0] px

4. Xác định top_center_y: Y tâm dòng bong bóng đầu tiên
   Xác định bottom_center_y: Y tâm dòng bong bóng cuối cùng

5. Suy ra số block (block_count): dựa trên số cụm marker theo chiều ngang
```

> **Cluster theo Y**: sắp xếp marker theo Y, dùng thuật toán gộp nhóm đơn giản — hai marker cách nhau ≤ 6 px được gộp vào cùng hàng.
>
> **Median** (trung vị): bền vững hơn mean (trung bình) với outlier. Nếu một khoảng cách bị lệch do marker phát hiện sai, median ít bị ảnh hưởng.

---

## Bước 8 — Xác định điểm neo và xây dựng vùng quan tâm (ROI)

**Module:** `omr_layout.py` → `_resolve_coordinate_anchors()`, `_build_rois_from_anchors()`

### 8a. Xác định 8 điểm neo (anchor points)

Hệ thống cần 8 điểm neo để định vị 3 vùng chính:

```
SID (MSSV):   sid_top_y, sid_bottom_y
Mã đề:        code_top_y, code_bottom_y
MCQ:          mcq_left_top, mcq_left_bottom
              mcq_right_top, mcq_right_bottom
```

Với mỗi điểm neo, tìm marker thực tế gần vị trí template nhất. Nếu không tìm được → dùng vị trí template (% theo chiều cao/rộng ảnh):

| Chế độ | MCQ top | line_h template |
|--------|---------|-----------------|
| Short-form (< 25 hàng/block) | 57.5% chiều cao | 1.52% chiều cao ≈ 21 px |
| Long-form (≥ 25 hàng/block) | 30.0% chiều cao | 2.20% chiều cao ≈ 31 px |

Nếu phải dùng fallback % → ghi cờ `COORD_ANCHOR_FALLBACK`.

### 8b. Tính line_h từ anchor

```
span_y = mcq_bottom_y − mcq_top_y
line_h = span_y ÷ (rows_per_block − 1)
line_h = clamp(line_h, 12.0, 44.0)  ← giới hạn hợp lý
```

> `clamp(x, a, b)` = max(a, min(b, x)): giữ x trong khoảng [a, b].

### 8c. Xây dựng ROI (Region of Interest — vùng quan tâm)

Từ 8 anchor, tính bounding box cho 3 vùng:

- **SID ROI**: hình chữ nhật bao quanh vùng MSSV
- **Exam Code ROI**: bao quanh vùng mã đề
- **MCQ ROI**: bao quanh toàn bộ vùng câu MCQ

> **ROI thứ 4 (tùy chọn, song song) — `ho_ten` (chữ viết tay):** Ngoài 3 ROI trên, hệ thống còn dựng thêm một ROI tùy chọn cho vùng họ tên viết tay, bằng hàm `_build_handwriting_rois()` trong module riêng `omr_handwriting.py` (gọi tại `omr_service.py`, ngay sau khi 3 ROI SID/Code/MCQ đã có). ROI này **không** dựa vào anchor/marker như 8a–8c ở trên — vì chữ viết tay không có fiducial marker cố định để căn — mà lấy từ cấu hình profile (`strategy.handwriting_fields.field_rois.ho_ten`, do người quản trị vẽ tay qua giao diện `OmrProfileRoiEditor`), hoặc suy luận heuristic tương đối theo vị trí SID ROI nếu là short-form. Kết quả ROI này được dùng ở Bước 15 để cắt và lưu ảnh vùng họ tên — chỉ lưu ảnh cho người chấm tự đọc, **không** chạy OCR (`ocr_engine: "disabled"`).

---

## Bước 9 — Tinh chỉnh ROI MCQ bằng template matching

**Module:** `omr_mcq.py` → `refine_mcq_roi()`

Dùng **template matching** (so khớp mẫu) để tìm chính xác hơn vị trí marker đặc trưng trong vùng MCQ:

```
1. Tạo template: hình chữ nhật đen ~ kích thước marker đặc trưng
2. Quét template trên binary_inv trong vùng MCQ (chỉ 58% trên cùng của trang)
3. Ngưỡng cascade: thử lần lượt 0.80 → 0.74 → 0.68 → 0.62 → 0.56
   → Dừng khi tìm được đủ match
4. Cluster match theo Y → tìm hàng trên cùng và dưới cùng có ≥ 3 match
5. Tái tính line_h:
   line_h = (y_hàng_dưới − y_hàng_trên) ÷ 16
   Điều kiện: line_h phải nằm trong [6.0, 72.0] px
```

> **Template matching**: `cv2.matchTemplate()` trượt template qua ảnh, tính độ tương đồng tại mỗi vị trí. Ngưỡng cascade giảm dần cho phép chấp nhận match kém hơn nếu ngưỡng cao không tìm được.
>
> **16 khoảng**: giữa hàng đầu và hàng cuối của block 20 hàng có 19 khoảng. Con số 16 là số khoảng đặc thù của mẫu phiếu trong hệ thống này (phụ thuộc thiết kế form).

Nếu tinh chỉnh thành công → ghi cờ `MCQ_TEMPLATE_REFINE_APPLIED`.

---

## Bước 10 — Giải mã MSSV (Student ID)

**Module:** `omr_numeric.py` → `_decode_numeric_columns()`

MSSV được tô dạng bubble số: mỗi chữ số là 1 cột, mỗi cột có 10 hàng (0–9).

```
1. Crop vùng SID ROI từ ảnh chuẩn
2. Phân tích từng cột:
   - Tính fill density của bong bóng tại mỗi hàng (0–9)
   - Chữ số được chọn = hàng có mật độ cao nhất
   - Nếu mật độ cao nhất < ngưỡng tối thiểu → "?" (không chắc chắn)
3. Hai lần thử (2 pass):
   - Pass 1: có hàng viết tay (skip hàng 0, dùng hàng 1–10)
   - Pass 2: không có hàng viết tay (dùng hàng 0–9)
4. Chọn pass tốt hơn dựa trên confidence tổng hợp
   Nếu tự động chuyển → ghi cờ SID_AUTO_SWITCH_ROW_MODE
```

**Đầu ra:** `"123456"` hoặc `"1?3456"` (dấu `?` cho chữ số không chắc)

---

## Bước 11 — Giải mã mã đề (Exam Code)

**Module:** `omr_numeric.py` → `_decode_numeric_columns()`

Quy trình tương tự MSSV nhưng:
- Chỉ 3 chữ số
- Không có hàng viết tay
- Mã đề dùng để tra bảng đáp án: `"001"` → đề số 1

Nếu không tìm thấy mã đề trong bảng đáp án → ghi cờ `ANSWER_CODE_NOT_FOUND`.

---

## Bước 12 — Giải mã câu MCQ

**Module:** `omr_mcq.py` → `_decode_mcq_with_map()`

Đây là bước phức tạp nhất. Với mỗi câu hỏi (từ 1 đến Q), hệ thống xác định học sinh đã tô đáp án nào.

### 12a. Định vị vị trí dòng

```
cy = top_center_y + row_idx × line_h
```

- `top_center_y`: tọa độ Y tâm dòng câu 1
- `row_idx`: chỉ số dòng trong block (0 = câu đầu block, 19 = câu cuối block 20 câu)
- `line_h`: khoảng cách dòng (pixel)

Cửa sổ lấy mẫu theo chiều dọc:

```
y1 = cy − 0.44 × line_h     ← biên trên
y2 = cy + 0.44 × line_h     ← biên dưới
→ Tổng chiều cao cửa sổ = 0.88 × line_h (88% khoảng cách dòng)
```

> Dùng 88% thay vì 100% để tránh lấy mẫu lẫn vào dòng kề trên/dưới.

### 12b. Tính điểm bong bóng — `_cell_score()`

Với mỗi lựa chọn (A, B, C, D), trích vùng pixel bong bóng và tính điểm tổng hợp:

```
fill_ratio = countNonZero(binary_cell) ÷ diện_tích_ô
           = tỷ lệ pixel đen sau nhị phân hóa

dark_mean  = max(0, 1 − mean(gray_cell) ÷ 255)
           = độ tối trung bình (0 = trắng hoàn toàn, 1 = đen hoàn toàn)

dark_p25   = max(0, 1 − percentile_25(gray_cell) ÷ 255)
           = độ tối tại phân vị 25 (25% pixel sáng nhất trong ô)
```

**Công thức điểm tổng hợp:**

```
score = 0.55 × fill_ratio
      + 0.30 × dark_mean
      + 0.15 × dark_p25
```

> **Tại sao kết hợp 3 thành phần?**
> - `fill_ratio` từ ảnh nhị phân có thể mất tín hiệu khi bút tô nhạt (threshold Otsu toàn cục có thể cắt bỏ vùng tô mờ)
> - `dark_mean` và `dark_p25` từ ảnh xám gốc vẫn giữ tín hiệu độ tối thực tế
> - Kết hợp: nếu nhị phân bị mất tín hiệu, ảnh xám bù lại
>
> **percentile_25**: giá trị mà 25% pixel trong ô tối hơn. Dùng percentile_25 (không phải min) để bớt nhạy với nhiễu điểm tối lẻ.

### 12c. Ngưỡng quyết định thích nghi — `dynamic_mark`

Thay vì dùng ngưỡng tĩnh cho tất cả câu, ngưỡng được tính riêng cho mỗi hàng câu:

```
row_mean    = mean(score[A], score[B], score[C], score[D])
             = điểm trung bình 4 lựa chọn trong hàng này

dynamic_mark = max(min_mark × 0.95, min(0.60, row_mean + 0.06))
             ← clamp giữa [min_mark×0.95, 0.60]
```

> Nếu cả 4 lựa chọn đều có score cao (ảnh sáng đều, mực đậm), ngưỡng tăng lên. Nếu score thấp (ảnh tối, tô nhạt), ngưỡng giảm xuống. Điều này bù cho sự khác biệt về điều kiện in ấn và ánh sáng.

### 12d. Noise gate — lọc nhiễu bề mặt

Trước khi chấp nhận một lựa chọn, kiểm tra thêm 2 điều kiện chống nhiễu:

```
noise_center_gate = max(0.42, row_center_mean + 0.10)  ← không vượt 0.92
noise_dark_gate   = max(0.30, row_dark_mean   + 0.08)  ← không vượt 0.85

noise_gate_ok = (best_center ≥ 0.85)            ← tâm ô rất đặc
             OR (score ≥ 0.55 AND center ≥ 0.72) ← strong_signal_bypass
             OR (center ≥ noise_center_gate
                 AND dark ≥ noise_dark_gate)      ← cả hai gate đều vượt
```

> **best_center**: fill_ratio của **vùng trung tâm** (56% giữa) bong bóng. Vết bẩn, đường kẻ thường ở rìa ô; tâm ô sạch hơn.
>
> **strong_signal_bypass**: nếu score ≥ 0.55 VÀ tâm ≥ 0.72 thì bỏ qua noise_dark_gate. Dùng cho bong bóng tô rõ ràng — không để dark_gate quá chặt làm false rejection.

### 12e. Bảng quyết định mỗi câu

| Tình huống | Điều kiện | Kết quả |
|------------|-----------|---------|
| Không có lựa chọn | Tất cả score < dynamic_mark | uncertain (?) |
| Tô hai đáp án (double mark) | ≥ 2 lựa chọn ≥ dynamic_mark | uncertain, trừ khi margin đủ lớn |
| Một lựa chọn rõ ràng | 1 lựa chọn ≥ dynamic_mark, noise_gate_ok | Chấp nhận = A/B/C/D |
| Tô nhạt (soft rescue) | score ≥ 0.38, margin ≥ 0.06, conf_ratio ≥ 1.18, noise_gate_ok | Chấp nhận dù dưới dynamic_mark |

> **margin** = điểm cao nhất − điểm cao thứ hai. Margin lớn → ô được tô vượt trội rõ ràng so với các ô còn lại.
>
> **conf_ratio** = điểm cao nhất ÷ điểm cao thứ hai. Tỷ số ≥ 1.18 → ô được tô sáng hơn các ô khác ít nhất 18%.
>
> **double mark**: học sinh tô 2 ô, hoặc có vết bẩn. Nếu margin giữa 2 ô ≥ max(min_margin, double_mark_gap=0.08) → hệ thống chọn ô cao điểm hơn. Nếu không → uncertain.

### 12f. Chất lượng hàng — `_row_candidate_quality()`

Điểm chất lượng tổng hợp của cả hàng (dùng để so sánh các cấu hình lưới trong Rescue):

```
quality = 1.10 × best_score
        + 1.45 × margin
        + 0.25 × best_center
        + 0.12 × best_dark
        − left_penalty
```

`left_penalty`: nếu lựa chọn A (cột trái) dẫn đầu nhưng margin < 0.08 → phạt 0.20 × (0.08 − margin). Cột A hay bị ảnh hưởng bởi đường kẻ dọc trái của phiếu.

---

## Bước 13 — MCQ Map Search Rescue (điều chỉnh lưới tự động)

**Module:** `omr_service.py` → phần sau `_decode_mcq_with_map()` baseline

### 13a. Điều kiện kích hoạt

```
gate = max(4, round(0.40 × Q))    ← Q = tổng số câu hỏi
```

Ví dụ: Q=40 → gate=16; Q=80 → gate=32.

Rescue **chỉ kích hoạt** khi số câu uncertain sau decode baseline ≥ gate. Ngưỡng 40% đảm bảo không kích hoạt do vài câu tô không rõ bình thường.

### 13b. Không gian tìm kiếm (54 ứng viên)

```
line_h scales = [1.00, 0.92, 0.88, 1.08, 0.84, 1.16]   ← 6 giá trị tỷ lệ line_h
shift_mults   = [0.0, -0.5, 0.5, -1.0, 1.0, -1.5, 1.5, -2.0, 2.0]  ← 9 giá trị dịch chuyển

Với mỗi (scale, shift_mult):
    cand_line_h = line_h × scale          ← nén/giãn khoảng cách dòng
    cand_shift  = shift_mult × cand_line_h ← dịch toàn bộ lưới lên/xuống
    → Gọi _decode_mcq_with_map() với tham số mới
    → Tổng: 6 × 9 = 54 lần decode
```

### 13c. Xếp hạng ứng viên

```
rank = (uncertain_count, double_mark_count, −quality)
```

Bộ ba này được so sánh theo từ điển (lexicographic): ưu tiên ít uncertain nhất, sau đó ít double mark, sau đó quality cao nhất.

### 13d. Tiêu chí chấp nhận bảo thủ

Chỉ thay baseline bằng ứng viên tốt nhất nếu đáp ứng ít nhất một trong:

1. Δuncertain ≥ 2 (giảm ít nhất 2 câu uncertain), **hoặc**
2. Δuncertain ≥ 1 **và** double_mark không tăng, **hoặc**
3. Δuncertain = 0 **và** double_mark giảm

Nếu không có ứng viên nào đạt → giữ nguyên baseline. Tiêu chí bảo thủ này ngăn hệ thống thực hiện điều chỉnh không cần thiết khi phiếu thực sự tô mờ (không phải lỗi lưới).

Nếu Rescue cải thiện được kết quả → ghi cờ `MCQ_MAP_SEARCH_RESCUE`.

---

## Bước 14 — Phát hiện drift và mở rộng ROI tự động

**Module:** `omr_service.py` → `_detect_q5_start_drift()`

Sau Rescue, hệ thống kiểm tra thêm một loại lỗi khác: **ROI MCQ bắt đầu quá thấp** — tức là `top_center_y` trỏ vào dòng câu 2 thay vì câu 1, khiến câu 1 bị bỏ qua và toàn bộ câu bị lệch chỉ số.

### Dấu hiệu nhận biết drift

```
Kiểm tra confidence của Q5–Q10:
  Nếu phần lớn Q5–Q10 có confidence thấp (< ngưỡng tối thiểu)
  TRONG KHI Q11+ có confidence bình thường
  → Suy ra: ROI bắt đầu thấp hơn thực tế khoảng 4–5 dòng
```

Đây là tình huống xảy ra khi marker đầu vùng MCQ bị phát hiện lệch xuống: `top_center_y` tính ra trỏ vào khoảng trống giữa các dòng thay vì tâm dòng đầu.

### Hành động mở rộng lên trên (auto-expand up)

```
new_top_center_y = top_center_y − 4 × line_h
new_MCQ_ROI.y   = MCQ_ROI.y    − 4 × line_h
new_MCQ_ROI.h   = MCQ_ROI.h    + 4 × line_h

→ Decode lại toàn bộ MCQ với ROI và top_center_y mới
→ Chỉ chấp nhận nếu quality mới ≥ quality baseline
```

Nếu mở rộng thành công → ghi cờ `MCQ_AUTO_EXPAND_UP`.

> **Tại sao 4 dòng?** Theo cấu trúc phiếu phổ biến, khoảng cách từ viền ROI đến dòng đầu tiên thường ≤ 4 × line_h. Mở rộng 4 dòng đủ để bắt trường hợp xấu nhất mà không quá lớn gây nhận nhầm vùng bên trên MCQ.

---

## Bước 15 — Chấm điểm và trả kết quả

**Module:** `omr_service.py`, `omr_scoring.py` → `_build_answer_compare()`

### 14a. Chấm điểm

```
Với mỗi câu trong graded_questions (có đáp án trong answer_key):
    Nếu user_answers[q] == answer_key[q] → correct_count += 1
    Nếu user_answers[q] == -1 → uncertain
    Nếu user_answers[q] != answer_key[q] AND != -1 → wrong
```

### 14b. Cấu trúc JSON đầu ra (các trường chính)

```json
{
  "score": 35,                          // Số câu đúng
  "student_id": "123456",               // MSSV phát hiện được
  "exam_code": "001",                   // Mã đề
  "answer_map": ["A","B","?","C",...],  // Đáp án phát hiện ("?" = uncertain)
  "uncertain_count": 2,                 // Số câu không chắc chắn
  "uncertain_questions": [5, 12],       // Danh sách số thứ tự câu uncertain
  "double_mark_questions": [7],         // Câu tô 2 đáp án

  "roi_boxes": {
    "student_id": {"x":85,"y":231,"w":255,"h":315},
    "mcq": {"x":75,"y":602,"w":855,"h":658},
    "handwriting": {"ho_ten": {"x":40,"y":60,"w":230,"h":75}}  // ROI chữ viết tay (nếu profile có cấu hình)
  },

  "handwriting_fields": {"ho_ten": ""},  // Luôn rỗng — không chạy OCR, chỉ crop ảnh

  "handwriting": {                      // Chi tiết crop chữ viết tay
    "enabled": true,
    "ocr_engine": "disabled",           // Cố ý disable, chờ tích hợp OCR/học sâu ở phiên bản sau
    "field_rois": {"ho_ten": {"x":40,"y":60,"w":230,"h":75}},
    "crop_images": {"ho_ten": "omr_hw_ho_ten_xxx.jpg"}
  },

  "warning_codes": [                    // Cờ cảnh báo kỹ thuật
    "MCQ_MAP_SEARCH_RESCUE",
    "MCQ_BLOCKS_INFERRED"
  ],

  "result_image": "omr_result_xxx.jpg",    // Ảnh kết quả có overlay
  "bubble_confidence_json": "bubble_xxx.json"  // Telemetry chi tiết
}
```

### 14c. Danh sách cờ cảnh báo (warning_codes) thường gặp

| Mã cờ | Ý nghĩa |
|-------|---------|
| `COORD_GLOBAL_WARP_FALLBACK` | Không tìm được 4 góc marker; chỉ resize, không warp |
| `COORD_ANCHOR_FALLBACK` | Không đủ marker fiducial; dùng vị trí mặc định (%) |
| `MCQ_BLOCKS_INFERRED` | Số block MCQ tự động suy luận từ marker |
| `MCQ_TEMPLATE_REFINE_APPLIED` | ROI MCQ được tinh chỉnh bằng template matching |
| `MCQ_MAP_SEARCH_RESCUE` | Rescue điều chỉnh line_h/top_shift thành công |
| `MCQ_AUTO_EXPAND_UP` | ROI MCQ được mở rộng lên trên do phát hiện drift |
| `MCQ_DOUBLE_MARK` | Có ít nhất 1 câu học sinh tô 2 đáp án |
| `SID_AUTO_SWITCH_ROW_MODE` | Tự động chuyển chế độ đọc MSSV (có/không hàng viết tay) |
| `ANSWER_CODE_NOT_FOUND` | Mã đề phát hiện không có trong bảng đáp án |
| `HANDWRITING_OUTPUT_DIR_FAILED` | Không tạo được thư mục lưu ảnh crop chữ viết tay |
| `HANDWRITING_CROP_WRITE_FAILED` | Lưu ảnh crop vùng họ tên thất bại |

---

## Tóm tắt các ngưỡng quan trọng

| Tham số | Giá trị | Ý nghĩa |
|---------|---------|---------|
| `min_mark` (static) | 0.55 | Ngưỡng tĩnh nhận dạng bong bóng |
| `dynamic_mark` | `max(min×0.95, min(0.60, mean+0.06))` | Ngưỡng thích nghi theo hàng |
| `soft_mark_floor` | 0.38 | Ngưỡng tối thiểu cho soft rescue |
| `min_margin` | 0.05 | Khoảng cách tối thiểu nhất/nhì |
| `double_mark_gap` | 0.08 | Margin để giải quyết double mark |
| `noise_center_floor` | 0.42 | Sàn gate tâm ô |
| `noise_dark_floor` | 0.30 | Sàn gate độ tối |
| `strong_signal_bypass` | score≥0.55 AND center≥0.72 | Bỏ qua dark gate khi tô rõ |
| `line_h` clamp | [12, 44] px | Giới hạn khoảng cách dòng hợp lý |
| Rescue gate | `max(4, round(0.40×Q))` | Ngưỡng kích hoạt Map Search |
| Rescue candidates | 6 × 9 = 54 | Số cấu hình lưới thử nghiệm |
