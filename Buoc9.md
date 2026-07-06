# Bước 9 — Tinh chỉnh ROI MCQ bằng template matching

> **Vị trí trong pipeline:** Bước 8 xây ROI MCQ bằng cách mở rộng lên trên hàng fiducial top (padding `-0.45×line_h`), vô tình kéo vào phần hướng dẫn làm bài in phía trên vùng MCQ. Bước 9 tìm chính xác vị trí hàng fiducial top và bottom bằng template matching, rồi đặt ROI bắt đầu ngay dưới hàng fiducial top — loại bỏ phần hướng dẫn đó.
>
> Chỉ chạy khi profile **không** cấu hình sẵn `mcq_roi` (tức `mcq_roi_cfg is None`). Hiện tại **tất cả 4 profile** trong hệ thống đều không cấu hình `mcq_roi` → bước này luôn được chạy.

**Module:** `be/app/services/omr/omr_mcq.py`
**Hàm chính:** `refine_mcq_roi()`
**Hàm hỗ trợ:** `_pick_anchor_template()`, `_cluster_match_rows()`, `_pick_three_markers_on_row()`

---

## 1. Tại sao cần bước này?

Phiếu thi thường có hàng fiducial marker in sẵn ngay **phía trên** vùng bong bóng MCQ. Phía trên hàng fiducial đó còn có phần hướng dẫn làm bài in trên phiếu.

Bước 8 xây ROI bằng công thức:
```python
mcq_y = top_anchor_y - 0.45 * line_h   # mở rộng LÊN TRÊN top anchor
```

Tức là ROI bắt đầu **trước** hàng fiducial top, có thể bao gồm cả phần hướng dẫn. Nếu phần hướng dẫn đó có nét đen (đường kẻ, chữ in đậm), bước decode MCQ sẽ đọc nhầm chúng là bong bóng được tô.

Bước 9 sửa lại: tìm chính xác vị trí Y của hàng fiducial top và bottom, rồi đặt:
```python
y_top = y_anchor_top + 5px   # bắt đầu DƯỚI hàng fiducial top
```

Comment trong code ghi rõ: *"Hard top boundary: start below top marker row to avoid instruction text overlap."*

**Lý do phụ:** template matching trên pixel cho tọa độ Y chính xác hơn centroid từ connected component analysis. Trung bình 3 marker cùng hàng còn ổn định hơn 1 điểm đơn lẻ.

Nếu template matching thất bại → ROI từ Bước 8 được giữ nguyên (không thay đổi).

---

## 2. Đầu vào (Input)

| Tham số | Kiểu | Giá trị tiêu biểu |
|---------|------|------------------|
| `source_img` | `np.ndarray` (BGR) | ảnh màu 1000×1400 |
| `gray_img` | `np.ndarray` | ảnh xám 1000×1400 |
| `mcq_roi` | `dict` | `{"x":75,"y":602,"w":855,"h":658}` từ Bước 8 |
| `top_padding_px` | `int` | 5 |
| `side_padding_px` | `int` | 10 |
| `bottom_padding_px` | `int` | 15 |

---

## 3. 9a — Mở rộng cửa sổ tìm kiếm

Template matching cần tìm marker **ngay sát biên trên/dưới** của ROI, và ROI từ Bước 8 có thể thấp hơn vị trí thực. Hệ thống mở rộng vùng tìm kiếm trước khi chạy matching:

```python
top_pad    = max(22, int(round(0.34 * h)))   # mở rộng lên trên 34% h ROI
bottom_pad = max(24, int(round(0.14 * h)))   # mở rộng xuống dưới 14% h ROI

search_window = gray_img[
    (y - top_pad) : (y + h + bottom_pad),    # hàng: mở rộng cả trên lẫn dưới
    x : (x + w)                               # cột: giữ nguyên X của ROI
]
```

Ví dụ với ROI `h=658`:
- `top_pad = max(22, 0.34×658)` = max(22, 224) = **224px**
- `bottom_pad = max(24, 0.14×658)` = max(24, 92) = **92px**
- Cửa sổ tìm kiếm cao = 658 + 224 + 92 = **974px**

**Tại sao mở rộng lên nhiều hơn xuống (34% vs 14%)?** Hàng fiducial top nằm **trên** vùng bong bóng — tức là nằm trên (hoặc ngay tại) biên trên của ROI Bước 8. Cần mở rộng lên để cửa sổ tìm kiếm bao trọn hàng fiducial đó. Xuống 14% chỉ để đảm bảo hàng fiducial bottom cũng nằm trong cửa sổ nếu ROI Bước 8 hơi ngắn.

---

## 4. 9b — Tìm template từ marker trong cửa sổ (`_pick_anchor_template`)

Template là **hình ảnh cắt ra từ một marker thực tế** trong ảnh — không phải template định sẵn. Hệ thống tự tìm:

```python
def _pick_anchor_template(search_gray):
    # Chỉ tìm trong 58% phần trên của cửa sổ tìm kiếm
    top_h = max(40, int(round(0.58 * h)))
    top_band = search_gray[:top_h, :]

    # Nhị phân hóa Otsu + khử nhiễu
    blur = GaussianBlur(top_band, 3×3)
    band_inv = Otsu(blur) + MorphOpen(2×2)

    # Tìm contour và lọc
    for cnt in findContours(band_inv):
        area = bw × bh
        if area < min_area or area > max_area: continue   # lọc theo diện tích
        if not (0.60 ≤ aspect ≤ 1.45): continue           # chỉ nhận hình gần vuông
        if fill_ratio < 0.50: continue                     # phải đặc ≥ 50%

        # Điểm số: ưu tiên đặc, to, ở giữa, ở trên
        score = 2.0×fill + 1.1×square_size + 0.45×center_bias − 0.38×y_norm
```

| Tiêu chí | Ngưỡng | Lý do |
|----------|--------|-------|
| Diện tích | 0.012%–0.60% top_band | Lọc nhiễu điểm nhỏ và vùng quá to |
| Aspect ratio | 0.60–1.45 | Chỉ nhận marker hình gần vuông |
| Fill ratio | ≥ 0.50 | Marker đen in sẵn phải đặc |
| Vùng tìm | Top 58% cửa sổ | Hàng marker đầu tiên nằm ở nửa trên |

**Điểm số template (`score`):**
- `2.0 × fill_ratio`: marker **đặc** được ưu tiên mạnh nhất
- `1.1 × square_size`: marker **to** được ưu tiên (size chuẩn hóa theo cửa sổ)
- `0.45 × center_bias`: marker ở **giữa chiều ngang** được ưu tiên hơn mép
- `-0.38 × y_norm`: marker ở **phần trên** được ưu tiên (càng dưới bị phạt)

Template được cắt ra với **padding 22%** xung quanh marker:

```python
pad = max(2, int(round(0.22 × max(bw, bh))))
template = top_band[by-pad : by+bh+pad, bx-pad : bx+bw+pad]
```

Padding giúp context xung quanh marker góp phần vào độ tương đồng khi matching — tránh khớp nhầm vào chi tiết nhỏ có hình dạng tương tự.

---

## 5. 9c — Template matching cascade + lọc kết quả

### Chạy template matching

```python
result = cv2.matchTemplate(search_gray, template, cv2.TM_CCOEFF_NORMED)
```

`TM_CCOEFF_NORMED` trả về ma trận điểm tương đồng trong [-1, 1]. Giá trị gần 1 = rất giống template.

### Cascade ngưỡng

Thử lần lượt 5 ngưỡng giảm dần, dừng khi tìm được ≥9 match:

```python
threshold_levels = [0.80, 0.74, 0.68, 0.62, 0.56]

for threshold in threshold_levels:
    ys, xs = np.where(result >= threshold)   # tìm tất cả điểm vượt ngưỡng
    candidates = sorted by score DESC
    picked = lọc qua NMS + fill check
    if len(picked) >= 9:
        matches = picked
        break                 # đủ rồi, không cần hạ ngưỡng thêm
```

**Tại sao cascade?** Marker có thể in nhạt (score thấp), hoặc ảnh bị blur. Ngưỡng cao tránh false positive; nếu không tìm đủ, hạ ngưỡng để không bỏ sót marker thật.

### Non-Maximum Suppression (NMS)

Cùng 1 marker thực tế tạo ra nhiều điểm liền nhau trong ma trận result. Loại bỏ trùng lặp:

```python
min_dist = max(6.0, 0.65 × min(template_h, template_w))

for cand in candidates (sorted by score DESC):
    cx = cand.x + 0.5 × template_w
    cy = cand.y + 0.5 × template_h
    if dist(cx, cy, bất kỳ picked nào) < min_dist: bỏ qua
    else: thêm vào picked
```

### Kiểm tra fill

Sau khi vượt qua NMS, kiểm tra thêm: vùng match phải **thực sự có pixel đen**:

```python
patch = search_inv[cy : cy+template_h, cx : cx+template_w]
fill_ratio = countNonZero(patch) / area
if fill_ratio < 0.36: bỏ qua   # loại vùng sáng (false positive)
```

Tối đa 220 match được giữ lại.

**Tại sao dừng ở 9, không phải 6?** Bước tiếp theo cần 2 hàng × 3 marker = tối thiểu 6 match. Nhưng 6 match phân bố sát nhau dễ bị cluster sai. 9 match cho dư để 1–2 match lạc chỗ vẫn không ảnh hưởng kết quả. Nếu tổng match < 6 sau toàn bộ cascade → thất bại, giữ ROI cũ.

---

## 6. 9d — Cluster match thành hàng và lọc

Chuyển tọa độ từ cửa sổ tìm kiếm về tọa độ ảnh gốc:

```python
global_matches = [(sx1 + local_cx, sy1 + local_cy, score) for each match]
```

Cluster theo Y tương tự Bước 7b (thuật toán greedy running mean):

```python
row_tol = max(4.0, 0.32 × min(template_h, template_w))
rows = _cluster_match_rows(global_matches, y_tol=row_tol)
```

**Lọc hàng yếu:**

```python
min_row_span = max(90.0, 0.42 × mcq_roi_w)   # span tối thiểu 42% chiều rộng MCQ

candidate_rows = [row for row in rows
    if len(row.points) >= 3                    # ≥ 3 match trong hàng
    and (max_x - min_x) >= min_row_span]       # trải rộng đủ
```

Điều kiện span ≥42% loại bỏ hàng chỉ có match ở một phía (do marker bị che hay ảnh bị cắt).

Cần ≥2 hàng sau lọc mới tiếp tục. Nếu không → thất bại.

---

## 7. 9e — Chọn top_row và bottom_row với 3 marker

### Chọn top_row

```python
top_row = candidate_rows[0]   # hàng Y nhỏ nhất (trên cùng)
top_markers = _pick_three_markers_on_row(top_row.points, x_left=mcq_x, x_right=mcq_x+mcq_w)
```

**Bên trong `_pick_three_markers_on_row`:**

Hàm tìm chính xác **3 marker đại diện** của một hàng fiducial (trái, giữa, phải):

```python
span = x_right - x_left

# 3 vị trí mong đợi theo thiết kế phiếu
expected_x = [
    x_left + 0.14 × span,    # cột câu hỏi số ~1
    x_left + 0.50 × span,    # cột câu hỏi số ~18
    x_left + 0.86 × span,    # cột câu hỏi số ~35
]

# Chọn match gần nhất cho mỗi vị trí (không tái sử dụng match đã chọn)
for target_x in expected_x:
    chọn match gần target_x nhất chưa được dùng
```

**Fallback nếu không khớp 3 vị trí:** sort tất cả match theo X, dedupe (loại các X cách nhau < 10% span), lấy đầu-giữa-cuối.

**Điều kiện cuối:** span của 3 marker phải ≥ max(90px, 38% span MCQ). Đảm bảo 3 marker thực sự trải đều qua vùng MCQ, không bị co cụm.

### Chọn bottom_row

```python
min_bottom_gap = max(80.0, 0.22 × search_window_h)

for row in reversed(candidate_rows[1:]):
    if row.cy - top_row.cy < min_bottom_gap: bỏ qua   # quá gần top_row
    maybe = _pick_three_markers_on_row(row.points, ...)
    if maybe is not None:
        bottom_markers = maybe
        break
```

Bắt đầu từ hàng **thấp nhất** và tìm lên trên, để bottom_row là xa top_row nhất có thể — tối đa hóa span để `line_h` chính xác hơn.

**Tại sao `min_bottom_gap = 22% × search_window_h`?** Ngăn tình huống top và bottom cùng là 2 hàng liền nhau do matching trùng: khoảng cách tối thiểu đảm bảo chúng không cùng 1 dải marker.

---

## 8. 9f — Tính anchor Y và kiểm tra line_h

```python
y_anchor_top    = mean(cy của top_markers)      # trung bình Y của 3 marker trên
y_anchor_bottom = mean(cy của bottom_markers)   # trung bình Y của 3 marker dưới

line_h = (y_anchor_bottom - y_anchor_top) / 16.0
```

**Tại sao chia 16?** Đây là hằng số được viết thẳng vào code, không có comment giải thích. Code không dùng `rows_per_block` để tính con số này — nó độc lập với cấu hình profile. Khả năng là người viết code hiệu chỉnh giá trị này dựa trên thiết kế cụ thể của phiếu thi đang dùng.

**Line_h chỉ dùng để:**
1. **Validation**: loại kết quả bất hợp lý — `if not (6.0 ≤ line_h ≤ 72.0): thất bại`
2. **Padding dưới**: `y_bottom += 0.45 × line_h + 15px`

Line_h này **KHÔNG thay thế** giá trị line_h từ Bước 8 (được dùng để decode MCQ). Nó chỉ phục vụ việc tính ROI trong bước này.

---

## 9. 9g — Xây dựng refined ROI

```python
# Biên dọc
y_top    = y_anchor_top + top_padding_px           # 5px dưới marker trên
y_bottom = y_anchor_bottom + max(15.0, bottom_padding_px)
y_bottom = max(y_bottom, y_anchor_bottom + 0.45*line_h + 15.0)

# Biên ngang: mở rộng side_padding_px (10px) so với ROI cũ
rx1 = max(0, mcq_x - side_padding_px)
rx2 = min(img_w, mcq_x + mcq_w + side_padding_px)

# Kiểm tra tối thiểu
if (ry2 - ry1) < 120: thất bại, giữ ROI cũ
```

**Padding top 5px (mặc định):** ROI bắt đầu ngay dưới marker hàng trên — marker nằm ở viền, không nằm trong vùng bong bóng, nên cần đẩy xuống một chút.

**Padding bottom = max(15px, bottom_padding_px + 0.45×line_h):** Đảm bảo hàng bong bóng cuối cùng được bao trọn kể cả khi tọa độ anchor lệch xuống.

**Tại sao giữ X từ ROI cũ + padding (không tính từ marker)?** Vị trí X của 3 marker top/bottom đại diện tốt cho vị trí Y, nhưng không đủ tin cậy cho X (chỉ có 3 điểm, có thể lệch trái/phải). Bước 8 đã tính X kỹ hơn từ block_bands.

```
ROI từ Bước 8:              ROI sau Bước 9:
┌──────────────────┐        ┌──────────────────┐
│  hướng dẫn làm   │        │  hướng dẫn làm   │  ← nằm ngoài ROI
│  bài (in sẵn)    │        │  bài (in sẵn)    │
│──────────────────│        ├──────────────────┤  ← y_anchor_top
│ [■][■][■] fid top│←──┐   │ [■][■][■] fid top│
│ bong bóng câu 1  │   │   │ bong bóng câu 1  │  ← ROI bắt đầu tại đây
│ bong bóng câu 2  │   │   │ bong bóng câu 2  │
│ ...              │   │   │ ...              │
│ bong bóng cuối   │   │   │ bong bóng cuối   │
│ [■][■][■] fid bot│   │   │ [■][■][■] fid bot│  ← y_anchor_bottom
└──────────────────┘   │   └──────────────────┘
  ROI bắt đầu từ       │     ROI bắt đầu từ
  top_anchor - 0.45×lh─┘     y_anchor_top + 5px
  (có thể bao phần hướng dẫn)
```

---

## 10. Điều kiện thất bại (giữ ROI cũ)

| Điều kiện | Lý do giữ ROI cũ |
|----------|-----------------|
| ROI nhỏ hơn 80×120 px | Quá nhỏ để làm việc |
| Không tìm được template | Không có marker đặc trong ảnh |
| Template < 8×8 px | Template không đủ thông tin |
| Tổng match < 6 | Marker quá ít, không đủ tin cậy |
| Ít hơn 2 hàng sau lọc | Không xác định được top và bottom |
| top_row không có 3 marker | Hàng trên không đủ marker |
| Không tìm được bottom_row | Hàng dưới không đủ marker/khoảng cách |
| `y_anchor_bottom ≤ y_anchor_top` | Thứ tự không hợp lệ |
| `line_h ∉ [6, 72]` | Khoảng cách dòng bất thường |
| Chiều cao refined ROI < 120px | Quá nhỏ để decode |

Khi thất bại: `meta["used"] = False`, `meta["reason"] = <lý do>`, ROI gốc từ Bước 8 được trả về nguyên.

Khi thành công: ghi cờ `MCQ_TEMPLATE_REFINE_APPLIED` trong warning_codes.

---

## 11. Đầu ra (Output)

```python
refined_roi = {
    "x": 65,    # mở rộng 10px so với mcq_roi.x
    "y": 598,   # = y_anchor_top + 5
    "w": 875,   # mở rộng 20px so với mcq_roi.w
    "h": 672,   # từ y_top đến y_bottom
}

refined_crop = source_img[ry1:ry2, rx1:rx2]   # ảnh cắt tương ứng

meta = {
    "used": True,
    "reason": "ok",
    "y_anchor_top": 602.5,
    "y_anchor_bottom": 1250.3,
    "line_h": 40.5,           # chỉ dùng nội bộ cho padding
    "matches": 87,
    "top_row_centers": [{"x":122.1,"y":602.5}, {"x":511.3,"y":602.8}, {"x":901.2,"y":602.3}],
    "bottom_row_centers": [{"x":122.5,"y":1250.3}, {"x":511.7,"y":1250.1}, {"x":901.0,"y":1250.4}],
    "refined_roi": {same as refined_roi above},
}
```

---

## 12. Sơ đồ luồng

```
mcq_roi từ Bước 8 + gray_img
        │
        ▼
  9a — Mở rộng cửa sổ tìm kiếm
        lên 34%h + xuống 14%h của ROI
        │
        ▼
  9b — _pick_anchor_template()
        Chỉ xét top 58% cửa sổ
        Binarize → findContours
        Lọc aspect, fill, diện tích
        Score = 2×fill + 1.1×size + 0.45×center − 0.38×y
        Cắt marker tốt nhất + 22% padding → template
        │
        │ Nếu template = None → return ROI cũ
        ▼
  9c — matchTemplate (TM_CCOEFF_NORMED)
        Cascade ngưỡng [0.80 → 0.56]
        Mỗi ngưỡng:
          sort DESC → NMS (min_dist=65% template size)
          fill check ≥ 0.36
        Dừng khi ≥ 9 match
        │
        │ Nếu < 6 match → return ROI cũ
        ▼
  9d — Chuyển về tọa độ ảnh gốc
        _cluster_match_rows(y_tol = 32% template size)
        Lọc hàng: count ≥ 3 AND span ≥ 42% MCQ width
        │
        │ Nếu < 2 hàng → return ROI cũ
        ▼
  9e — top_row = hàng đầu tiên
        _pick_three_markers_on_row() → 3 marker tại 14%, 50%, 86% span
        bottom_row = hàng cuối xa top ≥ 22% search_h
        _pick_three_markers_on_row()
        │
        │ Nếu top/bottom không đủ 3 marker → return ROI cũ
        ▼
  9f — y_anchor_top = mean(Y của top_markers)
        y_anchor_bottom = mean(Y của bottom_markers)
        line_h = span / 16  → kiểm tra [6, 72]
        │
        │ Nếu line_h bất hợp lý → return ROI cũ
        ▼
  9g — Refined ROI:
        y_top = y_anchor_top + 5px
        y_bottom = y_anchor_bottom + max(15, 0.45×line_h+15)
        x: mcq_roi.x ± 10px
        Kiểm tra chiều cao ≥ 120px
        │
        ▼
  refined_roi + refined_crop + meta
        │
        ├──→ Bước 10: giải mã MSSV (dùng sid_roi từ Bước 8, không đổi)
        ├──→ Bước 11: giải mã mã đề (dùng code_roi từ Bước 8, không đổi)
        └──→ Bước 12: giải mã MCQ trong refined_roi
```

---

## 13. Bảng ngưỡng

| Tham số | Giá trị | Ý nghĩa |
|---------|---------|---------|
| Top fill in search window | Top 58% | Vùng tìm template |
| Template pad | 22% × max(w,h) | Context xung quanh marker |
| Cascade thresholds | 0.80→0.74→0.68→0.62→0.56 | Ngưỡng TM_CCOEFF_NORMED |
| Min matches để dừng | ≥ 9 | Đủ tin cậy để cluster |
| Min matches tổng | ≥ 6 | Ngưỡng tối thiểu tiếp tục |
| NMS radius | 65% × min(tw,th) | Dedup match trùng |
| Fill check | ≥ 0.36 | Đảm bảo vùng match có pixel đen |
| Min row span | 42% × mcq_w | Hàng marker phải trải đều |
| Min 3-marker span | 38% × span | 3 marker không bị co cụm |
| Min bottom gap | 22% × search_h | Top và bottom cách nhau đủ xa |
| line_h valid | [6, 72] px | Khoảng dòng hợp lý |
| Min refined height | 120 px | ROI đủ lớn để decode |
