# Bước 8 — Xác định điểm neo và xây dựng vùng quan tâm (ROI)

> **Vị trí trong pipeline:** Bước 7 trả ra geometry (line_h, block_count, block_bands...). Bước 8 dùng geometry đó cùng với danh sách `markers` để tìm **tọa độ pixel chính xác** của 3 vùng trên phiếu: MSSV, mã đề, MCQ — rồi tạo ra 3 hình chữ nhật (ROI) bao quanh từng vùng, để Bước 10–12 cắt ra và đọc.

**Module:** `be/app/services/omr/omr_layout.py`
**Hàm chính:**
- `_resolve_coordinate_anchors()` — tìm 8 điểm neo pixel
- `_build_rois_from_anchors()` — từ 8 điểm neo tạo 3 ROI box

---

## 1. Tại sao cần bước này?

Sau Bước 7, hệ thống biết `line_h`, `block_count`, `top_center_y`... nhưng vẫn chưa biết:
- Vùng MSSV nằm ở hình chữ nhật nào trên ảnh?
- Vùng mã đề bắt đầu/kết thúc ở pixel nào?
- Bounding box của toàn bộ khu vực MCQ là gì?

Bước 8 trả lời bằng cách **tìm marker thực tế gần vị trí mong đợi** để lấy tọa độ chính xác, thay vì dùng tọa độ cứng từ profile. Nếu không tìm được marker → fallback về vị trí mặc định theo %.

---

## 2. Đầu vào (Input)

| Tham số | Kiểu | Nguồn |
|---------|------|-------|
| `img_w`, `img_h` | `int` | 1000, 1400 |
| `markers` | `list[dict]` | Bước 6 |
| `rows_per_block` | `int` | profile |
| `sid_roi_cfg` | `dict` hoặc `None` | profile (nếu có) |
| `code_roi_cfg` | `dict` hoặc `None` | profile (nếu có) |
| `mcq_roi_cfg` | `dict` hoặc `None` | profile (nếu có) |
| `block_count_hint` | `Optional[int]` | profile |

`sid_roi_cfg / code_roi_cfg / mcq_roi_cfg`: nếu profile cấu hình sẵn vị trí ROI (pixel hoặc tỷ lệ 0–1), hệ thống dùng ngay, bỏ qua bước detect marker cho vùng đó.

---

## 3. Khái niệm: "điểm neo" (anchor) là gì?

Anchor là **tọa độ pixel (cx, cy)** của 1 marker trên ảnh, được dùng làm mốc để tính ROI.

Bước 8 cần 8 anchor, tương ứng 4 vùng góc của 3 khu vực:

```
Ảnh 1000×1400:

  ┌────────────────────────────────────────┐
  │  [SID top anchor]   [Code top anchor]  │  ← ~4–5% từ trên
  │  ┌──────────┐       ┌──────┐           │
  │  │   MSSV   │       │ Mã đề│           │
  │  └──────────┘       └──────┘           │
  │  [SID bot anchor]   [Code bot anchor]  │  ← ~25–26% từ trên
  │                                        │
  │  [MCQ left-top]─────────[MCQ right-top]│  ← ~57% (short) hoặc 30% (long)
  │  │                                   │ │
  │  │        Vùng MCQ                   │ │
  │  │                                   │ │
  │  [MCQ left-bot]─────────[MCQ right-bot]│  ← ~88–94% từ trên
  └────────────────────────────────────────┘
```

Mỗi anchor được xác định theo thứ tự ưu tiên:
1. **Profile ROI config** → tính anchor từ config (bỏ qua detect)
2. **Marker thực tế** → tìm marker gần vị trí mong đợi nhất
3. **Fallback %** → dùng vị trí mặc định (% của ảnh)

---

## 4. 8a — Vị trí mong đợi mặc định (`_default_anchor_percent`)

Trước khi tìm marker, hệ thống tính **vị trí % mong đợi** của 8 anchor dựa trên `long_form_mode`:

```python
def _default_anchor_percent(rows_per_block, block_count_hint):
    long_form_mode = (rows_per_block >= 25 or block_count_hint >= 4)

    if long_form_mode:
        mcq_top  = 0.30     # MCQ bắt đầu từ 30% chiều cao
        line_h   = 0.0220   # khoảng dòng = 2.2% chiều cao ≈ 31px
        mcq_left = 0.16
        mcq_right= 0.94
    else:
        mcq_top  = 0.575    # MCQ bắt đầu từ 57.5% chiều cao
        line_h   = 0.0152   # khoảng dòng = 1.52% chiều cao ≈ 21px
        mcq_left = 0.20
        mcq_right= 0.77

    mcq_bottom = mcq_top + (rows - 1) * line_h
```

Vị trí cố định (không phụ thuộc long/short form):

| Anchor | X% | Y% |
|--------|----|----|
| `sid_top` | 57% | 4% |
| `sid_bottom` | 57% | 26% |
| `code_top` | 80% | 4% |
| `code_bottom` | 80% | 25% |

**Tại sao cần vị trí % mặc định?** Đây là "bản đồ chỗ ngồi dự kiến" — nếu không tìm được marker thật, ít nhất hệ thống có chỗ để fallback. Nếu profile cung cấp `sid_roi_cfg`, vị trí % này được ghi đè bằng tọa độ tính từ config.

---

## 5. 8b — Tìm anchor từ marker (`_resolve_coordinate_anchors`)

### 5.1 Override vị trí mong đợi từ profile config

Nếu profile có `sid_roi_cfg`, `code_roi_cfg`, `mcq_roi_cfg`, hệ thống tính lại vị trí % mong đợi từ đó:

```python
if sid_roi_cfg is not None:
    expected["sid_top"] = (
        (sid_roi_cfg["x"] + sid_roi_cfg["w"] * 0.52) / img_w,  # X: giữa vùng SID
        (sid_roi_cfg["y"] + sid_roi_cfg["h"] * 0.05) / img_h,  # Y: 5% từ trên ROI
    )
    expected["sid_bottom"] = (
        (sid_roi_cfg["x"] + sid_roi_cfg["w"] * 0.52) / img_w,
        (sid_roi_cfg["y"] + sid_roi_cfg["h"] * 0.95) / img_h,  # Y: 95% từ trên ROI
    )
```

Ý nghĩa: anchor top/bottom của SID nằm ở cùng X (giữa vùng SID), lần lượt ở 5% và 95% chiều cao ROI.

### 5.2 Lọc marker thành 2 nhóm

```python
# Nhóm 1: square_markers — marker hình vuông rõ nét
square_markers = [m for m in markers
    if area >= 40 and size >= 6
    and 0.72 <= aspect <= 1.40
    and 4 <= vertices <= 6
    and fill >= 0.58]

if len(square_markers) >= 12:
    candidate_markers = square_markers   # dùng nhóm chất lượng cao nếu đủ nhiều

# Nhóm 2: fiducial_markers — marker đen đặc (ưu tiên cho SID/code)
fiducial_markers = [m for m in candidates
    if fill >= 0.86 and size >= 11 and circularity <= 0.90]
```

Lý do có 2 nhóm: anchor SID/code cần marker chất lượng cao (fiducial). Anchor MCQ chấp nhận marker kém hơn (bubble rows).

### 5.3 Tìm anchor SID và code: `_pick_anchor_marker`

```python
for name in ("sid_top", "sid_bottom", "code_top", "code_bottom"):
    target = expected[name]                        # (x%, y%) mong đợi
    max_dist = 0.085 if "sid" else 0.075           # bán kính tìm kiếm (% đường chéo)

    picked = _pick_anchor_marker(fiducial_markers, target, ...)
    if picked is None:
        picked = _pick_anchor_marker(candidate_markers, target, ...)  # thử rộng hơn
    if picked is None:
        fallback_used = True
        anchors[name] = (target[0]*img_w, target[1]*img_h)  # dùng vị trí % mặc định
    else:
        anchors[name] = (picked_cx, picked_cy)
```

**Bên trong `_pick_anchor_marker`:**

```python
def _pick_anchor_marker(markers, target_xy, img_w, img_h, max_norm_dist=0.15):
    tx = target_xy[0] * img_w   # tọa độ pixel mong đợi
    ty = target_xy[1] * img_h
    diag = hypot(img_w, img_h)  # đường chéo ảnh ≈ 1720px

    for marker in markers:
        dist = hypot(cx-tx, cy-ty) / diag      # khoảng cách chuẩn hóa
        if dist > max_norm_dist: continue        # loại marker quá xa

        score = (1-dist)*2.0 + area_ratio*650 + fill*0.12
        # ưu tiên: gần > to > đặc
```

Công thức điểm: marker **gần** được ưu tiên nhất (hệ số 2), **lớn** (area_ratio×650) quan trọng thứ hai, **đặc** (fill×0.12) ít quan trọng nhất. Kết quả: chọn marker hình vuông đen to nhất gần vị trí mong đợi.

**Bán kính tìm kiếm `max_norm_dist`:**

- SID anchor: 8.5% đường chéo ≈ 146px
- Code anchor: 7.5% đường chéo ≈ 129px
- MCQ anchor (fallback): 9% đường chéo ≈ 155px

Nếu không có marker nào trong bán kính đó → fallback về vị trí %.

### 5.4 Tìm anchor MCQ: theo hàng (row-based)

Anchor MCQ phức tạp hơn vì phải tìm **hàng trên cùng** và **hàng dưới cùng** của vùng MCQ, không phải marker đơn lẻ:

```python
mcq_rows = _cluster_marker_rows(
    candidate_markers,
    min_x=img_w * 0.12,
    max_x=img_w * 0.90,
    min_y=img_h * (0.30 if long_form else 0.46),
    max_y=img_h * 0.96,
    y_tol=6.0,
)

# Chỉ giữ hàng đủ dày đặc và rải rộng
mcq_rows = [row for row in mcq_rows
    if row["count"] >= 4                                    # ≥ 4 marker trong hàng
    and (row["x_max"] - row["x_min"]) >= img_w * 0.48]    # span ≥ 48% chiều rộng
```

**Chọn top_row:** hàng có Y gần `exp_top_y` nhất:

```python
top_row = min(mcq_rows, key=lambda r: abs(r["cy"] - exp_top_y))
```

**Chọn bottom_row:** hàng nằm dưới top_row ít nhất 70% khoảng span mong đợi, rồi gần `exp_top_y + exp_span` nhất:

```python
bottom_candidates = [r for r in mcq_rows
    if r["cy"] > top_row["cy"] + 0.70 * exp_span]

target_bottom = top_row["cy"] + exp_span
bottom_row = min(bottom_candidates, key=lambda r: abs(r["cy"] - target_bottom))
```

**Tại sao phải > 70% exp_span?** Để tránh chọn nhầm 1 hàng bong bóng giữa vùng MCQ làm bottom_row — bottom thật phải nằm gần cuối vùng MCQ, không phải giữa.

**Gán anchor tùy chế độ:**

```python
if top_row and bottom_row:
    if long_form_mode:
        # Long form: dùng X từ profile %, Y từ marker thực tế
        anchors["mcq_left_top"]    = (exp_left_x,          top_row["cy"])
        anchors["mcq_right_top"]   = (exp_right_x,         top_row["cy"])
        anchors["mcq_left_bottom"] = (exp_left_x,          bottom_row["cy"])
        anchors["mcq_right_bottom"]= (exp_right_x,         bottom_row["cy"])
    else:
        # Short form: dùng cả X và Y từ marker thực tế
        anchors["mcq_left_top"]    = (top_row["x_min"],    top_row["cy"])
        anchors["mcq_right_top"]   = (top_row["x_max"],    top_row["cy"])
        anchors["mcq_left_bottom"] = (bottom_row["x_min"], bottom_row["cy"])
        anchors["mcq_right_bottom"]= (bottom_row["x_max"], bottom_row["cy"])
```

**Tại sao long_form dùng X từ profile?** Phiếu long_form có nhiều block trải rộng hơn, marker ở hàng trên/dưới có thể không đại diện cho toàn bộ bề rộng. Profile % đáng tin hơn marker đơn lẻ cho trường hợp này.

---

## 6. 8c — Xây ROI từ anchor (`_build_rois_from_anchors`)

### 6.1 SID ROI

```python
sid_anchor_top_y    = min(sid_top[1], sid_bottom[1])
sid_anchor_bottom_y = max(sid_top[1], sid_bottom[1])
sid_span = max(120.0, abs(sid_bottom_y - sid_top_y))   # tối thiểu 120px cao
sid_digit_diam = max(10.0, sid_span / 10.0)            # đường kính 1 chữ số ≈ span/10
```

**Tính X và W:**

```python
sid_x = sid_top[0] + img_w * 0.004      # lùi vào trong 0.4% so với anchor
sid_w = img_w * 0.20                     # rộng 20% ảnh

# Inset ngang (thu hẹp để loại border):
sid_horizontal_inset = max(7.0, 0.25 * sid_digit_diam)
if sid_w - 2*inset >= 120:
    sid_x += inset
    sid_w -= 2 * inset
```

**Tính Y và H:**

```python
sid_vertical_inset = max(4.0, 0.18 * sid_digit_diam)
sid_y = sid_anchor_top_y + sid_vertical_inset    # lùi vào trong từ anchor top
sid_h = sid_anchor_bottom_y - sid_vertical_inset - sid_y
```

Inset dọc và ngang giúp ROI không ôm sát marker (tránh lấy nhầm border marker vào vùng đọc số).

### 6.2 Code ROI

```python
code_span = max(120.0, abs(code_bottom[1] - code_top[1]))
code_w = img_w * 0.10                    # chỉ 10% chiều rộng (mã đề hẹp hơn SID)
code_h = max(200.0, code_span + img_h * 0.010)
code_x = code_top[0] + img_w * 0.008
code_y = min(code_top[1], code_bottom[1]) + img_h * 0.006
```

### 6.3 Giải quyết chồng lấn SID vs Code

```python
min_gap = max(8, int(img_w * 0.010))    # khoảng cách tối thiểu = 10px
sid_right = sid_roi["x"] + sid_roi["w"]
code_left = code_roi["x"]

if sid_right + min_gap > code_left:     # 2 vùng bị chồng lên nhau
    # Thử thu hẹp SID trước
    target_sid_w = code_left - min_gap - sid_roi["x"]
    if target_sid_w >= 120:
        sid_roi["w"] = target_sid_w
    else:
        # Dịch code sang phải
        code_roi["x"] = sid_right + min_gap
        sid_roi["w"] = max(120, code_roi["x"] - min_gap - sid_roi["x"])
```

Ưu tiên thu hẹp SID trước vì mã đề hẹp hơn, dịch code sang phải ít ảnh hưởng hơn.

### 6.4 MCQ ROI và line_h

**Tính span và line_h từ 4 anchor MCQ:**

```python
top_y    = 0.5 * (left_top[1]    + right_top[1])     # trung bình Y của 2 anchor trên
bottom_y = 0.5 * (left_bottom[1] + right_bottom[1])  # trung bình Y của 2 anchor dưới
span_y   = max(12.0, bottom_y - top_y)
line_h   = span_y / (rows_per_block - 1)
```

Dùng trung bình 2 anchor cùng phía (trái+phải) để line_h không bị lệch nếu 1 anchor bị lệch.

**Validation — kiểm tra và sửa nếu anchor bất hợp lý:**

```python
# Kiểm tra left_x hợp lệ
if not (img_w*0.08 <= left_x <= img_w*0.35):
    left_x = default_left    # fallback về %

# Kiểm tra span hợp lý (so với kỳ vọng từ profile)
expected_span = (rows_per_block - 1) * default_line_h
if measured_span < 0.80*expected_span or measured_span > 1.10*expected_span:
    bottom_y = top_y + expected_span   # sửa bottom_y
```

**Xây MCQ ROI với padding:**

```python
mcq_x = left_x  - 0.40 * line_h   # mở rộng trái  0.4 dòng
mcq_w = (right_x - left_x) + 0.80 * line_h  # mở rộng hai bên
mcq_y = top_y   - 0.45 * line_h   # mở rộng trên  0.45 dòng
mcq_h = span_y  + 0.90 * line_h   # mở rộng dưới  0.45 dòng
```

Padding 0.4–0.45 × `line_h` đảm bảo ROI bao trọn bong bóng hàng đầu và hàng cuối kể cả khi tọa độ anchor lệch 1–2px.

**Clamp line_h:**

```python
line_h = max(12.0, min(44.0, line_h))   # short_form: [12, 44] px
line_h = max(12.0, min(60.0, line_h))   # long_form:  [12, 60] px
```

---

## 7. Đầu ra (Output)

### `_resolve_coordinate_anchors` trả về:

```python
anchors = {
    "sid_top":          (cx, cy),   # pixel
    "sid_bottom":       (cx, cy),
    "code_top":         (cx, cy),
    "code_bottom":      (cx, cy),
    "mcq_left_top":     (cx, cy),
    "mcq_right_top":    (cx, cy),
    "mcq_left_bottom":  (cx, cy),
    "mcq_right_bottom": (cx, cy),
}
fallback_used = True / False        # True nếu có anchor nào phải dùng fallback %
```

### `_build_rois_from_anchors` trả về:

```python
rois = {
    "sid":  {"x": 85,  "y": 56,  "w": 192, "h": 315},   # vùng MSSV
    "code": {"x": 812, "y": 56,  "w": 95,  "h": 326},   # vùng mã đề
    "mcq":  {"x": 75,  "y": 602, "w": 855, "h": 658},   # vùng MCQ
}
line_h = 21.4   # khoảng cách dòng MCQ (pixel)
```

`line_h` này là giá trị cuối cùng dùng cho Bước 12 (decode MCQ).

---

## 8. Sơ đồ luồng

```
markers + profile config
        │
        ▼
  8a — _default_anchor_percent()
        Tính 8 vị trí % mong đợi theo long/short form
        Nếu có profile ROI cfg → override vị trí % tương ứng
        │
        ▼
  8b — _resolve_coordinate_anchors()
        │
        ├─ Lọc markers → square_markers, fiducial_markers
        │
        ├─ SID top/bottom, Code top/bottom:
        │    _pick_anchor_marker(fiducial) → tìm marker gần nhất
        │    nếu None → _pick_anchor_marker(candidates)
        │    nếu vẫn None → fallback_used=True, dùng vị trí %
        │
        └─ MCQ 4 corners:
             _cluster_marker_rows() → lọc hàng (count≥4, span≥48%)
             top_row = hàng gần exp_top_y nhất
             bottom_row = hàng dưới 70% span, gần exp_bottom_y nhất
             short_form → dùng x_min/x_max thực từ hàng
             long_form  → dùng X từ profile %, Y từ hàng
             Không tìm được → _pick_anchor_marker mỗi góc
        │
        ▼
  anchors dict (8 điểm pixel) + fallback_used
        │
        ▼
  8c — _build_rois_from_anchors()
        │
        ├─ SID ROI: anchor top/bottom → span → x,y,w,h với inset
        │
        ├─ Code ROI: anchor top/bottom → x,y,w,h
        │
        ├─ Overlap check: SID right > Code left → thu hẹp hoặc dịch
        │
        └─ MCQ ROI:
             top_y = mean(left_top.y, right_top.y)
             bottom_y = mean(left_bottom.y, right_bottom.y)
             line_h = span_y / (rows-1)
             Validate: check left_x, right_x, span hợp lý
             Fallback từng phần nếu bất hợp lý
             x,y,w,h với padding 0.4–0.45 × line_h
             Clamp line_h [12,44] hoặc [12,60]
        │
        ▼
  rois = {"sid":..., "code":..., "mcq":...}
  line_h (float)
        │
        ├──→ Bước 9: template matching tinh chỉnh MCQ ROI
        ├──→ Bước 10: giải mã MSSV trong sid_roi
        ├──→ Bước 11: giải mã mã đề trong code_roi
        └──→ Bước 12: giải mã MCQ trong mcq_roi với line_h
```

---

## 9. Bảng tóm tắt ngưỡng

| Tham số | Short form | Long form | Ý nghĩa |
|---------|-----------|----------|---------|
| MCQ top Y | 57.5% | 30.0% | Vị trí % mặc định hàng đầu MCQ |
| line_h default | 1.52% ≈ 21px | 2.20% ≈ 31px | Khoảng dòng mặc định |
| MCQ left X | 20% | 16% | Biên trái mặc định MCQ |
| MCQ right X | 77% | 94% | Biên phải mặc định MCQ |
| line_h clamp | [12, 44] px | [12, 60] px | Giới hạn an toàn |
| SID bán kính tìm | 8.5% đường chéo | ← | ≈ 146px |
| Code bán kính tìm | 7.5% đường chéo | ← | ≈ 129px |
| MCQ span hợp lệ | [80%, 110%] | [70%, 125%] | So với expected_span |
| MCQ padding | 0.40–0.45 × line_h | ← | Buffer xung quanh ROI |

---

## 10. Phụ lục — ROI họ tên (handwriting), một nhánh song song

Ngoài 3 ROI (SID, Code, MCQ) do `omr_layout.py` dựng như mô tả ở trên, hệ thống còn có thể dựng thêm **một ROI thứ 4, tùy chọn**: vùng chữ viết tay "họ tên" (`ho_ten`). ROI này **không thuộc luồng anchor/marker của Bước 8** — nó được xử lý bởi một module riêng, `be/app/services/omr/omr_handwriting.py`, và gọi song song ngay sau khi 3 ROI SID/Code/MCQ đã có (`omr_service.py:1287-1296`, hàm `_build_handwriting_rois()`).

### 10.1 Vì sao không dùng anchor/marker như SID/Code/MCQ?

Toàn bộ kỹ thuật 8a–8c ở trên (marker fiducial, `_pick_anchor_marker`, cluster hàng bong bóng...) đều dựa trên việc phiếu có **marker in sẵn** làm mốc. Vùng chữ viết tay không có marker nào cả — học sinh viết tự do trong một khung trống — nên không có "điểm neo" nào để tự động dò tìm bằng connected components hay template matching. ROI này bắt buộc phải đến từ nguồn khác.

### 10.2 Nguồn ROI theo thứ tự ưu tiên

```
1. Cấu hình profile: strategy.handwriting_fields.field_rois.ho_ten
   → Người quản trị tự vẽ 4 góc vùng họ tên trên giao diện OmrProfileRoiEditor
   → Lưu tọa độ tuyệt đối hoặc tỷ lệ 0–1 vào profile, dùng trực tiếp mỗi lần chấm

2. Suy luận heuristic (chỉ khi short-form, không có cấu hình profile):
   _infer_short_form_ho_ten_roi(sid_roi, mcq_roi, ...)
   → Đặt ROI ngay bên trái vùng SID ROI đã dựng ở mục 6.1,
     dựa trên quan sát: các mẫu phiếu A4 20/40/50 câu luôn đặt khung họ tên
     sát bên trái lưới MSSV.
   → Điều kiện: mcq_top phải < 42% chiều cao ảnh (tức short-form),
     nếu không thỏa hoặc là long-form → không suy luận, bỏ qua ROI này.

3. Không tìm được cả hai → không có ROI ho_ten, tính năng crop bị bỏ qua cho lần chấm đó.
```

### 10.3 Việc ROI này dùng để làm gì (Bước 15)

Sau khi có ROI, `_extract_handwriting_crops()` (cùng module `omr_handwriting.py`) cắt vùng ảnh tương ứng từ `img_std`, làm sạch viền bằng `_trim_ink_bounding_box()` (kết hợp Otsu + adaptive threshold để tìm bounding box sát nét mực), rồi lưu thành file `omr_hw_ho_ten_<run_tag>.jpg` trong `output_folder`.

Kết quả này **chỉ là ảnh crop** — trường `values.ho_ten` trong JSON đầu ra luôn là chuỗi rỗng `""`, và `ocr_engine` luôn ghi `"disabled"`. Hệ thống chưa chạy nhận dạng chữ viết tay tự động; người chấm bài xem trực tiếp ảnh crop trên giao diện (`StatsPanel`) để tự đọc tên. Đây là tính năng dở dang có chủ đích — chuẩn bị hạ tầng ROI/crop để làm nền cho việc tích hợp mô hình OCR/học sâu ở phiên bản sau, không phải code thừa bị bỏ quên.

### 10.4 Vị trí code

| Thành phần | File | Vị trí |
|---|---|---|
| Parse cấu hình + xây ROI | `omr_handwriting.py` | `_parse_handwriting_config()`, `_build_handwriting_rois()` |
| Suy luận ROI short-form | `omr_handwriting.py` | `_infer_short_form_ho_ten_roi()` |
| Cắt ảnh + lưu file | `omr_handwriting.py` | `_extract_handwriting_crops()`, `_trim_ink_bounding_box()` |
| Gọi trong pipeline chính | `omr_service.py` | dòng 1287–1296 (xây ROI), dòng 1901–1909 (cắt + lưu ảnh) |
| Vẽ ROI trên UI cấu hình | `fe/.../OmrProfileRoiEditor.tsx` | — |
| Hiển thị ảnh crop cho người chấm | `fe/.../StatsPanel.tsx` | — |
