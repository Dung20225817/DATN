# Bước 12 — Giải mã câu MCQ

> **Vị trí trong pipeline:** Chạy sau Bước 9 (tinh chỉnh ROI), song song với Bước 10 và 11. Đây là bước phức tạp nhất — duyệt từng câu hỏi, định vị ô bong bóng, tính điểm, và quyết định đáp án hoặc "uncertain".

**Module:** `be/app/services/omr/omr_mcq.py`
**Hàm chính:** `_decode_mcq_with_map()`

---

## 1. Tại sao cần bước này?

Sau tất cả các bước định vị trước đó, hệ thống đã có:
- `mcq_roi`: hình chữ nhật bao vùng MCQ
- `top_center_y`: tọa độ Y tâm dòng câu 1
- `line_h`: khoảng cách giữa 2 dòng (pixel)
- `left_x`, `right_x`: biên ngang MCQ
- `block_bands`: vị trí từng block theo chiều ngang

Bước 12 dùng tất cả thông số trên để **đọc từng câu một** — tìm đúng ô bong bóng của câu đó trong ảnh và xác định học sinh tô ô nào.

---

## 2. Đầu vào (Input)

| Tham số | Kiểu | Giá trị tiêu biểu |
|---------|------|------------------|
| `gray_img` | `np.ndarray` | ảnh xám 1000×1400 |
| `binary_inv` | `np.ndarray` | ảnh nhị phân 1000×1400 |
| `roi` | `dict` | `{"x":75,"y":602,"w":855,"h":658}` |
| `questions` | `int` | 40 (tổng số câu) |
| `choices` | `int` | 4 (A/B/C/D) hoặc 5 (A–E) |
| `rows_per_block` | `int` | 20 |
| `block_count` | `int` | 2 |
| `left_x`, `right_x` | `float` | biên ngang vùng bubble |
| `top_center_y` | `float` | Y tâm dòng câu 1 |
| `line_h` | `float` | ~21.0 px |
| `decode_cfg` | `dict` | tham số ngưỡng từ profile |
| `top_shift_px` | `float` | 0.0 (dịch dọc thêm từ Rescue) |
| `block_bands` | `list` | vị trí X từng block |

---

## 3. 12a — Xác định block và dòng của mỗi câu

Với Q câu và `rows_per_block` câu/block:

```python
for q_idx in range(total_questions):
    block_idx = q_idx // rows_per_block   # block nào (0, 1, 2...)
    row_idx   = q_idx % rows_per_block    # dòng thứ mấy trong block (0–19)
```

Ví dụ: `questions=40`, `rows_per_block=20`, `block_count=2`:
- Câu 1–20: `block_idx=0`, `row_idx=0–19`
- Câu 21–40: `block_idx=1`, `row_idx=0–19`

Nếu `block_idx >= block_count` (số câu vượt quá số block): ghi `selected=-1` (uncertain), không đọc.

---

## 4. 12b — Áp quy luật lưới vào từng câu cụ thể

Các bước trước (7–9) tìm ra **tham số chung** của lưới: dòng đầu ở đâu (`top_center_y`), mỗi dòng cách nhau bao nhiêu (`line_h`). Bước này **dùng tham số đó để tính pixel Y của riêng từng câu**:

```python
cy = top_center_y + row_idx × line_h + top_shift_px + row_offsets[row_idx]
```

Ví dụ: `top_center_y=615`, `line_h=21`:
- Câu 1 (`row_idx=0`): `cy = 615`
- Câu 5 (`row_idx=4`): `cy = 615 + 4×21 = 699`
- Câu 20 (`row_idx=19`): `cy = 615 + 19×21 = 1014`

Hai thành phần bổ sung:
- `top_shift_px`: offset toàn cục dịch toàn bộ lưới lên/xuống (chỉ khác 0 khi Rescue hoặc Drift fix chỉnh lại)
- `row_offsets[row_idx]`: offset riêng từng dòng từ profile (thường toàn 0)

**Cửa sổ lấy mẫu** quanh `cy` (88% khoảng cách dòng — tránh lấy nhầm pixel dòng kề):

```
y1 = cy − 0.44 × line_h
y2 = cy + 0.44 × line_h
```

---

## 5. 12c — Xác định vùng X của từng block và từng lựa chọn

Mỗi block chiếm một dải X (`band_left` → `band_right`). Từ dải đó chia đều thành `choices` ô:

```python
# Inset: thu vào trong một chút để tránh đường kẻ
x_inset = 0.08  # nếu dùng custom block_bands
x_inset = 0.14  # nếu tính đều từ left_x/right_x

choice_w = (band_right - band_left) / choices

# Ô thứ c (c = 0,1,2,3 ~ A,B,C,D):
cx1 = band_left + c × choice_w + x_inset × choice_w
cx2 = band_left + (c+1) × choice_w - x_inset × choice_w
```

Ví dụ: `band_left=85`, `band_right=475`, `choices=4`, `x_inset=0.14`:
- `choice_w = 390 / 4 = 97.5 px`
- Ô A: `cx1=85 + 0×97.5 + 13.65 = 98.65`, `cx2=85 + 97.5 - 13.65 = 168.85`
- Ô B: `cx1=168.15`, `cx2=252.35`
- ...

---

## 6. 12d — Tính điểm mỗi ô bong bóng

Với mỗi ô `(cx1, y1, cx2, y2)`, trích 2 vùng con:

```
_extract_cell(..., inner_ratio=0.78) → cell_gray, cell_bin  (vùng 78% trung tâm)
_extract_cell(..., inner_ratio=0.56) → _,         center_bin (vùng 56% trung tâm)
```

Lọc nhiễu đường kẻ mảnh trên binary (`_filter_binary_cell_noise`):
```
Erode với kernel 2×2 → xóa vết mực mảnh
Loại connected component nhỏ hơn 1.2% diện tích ô
(center_bin dùng ngưỡng lỏng hơn: 0.65 × 1.2% = 0.78%)
```

Tính 4 chỉ số đo độ tô:

| Chỉ số | Công thức | Nguồn |
|--------|-----------|-------|
| `density` | `countNonZero(cell_bin) / area` | binary_inv |
| `center_density` | `countNonZero(center_bin) / area` | binary_inv, vùng 56% |
| `darkness` | `0.55×fill + 0.30×dark_mean + 0.15×dark_p25` | gray_img |
| `dark_p20` | `1 − percentile_20(gray) / 255` | gray_img |

**Điểm tổng hợp mỗi ô:**

```python
score = 0.90 × density + 0.10 × darkness
```

Trọng số nghiêng về `density` (binary) vì đơn giản và ổn định. `darkness` (gray) bù khi nhị phân hóa cắt mất tín hiệu mực nhạt.

> Chế độ `density_only_scoring=True` (debug): `score = density`, bỏ qua darkness.

---

## 7. 12e — Local grid search (chỉ long-form, block giữa)

Phiếu long-form (nhiều câu, nhiều block) thường bị lệch in nhẹ theo chiều ngang — mỗi block có thể lệch vài pixel so với tính toán. **Local grid search** tự động tìm offset tốt nhất cho từng block:

Chỉ kích hoạt khi `local_grid_search=True` và block là block giữa (`1 ≤ block_idx ≤ N-2`, không phải block đầu hoặc cuối).

Với mỗi câu trong block đó, thử **5 offset X × 3 offset Y = 15 vị trí** xung quanh vị trí tính toán:

```
X offsets: 0, ±0.25×shift_span, ±0.50×shift_span   (shift_span ≈ 0.3×choice_w)
Y offsets: 0, ±y_shift_step                           (y_shift_step ≈ 0.06×line_h)
```

Với mỗi vị trí thử, tính `quality` của cả hàng, chọn vị trí cho quality cao nhất (trừ penalty tỉ lệ với độ dịch chuyển).

**`block_anchor_shifts`:** tích lũy offset trung bình của block theo exponential moving average (momentum=0.20). Câu tiếp theo trong cùng block dùng offset tích lũy này làm điểm xuất phát — giúp bù lệch nhất quán theo chiều dọc phiếu.

---

## 8. 12f — Ngưỡng quyết định thích nghi (`dynamic_mark`)

Thay vì dùng ngưỡng tĩnh `min_mark` (mặc định 0.55) cho mọi câu:

```python
row_mean = mean(density_scores[A], [B], [C], [D])

dynamic_mark = max(min_mark × 0.95, min(0.60, row_mean + 0.06))
```

- Nếu `row_mean` thấp (tô nhạt, ảnh tối): `dynamic_mark` giảm theo → không bỏ sót câu tô nhạt
- Nếu `row_mean` cao (mực đậm): `dynamic_mark` tăng → tránh chấp nhận vết bẩn

Giới hạn: không thấp hơn `min_mark × 0.95` và không vượt 0.60.

---

## 9. 12g — Noise gate: lọc tín hiệu giả

Trước khi chấp nhận một lựa chọn, kiểm tra xem tín hiệu có đến từ **tô thật** không (không phải đường kẻ, bụi, vết gấp):

```python
noise_center_gate = min(0.92, max(noise_center_floor=0.42, row_center_mean + 0.10))
noise_dark_gate   = min(0.85, max(noise_dark_floor=0.30,   row_dark_mean   + 0.08))

noise_gate_ok =
    very_high_center (best_center ≥ 0.85)        ← tâm ô rất đặc → rõ ràng là tô
    OR strong_signal_bypass                        ← score ≥ 0.55 AND center ≥ 0.72
    OR (best_center ≥ noise_center_gate
        AND best_dark ≥ noise_dark_gate)           ← cả 2 gate đều vượt
```

**`center_density`**: fill_ratio của vùng 56% trung tâm ô. Đường kẻ, nhiễu thường nằm ở rìa; tô thật lấp đầy từ trong ra. Nếu chỉ có tín hiệu ở rìa mà tâm trống → nhiễu.

**`strong_signal_bypass`**: nếu `score ≥ 0.55` và `center ≥ 0.72` → tín hiệu đủ mạnh, bỏ qua `noise_dark_gate`. Ngăn việc dark gate quá chặt từ chối bong bóng tô rõ ràng.

Nếu `noise_gate_ok = False` và đã chọn được ô: hủy kết quả, ghi `noise_rejected=True`, câu trở thành uncertain.

---

## 10. 12h — Bảng quyết định mỗi câu

```
Trường hợp 1: best_score < dynamic_mark
    → Không có ô nào đủ đậm
    → Thử soft rescue:
       soft_threshold = max(soft_mark_floor=0.38, row_mean + 0.08)
       Nếu best_score ≥ soft_threshold
          AND margin ≥ 0.06, conf_ratio ≥ 1.18, quality ≥ 0.42
          AND noise_gate_ok
       → Chấp nhận (resolved_by_soft=True)
       Ngược lại → uncertain

Trường hợp 2: is_double_mark (≥ 2 ô đạt dynamic_mark)
    → Học sinh tô 2 đáp án (hoặc vết bẩn)
    → Nếu margin ≥ max(min_margin=0.05, double_mark_gap=0.08)
       AND conf_ratio ≥ min_conf_ratio
    → Chọn ô cao điểm nhất (resolved_by_highest=True)
    → Ngược lại → uncertain + ghi double_mark_questions

Trường hợp 3: đúng 1 ô đạt dynamic_mark
    → Nếu margin < min_margin AND conf_ratio < min_conf_ratio
    → uncertain (2 ô quá gần nhau)
    → Ngược lại → Chấp nhận

Sau mọi trường hợp: nếu selected ≥ 0 nhưng noise_gate_ok=False → uncertain
```

---

## 11. 12i — Quality score của hàng

Điểm tổng hợp chất lượng (`candidate_quality`) của cả hàng — dùng trong Bước 13 (Rescue) để so sánh các cấu hình lưới:

```python
quality = 1.10 × best_density_score
        + 1.45 × margin              ← margin quan trọng nhất: phân biệt rõ ô được tô
        + 0.25 × best_center         ← tín hiệu tâm ô
        + 0.12 × best_dark_p20       ← độ tối ảnh xám
        − left_penalty               ← phạt nếu A dẫn đầu với margin thấp
```

`left_penalty`: nếu lựa chọn A (cột trái nhất) dẫn đầu nhưng margin < 0.08:
```
left_penalty = 0.20 × (0.08 − margin)
```
Cột A hay bị ảnh hưởng bởi đường kẻ dọc trái của phiếu — phạt thêm nếu margin yếu.

---

## 12. Đầu ra (Output)

```python
{
    "user_answers": [2, 0, 3, 1, -1, ...],  # index 0=A,1=B,2=C,3=D; -1=uncertain
    "answer_confidences": [0.72, 0.85, ...], # density score của ô được chọn
    "uncertain_questions": [5, 12, 18],      # số thứ tự câu uncertain (1-indexed)
    "double_mark_questions": [7],             # câu tô 2 đáp án không giải quyết được
    "line_h": 21.3,                           # line_h thực dùng
    "left_x": 88.0,
    "right_x": 912.0,
    "top_center_y": 615.4,
    "rows": [                                 # chi tiết từng câu (dùng cho Rescue + debug)
        {
            "question": 1,
            "block": 1, "row": 1,
            "scores": [0.05, 0.73, 0.04, 0.06],  # density A,B,C,D
            "center_scores": [0.03, 0.68, 0.02, 0.04],
            "dark_p20_scores": [0.12, 0.81, 0.10, 0.13],
            "best_score": 0.73, "second_score": 0.06,
            "margin": 0.67,
            "selected": 1, "selected_label": "B",
            "threshold": 0.549, "soft_threshold": 0.457,
            "noise_center_gate": 0.42, "noise_dark_gate": 0.30,
            "noise_gate_passed": true,
            "uncertain": false, "double_mark": false,
            "resolved_by_highest": false, "resolved_by_soft": false,
            "noise_rejected": false,
            "grid_search_used": false,
            "local_x_shift_px": 0.0, "local_y_shift_px": 0.0,
            "candidate_quality": 1.88,
            "cell_boxes": [[88,603,145,628], ...],
            "cell_centroids": [[116.5, 615.0], ...],
            "selected_centroid": [252.3, 615.0]
        },
        ...
    ]
}
```

---

## 13. Sơ đồ luồng (mỗi câu)

```
q_idx → block_idx, row_idx
        │
        ▼
  12a — cy = top_center_y + row_idx×line_h + shifts
         y1 = cy − 0.44×line_h
         y2 = cy + 0.44×line_h
        │
        ▼
  12b — X band của block → chia thành choices ô
         choice_w = band_width / choices
         cx1, cx2 cho mỗi lựa chọn (với inset 8% hoặc 14%)
        │
        ▼
  12c — Mỗi ô:
         _extract_cell(inner_ratio=0.78) → cell_gray, cell_bin
         _extract_cell(inner_ratio=0.56) → center_bin
         _filter_binary_cell_noise(erode 2×2)
         density, center_density, darkness, dark_p20
         score = 0.90×density + 0.10×darkness
        │   [nếu local_grid_search: thử 15 vị trí, lấy quality cao nhất]
        ▼
  12d — Ngưỡng thích nghi:
         dynamic_mark = max(min_mark×0.95, min(0.60, row_mean+0.06))
        │
        ▼
  12e — Noise gate:
         noise_center_gate, noise_dark_gate
         noise_gate_ok?
        │
        ▼
  12f — Quyết định:
         best_score < dynamic_mark → soft rescue?  → selected hoặc -1
         double_mark?              → margin lớn?   → selected hoặc -1
         1 ô vượt ngưỡng           → margin ok?    → selected hoặc -1
         noise_gate fail           → -1
        │
        ▼
  user_answers[q_idx] = selected  (0–3 hoặc -1)
```

---

## 14. Các tham số cấu hình (decode_cfg)

| Tham số | Mặc định | Ý nghĩa |
|---------|---------|---------|
| `min_mark_density` | 0.55 | Ngưỡng tối thiểu `density` |
| `min_margin` | 0.05 | Margin tối thiểu nhất/nhì |
| `min_conf_ratio` | 1.10 | Tỷ lệ nhất/nhì tối thiểu |
| `double_mark_gap` | 0.08 | Margin để giải quyết double mark |
| `adaptive_threshold` | True | Dùng `dynamic_mark` hay ngưỡng tĩnh |
| `soft_mark_floor` | 0.38 | Ngưỡng tối thiểu cho soft rescue |
| `noise_center_floor` | 0.42 | Sàn noise center gate |
| `noise_dark_floor` | 0.30 | Sàn noise dark gate |
| `local_grid_search` | False (long-form: True) | Bật local grid search |
| `thin_noise_erode_iter` | 1 | Số lần erode lọc nhiễu mảnh |
