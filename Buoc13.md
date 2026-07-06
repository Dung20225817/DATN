# Bước 13 — MCQ Map Search Rescue

> **Vị trí trong pipeline:** Chạy ngay sau Bước 12 (decode MCQ lần đầu). Nếu kết quả lần đầu có quá nhiều câu uncertain, bước này thử lại 54 cấu hình lưới khác nhau để tìm cấu hình cho kết quả tốt hơn.

**Module:** `be/app/services/omr/omr_service.py` (dòng 1641–1777)
**Hàm gọi lại:** `_decode_mcq_with_map()` (từ `omr_mcq.py`)

---

## 1. Tại sao cần bước này?

Bước 12 decode MCQ dựa trên `line_h` và `top_center_y` tính từ marker. Nhưng marker có thể bị phát hiện sai vị trí vài pixel, dẫn đến:

- `line_h` hơi sai → lưới ngày càng lệch theo chiều dọc, câu cuối block bị lệch nhiều nhất
- `top_center_y` hơi sai → toàn bộ lưới dịch lên/xuống

Khi một trong hai sai, nhiều câu bị đọc ở vùng trống giữa 2 dòng → không có tín hiệu tô nào đạt ngưỡng → uncertain hàng loạt.

Rescue không cố sửa marker hay tính lại từ đầu. Thay vào đó: **thử nhiều bộ (`line_h`, `top_shift`) khác nhau, xem bộ nào decode ra ít uncertain nhất, rồi dùng bộ đó**.

---

## 2. Điều kiện kích hoạt

```python
search_uncertain_gate = max(4, round(0.40 × questions))

kích hoạt khi:
  initial_uncertain >= search_uncertain_gate
  AND mcq_rescue_disabled = False
  AND short_form_bands_locked = False
```

| `questions` | gate |
|-------------|------|
| 20 | 8 |
| 40 | 16 |
| 80 | 32 |
| 120 | 48 |

**40% uncertain** mới kích hoạt — ngưỡng cao đảm bảo không Rescue khi phiếu thực sự tô mờ (5–10 câu uncertain là bình thường).

**Các trường hợp không chạy Rescue:**

| Lý do | `reason` |
|-------|---------|
| uncertain < gate | `"below-uncertain-gate"` |
| Profile tắt Rescue | `"disabled-by-profile"` |
| Long-form mode (Rescue bị tắt tự động) | `"disabled-by-long-form"` |
| Short-form với block_bands cố định từ marker | `"locked-by-short-form-block-bands"` |

---

## 3. Baseline (xuất phát điểm)

Trước khi thử, ghi lại kết quả baseline từ Bước 12:

```python
baseline_uncertain = len(mcq_result["uncertain_questions"])
baseline_double    = len(mcq_result["double_mark_questions"])
baseline_quality   = _mcq_quality(mcq_result)

baseline_rank = (baseline_uncertain, baseline_double, −baseline_quality)
```

**`_mcq_quality()`** — điểm chất lượng tổng hợp của toàn bộ kết quả:

```python
quality = confident_count × 10      ← số câu có đáp án rõ
        + first4_confident × 6      ← bonus câu 1–4 (thường bị ảnh hưởng nhất khi lưới lệch)
        + sum(margins) × 70         ← tổng margin tất cả câu
```

Rank được so sánh **lexicographic** (từ điển): ưu tiên ít uncertain nhất → ít double mark nhất → quality cao nhất.

---

## 4. Không gian tìm kiếm: 54 ứng viên

```python
line_scales      = [1.00, 0.92, 0.88, 1.08, 0.84, 1.16]   # 6 tỷ lệ
shift_multipliers = [0.0, -0.5, 0.5, -1.0, 1.0, -1.5, 1.5, -2.0, 2.0]  # 9 độ dịch

tổng: 6 × 9 = 54 lần decode
```

Với mỗi cặp `(scale, shift_mul)`:

```python
cand_line_h   = clamp(line_h × scale, 6.0, 44.0)
cand_shift_px = shift_mul × cand_line_h

# Gọi lại decode với tham số mới
cand_result = _decode_mcq_with_map(...,
    line_h=cand_line_h,
    top_shift_px=cand_shift_px,   # dịch toàn bộ lưới lên/xuống
    # left_x, right_x, top_center_y, block_bands: giữ nguyên
)
```

**Ý nghĩa vật lý của `shift_mul`:**

| `shift_mul` | `cand_shift_px` (với `line_h=21`) | Ý nghĩa |
|-------------|-----------------------------------|---------|
| 0.0 | 0 px | Giữ nguyên vị trí baseline |
| −0.5 | −10.5 px | Dịch lưới lên nửa dòng |
| +0.5 | +10.5 px | Dịch lưới xuống nửa dòng |
| −1.0 | −21 px | Dịch lưới lên 1 dòng |
| +2.0 | +42 px | Dịch lưới xuống 2 dòng |

**Ý nghĩa vật lý của `scale`:**

| `scale` | Tác dụng |
|---------|----------|
| 1.00 | Giữ nguyên `line_h` |
| 0.92 | Nén lưới 8% — các dòng gần nhau hơn |
| 1.08 | Giãn lưới 8% — các dòng xa nhau hơn |
| 0.84 | Nén mạnh 16% |
| 1.16 | Giãn mạnh 16% |

Kết hợp scale + shift bao phủ được cả hai loại lỗi:
- Lỗi `line_h` đơn thuần: scale ≠ 1.0 sẽ tìm được
- Lỗi `top_center_y` đơn thuần: shift ≠ 0 sẽ tìm được
- Cả hai kết hợp: kết hợp scale + shift tìm được

---

## 5. Xếp hạng và chọn ứng viên tốt nhất

Với mỗi ứng viên:

```python
cand_rank = (cand_uncertain, cand_double, −cand_quality)
if cand_rank < best_rank:
    best_rank = cand_rank
    best_result = cand_result
    best_line_h = cand_line_h
    best_top_shift_px = cand_shift_px
```

So sánh tuple theo từ điển: `(3, 1, −5.2) < (5, 0, −6.1)` vì 3 < 5 ở phần tử đầu.

---

## 6. Tiêu chí chấp nhận bảo thủ

Dù tìm được ứng viên "tốt hơn" về rank, chỉ thay baseline khi đáp ứng **ít nhất một** trong 3 điều kiện:

```python
uncertain_gain = baseline_uncertain - best_uncertain

if   uncertain_gain >= 2:                                  # giảm ≥ 2 uncertain
    improved = True
elif uncertain_gain >= 1 and best_double <= baseline_double:  # giảm 1 uncertain, double không tăng
    improved = True
elif uncertain_gain == 0 and best_double < baseline_double:   # uncertain bằng nhau, double giảm
    improved = True
```

**Tại sao cần điều kiện bảo thủ này?**

Rank tuple có thể "tốt hơn" chỉ vì `quality` tăng nhẹ (phần tử thứ 3), dù uncertain không giảm. Trường hợp đó không cần điều chỉnh — phiếu có thể thực sự tô mờ, không phải do lỗi lưới. Điều kiện bảo thủ ngăn hệ thống chỉnh lưới không cần thiết, vốn có thể làm sai câu đang đúng.

---

## 7. Áp dụng kết quả Rescue

Nếu `improved = True`:

```python
mcq_result    = best_result
line_h        = best_line_h
top_center_y  = top_center_y + best_top_shift_px   # cập nhật tọa độ thực tế

# Clamp top_center_y vào trong ROI (tránh ra ngoài biên)
top_guard = max(3.0, 0.35 × line_h)
top_center_y = clamp(top_center_y, mcq_roi.y + top_guard, mcq_roi.y + mcq_roi.h - top_guard)

warning_codes.append("MCQ_MAP_SEARCH_RESCUE")
```

`line_h` và `top_center_y` mới được dùng xuyên suốt cho Bước 14 (Drift detection) nếu cần chạy tiếp.

---

## 8. Đầu vào và đầu ra

### Đầu vào

| | Nguồn |
|-|-------|
| `mcq_result` từ Bước 12 | baseline decode |
| `line_h`, `top_center_y` | từ Bước 9 (tinh chỉnh) |
| `mcq_roi`, `left_x`, `right_x` | từ Bước 8 |
| `mcq_block_bands` | từ Bước 7 (block bands) |
| `questions`, `choices`, `rows_per_block`, `block_count` | từ profile |
| `mcq_cfg` | cấu hình decode |

### Đầu ra

| Trường hợp | Kết quả |
|------------|---------|
| Rescue thành công | `mcq_result` mới, `line_h` mới, `top_center_y` mới, cờ `MCQ_MAP_SEARCH_RESCUE` |
| Rescue không cải thiện | `mcq_result` giữ nguyên, không có cờ |
| Rescue không chạy | `mcq_result` giữ nguyên, không có cờ |

**`mcq_map_search_meta`** — metadata ghi vào JSON debug:

```json
{
  "used": true,
  "reason": "improved",
  "tested": 54,
  "initial_uncertain": 20,
  "final_uncertain": 3,
  "line_h_before": 21.4,
  "line_h_after": 18.7,
  "top_center_y_before": 615.0,
  "top_center_y_after": 604.5,
  "top_shift_px": -10.5,
  "block_bands_used": true
}
```

---

## 9. Sơ đồ luồng

```
Bước 12: mcq_result (baseline)
        │
        ▼
initial_uncertain ≥ gate (40% × Q)?
        │ Không → giữ nguyên, không làm gì
        │ Có
        ▼
Tính baseline_rank = (uncertain, double_mark, −quality)

Duyệt 6 line_scales × 9 shift_mults = 54 ứng viên:
  mỗi ứng viên: _decode_mcq_with_map(cand_line_h, cand_shift_px)
  → tính cand_rank
  → nếu cand_rank < best_rank: cập nhật best
        │
        ▼
best_rank < baseline_rank?
  Không → mcq_result giữ nguyên
  Có → kiểm tra điều kiện bảo thủ:
        uncertain_gain ≥ 2?       → improved
        uncertain_gain ≥ 1 và double không tăng? → improved
        double giảm?              → improved
        │ Không → mcq_result giữ nguyên
        │ Có
        ▼
mcq_result = best_result
line_h = best_line_h
top_center_y += best_top_shift_px
→ ghi cờ MCQ_MAP_SEARCH_RESCUE
        │
        ▼
Bước 14: Drift detection (dùng mcq_result và line_h mới)
```

---

## 10. Ví dụ minh họa

Giả sử phiếu 40 câu, `line_h` thực = 21px nhưng tính ra 23px (sai 2px):

```
Baseline: line_h=23, top_center_y=615, top_shift=0
→ Câu 1: cy=615, câu 20: cy=615+19×23=1052
→ Thực tế câu 20 ở 615+19×21=1014 → lệch 38px → nằm giữa 2 dòng → uncertain

Rescue thử scale=0.92: cand_line_h = 23×0.92 = 21.16 ≈ đúng
→ Câu 20: cy=615+19×21.16=1017 → gần đúng → đọc được
→ uncertain giảm từ 20 → 2
→ uncertain_gain = 18 ≥ 2 → improved = True
→ line_h cập nhật = 21.16, ghi MCQ_MAP_SEARCH_RESCUE
```
