# Bước 14 — Phát hiện drift và mở rộng ROI tự động

> **Vị trí trong pipeline:** Chạy ngay sau Bước 13 (Map Search Rescue). Kiểm tra một loại lỗi mà Rescue không xử lý được: ROI MCQ bắt đầu quá thấp nên bỏ sót các câu đầu tiên hoàn toàn. Nếu phát hiện, tự động mở rộng ROI lên và decode lại.

**Module:** `be/app/services/omr/omr_service.py` (dòng 1780–1828)
**Hàm phát hiện:** `_detect_q5_start_drift()` trong `omr_mcq.py` (dòng 1713–1755)

---

## 1. Tại sao cần bước này — và tại sao Rescue (Bước 13) không đủ?

**Bước 13** điều chỉnh `line_h` và dịch (`top_shift_px`) lưới lên/xuống — nhưng vẫn **trong phạm vi ROI hiện tại**. Nếu ROI bắt đầu quá thấp, dù lưới được căn chỉnh hoàn hảo, các câu nằm phía trên ROI vẫn không được đọc.

**Kịch bản drift xảy ra:**

```
Thực tế phiếu:                    ROI hiện tại bắt đầu tại đây:
┌────────────────┐                         │
│ Câu 1 ← bị bỏ │  ← nằm ngoài ROI        │
│ Câu 2 ← bị bỏ │  ← nằm ngoài ROI        │
│ Câu 3 ← bị bỏ │  ← nằm ngoài ROI        │
│ Câu 4 ← bị bỏ │  ← nằm ngoài ROI        │
├────────────────┤ ←───────────────── mcq_roi.y (bắt đầu muộn)
│ Câu 5         │  ← hệ thống đọc là "Câu 1"
│ Câu 6         │  ← hệ thống đọc là "Câu 2"
│ ...           │
└────────────────┘
```

Kết quả: hệ thống đọc câu 5 là câu 1, câu 6 là câu 2 ... — toàn bộ câu bị **lệch index 4**. Câu "1–4" của hệ thống đọc vùng trống hoặc vùng in (đường kẻ, hướng dẫn) → uncertain hoặc false positive A.

**Tại sao Rescue (Bước 13) không xử lý được?**

Rescue dịch lưới qua `top_shift_px` tối đa ±2 × `line_h`. Nhưng quan trọng hơn: **ROI không thay đổi**. Trong `_decode_mcq_with_map`, tọa độ Y luôn bị clamp vào biên ROI:

```python
y1 = max(roi["y"], cy - 0.44 × line_h)
```

Dù dịch `top_center_y` lên bao nhiêu, nếu điểm đó nằm ngoài `roi["y"]` → bị clip về đúng biên ROI → vẫn đọc cùng một vùng. Cần **mở rộng bản thân ROI** thì mới đọc được phần phía trên.

---

## 2. Phát hiện drift — `_detect_q5_start_drift()`

Hàm không nhìn vào ảnh — chỉ nhìn vào **kết quả decode** của Bước 12 và tìm mẫu bất thường ở 4 câu đầu so với 4 câu tiếp theo:

```python
first_rows = rows[:4]    # câu "1–4" (theo hệ thống, thực chất đọc vùng sai)
next_rows  = rows[4:8]   # câu "5–8" (theo hệ thống, thực chất đọc đúng bong bóng)
```

Tính 5 chỉ số trên kết quả decode:

```python
weak_first     = số câu trong first_rows bị uncertain HOẶC best_score < min_mark_score
strong_next    = số câu trong next_rows có đáp án rõ (selected ≥ 0 VÀ score ≥ min_mark_score)
left_bias      = ≥ 3/4 câu đầu chọn lựa chọn A (index 0)
weak_margin    = mean(margin của 4 câu đầu) < 0.035
next_confident = ≥ 2 câu trong next_rows có đáp án
low_diversity  = các câu đầu không uncertain đều chọn cùng 1 đáp án (≤ 1 giá trị khác nhau)
```

**Hai pattern kích hoạt drift:**

### Pattern A — false positive A + margin yếu

```
left_bias AND weak_margin AND next_confident
```

Khi lưới đọc vào vùng phía trên MCQ (khoảng trắng, vùng in hướng dẫn), cột A nằm sát **đường kẻ dọc trái** của phiếu:

```
│ A  B  C  D
│○  ○  ○  ○   ← bong bóng câu 1 thật (dưới ROI, chưa đọc được)
↑ đường kẻ dọc trái
```

Đường kẻ dọc làm cột A có tín hiệu đen nhẹ hơn B/C/D → hệ thống chọn A cho nhiều câu "ảo" trong vùng không có bong bóng. Nhưng **margin rất thấp** (< 0.035) vì đường kẻ không đậm như bong bóng tô thật — A chỉ nhỉnh hơn B/C/D vài điểm. Từ câu 5+ (thực ra là câu 1 thật), bong bóng rõ ràng → đọc được bình thường.

Ba điều kiện cùng xảy ra (`left_bias + weak_margin + next_confident`) là dấu hiệu đặc trưng của drift dạng này, khác với trường hợp học sinh thật sự tô A liên tục (margin thường cao hơn nhiều).

### Pattern B — yếu đầu, mạnh sau (dạng tổng quát)

```
# Phiếu ≥ 8 câu:
weak_first >= 3 AND strong_next >= 3

# Phiếu < 8 câu:
weak_first >= max(2, first_count-1) AND low_diversity AND next_margin_mean > 0.010
```

Pattern đơn giản hơn, không cần suy luận về đường kẻ: sự tương phản rõ rệt giữa "đầu yếu — sau mạnh" đủ để nghi ngờ drift, dù vùng phía trên ROI không có đường kẻ gây thiên lệch A.

### Có cần thiết không?

Cần, vì đây là loại lỗi Rescue không thể sửa (ROI boundary là giới hạn cứng). Xảy ra trong thực tế khi marker đầu vùng MCQ bị phát hiện lệch xuống vài dòng.

Tuy nhiên hàm này **hoàn toàn heuristic** — không có bằng chứng toán học, chỉ dựa trên quan sát thực tế về mẫu phiếu. Vì vậy kết quả expand chỉ được chấp nhận khi `quality(retry) ≥ quality(current)` — nếu expand không cải thiện thì bỏ qua, không làm hỏng kết quả đang có.

---

## 3. Trình tự xử lý

### Bước 14a — Chạy phát hiện

```python
drift_suspected = _detect_q5_start_drift(mcq_result, min_mark_score)
```

Nếu `drift_suspected = True` → ghi cờ `MCQ_COORD_DRIFT`.

### Bước 14b — Mở rộng ROI lên trên

```python
expand_lines = 4
expand_px = round(4 × line_h)

# Kéo cạnh trên ROI lên
y_new        = max(0, mcq_roi["y"] − expand_px)
gained       = mcq_roi["y"] − y_new       # số pixel thực sự kéo được (bị chặn nếu cạnh trên ảnh)
retry_roi["y"] = y_new
retry_roi["h"] = min(HEIGHT_IMG − y_new, mcq_roi["h"] + gained)
```

Ví dụ: `line_h=21`, `mcq_roi.y=400`:
- `expand_px = 4 × 21 = 84 px`
- `y_new = 400 − 84 = 316`
- `retry_roi = {"y": 316, "h": h + 84, ...}`

### Bước 14c — Dịch `top_center_y` lên tương ứng

```python
retry_top_center = top_center_y − 4 × line_h

# Safety: không để top_center vượt ra ngoài ROI mới
if retry_top_center < retry_roi["y"] + 0.25 × line_h:
    retry_top_center = retry_roi["y"] + 0.5 × line_h
```

### Bước 14d — Decode lại với ROI và `top_center_y` mới

```python
retry = _decode_mcq_with_map(...,
    roi=retry_roi,
    top_center_y=retry_top_center,
    line_h=line_h,         # giữ nguyên line_h
    top_shift_px=0.0,
)
```

`line_h` không thay đổi vì khoảng cách dòng vẫn đúng — chỉ vị trí bắt đầu sai.

### Bước 14e — Chấp nhận nếu tốt hơn

```python
if _mcq_quality(retry) >= _mcq_quality(mcq_result):
    mcq_result   = retry
    mcq_roi      = retry_roi
    top_center_y = retry_top_center
    warning_codes.append("MCQ_AUTO_EXPAND_UP")
else:
    # Ghi warning nhưng giữ kết quả cũ
    warnings.append("Phat hien drift nhung thu nghiem mo rong len khong tot hon decode hien tai.")
```

Điều kiện `quality ≥ quality_baseline` (không chặt như Rescue): chỉ cần không tệ hơn là chấp nhận — vì drift đã được xác nhận, không cần bảo thủ như Rescue.

---

## 4. Khác biệt với Bước 13 (Rescue)

| | Bước 13 — Map Search Rescue | Bước 14 — Drift Expand |
|-|----------------------------|------------------------|
| **Vấn đề** | `line_h` sai hoặc `top_center_y` lệch nhẹ | ROI bắt đầu quá thấp, câu đầu nằm ngoài ROI |
| **Cách sửa** | Thử 54 bộ (`line_h_scale`, `top_shift`) trong ROI hiện tại | Mở rộng ROI lên 4 × `line_h`, dịch `top_center_y` |
| **`mcq_roi`** | Không thay đổi | Thay đổi (y thấp hơn, h lớn hơn) |
| **`line_h`** | Có thể thay đổi (scale ≠ 1.0) | Giữ nguyên |
| **Điều kiện chấp nhận** | Bảo thủ: uncertain giảm ≥ 2, hoặc ... | Đơn giản: quality ≥ quality_baseline |
| **Kích hoạt khi** | uncertain ≥ 40% × Q | Bất kỳ khi nào pattern drift phát hiện được |

---

## 5. Đầu vào và đầu ra

### Đầu vào

| Tham số | Nguồn |
|---------|-------|
| `mcq_result` | Kết quả Bước 12 (hoặc đã được Rescue cập nhật ở Bước 13) |
| `line_h` | Sau Rescue |
| `top_center_y` | Sau Rescue |
| `mcq_roi` | Hình chữ nhật ROI MCQ hiện tại |
| `min_mark_score` | Từ `mcq_cfg` |

### Đầu ra

| Trường hợp | Kết quả |
|------------|---------|
| Không phát hiện drift | Không làm gì, `mcq_result` giữ nguyên |
| Drift phát hiện nhưng Rescue bị khóa | Ghi cờ `MCQ_COORD_DRIFT`, không expand |
| Drift + expand tốt hơn | `mcq_result` mới, `mcq_roi` mới (y nhỏ hơn, h lớn hơn), `top_center_y` mới, cờ `MCQ_AUTO_EXPAND_UP` |
| Drift + expand không tốt hơn | Ghi cờ `MCQ_COORD_DRIFT`, giữ nguyên kết quả cũ |

---

## 6. Sơ đồ luồng

```
mcq_result (từ Bước 12 hoặc 13)
        │
        ▼
_detect_q5_start_drift():
  first_rows = rows[0:4]
  next_rows  = rows[4:8]
  ├─ Pattern A: left_bias AND weak_margin AND next_confident
  └─ Pattern B: weak_first ≥ 3 AND strong_next ≥ 3
        │ Không → kết thúc, giữ nguyên
        │ Có → drift_suspected = True, ghi MCQ_COORD_DRIFT
        ▼
expand_px = 4 × line_h
retry_roi = mcq_roi.y − expand_px, mcq_roi.h + expand_px
retry_top_center = top_center_y − 4 × line_h
        │
        ▼
_decode_mcq_with_map(roi=retry_roi, top_center_y=retry_top_center, line_h=line_h)
        │
        ▼
quality(retry) ≥ quality(current)?
  Không → giữ nguyên mcq_result
  Có    → mcq_result = retry
          mcq_roi = retry_roi
          top_center_y = retry_top_center
          ghi MCQ_AUTO_EXPAND_UP
        │
        ▼
Bước 15: Chấm điểm & trả kết quả JSON
```
