# Bước 10 — Giải mã MSSV (Student ID)

> **Vị trí trong pipeline:** Có ROI MSSV từ Bước 8. Bước 10 cắt vùng đó ra, chia thành lưới ô, tính điểm từng ô, rồi ghép lại thành chuỗi chữ số MSSV.

**Module:** `be/app/services/omr/omr_numeric.py`
**Hàm chính:** `_decode_numeric_columns()`
**Gọi từ:** `be/app/services/omr/omr_service.py` (dòng 1300–1314)

---

## 1. Tại sao cần bước riêng?

MSSV được tô dạng **bubble số**: mỗi chữ số là 1 cột, mỗi cột có 10 hàng (số 0–9). Học sinh tô vào hàng tương ứng với chữ số muốn điền. Khác với MCQ (chọn 1 trong A/B/C/D), mỗi cột MSSV là 1 quyết định độc lập.

```
Ví dụ MSSV = "2151063":

Cột:   [0]  [1]  [2]  [3]  [4]  [5]  [6]
Hàng 0  .    .    .    .    .    ●    .
Hàng 1  .    ●    .    .    .    .    .
Hàng 2  ●    .    .    .    .    .    .
Hàng 3  .    .    .    .    .    .    .
Hàng 4  .    .    .    .    .    .    .
Hàng 5  .    .    ●    .    .    .    .
Hàng 6  .    .    .    .    .    .    ●
Hàng 7  .    .    .    .    .    .    .
Hàng 8  .    .    .    .    .    .    .
Hàng 9  .    .    .    ●    .    .    .

→ Đọc hàng có dấu ● mỗi cột: 2, 1, 5, 9(→sai?), 0, 6, 3? (ví dụ minh hoạ)
```

---

## 2. Đầu vào (Input)

| Tham số | Kiểu | Giá trị tiêu biểu |
|---------|------|------------------|
| `gray_img` | `np.ndarray` | ảnh xám 1000×1400 |
| `binary_inv` | `np.ndarray` | ảnh nhị phân 1000×1400 |
| `roi` | `dict` | `{"x":85,"y":56,"w":192,"h":315}` từ Bước 8 |
| `digits` | `int` | 7 (số chữ số MSSV, từ profile) |
| `has_write_row` | `bool` | `True`/`False` (phiếu có hàng viết tay không) |
| `row_offsets` | `list[int]` | `[0,0,...]` (offset từ profile, thường toàn 0) |

---

## 3. 10a — Chia lưới ROI thành ô

ROI MSSV được chia đều thành lưới `digits × total_rows`:

```python
total_rows = 11 if has_write_row else 10

row_edges = np.linspace(y, y+h, total_rows+1)   # total_rows+1 điểm → total_rows khoảng
col_edges = np.linspace(x, x+w, digits+1)        # digits+1 điểm   → digits khoảng
```

Ví dụ: `digits=7`, `has_write_row=False`, ROI `h=315`:
- 10 hàng × 7 cột = 70 ô
- Chiều cao mỗi ô = 315 / 10 = **31.5 px**
- Chiều rộng mỗi ô = 192 / 7 = **27.4 px**

Mỗi ô `(r, c)` có tọa độ:
```
x1 = col_edges[c],   x2 = col_edges[c+1]
y1 = row_edges[r],   y2 = row_edges[r+1]
```

---

## 4. 10b — Tính điểm từng ô (`_cell_score`)

Với mỗi ô trong lưới, tính điểm bằng `_cell_score()` (dùng chung với MCQ):

```python
cell_gray, cell_bin = _extract_cell(..., inner_ratio=0.74)
score = _cell_score(cell_gray, cell_bin)
```

`inner_ratio=0.74`: chỉ lấy **74% vùng trung tâm** của ô, bỏ rìa 13% mỗi bên — tránh đọc nhầm đường kẻ ô tính điểm.

Công thức điểm (từ `_cell_score`):
```
score = 0.55 × fill_ratio          ← tỷ lệ pixel đen trong ô (từ binary_inv)
      + 0.30 × dark_mean           ← độ tối trung bình (từ gray_img)
      + 0.15 × dark_p25            ← percentile 25 độ tối (ổn định hơn mean)
```

Kết quả: `score_matrix` shape `(total_rows, digits)` — ma trận điểm của toàn bộ lưới.

**`row_offsets`:** nếu profile khai báo offset, mỗi cột `c` đọc hàng `r + offsets[c]` thay vì hàng `r`. Dùng để bù lệch in của từng cột trên một số mẫu phiếu cụ thể.

---

## 5. 10c — Chọn chữ số tốt nhất mỗi cột

Với mỗi cột, chỉ xét phạm vi hàng hợp lệ (`valid_start` đến `valid_start + 10`):

```python
valid_start = 1 if has_write_row else 0
valid_scores = col_scores[valid_start : valid_start + 10]   # 10 hàng số 0–9

best_rel  = argmax(valid_scores)      # chỉ số 0–9 của chữ số tốt nhất
best      = valid_scores[best_rel]    # điểm cao nhất
second    = second_highest(valid_scores)

conf = best / max(1e-4, second)       # tỷ lệ nhất/nhì, capped ở 9.99
```

**Quyết định:**
- Nếu `best < 0.07` → ghi `"?"` (ô quá trắng, học sinh chưa tô)
- Nếu không → ghi `str(best_rel)` (chữ số 0–9)

**`conf`** (confidence): tỷ lệ điểm cao nhất / điểm cao thứ nhì trong cột.
- `conf = 3.0` → ô được tô rõ gấp 3 lần ô kế tiếp → tin cậy cao
- `conf ≈ 1.0` → 2 ô gần bằng nhau → không chắc chắn

---

## 6. 10d — has_write_row: hai chế độ lưới

Một số mẫu phiếu có thêm **hàng viết tay** ở đầu mỗi cột (học sinh viết chữ số bằng bút trước, rồi tô bong bóng bên dưới). Hàng viết tay này in không chuẩn và không phải bong bóng — không nên đọc.

```
has_write_row=False (10 hàng):    has_write_row=True (11 hàng):
┌───────┐                          ┌───────┐
│ ○ = 0 │  hàng 0                 │ _____ │  hàng 0 = hàng viết tay (bỏ qua)
│ ○ = 1 │  hàng 1                 │ ○ = 0 │  hàng 1  ← valid_start = 1
│ ○ = 2 │  hàng 2                 │ ○ = 1 │  hàng 2
│  ...  │                          │  ...  │
│ ○ = 9 │  hàng 9                 │ ○ = 9 │  hàng 10
└───────┘                          └───────┘
```

`valid_start = 1 if has_write_row else 0` — bỏ qua hàng 0 nếu là hàng viết tay.

---

## 7. 10e — Chạy 2 pass và tự động chọn

Hệ thống **luôn chạy cả 2 chế độ** rồi so sánh, không chạy 1 lần rồi thôi:

```python
sid_result_with_write    = _decode_numeric_columns(..., has_write_row=True)
sid_result_without_write = _decode_numeric_columns(..., has_write_row=False)
```

**Chọn primary vs alt:**
- `sid_has_write_row` (từ profile) quyết định cái nào là primary
- Primary được dùng trước. Alt chỉ dùng nếu điều kiện auto-switch kích hoạt

**Tại sao phải chạy 2 pass?** Nếu lưới chia sai (phiếu có write_row nhưng code nghĩ không có, hoặc ngược lại), toàn bộ các chữ số đọc ra đều lệch 1 hàng — mỗi chữ số sai đi 1 đơn vị. Chạy 2 pass cho phép phát hiện và tự sửa lỗi này.

### Điều kiện auto-switch (khi `sid_has_write_row=True`, primary=with_write):

| Điều kiện | Ý nghĩa |
|----------|---------|
| `alt_ok and not primary_ok` | Alt đọc được, primary bị "?" → dùng alt |
| `alt_conf ≥ primary_conf + 0.08` | Alt tự tin hơn đáng kể |
| Primary bắt đầu `"99..."` và alt không | "99" là dấu hiệu lệch hàng (hàng viết tay bị đọc nhầm là "9") |
| `shift_delta ∈ {1, 9}` và `alt_conf ≥ primary_conf − 0.15` | Toàn bộ chữ số lệch đúng 1 hàng |

**`shift_delta`:** kiểm tra xem tất cả chữ số giữa primary và alt có lệch đều cùng 1 giá trị (mod 10) không:
```python
deltas = {(alt_digit - primary_digit) % 10 for each digit}
# Nếu tất cả deltas bằng nhau → shift_delta = giá trị đó
# shift_delta = 1: alt = primary + 1 mỗi chữ số → primary bị lệch lên 1 hàng
# shift_delta = 9: alt = primary - 1 mỗi chữ số → primary bị lệch xuống 1 hàng
```

### Điều kiện auto-switch (khi `sid_has_write_row=False`, primary=without_write):

Đơn giản hơn: chỉ switch khi `alt_ok and not primary_ok` — primary thất bại hoàn toàn, alt đọc được.

Nếu switch → ghi cờ `SID_AUTO_SWITCH_ROW_MODE`.

---

## 8. Tính `status` và `confidence`

```python
mean_conf = mean(conf của tất cả các cột)
status = "ok" if ("?" not in value and mean_conf >= 1.03) else "uncertain"
```

`mean_conf ≥ 1.03`: trung bình tỷ lệ nhất/nhì ≥ 1.03 — mỗi cột có ô tô đậm hơn ô kế tiếp ít nhất 3%. Ngưỡng rất thấp, chỉ loại trường hợp hoàn toàn không có tín hiệu.

---

## 9. Đầu ra (Output)

```python
{
    "value":    "2151063",     # chuỗi chữ số MSSV, "?" nếu có cột không chắc
    "status":   "ok",          # "ok" hoặc "uncertain"
    "confidence": 2.84,        # trung bình tỷ lệ nhất/nhì của 7 cột
    "scores": [[...]],         # score_matrix (total_rows × digits) — để debug
    "selected_cells": [        # vị trí pixel của ô được chọn mỗi cột
        {"digit_index": 0, "selected_digit": 2, "cell_box": [85,118,112,149], "score": 0.71},
        ...
    ]
}
```

---

## 10. Sơ đồ luồng

```
gray_img + binary_inv + sid_roi
        │
        ▼
  10a — Chia lưới
        col_edges = linspace(x, x+w, digits+1)
        row_edges = linspace(y, y+h, total_rows+1)
        total_rows = 10 (no write row) hoặc 11 (has write row)
        │
        ▼
  10b — Tính score_matrix (total_rows × digits)
        Mỗi ô: _extract_cell(inner_ratio=0.74) → _cell_score()
        score = 0.55×fill + 0.30×dark_mean + 0.15×dark_p25
        Áp dụng row_offsets nếu có
        │
        ▼
  10c — Mỗi cột: lấy valid_scores[valid_start : valid_start+10]
        best = argmax → chữ số 0–9
        conf = best / second
        best < 0.07 → "?"
        │
        ▼
  Ghép value = "2151063" hoặc "21?1063"

────────────────────────────────────────────────────────
Chạy 2 lần song song (has_write_row=True và False):

  sid_result_with_write    ←── has_write_row=True  (valid_start=1)
  sid_result_without_write ←── has_write_row=False (valid_start=0)
        │
        ▼
  So sánh: primary (theo profile) vs alt
  Auto-switch nếu:
    - primary có "?", alt không có
    - alt_conf cao hơn rõ rệt
    - primary bắt đầu "99"
    - shift_delta = 1 hoặc 9 (toàn bộ cột lệch 1 hàng)
        │
        ▼
  sid_result cuối cùng
  (+ ghi cờ SID_AUTO_SWITCH_ROW_MODE nếu đã switch)
        │
        └──→ Bước 15: ghép vào kết quả JSON cuối
```

---

## 11. Khác biệt với giải mã mã đề (Bước 11)

Mã đề dùng cùng hàm `_decode_numeric_columns` nhưng:

| | MSSV (Bước 10) | Mã đề (Bước 11) |
|-|----------------|-----------------|
| `digits` | 7 (từ profile) | 3 (cố định) |
| `has_write_row` | True hoặc False (từ profile) | luôn False |
| 2-pass decode | Có, với auto-switch | Không, chạy 1 lần |
| `row_offsets` | Từ profile | Không dùng |
| ROI | `sid_roi` | `code_roi` |
