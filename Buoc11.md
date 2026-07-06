# Bước 11 — Giải mã mã đề (Exam Code)

> **Vị trí trong pipeline:** Chạy song song với Bước 10 (MSSV). Dùng cùng hàm `_decode_numeric_columns()` nhưng với cấu hình khác: 3 chữ số, không có hàng viết tay, không có 2-pass. Sau khi đọc xong chuỗi 3 chữ số, tra bảng đáp án để chọn đúng đáp án cho đề thi này.

**Module giải mã:** `be/app/services/omr/omr_numeric.py` → `_decode_numeric_columns()`
**Module tra đáp án:** `be/app/services/omr/omr_service.py` (dòng 1840–1870)

---

## 1. Tại sao cần giải mã mã đề?

Trong thi cử có nhiều mã đề khác nhau (đề 001, 002, 003…). Mỗi đề có thứ tự câu hỏi và đáp án riêng. Nếu không đọc được mã đề, hệ thống không biết dùng bảng đáp án nào để chấm — kết quả chấm sẽ sai.

Học sinh tô mã đề theo dạng bubble số, hoàn toàn giống MSSV nhưng chỉ có **3 cột** (3 chữ số). Ví dụ mã đề 132:

```
Cột: [0]  [1]  [2]
     [1]  [3]  [2]   ← học sinh tô các hàng này
```

---

## 2. Đầu vào (Input)

| Tham số | Giá trị cụ thể |
|---------|---------------|
| `gray_img` | ảnh xám 1000×1400 |
| `binary_inv` | ảnh nhị phân 1000×1400 |
| `roi` | `code_roi` từ Bước 8 (`{"x":..., "y":..., "w":..., "h":...}`) |
| `digits` | **3** (cố định — mã đề luôn 3 chữ số) |
| `has_write_row` | **False** (cố định — không có hàng viết tay) |
| `row_offsets` | **None** (không dùng) |

```python
# omr_service.py dòng 1372–1379
code_result = _decode_numeric_columns(
    gray_norm,
    binary_inv,
    code_roi,
    digits=3,
    has_write_row=False,
    row_offsets=None,
)
```

---

## 3. Trình tự tính toán

Bước 11 dùng y hệt cơ chế của Bước 10 (xem Buoc10.md để biết chi tiết), nhưng đơn giản hơn nhiều vì không có 2-pass.

### 11a — Chia lưới 3 × 10

```
total_rows = 10  (has_write_row=False → không có hàng viết tay)

row_edges = linspace(y, y+h, 11)   → 10 hàng
col_edges = linspace(x, x+w,  4)   → 3 cột

Lưới: 3 cột × 10 hàng = 30 ô
```

### 11b — Tính điểm từng ô

Mỗi ô được trích 74% vùng trung tâm rồi tính:

```
score = 0.55 × fill_ratio
      + 0.30 × dark_mean
      + 0.15 × dark_p25
```

Kết quả: `score_matrix` shape `(10, 3)`.

### 11c — Chọn chữ số mỗi cột

```python
valid_start = 0   # không có write_row
valid_scores = col_scores[0:10]   # toàn bộ 10 hàng đều hợp lệ

best_rel = argmax(valid_scores)   # chữ số 0–9
conf     = best / second          # độ tin cậy

if best < 0.07: → "?"
else:           → str(best_rel)
```

→ Ghép 3 chữ số thành chuỗi, ví dụ: `"132"`, `"007"`, `"1?2"`.

### 11d — Chuẩn hóa mã đề (`_normalize_exam_code_key`)

Chuỗi vừa đọc được qua một bước chuẩn hóa để so khớp với bảng đáp án:

```python
def _normalize_exam_code_key(raw) -> str:
    digits = "".join(ch for ch in text if ch.isdigit())
    if digits:
        if len(digits) <= 3:
            return digits.zfill(3)   # "7" → "007", "32" → "032", "132" → "132"
        return digits
    return text.upper()              # nếu không có chữ số → uppercase
```

Mục đích: đảm bảo `"7"` và `"007"` cùng tra ra một đáp án. Người ra đề có thể nhập "7" hay "007" vào hệ thống — đều được chấp nhận.

### 11e — Tra bảng đáp án (`answer_key_by_code`)

Sau khi đọc và chuẩn hóa mã đề, hệ thống tra vào bảng đáp án theo code:

```python
detected_exam_code_key = _normalize_exam_code_key(code_result["value"])
# Ví dụ: "132"

if detected_exam_code_key in normalized_code_map:
    selected_answer_key = normalized_code_map["132"]   # lấy bảng đáp án đề 132
else:
    selected_answer_key = []                            # không tìm thấy → không chấm được
    warning_codes.append("ANSWER_CODE_NOT_FOUND")
```

**`answer_key_by_code`**: dict do người dùng truyền vào API, ví dụ:
```json
{
  "132": [1, 3, 2, 4, 1, ...],
  "231": [3, 1, 4, 2, 3, ...],
  "312": [2, 4, 1, 3, 2, ...]
}
```

Nếu `answer_key_by_code` không được truyền vào (null) → bỏ qua bước tra, dùng `answer_key` chung cho tất cả đề.

---

## 4. Không có 2-pass — tại sao?

Bước 10 (MSSV) cần 2-pass vì MSSV của mỗi trường có thể có hoặc không có hàng viết tay — cần thử cả hai để không đọc sai 1 hàng.

Mã đề **không bao giờ có hàng viết tay** — thiết kế phiếu chuẩn không có ô viết mã đề bằng tay. Do đó `has_write_row=False` là cố định và không cần chạy lần thứ hai.

---

## 5. Đầu ra (Output)

### Từ `_decode_numeric_columns()`:

```python
{
    "value":      "132",     # chuỗi 3 chữ số mã đề ("?" nếu cột không chắc)
    "status":     "ok",      # "ok" hoặc "uncertain"
    "confidence": 3.21,      # trung bình tỷ lệ nhất/nhì của 3 cột
    "scores": [[...]],       # score_matrix (10 × 3) — để debug
    "selected_cells": [...]  # vị trí pixel của ô được chọn
}
```

### Từ bước tra đáp án:

| Biến | Ý nghĩa |
|------|---------|
| `detected_exam_code_raw` | Chuỗi thô đọc ra: `"132"` |
| `selected_answer_key` | Bảng đáp án được chọn (list[int]) hoặc `[]` nếu không tìm thấy |
| `selected_answer_code` | Mã đề được dùng, hiển thị trong JSON đầu ra |
| `answer_key_source` | `"assignment-code-map"` hoặc `"manual-or-file"` |

Nếu không tìm được mã đề → `ANSWER_CODE_NOT_FOUND` → `selected_answer_key = []` → `correct_count = 0` → `score = 0`.

---

## 6. Sơ đồ luồng

```
gray_img + binary_inv + code_roi
        │
        ▼
  11a — Chia lưới 3 cột × 10 hàng
        (has_write_row=False, valid_start=0)
        │
        ▼
  11b — Tính score_matrix (10 × 3)
        _extract_cell(inner_ratio=0.74) → _cell_score()
        │
        ▼
  11c — Mỗi cột: argmax(10 hàng) → chữ số
        best < 0.07 → "?"
        conf = best / second
        │
        ▼
  Ghép value = "132" hoặc "1?2"
        │
        ▼
  11d — Chuẩn hóa: "7" → "007", "32" → "032"
        │
        ▼
  11e — Tra answer_key_by_code["132"]
        ├── Tìm thấy → selected_answer_key = [1,3,2,...]
        └── Không tìm thấy → [] + ANSWER_CODE_NOT_FOUND
        │
        ▼
  Bước 12: chấm MCQ bằng selected_answer_key
```

---

## 7. So sánh Bước 10 (MSSV) và Bước 11 (mã đề)

| | MSSV (Bước 10) | Mã đề (Bước 11) |
|-|----------------|-----------------|
| `digits` | 7 (từ profile) | 3 (cố định) |
| `has_write_row` | True hoặc False (từ profile) | luôn False |
| 2-pass decode | Có, với 6 điều kiện auto-switch | Không — chạy 1 lần |
| `row_offsets` | Từ profile | Không dùng |
| ROI | `sid_roi` từ Bước 8 | `code_roi` từ Bước 8 |
| Mục đích kết quả | Định danh học sinh | Chọn bảng đáp án để chấm |
| Khi thất bại | `student_id = "?"` | `score = 0` + cờ `ANSWER_CODE_NOT_FOUND` |
