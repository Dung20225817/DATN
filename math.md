# Toán học đằng sau Otsu Threshold

> Tài liệu này giải thích từng công thức được dùng trong thuật toán Otsu từ nền tảng — không giả định biết trước thống kê.

---

## 1. Mean (Trung bình)

### Công thức

```
mean = (x₁ + x₂ + ... + xₙ) / n = Σxᵢ / n
```

### Ý nghĩa

"Giá trị đại diện" cho cả nhóm — nếu mọi pixel trong nhóm đều bằng nhau, chúng sẽ bằng mean.

### Ví dụ

Nhóm pixel mực: `[58, 60, 62, 60]`

```
mean = (58 + 60 + 62 + 60) / 4 = 240 / 4 = 60
```

---

## 2. Deviation (Độ lệch)

### Công thức

```
deviation_i = xᵢ − mean
```

### Ý nghĩa

Mỗi pixel lệch bao nhiêu so với trung bình của nhóm.

### Ví dụ

Nhóm `[58, 60, 62, 60]`, mean = 60:

```
deviation₁ = 58 − 60 = −2
deviation₂ = 60 − 60 =  0
deviation₃ = 62 − 60 = +2
deviation₄ = 60 − 60 =  0
```

Nếu cộng tất cả deviation lại: `−2 + 0 + 2 + 0 = 0` — luôn bằng 0 theo định nghĩa của mean. Nên không thể dùng tổng deviation để đo độ "tản mát".

---

## 3. Variance (Phương sai)

### Công thức

```
var = Σ(xᵢ − mean)² / n
```

### Tại sao lại bình phương?

Bình phương giải quyết hai vấn đề:

**Vấn đề 1 — độ lệch âm và dương triệt tiêu nhau:**

```
Nhóm A: [40, 80]   mean = 60   deviations: −20, +20   tổng = 0
Nhóm B: [59, 61]   mean = 60   deviations:  −1,  +1   tổng = 0
```

Hai nhóm có mức độ tản mát rất khác nhau nhưng tổng deviation đều bằng 0.

Sau khi bình phương: nhóm A = (400 + 400)/2 = 400, nhóm B = (1 + 1)/2 = 1 → phân biệt được.

**Vấn đề 2 — phạt nặng hơn khi lệch xa:**

Nếu dùng giá trị tuyệt đối `|deviation|`, lệch 10 chỉ "tệ hơn" lệch 5 hai lần (10 vs 5).

Với bình phương: lệch 10 tệ hơn lệch 5 **bốn lần** (100 vs 25). Điều này phản ánh đúng hơn thực tế — một pixel lệch xa mean rất bất thường, không chỉ "đôi chút tệ hơn".

### Ví dụ

Nhóm mực `[58, 60, 62, 60]`, mean = 60:

```
(58−60)² = (−2)² = 4
(60−60)² = ( 0)² = 0
(62−60)² = (+2)² = 4
(60−60)² = ( 0)² = 0

var = (4 + 0 + 4 + 0) / 4 = 8 / 4 = 2
```

Nhóm nền `[195, 200, 205, 200]`, mean = 200:

```
(195−200)² = 25
(200−200)² =  0
(205−200)² = 25
(200−200)² =  0

var = (25 + 0 + 25 + 0) / 4 = 50 / 4 = 12.5
```

### Đọc kết quả

- **var nhỏ** → các pixel trong nhóm gần nhau, nhóm "thuần"
- **var lớn** → các pixel trải rộng, nhóm "hỗn loạn"

Nhóm mực (var=2) thuần hơn nhóm nền (var=12.5) trong ví dụ này.

---

## 4. Weighted Intra-class Variance — σ²_within (Công thức Otsu gốc)

### Công thức

```
σ²_within(T) = w₁ × var₁  +  w₂ × var₂
```

Trong đó:
- `T` = ngưỡng đang thử
- `w₁` = tỷ lệ số pixel trong nhóm 1 (< T) = n₁ / n_total
- `w₂` = tỷ lệ số pixel trong nhóm 2 (≥ T) = n₂ / n_total
- `var₁`, `var₂` = phương sai của từng nhóm

### Tại sao cần trọng số w?

Nếu không có w, nhóm 1 pixel và nhóm 1000 pixel ảnh hưởng như nhau. Đây là vô lý:

```
Ví dụ cắt sai (T quá thấp):
  Nhóm 1: [60]              n₁=1,  var₁=0
  Nhóm 2: [58,62,60,200,200,198,202,200]  n₂=8,  var₂ rất lớn

Không có w: σ² = 0 + var₂_lớn = lớn  ← đúng hướng nhưng...
Có w:       σ² = (1/9)×0 + (8/9)×var₂_lớn  ← nhóm lớn ảnh hưởng nhiều hơn ✓
```

Trọng số đảm bảo nhóm nhiều pixel đóng góp nhiều hơn vào tổng phương sai.

### Ví dụ đầy đủ — thử T = 100

Ảnh 8 pixel: `[58, 60, 62, 60, 195, 200, 205, 200]`  
(4 pixel mực, 4 pixel nền)

```
Nhóm 1 (< 100): [58, 60, 62, 60]
  n₁ = 4,  w₁ = 4/8 = 0.5
  mean₁ = 60,  var₁ = 2

Nhóm 2 (≥ 100): [195, 200, 205, 200]
  n₂ = 4,  w₂ = 4/8 = 0.5
  mean₂ = 200,  var₂ = 12.5

σ²_within(100) = 0.5 × 2  +  0.5 × 12.5  =  1 + 6.25  =  7.25
```

### Ví dụ đầy đủ — thử T = 50 (cắt sai)

```
Nhóm 1 (< 50): rỗng
  w₁ = 0

Nhóm 2 (≥ 50): tất cả 8 pixel
  w₂ = 1.0
  mean₂ = (58+60+62+60+195+200+205+200)/8 = 1040/8 = 130
  var₂  = [(58−130)²+(60−130)²+(62−130)²+(60−130)²
           +(195−130)²+(200−130)²+(205−130)²+(200−130)²] / 8
        = [5184+4900+4624+4900+4225+4900+5625+4900] / 8
        = 39258 / 8 = 4907

σ²_within(50) = 0 + 1.0 × 4907 = 4907   ← rất cao
```

Khi cắt sai (T=50), một nhóm chứa cả mực lẫn nền → phương sai khổng lồ. Khi cắt đúng (T=100), mỗi nhóm thuần → phương sai thấp.

---

## 5. Inter-class Variance — σ²_between (Công thức OpenCV dùng thực tế)

### Công thức

```
σ²_between(T) = w₁ × w₂ × (μ₁ − μ₂)²
```

Trong đó μ₁, μ₂ là mean của hai nhóm.

### Tại sao OpenCV dùng công thức này thay vì σ²_within?

Để tính `σ²_within`, mỗi lần thử T cần:
1. Tính mean₁ từ đầu
2. Tính var₁ từ đầu (duyệt qua n₁ pixel)
3. Tính mean₂, var₂ tương tự
4. Cộng lại

Với ảnh 12MP (12 triệu pixel) × 256 giá trị T = ~3 tỷ phép tính.

`σ²_between` chỉ cần mean của hai nhóm, không cần variance — tính được bằng cách dùng **prefix sum** (tích lũy từ histogram), toàn bộ chỉ cần 256 phép tính.

### Ví dụ — thử T = 100

```
μ₁ = 60 (mean nhóm mực),  w₁ = 0.5
μ₂ = 200 (mean nhóm nền), w₂ = 0.5

σ²_between(100) = 0.5 × 0.5 × (60 − 200)²
                = 0.25 × 19600
                = 4900
```

### Ví dụ — thử T = 50

```
μ₁ = không xác định (nhóm rỗng), w₁ = 0
μ₂ = 130,  w₂ = 1.0

σ²_between(50) = 0 × 1.0 × (undefined − 130)² = 0   ← nhóm rỗng đóng góp 0
```

---

## 6. Tại sao hai công thức tương đương?

### Hệ thức nền tảng

```
σ²_total = σ²_within(T) + σ²_between(T)
```

`σ²_total` là phương sai của **toàn bộ ảnh** — không phụ thuộc vào T (một con số cố định).

Vì `σ²_total` cố định:
```
Khi σ²_within giảm → σ²_between tăng (và ngược lại)
```

Nên **tối thiểu hóa σ²_within** hoàn toàn giống **tối đa hóa σ²_between**. Cả hai cho ra cùng T*.

### Kiểm tra bằng số

| T | σ²_within | σ²_between | σ²_within + σ²_between |
|---|-----------|------------|------------------------|
| 50 | 4907 | 0 | 4907 |
| 100 | 7.25 | 4900 | 4907.25 ≈ 4907 |
| 150 | 7.25 | 4900 | 4907 |

σ²_total ≈ 4907 ở mọi T (sai số nhỏ do làm tròn).

Khi T=100: σ²_within thấp nhất (7.25) và σ²_between cao nhất (4900) → cùng kết luận T*=100.

---

## 7. Walkthrough hoàn chỉnh — trường hợp mực mờ

Ảnh có mực phai: `[145, 150, 148, 152, 248, 250, 252, 250]`

Otsu thử T = 127 (ngưỡng cố định thông thường):

```
Nhóm 1 (< 127): rỗng — tất cả pixel mực (145–152) đều > 127
Nhóm 2 (≥ 127): tất cả 8 pixel trộn lẫn
  mean₂ = (145+150+148+152+248+250+252+250)/8 = 1595/8 = 199.4
  var₂ rất lớn (trộn mực ~149 và nền ~250)

σ²_within(127) = 0 + 1.0 × var_lớn = CAO
σ²_between(127) = 0 (nhóm 1 rỗng, w₁=0)
```

Otsu thử T = 200:

```
Nhóm 1 (< 200): [145, 150, 148, 152]
  w₁ = 0.5,  mean₁ = 148.75
  var₁ = [(145−148.75)²+(150−148.75)²+(148−148.75)²+(152−148.75)²] / 4
        = [14.06 + 1.56 + 0.56 + 10.56] / 4 = 26.75/4 = 6.69

Nhóm 2 (≥ 200): [248, 250, 252, 250]
  w₂ = 0.5,  mean₂ = 250
  var₂ = [(248−250)²+(250−250)²+(252−250)²+(250−250)²] / 4
        = [4+0+4+0] / 4 = 2

σ²_within(200) = 0.5×6.69 + 0.5×2 = 3.35 + 1 = 4.35   ← THẤP NHẤT
σ²_between(200) = 0.5×0.5×(148.75−250)² = 0.25×10251 = 2562.8  ← CAO NHẤT
```

Otsu duyệt qua 256 giá trị T, T=200 cho σ²_within thấp nhất → T* = 200.

Không cần biết trước mực mờ — chỉ nhìn histogram của ảnh hiện tại và tìm điểm chia tốt nhất.

---

## Tóm tắt công thức

| Công thức | Dùng để | Ghi chú |
|-----------|---------|---------|
| `mean = Σxᵢ/n` | Giá trị đại diện nhóm | Nền tảng |
| `var = Σ(xᵢ−mean)²/n` | Đo độ tản mát trong nhóm | Bình phương để tránh triệt tiêu dấu và phạt nặng lệch xa |
| `σ²_within = w₁×var₁ + w₂×var₂` | Đánh giá chất lượng ngưỡng T — Otsu gốc | Tối thiểu hóa → tìm T* |
| `σ²_between = w₁×w₂×(μ₁−μ₂)²` | Tương đương, nhanh hơn | OpenCV dùng cái này |
| `σ²_total = σ²_within + σ²_between` | Chứng minh hai công thức tương đương | σ²_total cố định theo T |
