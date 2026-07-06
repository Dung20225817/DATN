# Bước 5 — Chuyển ảnh xám và nhị phân hóa

> **Vị trí trong pipeline:** Bước 4 trả ra ảnh BGR 1000×1400 đã thẳng hình học. Bước 5 chuyển ảnh đó thành ảnh đen/trắng thuần túy để mọi bước sau chỉ làm việc với 2 giá trị 0 và 255.

**Module:** `be/app/services/omr/omr_preprocess.py`
**Hàm chính:** `_binarize()` (dòng 205–236)

---

## 1. Thông tin đầu vào (Input)

| Tham số | Giá trị | Nguồn |
|---------|---------|-------|
| `gray_img` | Ảnh xám 1000×1400 | Chuyển từ BGR của Bước 4 |

---

## 2. Tại sao cần chuyển sang đen/trắng?

Ảnh màu BGR sau Bước 4 có mỗi pixel gồm 3 kênh (Blue, Green, Red), mỗi kênh 8 bit → 256³ ≈ 16 triệu màu có thể.

Các bước sau không cần màu, chỉ cần biết "chỗ này có mực in không":

- **fill ratio bong bóng** (Bước 12): đếm pixel đen trong ô / diện tích ô → cần ảnh 0/255
- **connected components** (Bước 6): tìm vùng pixel liên thông → chỉ có nghĩa trên ảnh nhị phân
- **template matching** (Bước 9): so khớp mẫu vuông đen → ảnh nhị phân tránh bị nhiễu bởi màu sắc

Ảnh nhị phân: pixel đen (mực in, bong bóng tô) = **255**, nền trắng = **0**.

> Lý do đảo ngược (đen = 255 thay vì 0): dễ tính fill ratio — `countNonZero(ô)` đếm thẳng pixel có mực mà không cần đổi dấu.

---

## 3. 5a — Chuyển sang ảnh xám (grayscale)

```python
gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
```

OpenCV áp công thức:

```
Y = 0.114×B + 0.587×G + 0.299×R
```

(Chuẩn ITU-R BT.601 — trọng số khác nhau vì mắt người nhạy với từng màu khác nhau)

**Tại sao trọng số Green (0.587) cao nhất?**

Mắt người có 3 loại tế bào cảm thụ màu (S, M, L). Tế bào M (nhạy với xanh lá ~550nm) đông nhất và nhạy nhất với cường độ sáng. Vì vậy cùng một pixel, kênh Green đóng góp nhiều nhất vào cảm giác "sáng hay tối" của mắt người.

**Tại sao OpenCV dùng BGR thay vì RGB?**

Quyết định lịch sử từ thời Windows API dùng thứ tự BGR. Không ảnh hưởng kết quả vì `COLOR_BGR2GRAY` đã biết thứ tự kênh và dùng đúng trọng số cho từng kênh.

---

## 4. 5b — Chuẩn hóa background

```python
k = max(31, (min(h, w) // 9) | 1)          # dòng 211
bg = cv2.morphologyEx(gray_img,
         cv2.MORPH_CLOSE,
         cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))   # dòng 212
gray_norm = cv2.divide(gray_img, bg, scale=255)                  # dòng 213
gray_norm = cv2.GaussianBlur(gray_norm, (3, 3), 0)               # dòng 214
```

### Tại sao phải chuẩn hóa — vấn đề ánh sáng không đều

Ảnh chụp điện thoại thường có góc gần đèn sáng hơn, góc xa tối hơn. Nếu dùng một ngưỡng T duy nhất cho toàn ảnh thì:

```
Góc sáng (pixel nền = 220):  pixel nền vẫn < T nếu T = 200 → bị coi là mực in (sai)
Góc tối  (pixel mực = 100):  pixel mực > T nếu T = 80  → bị coi là nền (sai)
```

Giải pháp: ước tính độ sáng nền tại mỗi vùng rồi "chia ra" → toàn ảnh có cùng độ sáng nền.

### Bước 1 — Tính kích thước kernel k

```
h=1400, w=1000
k = max(31, (min(1400, 1000) // 9) | 1)
  = max(31, (1000 // 9) | 1)
  = max(31, 111 | 1)
  = max(31, 111)
  = 111
```

- `// 9`: kernel cần đủ lớn để "nhìn" qua cả vùng bóng sáng (thường lan rộng ~1/10 ảnh), nhưng không quá lớn để gây méo
- `| 1`: đảm bảo k lẻ — kernel morphology của OpenCV yêu cầu kích thước lẻ để có điểm trung tâm rõ ràng
- `max(31, ...)`: ảnh nhỏ (<279px) vẫn dùng kernel tối thiểu 31px

### Bước 2 — Morphological Close → ước tính background

**Morphological Close = Dilate rồi Erode** với kernel ellipse 111×111:

```
Dilate:  mỗi pixel lấy giá trị MAX trong vùng 111×111 xung quanh
         → Vùng sáng lan rộng ra, lấp đầy chi tiết tối nhỏ (chữ in, bong bóng)

Erode:   mỗi pixel lấy giá trị MIN trong vùng 111×111 xung quanh
         → Thu nhỏ vùng sáng về kích thước ban đầu

Kết quả (bg): ảnh mà mọi chi tiết nhỏ hơn 111px đã bị xóa → chỉ còn gradient sáng nền
```

Trực quan:

```
Ảnh gốc (xám):      Sau Close (bg):
  ░░▓░░░░░░         ░░░░░░░░░    ← chi tiết ▓ (bong bóng ~8px) bị xóa
  ░░░░░░░░░         ░░░░░░░░░    ← chỉ còn gradient ánh sáng nền
  ░▓▓▓░░░░░         ░░░░░░░░░
```

**Tại sao dùng hình elipse thay vì hình vuông?**

Kernel elipse mượt hơn ở góc, tránh hiệu ứng "vuông góc" nhân tạo khi ước tính vùng sáng.

### Bước 3 — Chia để chuẩn hóa

```
gray_norm = gray_img / bg × 255
```

Ví dụ số cụ thể:

```
Góc sáng: pixel nền gray=210, bg=220  → 210/220×255 = 243  (chuẩn hóa lên gần 255)
Góc tối:  pixel nền gray=130, bg=145  → 130/145×255 = 229  (chuẩn hóa lên gần 255)

Góc sáng: pixel mực gray=60,  bg=220  → 60/220×255  = 70   (thấp, là mực)
Góc tối:  pixel mực gray=40,  bg=145  → 40/145×255  = 70   (thấp, là mực)
```

Sau chuẩn hóa: nền ≈ 230–250, mực ≈ 60–80 trên toàn ảnh — bất kể góc ảnh tối hay sáng.

**Tại sao chia (không trừ)?**

Nếu trừ: `gray - bg` → nền sẽ ra gần 0 ở cả vùng sáng lẫn tối, nhưng mực cũng bị kéo âm ở vùng tối → phức tạp và thiếu ổn định. Chia tương đối (tỉ lệ) hoạt động tốt hơn vì bù được cả sự thay đổi cường độ tuyệt đối.

### Bước 4 — Gaussian Blur 3×3

```python
gray_norm = cv2.GaussianBlur(gray_norm, (3, 3), 0)
```

Làm mờ nhẹ để giảm nhiễu pixel lẻ trước khi threshold. Kernel 3×3 đủ nhỏ để không làm mờ các cạnh quan trọng (viền bong bóng ~1–2px).

### Đây không phải lần 2 — Bước 2 và Bước 5 chuẩn hóa hai ảnh khác nhau hoàn toàn

Bước 2 chuẩn hóa ảnh gốc (4000×3000 chưa warp) với **mục đích duy nhất là tìm 4 ô vuông đen**. Sau khi tìm được 4 tọa độ góc, ảnh chuẩn hóa đó bị bỏ — không đi theo sang các bước sau.

Bước 4 lấy ảnh gốc BGR **chưa chuẩn hóa** → warp → resize → ra ảnh 1000×1400. Ảnh này chưa bao giờ được chuẩn hóa.

```
Bước 2: chuẩn_hóa(ảnh_gốc) → tìm 4 góc → bỏ ảnh đã chuẩn hóa
                                                │
Bước 4: warp(ảnh_gốc_gốc) ────────────────────▼ → ảnh 1000×1400 (chưa chuẩn hóa)
                                                │
Bước 5: chuẩn_hóa(ảnh 1000×1400) ← lần đầu tiên trên ảnh này
```

Vì vậy chuẩn hóa ở Bước 5 không phải "làm lại" — đây là lần đầu tiên ảnh 1000×1400 được chuẩn hóa.

---

## 5. 5c — Nhị phân hóa

```python
otsu_value, otsu_inv = cv2.threshold(
    gray_norm, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)   # dòng 216
```

**Ý tưởng:** tìm ngưỡng T sao cho hai nhóm pixel (< T và ≥ T) nội bộ đồng đều nhất.

Ảnh xám sau chuẩn hóa có histogram điển hình gồm 2 đỉnh:

```
Số pixel
  │     ╭──╮              ╭──╮
  │    ╱    ╲            ╱    ╲
  │   ╱      ╲          ╱      ╲
  └──────────────────────────────→ Giá trị xám (0–255)
        ↑ mực (60–80)    ↑ nền (230–250)
                   ↑
               Otsu chọn đây = T ≈ 150
```

Otsu duyệt mọi giá trị T từ 0–255, với mỗi T tính phương sai trong nhóm, chọn T có phương sai nhỏ nhất — tức ranh giới giữa 2 nhóm rõ nhất.

`THRESH_BINARY_INV`: pixel < T (mực tối) → **255**, pixel ≥ T (nền sáng) → **0**.

Nhờ bước cân bằng nền ở 5b, histogram luôn có 2 đỉnh tách biệt rõ bất kể góc ảnh tối hay sáng → Otsu chọn ngưỡng ổn định mà không cần can thiệp thủ công.

---

## 6. 5d — Khử nhiễu (Morphological Open)

```python
binary_inv = cv2.morphologyEx(
    binary_inv, cv2.MORPH_OPEN,
    np.ones((2, 2), np.uint8), iterations=1)   # dòng 234
```

**Morphological Open = Erode rồi Dilate** với kernel 2×2:

```
Erode:  mỗi pixel trắng (255) chỉ giữ lại nếu tất cả pixel trong 2×2 xung quanh cũng trắng
        → Vùng trắng nhỏ hơn 2×2 px bị xóa hoàn toàn (nhiễu điểm lẻ)

Dilate: vùng trắng còn lại được phục hồi về kích thước ban đầu
```

**Tại sao kernel nhỏ (2×2)?**

```
Nhiễu điểm lẻ:   1–2 px → bị xóa hoàn toàn bởi erode 2×2  ✓
Bong bóng tô:    ~8–12 px rộng → sau erode 2×2 vẫn còn ~6–10 px → dilate phục hồi ✓
Đường kẻ mỏng:   1 px rộng → bị xóa — nhưng không sao, đường kẻ không cần giữ ở đây
```

---

## 7. Đầu ra (Output)

```python
return gray_norm, binary_inv, {"mode": mode, "otsu_value": float(otsu_value)}
```

| Biến | Kiểu | Dùng ở đâu sau này |
|------|------|-------------------|
| `gray_norm` | `np.ndarray` (1400, 1000) uint8 | Bước 12: tính `dark_mean`, `dark_p25` từ ảnh xám gốc (không bị mất tín hiệu khi bút tô nhạt) |
| `binary_inv` | `np.ndarray` (1400, 1000) uint8, giá trị 0/255 | Bước 6: tìm marker; Bước 9: template matching; Bước 12: tính `fill_ratio` |
| `otsu_value` | `float` | Ghi log, debug |

---

## 8. Sơ đồ luồng

```
gray_img (ảnh xám 1000×1400)
        │
        ▼
  Morphological Close (ellipse 111×111)
        │
        ▼  bg = ảnh ước tính gradient ánh sáng nền
        │
  gray / bg × 255
        │
        ▼  gray_norm = ảnh xám đã cân bằng ánh sáng
        │
  Gaussian Blur 3×3
        │
        │
        ▼
  Otsu threshold (1 ngưỡng T toàn ảnh)
        │
        ▼
          Morphological Open (2×2, 1 lần)
                       │
                       ▼
          binary_inv: ảnh 0/255
          mực in / bong bóng tô = 255
          nền trắng              = 0
```
