# Bước 3 — Phát hiện trang bằng contour (Phương án dự phòng)

> **Vị trí trong pipeline:** Bước này chỉ được gọi khi Bước 2 (phát hiện 4 corner marker đen) **thất bại**.
> Mục tiêu vẫn giống Bước 2: tìm 4 điểm góc của trang giấy để warp phối cảnh (Bước 4).

**Module:** `be/app/services/omr/omr_preprocess.py`
**Hàm chính:** `_find_page_quad_by_contour()` (dòng 100–138)
**Hàm gọi:** `_detect_page_quad()` (dòng 141–152)

---

## 1. Thông tin đầu vào (Input)

| Tham số | Kiểu | Mô tả |
|---------|------|-------|
| `gray_img` | `np.ndarray` (2D, uint8) | Ảnh **xám** (grayscale) của tờ phiếu gốc, chưa qua warp hay resize |

- **Nguồn gốc `gray_img`:** ảnh gốc BGR (`cv2.imread`) đã được chuyển sang xám qua `cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)` trước khi truyền vào `_detect_page_quad()`.
- **Kích thước:** tuỳ thuộc camera điện thoại, thường từ 2000–6000 px mỗi chiều — **không** cố định.
- **Điều kiện kích hoạt Bước 3:** `_detect_page_quad()` đã gọi `_detect_page_corners_from_black_square_markers(gray_img)` (Bước 2) và nhận về `None`, nghĩa là không tìm đủ 4 corner marker đen.

```python
# omr_preprocess.py — _detect_page_quad() dòng 141-152
def _detect_page_quad(gray_img):
    marker_pts = omr_marker_utils._detect_page_corners_from_black_square_markers(gray_img)
    if marker_pts is not None:                  # Bước 2 thành công → dùng luôn
        ...
        return _order_quad_points(arr), "corner-markers"

    contour_pts = _find_page_quad_by_contour(gray_img)   # ← Bước 3 được gọi ở đây
    if contour_pts is not None:
        return contour_pts, "page-contour"

    return None, "none"                         # Cả hai đều thất bại
```

---

## 2. Tại sao cần Bước 3? — Lý do tồn tại của phương án dự phòng

Bước 2 dựa vào 4 **ô vuông đen in sẵn** ở 4 góc phiếu. Bước 2 thất bại khi:

| Tình huống | Hậu quả |
|-----------|---------|
| Góc phiếu bị che tay / bàn | Marker không xuất hiện trong ảnh |
| Ảnh chụp quá gần → cắt xén góc | Marker bị cắt khỏi frame |
| Ánh sáng cực đoan → marker mờ đi | Bước nhị phân hóa không tách được marker |
| Phiếu bị nhăn → marker biến dạng | Không đạt ngưỡng fill/solidity |

Thay vì bỏ cuộc, Bước 3 khai thác một đặc trưng **khác nhưng vẫn ổn định**: **viền ngoài của tờ giấy trắng** tương phản với nền bàn/sàn tối hơn. Viền giấy tạo ra gradient pixel mạnh → Canny Edge Detection phát hiện được → `findContours` tìm ra đường bao → xấp xỉ thành tứ giác.

---

## 3. Luồng xử lý chi tiết

```
gray_img (bất kỳ kích thước)
    │
    ▼ 3a. Gaussian Blur 5×5
Làm mờ nhiễu trước khi tính gradient
    │
    ▼ 3b. Canny Edge Detection (60 / 180)
Ảnh biên (edges) — pixel trắng = cạnh mạnh
    │
    ▼ 3c. Dilation 3×3 (1 lần)
Nối các đoạn biên bị hở nhỏ
    │
    ▼ 3d. findContours (RETR_EXTERNAL)
Danh sách contour, sắp xếp diện tích giảm dần
    │
    ▼ 3e. Lọc top-10, bỏ qua contour < 18% diện tích ảnh
    │
    ▼ 3f. approxPolyDP (epsilon = 2% chu vi)
    │    ├─ Nếu ra 4 đỉnh → trả về tứ giác ✓
    │    └─ Không ra 4 đỉnh ↓
    ▼ 3g. minAreaRect + boxPoints (fallback trong fallback)
         Nếu box_area ≥ 16% → trả về tứ giác ✓
         Không đạt → thử contour tiếp theo
    │
    ▼ 3h. _order_quad_points
Sắp xếp 4 điểm: TL → TR → BR → BL
    │
    ▼ Output: np.ndarray (4, 2) float32  hoặc  None
```

---

## 4. Giải thích từng bước con

### 3a. Gaussian Blur 5×5

```python
blur = cv2.GaussianBlur(gray_img, (5, 5), 0)
```

**Thuật toán:** Tích chập ảnh với kernel Gaussian 5×5. Mỗi pixel mới = trung bình có trọng số Gaussian của 25 pixel lân cận. Trọng số giảm dần theo khoảng cách (phân phối chuẩn 2D).

**Trọng số cụ thể lấy từ đâu — công thức Gaussian 2D:**

```
G(x, y) = 1/(2π·σ²) × exp( −(x² + y²) / (2σ²) )
```

`σ` quyết định độ "phình" của quả chuông: σ nhỏ → trọng số dồn sát tâm (blur nhẹ); σ lớn → trọng số lan rộng (blur mạnh). Với kernel 5×5, `sigma=0` được OpenCV quy đổi thành `σ ≈ 1.1` (xem công thức bên dưới).

**OpenCV không nhân trực tiếp ma trận 5×5 — đây là điểm hay bị bỏ qua.** Vì `G(x,y) = G(x) × G(y)` (Gaussian 2D tách được thành tích 2 Gaussian 1D), `cv2.GaussianBlur` tính theo 2 bước:

1. **Kernel 1D (5 số)** theo công thức trên với `y=0`, offset `x = −2,−1,0,1,2`:

```
G(x) thô:  exp(−4/2.42)  exp(−1/2.42)  exp(0)  exp(−1/2.42)  exp(−4/2.42)
        =    0.1915         0.6615       1.0       0.6615        0.1915
```

Chia cho tổng (2.706) để chuẩn hóa (tổng trọng số phải bằng 1, nếu không ảnh sẽ bị sáng/tối lệch đi so với gốc):

```
w = [0.0708, 0.2445, 0.3696, 0.2445, 0.0708]   ← kernel 1D, tổng = 1.0
```

2. **Áp dụng kernel 1D này 2 lần riêng biệt:** trượt ngang qua từng hàng (blur theo trục X), rồi trượt dọc qua kết quả đó (blur theo trục Y):

```
Ảnh gốc → [lọc ngang bằng 5 số w] → ảnh tạm → [lọc dọc bằng 5 số w] → ảnh blur
```

Kết quả **giống hệt về mặt toán học** với việc nhân ma trận 5×5 đầy đủ (`K[i][j] = w[i] × w[j]`), nhưng rẻ hơn nhiều: 2 lần × 5 phép nhân-cộng = **10 phép tính/pixel**, thay vì ma trận 5×5 đầy đủ = **25 phép tính/pixel**. Đây là lý do Gaussian Blur luôn nhanh hơn các loại lọc không tách được (như lọc trung vị — median blur).

Ma trận 5×5 tương đương (chỉ để hình dung, không phải cách OpenCV tính thật):

```
        x=-2    x=-1    x=0     x=1     x=2
y=-2  0.0050  0.0173  0.0262  0.0173  0.0050
y=-1  0.0173  0.0598  0.0904  0.0598  0.0173
y=0   0.0262  0.0904  0.1366  0.0904  0.0262   ← đỉnh chuông ở tâm
y=1   0.0173  0.0598  0.0904  0.0598  0.0173
y=2   0.0050  0.0173  0.0262  0.0173  0.0050
```

Tổng 25 ô = 1.0 (bảo toàn độ sáng trung bình sau khi blur).

**Ví dụ tính số — vì sao blur "làm mất" nhiễu:** giả sử trong vùng 5×5 quanh 1 pixel có 1 điểm nhiễu cảm biến (hot pixel) giá trị 330, lệch hẳn khỏi nền ~203 xung quanh, nằm ở offset `(dy=−1, dx=−1)` so với tâm — trọng số tương ứng `w(−1)×w(−1) = 0.2445×0.2445 ≈ 0.0598`:

```
Độ lệch của pixel nhiễu so với nền = 330 − 203 = 127
Đóng góp riêng của pixel này vào giá trị blur ở tâm
  = trọng số × độ lệch = 0.0598 × 127 ≈ 7.6
```

Pixel nhiễu lệch **+127** so với nền, nhưng sau blur chỉ còn kéo giá trị trung tâm lệch đi **+7.6** — khoảng 94% độ lệch bị triệt tiêu. Đây chính là cơ chế đứng sau việc "không làm mờ trước thì Canny phát hiện hàng nghìn biên nhiễu" (xem bên dưới): Sobel tính hiệu số giữa các pixel liền kề (mục 3b) — để nguyên pixel 330 thì hiệu số với hàng xóm ~203 là 127, đủ để Canny coi là "cạnh"; sau blur, hiệu số đó co lại chỉ còn vài đơn vị, dưới ngưỡng 60 → không tạo cạnh giả.

Ngược lại, 1 pixel ở góc xa tâm (`dy=2, dx=2`) chỉ có trọng số `0.0708×0.0708 ≈ 0.005` — gần như không ảnh hưởng đến pixel trung tâm. Đây là lý do Gaussian tốt hơn lọc trung bình cộng (box blur, trọng số đều `1/25=0.04` cho mọi pixel bất kể xa gần): Gaussian tin tưởng pixel gần tâm hơn nhiều so với pixel ở rìa vùng lọc, nên giữ được cạnh sắc nét hơn box blur cùng kích thước.

**Xử lý biên ảnh:** tại pixel nằm sát mép ảnh (cách mép < 2px), kernel 5×5 sẽ "thò" ra ngoài ảnh — không có pixel thật ở đó. `cv2.GaussianBlur` mặc định dùng `borderType=BORDER_REFLECT_101`: phản chiếu pixel qua mép ảnh làm giá trị giả (pixel ở vị trí `-1` lấy giá trị pixel ở vị trí `1`, không lặp lại chính mép), tránh tạo viền tối/sáng giả ở vài pixel rìa ảnh.

**Tại sao cần làm mờ trước Canny?**

Canny phát hiện biên bằng cách tính **gradient** (đạo hàm) cường độ pixel. Nhiễu kỹ thuật số (sensor noise, JPEG artifact) tạo ra gradient giả trên mọi pixel — không làm mờ trước thì Canny phát hiện hàng nghìn biên nhiễu, che lấp viền thực của tờ giấy.

**Tại sao kernel 5×5 (không phải 3×3 hay 7×7)?**

- `3×3`: quá nhỏ → không loại được nhiễu JPEG granular
- `7×7` hoặc lớn hơn: làm mờ quá mức → các góc nhọn của tờ giấy bị bo tròn → Canny mất chính xác tại góc
- `5×5`: cân bằng — loại được nhiễu vừa đủ, vẫn giữ cạnh sắc nét

**Tham số `sigma=0`:** OpenCV tự tính `sigma = 0.3 × ((ksize−1) × 0.5 − 1) + 0.8` → với kernel 5×5 cho sigma ≈ 1.1. Không cần đặt thủ công.

---

### 3b. Canny Edge Detection — ngưỡng thấp 60, ngưỡng cao 180

```python
edges = cv2.Canny(blur, 60, 180)
```

---

#### Trực giác cốt lõi: Cạnh là gì?

Trong ảnh số, **cạnh** (edge) là nơi cường độ pixel **thay đổi đột ngột** từ tối sang sáng (hoặc ngược lại).

```
Vùng bình thường (không có cạnh):
... 200  202  199  201  203 ...   ← thay đổi nhỏ = vùng đồng đều

Tại cạnh tờ giấy:
... 210  208  205   40   38   35 ...
                  ↑
              đây là cạnh — pixel nhảy từ ~205 xuống ~40
```

**Gradient** đo tốc độ thay đổi đó. Gradient cao = thay đổi nhanh = có cạnh. Canny phát hiện cạnh bằng cách tìm những chỗ gradient lớn, rồi làm sạch kết quả qua 4 bước.

---

#### Bước 1 nội bộ — Tính Gradient bằng Sobel

Câu hỏi đặt ra cho mỗi pixel: **"Pixel này có thay đổi nhiều so với xung quanh không?"**

Câu trả lời đơn giản nhất: lấy pixel bên phải trừ pixel bên trái.

```
thay đổi ngang tại x  =  I[x+1] − I[x−1]
```

Ví dụ:

```
Dãy pixel:  ... 202   205   40   38 ...
                       ↑ x đang xét
thay đổi = 40 − 202 = −162   ← lớn → có cạnh tại đây

Dãy pixel:  ... 200   202   205 ...
                       ↑ x đang xét
thay đổi = 205 − 200 = 5     ← nhỏ → vùng phẳng, không có cạnh
```

Đó là toàn bộ ý tưởng. Cần tính giá trị này tại **mọi pixel** trong ảnh — theo cả chiều ngang (Gx) và chiều dọc (Gy):

```
Gx = thay đổi ngang = I[x+1] − I[x−1]   ← phát hiện cạnh DỌC
Gy = thay đổi dọc  = I[y+1] − I[y−1]   ← phát hiện cạnh NGANG
```

---

##### Kernel là gì — cách đóng gói phép tính trên vào một ma trận

Thay vì viết `I[x+1] − I[x−1]` thành code dài, người ta đóng gói phép tính này vào một **ma trận nhỏ gọi là kernel**:

```
I[x+1] − I[x−1]
= (+1) × I[x+1]  +  (0) × I[x]  +  (−1) × I[x−1]
           ↑                ↑                ↑
        hệ số phải     hệ số giữa       hệ số trái
                       (không cần)

→ viết gọn thành kernel 1 hàng:  [ −1,  0,  +1 ]
```

Kernel chỉ là cách ghi lại bộ hệ số. Đọc kernel `[−1, 0, +1]` nghĩa là: *"lấy pixel bên phải nhân +1, pixel giữa nhân 0, pixel bên trái nhân −1, rồi cộng lại"* — tức là phải trừ trái.

Sobel X mở rộng 1 hàng này thành 3 hàng để tính trên cả hàng trên và hàng dưới pixel đang xét (giải thích ở phần sau), nên thành ma trận 3×3:

```
Kernel Sobel X (Gx):    Kernel Sobel Y (Gy):
┌─────────────────┐     ┌─────────────────┐
│ −1   0   +1     │     │ +1   +2   +1    │
│ −2   0   +2     │     │  0    0    0    │
│ −1   0   +1     │     │ −1   −2   −1    │
└─────────────────┘     └─────────────────┘
  (phải − trái)           (trên − dưới)
  phát hiện cạnh DỌC      phát hiện cạnh NGANG
```

Kernel chỉ là bộ trọng số — nó không tự làm gì. Cơ chế áp dụng nó lên ảnh gọi là **tích chập**.

---

##### Tích chập (convolution) — cơ chế trượt và nhân

Tích chập = **trượt kernel qua từng pixel của ảnh**, tại mỗi pixel thực hiện 3 bước:

```
1. Đặt kernel 3×3 lên pixel đó (kernel nằm giữa, bao quanh pixel)
2. Nhân từng ô của kernel với pixel tương ứng trong ảnh
3. Cộng tất cả 9 tích lại → đó là giá trị đầu ra tại pixel đó
```

**Ví dụ cụ thể tại pixel P (giữa = 203), vùng cạnh giấy:**

```
Vùng ảnh 3×3:          Kernel Sobel X:        Nhân từng ô:
┌───────────────┐       ┌─────────────┐        ┌───────────────────────┐
│ a=200 b=205 c=40│      │ −1   0   +1 │        │ 200×(−1) 205×0  40×(+1)│
│ d=202 e=203 f=38│  ×   │ −2   0   +2 │   =    │ 202×(−2) 203×0  38×(+2)│
│ g=198 h=201 i=42│      │ −1   0   +1 │        │ 198×(−1) 201×0  42×(+1)│
└───────────────┘       └─────────────┘        └───────────────────────┘

Gx = (−200 +   0 +  40)
   + (−404 +   0 +  76)
   + (−198 +   0 +  42)
   = −644   ← rất lớn → có cạnh dọc tại P!
```

**Quan sát:** Cột giữa (b, e, h) có hệ số 0 → không đóng góp gì. Chỉ có:
- Cột trái × (−1, −2, −1) → bị trừ đi
- Cột phải × (+1, +2, +1) → được cộng vào
- Kết quả = **(cột phải − cột trái)** = đạo hàm theo chiều ngang ✓

Nếu toàn vùng đồng màu (~200 khắp nơi):

```
Gx = (−200+0+200) + (−400+0+400) + (−200+0+200) = 0   ← không có cạnh
```

---

##### Tại sao Sobel có 3 hàng thay vì chỉ 1 hàng `[−1, 0, +1]`?

Nếu chỉ dùng 1 hàng, ta tính đạo hàm trên **đúng 1 dòng pixel** — rất nhạy với nhiễu tại dòng đó.

Sobel tính đạo hàm trên **3 dòng song song** rồi cộng có trọng số:

```
Hàng trên:  [−1, 0, +1]  × 1   ← đóng góp nhỏ
Hàng giữa:  [−1, 0, +1]  × 2   ← đóng góp lớn hơn (pixel đang xét)
Hàng dưới:  [−1, 0, +1]  × 1   ← đóng góp nhỏ

Gx = 1×(đạo hàm hàng trên) + 2×(đạo hàm hàng giữa) + 1×(đạo hàm hàng dưới)
```

Trọng số 1-2-1 là xấp xỉ Gaussian 1D — làm mịn nhẹ theo chiều dọc trước khi tính đạo hàm theo chiều ngang, giúp kết quả ít bị ảnh hưởng bởi nhiễu một dòng đơn lẻ.

---

**Tổng hợp Gx và Gy thành G và θ:**

```
G = √(Gx² + Gy²)
θ = atan2(Gy, Gx)
```

**G — mức độ tương phản tại pixel đó**

G đo hai bên pixel khác nhau nhiều cỡ nào. Hình dung như độ dốc của một con dốc:

```
Vùng phẳng (nền giấy):   Cạnh nhẹ (chữ in):   Cạnh sắc (viền giấy):
200  201  202             200  180  150          200  205   40
G ≈ 2 → không có cạnh    G ≈ 50 → cạnh mờ      G ≈ 644 → cạnh rõ nét
```

G lớn = hai bên pixel khác nhau nhiều = viền sắc nét.
G nhỏ = hai bên gần giống nhau = không có viền.

**θ — hướng đi về phía sáng hơn**

θ không phải góc của cạnh. θ là hướng bạn phải đi để pixel càng lúc càng sáng hơn. Đo từ hướng phải (0°), ngược chiều kim đồng hồ:

```
         90° (lên)
              ↑
180° (trái) ←   → 0° (phải)
              ↓
         270° (xuống)
```

Ví dụ Gx = −644, Gy = −12:

```
Ảnh tại vùng cạnh giấy:
pixel trái ≈ 205 (sáng)  |  pixel phải ≈ 40 (tối)
                          ↑ cạnh dọc

Muốn đi về phía sáng hơn → phải đi sang TRÁI = 180°
Gy = −12 (rất nhỏ) → lệch xuống dưới một chút → θ = 181°

Cạnh nằm VUÔNG GÓC với hướng sáng dần:
hướng sáng dần = 181° (gần như trái)
→ cạnh = gần như thẳng đứng ✓
```

θ được dùng ở bước NMS tiếp theo: để biết nhìn về hướng nào khi so sánh xem pixel hiện tại có phải cực đại không.

> **Tại sao cần cả Gx lẫn Gy?** Gx phát hiện cạnh dọc, Gy phát hiện cạnh ngang. Kết hợp `G = √(Gx²+Gy²)` phát hiện cạnh theo mọi hướng.

---

#### Bước 2 nội bộ — Non-Maximum Suppression (NMS): làm mỏng biên xuống 1 pixel

**Vấn đề sau bước tính gradient:** Mỗi cạnh là một dải rộng vài pixel:

```
G của mỗi pixel tại vùng cạnh tờ giấy:

pixel:  ...  P1   P2   P3   P4   P5  ...
G:      ...  120  380  644  410   95  ...
                        ↑
                   cạnh thật chỉ ở đây (G lớn nhất)
                   nhưng P2, P4 cũng có G cao → biên dày 3 pixel
```

Nếu giữ tất cả → biên dày → `findContours` sau này cho ra contour phập phù.

**Giải pháp:** Tại mỗi pixel, so sánh G của nó với 2 pixel lân cận **dọc theo hướng θ**. Chỉ giữ nếu là cực đại cục bộ, còn lại đặt về 0.

```
Tại P3 (G=644, θ=90° — hướng dọc):
  → So sánh với pixel phía trên (G=380) và phía dưới (G=410)
  → 644 > 380 và 644 > 410 → P3 là cực đại → GIỮ

Tại P2 (G=380, θ=90°):
  → So sánh với pixel phía trên (G=120) và phía dưới (G=644)
  → 380 < 644 → P2 không phải cực đại → XÓA (đặt về 0)

Kết quả:
pixel:  ...  P1   P2   P3   P4   P5  ...
G:      ...   0    0   644   0    0  ...   ← biên mỏng 1 pixel!
```

---

#### Bước 3 & 4 nội bộ — Double Threshold + Hysteresis Tracking: lọc nhiễu

**Vấn đề:** Ngay cả sau Gaussian blur, nhiễu vẫn tạo ra gradient nhỏ khắp nơi — nếu giữ tất cả thì biên đầy "chấm muỗi".

**Giải pháp — 2 ngưỡng:**

```
G > 180        →  BIÊN MẠNH (strong)  — chắc chắn là cạnh thật
60 < G ≤ 180   →  BIÊN YẾU  (weak)   — có thể là cạnh, có thể là nhiễu
G ≤ 60         →  LOẠI BỎ   (noise)  — không phải cạnh
```

**Hysteresis Tracking** — quy tắc chọn biên yếu:

```
Biên yếu được GIỮ nếu: có ít nhất 1 biên MẠNH trong 8 ô lân cận
Biên yếu bị XÓA nếu:  hoàn toàn cô lập (8 ô lân cận đều là weak hoặc noise)
```

**Ví dụ — cạnh giấy thật (được giữ):**

```
[STRONG=644]─[weak=180]─[weak=150]─[weak=90]─[noise=40]
      ↑             ↑          ↑          ↑          ↑
  luôn giữ    kề STRONG   kề weak    kề weak    cô lập
               → giữ      đã giữ     đã giữ     → xóa
                           → giữ      → giữ

Kết quả: cả đoạn cạnh được giữ liên tục ✓
```

**Ví dụ — nhiễu cô lập (bị xóa):**

```
[noise=30] [weak=90] [noise=25] [weak=110] [noise=20]
                ↑                    ↑
          không kề STRONG       không kề STRONG
              → xóa                → xóa   ✓
```

**Tại sao ngưỡng thấp = 60, cao = 180?**

```
Viền tờ giấy (tương phản cao):   G ≈ 200–600 → vượt ngưỡng 180 → STRONG ✓
Nội dung in trên phiếu (vừa):    G ≈ 80–200  → weak, được kéo theo viền ✓
Nhiễu JPEG/sensor:               G ≈ 10–50   → dưới 60 → bị loại ✓
```

Tỷ lệ 1:3 (60:180) là quy tắc kinh nghiệm từ paper gốc của Canny. Ngưỡng cao 180 lọc được nhiễu mà không bỏ sót viền giấy; ngưỡng thấp 60 đủ thấp để "kéo theo" phần biên yếu dọc cạnh giấy dài.

---

#### Tổng kết luồng Canny

```
gray (sau Gaussian Blur)
        │
        ▼ Sobel Gx, Gy
        G = √(Gx²+Gy²),  θ = atan2(Gy,Gx)
        Ảnh G:  ......░░▓████▓░░......   ← dải biên rộng
        │
        ▼ Non-Maximum Suppression (theo hướng θ)
        Ảnh NMS:  ......░░░░█░░░░......  ← 1 pixel mỏng
        │
        ▼ Double Threshold
        S = G > 180,  w = 60 < G ≤ 180,  . = G ≤ 60
        Ảnh:  ...S.w.w.S.S.w...w...
        │
        ▼ Hysteresis: w kề S → giữ; w cô lập → xóa
        edges:  ....████████.........   ← biên sạch, mỏng, liên tục
```

**Đầu ra `edges`:** ảnh nhị phân, pixel trắng (255) = biên phát hiện được, nền đen (0).

---

### 3c. Dilation 3×3, 1 lần lặp

```python
edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
```

**Thuật toán Dilation:** Với mỗi pixel trắng trong `edges`, tô trắng tất cả pixel trong vùng 3×3 xung quanh nó. Kết quả: các đường biên "phình ra" 1 pixel mỗi phía.

**Tại sao cần dilation sau Canny?**

Canny đôi khi tạo ra biên **không liên tục** — có chỗ bị hở 1–2 pixel do:
- Góc giấy hơi cong → gradient không đủ mạnh tại một vài điểm
- JPEG compression artifact làm gradient bị gián đoạn

`findContours` (bước tiếp theo) yêu cầu đường bao **liên thông** để tạo thành một contour. Nếu biên bị hở, trang giấy có thể bị chia thành nhiều contour nhỏ thay vì một contour lớn.

Dilation lấp hở 1–2 pixel → đảm bảo contour trang giấy liền mạch.

**Tại sao chỉ 1 lần lặp (không phải 2–3)?**

Dilation quá nhiều lần làm các contour gần nhau bị **hợp nhất** (merge) — đặc biệt nguy hiểm nếu có văn bản, đường kẻ, hay bảng gần viền trang. 1 lần đủ để lấp hở mà không gây merge.

---

### 3d. findContours — tìm đường bao ngoài

```python
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
```

**Thuật toán:** Duyệt toàn bộ ảnh `edges`, nhóm các pixel trắng liền thông thành từng **contour** (chuỗi điểm biên).

**`RETR_EXTERNAL`:** Chỉ trả về contour **ngoài cùng** — bỏ qua contour lồng bên trong. Ví dụ: viền bảng câu hỏi bên trong trang giấy sẽ không được trả về, chỉ có viền ngoài trang giấy.

> Tại sao không dùng `RETR_LIST` hay `RETR_TREE`? Vì chúng ta chỉ quan tâm đến hình dạng ngoài cùng (trang giấy). Đưa thêm contour nội bộ vào chỉ tốn thời gian lọc.

**`CHAIN_APPROX_SIMPLE`:** Nén đoạn thẳng trong contour. Ví dụ: cạnh thẳng nằm ngang dài 500 pixel chỉ được lưu bằng 2 điểm đầu-cuối thay vì 500 điểm. Tiết kiệm bộ nhớ đáng kể với ảnh phân giải cao.

---

### 3e. Lọc top-10 contour theo diện tích, ngưỡng ≥ 18%

```python
contours = sorted(contours, key=cv2.contourArea, reverse=True)
for cnt in contours[:10]:
    area = float(cv2.contourArea(cnt))
    if area < total_area * 0.18:
        continue
```

**Logic:**

1. **Sắp xếp giảm dần theo diện tích** — contour trang giấy gần như luôn là contour lớn nhất.
2. **Chỉ xét top-10** — bỏ qua hàng trăm contour nhỏ của chữ, đường kẻ, bong bóng câu trả lời.
3. **Ngưỡng 18% (0.18)** — nếu contour chiếm ít hơn 18% diện tích ảnh thì không thể là trang giấy.

**Tại sao 18%?**

- Phiếu trắc nghiệm A4 được chụp từ xa tối thiểu vài chục cm để vừa trang → trang giấy chiếm ít nhất 20–30% khung hình trong điều kiện bình thường.
- Ngưỡng 18% đặt hơi thấp hơn 20% để có dư địa với ảnh chụp xa hoặc góc nghiêng lớn làm trang trông nhỏ hơn thực.
- Không đặt thấp hơn nữa (ví dụ 10%) để tránh nhận nhầm bảng nội dung hay khung câu hỏi lớn.

---

### 3f. approxPolyDP — xấp xỉ đa giác (Douglas-Peucker)

```python
peri = float(cv2.arcLength(cnt, True))
approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
if len(approx) == 4:
    pts = approx.reshape(4, 2).astype(np.float32)
    return _order_quad_points(pts)
```

#### Vấn đề: findContours trả về quá nhiều điểm

Contour của viền tờ giấy có thể chứa **hàng trăm đến hàng nghìn điểm** vì nó theo sát từng pixel trên đường biên. Mục tiêu là tìm **4 góc** — nhưng làm sao biết trong hàng nghìn điểm đó điểm nào là góc?

#### Quy tắc duy nhất của thuật toán

> **Luôn nối HAI ĐẦU MÚT của đoạn đang xét. Sau đó đo khoảng cách từng điểm còn lại tới đường thẳng đó.**

Không có điều kiện nào khác về việc chọn điểm nào để nối.

#### Epsilon là gì

Epsilon là **ngưỡng khoảng cách tính bằng pixel**. Nó trả lời câu hỏi:

> *"Một điểm phải cách đường thẳng bao nhiêu pixel thì mới được coi là quan trọng?"*

- Điểm cách đường thẳng **> epsilon** → quan trọng (có thể là góc) → **giữ**
- Điểm cách đường thẳng **≤ epsilon** → không quan trọng (nằm trên cạnh thẳng) → **xóa**

#### Thuật toán — 3 bước, lặp lại trên từng đoạn

```
Bước 1: Nối hai đầu mút của đoạn đang xét thành đường thẳng
Bước 2: Đo khoảng cách từng điểm còn lại tới đường thẳng đó
         → Tìm điểm Pm có khoảng cách d lớn nhất
Bước 3:
  Nếu d > epsilon → Pm là góc, GIỮ LẠI
                     Chia đoạn thành 2 tại Pm
                     Áp dụng lại 3 bước này cho từng đoạn nhỏ
  Nếu d ≤ epsilon → mọi điểm giữa đều sát đường thẳng, XÓA HẾT
```

"Áp dụng lại 3 bước này cho từng đoạn nhỏ" = lặp lại đúng thao tác đó trên mảnh nhỏ hơn, không có gì phức tạp hơn.

#### Tại sao lần đầu nối điểm đầu với điểm cuối của contour?

Vì đó là hai đầu mút của đoạn đang xét. Lần đầu tiên, đoạn đang xét là **toàn bộ danh sách contour** từ P0 đến P(N−1):

```
contour = [P0, P1, P2, ..., P847]
              ↑                ↑
         đầu mút trái    đầu mút phải

→ Lần đầu: nối P0 với P847
```

#### Tại sao không nối thẳng với điểm xa nhất ngay?

Vì **ta chưa biết điểm nào xa nhất** — đó chính là thứ ta đang đi tìm. Phải vẽ đường P0→P847 trước, rồi mới đo để phát hiện ra điểm xa nhất là ai.

#### Ví dụ đầy đủ với tờ giấy 4 góc

```
Contour đi theo thứ tự:
[TL=P0, ...cạnh trên..., TR=P200, ...cạnh phải...,
 BR=P450, ...cạnh dưới..., BL=P700, ...cạnh trái..., P847≈TL]
```

**Lần 1 — đoạn [P0..P847], nối P0(TL)→P847(≈TL):**

```
P0(TL) ─────── P847(≈TL)   ← đường thẳng rất ngắn (gần nhau)
                                tất cả điểm còn lại đều lồi ra xa
→ xa nhất là BR(P450), cách ~700px >> epsilon
→ GIỮ BR, chia thành [P0..P450] và [P450..P847]
```

**Lần 2a — đoạn [P0..P450], nối P0(TL)→P450(BR):**

```
P0(TL)
  *  \
  *    \   ← đường chéo TL→BR
  *      \
           * P450(BR)
→ xa nhất là TR(P200), cách ~500px >> epsilon
→ GIỮ TR, chia thành [P0..P200] và [P200..P450]
```

**Lần 2b — đoạn [P450..P847], nối P450(BR)→P847(≈TL):**

```
→ xa nhất là BL(P700), cách ~500px >> epsilon
→ GIỮ BL, chia thành [P450..P700] và [P700..P847]
```

**Lần 3 — các đoạn còn lại đều là cạnh thẳng:**

```
[P0..P200]   → nối TL→TR → các điểm giữa cách <5px ≤ epsilon → XÓA HẾT
[P200..P450] → nối TR→BR → tương tự → XÓA HẾT
[P450..P700] → nối BR→BL → tương tự → XÓA HẾT
[P700..P847] → nối BL→TL → tương tự → XÓA HẾT
```

**Kết quả:** chỉ còn `[TL, TR, BR, BL]` = 4 góc ✓

#### Epsilon được tính như thế nào

```python
peri = float(cv2.arcLength(cnt, True))   # tổng độ dài chu vi contour (pixel)
# bên trong approxPolyDP:
epsilon = 0.02 * peri
```

`cv2.arcLength` cộng khoảng cách giữa tất cả các điểm liền nhau trong contour:

```
arcLength = dist(P0→P1) + dist(P1→P2) + ... + dist(P847→P0)
          = tổng pixel của đường viền

Ví dụ: peri = 15.000px
→ epsilon = 0.02 × 15.000 = 300px
```

Cả `d` (khoảng cách điểm tới đường thẳng) lẫn `epsilon` đều đơn vị pixel — so sánh trực tiếp được với nhau.

#### Tại sao dùng tỉ lệ (2%) thay vì số pixel cố định?

Vì ảnh đầu vào có kích thước khác nhau tùy camera và khoảng cách chụp. Nếu dùng epsilon cố định, cùng một ngưỡng sẽ hoạt động sai ở các kích thước khác nhau:

```
Tình huống A — ảnh chụp gần, tờ giấy to:
  chu vi contour = 20.000px
  Góc lồi ra: ~1.000px     Nhiễu cạnh thẳng: ~8px

Tình huống B — ảnh chụp xa, tờ giấy nhỏ:
  chu vi contour = 4.000px
  Góc lồi ra: ~200px       Nhiễu cạnh thẳng: ~2px
```

Thử dùng epsilon cố định = 50px:

```
Tình huống A: góc 1.000px >> 50px ✓  |  nhiễu 8px < 50px ✓   → hoạt động đúng
Tình huống B: góc 200px >> 50px ✓    |  nhiễu 2px < 50px ✓   → hoạt động đúng
```

Tưởng như ổn, nhưng thử epsilon cố định = 300px:

```
Tình huống A: góc 1.000px >> 300px ✓  |  nhiễu 8px < 300px ✓  → đúng
Tình huống B: góc 200px < 300px ✗     → góc bị xóa → ra tam giác!
```

Không có con số cố định nào an toàn cho mọi kích thước. Dùng tỉ lệ giải quyết điều này — epsilon tự co giãn cùng với kích thước contour:

```
Tình huống A: epsilon = 2% × 20.000 = 400px
  góc ~1.000px >> 400px ✓   nhiễu ~8px << 400px ✓

Tình huống B: epsilon = 2% × 4.000 = 80px
  góc ~200px >> 80px ✓      nhiễu ~2px << 80px ✓
```

#### Tại sao tỉ lệ là 2%, không phải 1% hay 5%?

Nhìn vào khoảng cách lồi của góc và nhiễu so với chu vi:

```
Góc vuông tờ giấy lồi ra ≈ 25–40% chiều dài cạnh
  = khoảng 6–10% chu vi (vì chu vi = 4 cạnh)

Nhiễu dọc cạnh thẳng lồi ra ≈ vài pixel
  = khoảng 0.01–0.05% chu vi
```

Cần chọn tỉ lệ nằm **giữa hai vùng** này:

```
0.01% chu vi ← nhiễu          góc → ~6% chu vi
     |_______________|___________________|
                     ↑
               vùng an toàn để đặt epsilon
               (bất kỳ % nào từ ~0.5% đến ~4%)

2% được chọn nằm chắc giữa vùng an toàn đó.
```

Nếu epsilon quá nhỏ (0.1%): nhiễu vượt ngưỡng → ra 20–30 đỉnh thay vì 4.
Nếu epsilon quá lớn (10%): góc không vượt ngưỡng → bị xóa → ra tam giác.

#### Điều kiện `len(approx) == 4`

Nếu kết quả sau thuật toán có đúng 4 điểm → đây là tứ giác = hình dạng trang giấy → trả về ngay.

---

### 3g. minAreaRect + boxPoints — fallback trong fallback

```python
rect = cv2.minAreaRect(cnt)
box = cv2.boxPoints(rect)
if box is not None and len(box) == 4:
    box = box.astype(np.float32)
    box_area = float(cv2.contourArea(box))
    if box_area >= total_area * 0.16:
        return _order_quad_points(box)
```

**Vị trí chính xác trong luồng — "fallback trong fallback" nghĩa là gì:** vòng lặp `for cnt in contours[:10]` (mục 3e) duyệt qua tối đa 10 contour lớn nhất. Với **mỗi** contour đó, `approxPolyDP` (3f) được thử trước; nếu không ra đúng 4 đỉnh, 3g được thử tiếp **trên cùng contour đó** trước khi bỏ cuộc và chuyển sang xét contour lớn thứ 2, thứ 3... trong top-10:

```
for cnt in top-10 contours:
    approx = approxPolyDP(cnt, ...)      ← 3f: thử trước
    if len(approx) == 4:
        return quad                       ✓ THÀNH CÔNG — dừng luôn
    else:
        rect = minAreaRect(cnt)           ← 3g: thử tiếp, VẪN TRÊN CÙNG cnt đó
        box = boxPoints(rect)
        if box_area đủ lớn:
            return quad                   ✓ THÀNH CÔNG (kém chính xác hơn)
        else:
            continue                      ✗ bỏ cnt này, sang cnt tiếp theo
```

**Khi nào được dùng?** Khi `approxPolyDP` không cho ra đúng 4 đỉnh — ví dụ trang giấy bị gấp góc, viền bị cắt, hay đường Canny bị nhiễu tạo ra đa giác 5–8 đỉnh.

**Mục đích — giải quyết vấn đề gì mà 3f không giải quyết được:** `approxPolyDP` chỉ trả về đúng 4 điểm khi contour thực sự có hình dạng gần tứ giác sạch. Trên thực tế, contour tờ giấy có thể bị:

- **Gấp góc** (1 góc bị bẻ cong) → thay vì 1 góc nhọn, contour có 2 điểm góc gần nhau → `approxPolyDP` giữ lại cả 2 → ra **5 đỉnh**.
- **Viền bị cắt khỏi khung hình** (chụp quá gần, 1 cạnh phiếu nằm ngoài ảnh) → contour bị "hở" 1 đoạn, `findContours` tự nối đường thẳng chỗ hở theo cách không tự nhiên → sinh thêm góc lạ → **5-6 đỉnh**.
- **Canny bị nhiễu cục bộ** (mục 3b) → viền bị "răng cưa" nhẹ ở 1 đoạn → dù đã qua `approxPolyDP`, đoạn răng cưa đó vẫn có thể sinh thêm 1-2 đỉnh phụ không vượt hẳn ngưỡng epsilon để bị xóa hết → **6-8 đỉnh**.

Trong mọi trường hợp trên, `len(approx) != 4` → nếu không có 3g, contour đó bị bỏ hoàn toàn. 3g tồn tại để không bỏ phí contour đó — vẫn ước lượng ra 1 tứ giác "tạm chấp nhận được" bằng 1 kỹ thuật khác hẳn.

**`cv2.minAreaRect(cnt)` hoạt động thế nào:** khác biệt cốt lõi so với `approxPolyDP` là nó **không quan tâm hình dạng thật** của contour — chỉ tìm 1 **hình chữ nhật xoay được, nhỏ nhất có thể, sao cho bao trọn tất cả các điểm của contour vào bên trong**. Vì luôn là hình chữ nhật (4 cạnh, 4 góc vuông), nó **luôn** cho ra đúng 4 điểm, bất kể contour đầu vào có 5, 6, hay 50 đỉnh. Trả về `(center, (width, height), angle)`.

Thuật toán bên trong (rotating calipers):
1. Tính convex hull của contour (bao lồi — đã học ở `Buoc2.md` mục 6.3 tiêu chí 7).
2. Định lý hình học: hình chữ nhật nhỏ nhất bao quanh 1 tập điểm lồi **luôn có ít nhất 1 cạnh trùng phương với 1 cạnh của bao lồi đó**. Nhờ định lý này, không cần thử vô số góc xoay — chỉ cần thử đúng bằng số cạnh của bao lồi.
3. Với mỗi cạnh của bao lồi: xoay hệ trục tọa độ sao cho cạnh đó nằm ngang, tính bounding box (hình chữ nhật thẳng trục) của toàn bộ điểm trong hệ trục đã xoay đó → ra 1 hình chữ nhật ứng viên, tính diện tích.
4. Lặp lại cho mọi cạnh của bao lồi, chọn hình chữ nhật có **diện tích nhỏ nhất**.

**`cv2.boxPoints(rect)`:** bước chuyển đổi hình học đơn giản — từ `(tâm, kích thước, góc xoay)`, tính ra tọa độ 4 góc thực tế bằng cách xoay 4 nửa-đường-chéo quanh tâm theo `angle`.

**Ví dụ minh họa — contour bị gấp góc:**

```
Contour thật (góc trên-phải bị gấp lõm vào 1 chút):

  TL●──────────●●────────●TR'      ← approxPolyDP giữ CẢ 2 điểm gần TR
    │                      │           (chỗ gấp) vì cả 2 đều "lồi ra" đủ xa
    │                      │           so với đường nối 2 điểm lân cận
    │                      │        → ra 5 đỉnh: TL, 2 điểm ở chỗ gấp, BR, BL
   BL●──────────────────●BR         → 3f thất bại (len(approx)=5)

minAreaRect bao quanh TOÀN BỘ các điểm trên (kể cả 2 điểm ở chỗ gấp):

  ┌──────────────────────┐   ← hình chữ nhật nhỏ nhất chứa hết
  │  TL           TR'     │      mọi điểm của contour, kể cả
  │                        │      phần lồi ra do gấp góc
  │                        │
  └──────────────────────┘
  → 4 góc của hình chữ nhật này ≈ 4 góc tờ giấy, dù không khớp
    tuyệt đối tại vùng bị gấp
```

**Tại sao ngưỡng là 16% (không phải 18%)?**

- `minAreaRect` tạo ra hình chữ nhật **bao quanh** contour, nên `box_area` luôn **≥** `area` của contour thực.
- Tuy nhiên với ảnh bị nhiễu, `box_area` có thể lớn hơn contour thực khá nhiều.
- Hạ ngưỡng xuống 16% (thay vì 18%) để chấp nhận trường hợp contour hơi nhỏ hơn nhưng box lại hợp lệ.
- Không hạ xuống dưới 15% để tránh nhận nhầm bảng nội dung lớn làm "trang".

---

### 3h. _order_quad_points — sắp xếp 4 góc theo thứ tự chuẩn

```python
return _order_quad_points(pts)   # hoặc _order_quad_points(box)
```

**Mục đích:** `cv2.getPerspectiveTransform()` (Bước 4) yêu cầu 4 điểm nguồn và 4 điểm đích phải **cùng thứ tự** (top-left → top-right → bottom-right → bottom-left). Nếu thứ tự sai, ma trận warp sẽ phản chiếu hoặc xoay ngược ảnh.

`_order_quad_points` sắp xếp 4 điểm bất kỳ về thứ tự chuẩn:
- **Top-Left (TL):** tổng x+y nhỏ nhất
- **Bottom-Right (BR):** tổng x+y lớn nhất
- **Top-Right (TR):** hiệu x-y lớn nhất
- **Bottom-Left (BL):** hiệu x-y nhỏ nhất

---

## 5. Đầu ra (Output)

| Trường hợp | Giá trị trả về | Ý nghĩa |
|-----------|---------------|---------|
| Thành công | `np.ndarray` shape `(4, 2)`, dtype `float32` | 4 điểm góc TL/TR/BR/BL của trang giấy |
| Thất bại | `None` | Không tìm được tứ giác hợp lệ |

Hàm `_detect_page_quad()` bọc thêm:
- Thành công: trả `(pts, "page-contour")` — chuỗi `"page-contour"` được ghi vào `warp_strategy` để debug
- Thất bại: trả `(None, "none")`

---

## 6. Hậu quả khi Bước 3 cũng thất bại

```python
# omr_service.py — sau khi gọi _warp_to_standard_layout()
if not global_warp_used:
    warnings.append("Khong tim thay 4 goc marker ro rang, da fallback ve resize thong thuong.")
    warning_codes.append("COORD_GLOBAL_WARP_FALLBACK")
```

Khi cả Bước 2 lẫn Bước 3 đều trả `None`:
1. `_warp_to_standard_layout()` **chỉ resize** ảnh về 1000×1400 — không có perspective warp
2. Cờ `COORD_GLOBAL_WARP_FALLBACK` được ghi vào `warning_codes` trong JSON kết quả
3. Nếu ảnh bị nghiêng/chụp xiên → ảnh resize cũng nghiêng → ROI câu MCQ, MSSV, mã đề sẽ lệch
4. Bước 8 (`COORD_ANCHOR_FALLBACK`) có thể xảy ra tiếp theo nếu marker fiducial cũng không tìm được

---

## 7. So sánh Bước 2 vs Bước 3

| Tiêu chí | Bước 2 — Corner Marker | Bước 3 — Contour |
|---------|----------------------|-----------------|
| **Đặc trưng khai thác** | 4 ô vuông đen in sẵn tại góc phiếu | Viền ngoài tờ giấy trắng |
| **Hàm chính** | `_detect_page_corners_from_black_square_markers()` | `_find_page_quad_by_contour()` |
| **Độ chính xác** | Cao — marker in cố định, vị trí pixel-perfect | Thấp hơn — phụ thuộc chất lượng biên Canny |
| **Điều kiện thất bại** | Marker bị che, mờ, cắt xén, biến dạng | Tờ giấy bị che > 82%, nền quá phức tạp, ảnh overexposed |
| **Độ bền với ánh sáng xấu** | Kém (cần nhị phân hóa chính xác để tách marker) | Tốt hơn (Canny làm việc trên gradient, ít nhạy với offset sáng) |
| **Kết quả** | 4 điểm chính xác, sai số < 5 px | 4 điểm ước lượng, sai số có thể 10–30 px |
| **Thứ tự ưu tiên** | Thứ 1 — luôn thử trước | Thứ 2 — chỉ dùng khi Bước 2 thất bại |

---

## 8. Sơ đồ code đầy đủ

```
omr_service.py
    └── _warp_to_standard_layout(img_bgr, ...)
            └── _detect_page_quad(gray_img)            ← omr_preprocess.py:141
                    │
                    ├─ [Thử Bước 2]
                    │   omr_marker_utils._detect_page_corners_from_black_square_markers(gray_img)
                    │   → Thành công: return (pts, "corner-markers")
                    │   → Thất bại:  None
                    │
                    └─ [Thử Bước 3 nếu Bước 2 thất bại]
                        _find_page_quad_by_contour(gray_img)   ← omr_preprocess.py:100
                            ├─ GaussianBlur(5×5)
                            ├─ Canny(60, 180)
                            ├─ dilate(3×3, 1 iter)
                            ├─ findContours(RETR_EXTERNAL)
                            ├─ sort by area DESC, top-10
                            ├─ filter area ≥ 18%
                            ├─ approxPolyDP(epsilon=2%) → 4 đỉnh?
                            │   ├─ Có → _order_quad_points → return pts
                            │   └─ Không → minAreaRect → boxPoints
                            │              box_area ≥ 16%?
                            │               ├─ Có → _order_quad_points → return pts
                            │               └─ Không → thử contour tiếp
                            └─ Hết contour → return None

        Nếu cả hai đều None → chỉ resize, không warp
        → warning_codes += ["COORD_GLOBAL_WARP_FALLBACK"]
```

---

## 9. Các ngưỡng quan trọng tóm tắt

| Tham số | Giá trị | Vị trí trong code | Lý do |
|---------|---------|------------------|-------|
| Gaussian kernel | `(5, 5)` | dòng 107 | Loại nhiễu vừa đủ, không bo góc |
| Canny thấp | `60` | dòng 108 | Biên yếu — nối cạnh giấy dài |
| Canny cao | `180` | dòng 108 | Biên mạnh — viền trang rõ nét |
| Dilation kernel | `(3, 3)`, 1 lần | dòng 109 | Lấp hở 1–2 px, không merge contour |
| Top-N contour | `10` | dòng 116 | Bỏ qua contour nhỏ lẻ |
| Diện tích tối thiểu (contour) | `0.18 × total` | dòng 118 | Loại bảng/chữ/đường kẻ |
| epsilon approxPolyDP | `0.02 × peri` | dòng 125 | 2% chu vi → tứ giác đúng 4 đỉnh |
| Diện tích tối thiểu (box) | `0.16 × total` | dòng 135 | Hơi lỏng hơn để bù box vs contour |
