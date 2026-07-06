# Bước 4 — Warp phối cảnh và chuẩn hóa kích thước

> **Vị trí trong pipeline:** Sau khi Bước 2 hoặc 3 tìm được 4 điểm góc tờ giấy, Bước 4 dùng 4 điểm đó để "kéo thẳng" ảnh về góc nhìn trực diện, rồi resize về kích thước chuẩn để mọi bước sau xử lý trên cùng một hệ tọa độ.

**Module:** `be/app/services/omr/omr_preprocess.py`
**Hàm chính:** `_warp_to_standard_layout()` (dòng 93–130)
**Caller:** `omr_service.py` dòng 999, trong `process_omr_exam()`

---

## 1. Danh sách hàm được dùng trong Bước 4 — mỗi hàm để làm gì

Tra cứu nhanh trước khi đọc mạch trace ở mục 2. Mỗi hàm chỉ giải thích ngắn gọn **dùng để làm gì** — mục 2 sẽ đi qua chúng theo đúng thứ tự thực thi, mục 4-6 giải thích sâu cách tính toán bên trong.

| # | Hàm | Mục đích ngắn gọn | Input | Output |
|---|---|---|---|---|
| 1 | `cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)` | Chuyển ảnh màu (3 kênh BGR) thành ảnh xám (1 kênh) | Ảnh BGR bất kỳ kích thước | Ảnh xám cùng kích thước |
| 2 | `_detect_page_quad(gray_img)` | Tìm 4 điểm góc trang giấy — thử marker (Bước 2) trước, thất bại thì thử contour (Bước 3) | Ảnh xám | `(quad, quad_strategy)` — `quad` là 4 điểm hoặc `None` |
| 3 | `cv2.getPerspectiveTransform(src, dst)` | Tính ra 1 ma trận mô tả phép biến đổi biến 4 điểm `src` thành 4 điểm `dst` tương ứng | 2 mảng 4 điểm (`src=quad`, `dst`=4 góc hình chữ nhật chuẩn) | Ma trận 3×3 |
| 4 | `cv2.warpPerspective(img, matrix, size)` | Áp dụng ma trận đó lên toàn bộ ảnh, tạo ảnh mới đã "duỗi thẳng" | Ảnh gốc, ma trận, kích thước ảnh đích | Ảnh mới đã warp, đúng kích thước `size` |
| 5 | `_order_quad_points(pts)` | Sắp xếp lại 4 điểm bất kỳ về đúng thứ tự chuẩn TL→TR→BR→BL | Mảng 4 điểm `(x,y)`, thứ tự bất kỳ | Mảng 4 điểm đã sắp `[TL,TR,BR,BL]` |
| 6 | `_norm_quad_from_points(quad, img_w, img_h)` | Đổi tọa độ 4 điểm quad từ pixel tuyệt đối sang tỉ lệ `[0,1]`, chỉ để lưu debug | `quad` (pixel), kích thước ảnh gốc | `dict` 4 điểm, mỗi giá trị trong `[0,1]` |
| 7 | `cv2.resize(img, size, interpolation)` | Co giãn ảnh về đúng kích thước pixel mong muốn | Ảnh, kích thước đích, phương pháp nội suy | Ảnh đã co giãn |

**2 hàm KHÔNG thuộc Bước 4, chỉ nhắc tới ở mục 9 để so sánh:** `_resolve_coordinate_anchors()` và `_default_anchor_percent()` (`be/app/services/omr/omr_layout.py`) chạy ở 1 bước **sau** Bước 4, không phải hàm của `_warp_to_standard_layout()`.

---

## 2. Đi tuần tự theo code — từng dòng một

Toàn bộ hàm `_warp_to_standard_layout()` chỉ có 1 luồng thẳng từ trên xuống, đúng 1 điểm rẽ nhánh (`if quad is not None:`). Đi đúng theo thứ tự dòng chạy thật:

```python
def _warp_to_standard_layout(img_bgr, width_img, height_img, a4_warp_w, a4_warp_h):
```

**Input của cả hàm:** `img_bgr` (ảnh màu gốc, kích thước bất kỳ — từ `cv2.imread`), `width_img=1000`, `height_img=1400`, `a4_warp_w=2480`, `a4_warp_h=3508` (3 hằng số cuối định nghĩa ở `omr_service.py`).

### Dòng 100-108 — khởi tạo giá trị mặc định

```python
src_h, src_w = img_bgr.shape[:2]        # dòng 100 — đọc kích thước ảnh gốc
strategy = "resize-only"                 # dòng 101 — giá trị mặc định, dùng nếu warp KHÔNG xảy ra
global_warp_used = False                 # dòng 102 — mặc định False
info = {"source": {...}, "target": {...}}  # dòng 103-106 — chỉ ghi kích thước nguồn/đích, chưa có quad
working = img_bgr                        # dòng 108 — mặc định "ảnh đang xử lý" = ảnh gốc, chưa đổi gì
```

**Vì sao khởi tạo trước khi biết quad có tìm được hay không?** Đây chính là cơ chế fallback: nếu nhánh warp bên dưới (dòng 112-126) không chạy (vì `quad is None`), 4 biến `strategy`, `global_warp_used`, `working`, `info` vẫn giữ nguyên giá trị mặc định này — hàm vẫn trả về được kết quả hợp lệ (`"resize-only"`, `False`, ảnh gốc chưa warp) thay vì lỗi. Đây là nội dung mục 8 ("Khi Bước 2/3 thất bại") — chỉ là hệ quả trực tiếp của việc gán mặc định ở đây, không phải code riêng biệt nào khác.

### Dòng 110 — chuyển ảnh sang xám

```python
gray_src = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
```

**Input:** `img_bgr` (biến đã có sẵn từ tham số hàm). **Output:** `gray_src` — ảnh cùng kích thước, chỉ còn 1 kênh xám (hàm #1 ở mục 1). **Tại sao cần:** các thuật toán phát hiện góc ở Bước 2/3 chỉ cần độ sáng (để tính ngưỡng nhị phân, gradient Canny...), không cần thông tin màu — chuyển sang xám giảm khối lượng tính toán và khớp đúng input mà `Buoc2.md`/`Buoc3.md` mô tả (biến `gray_img` trong 2 tài liệu đó chính là `gray_src` ở đây).

### Dòng 111 — nhận dữ liệu từ Bước 2/3 (điểm giao thoa quan trọng nhất)

```python
quad, quad_strategy = _detect_page_quad(gray_src)
```

**Input:** `gray_src` (vừa tạo ở dòng 110). **Output:** 1 tuple gồm 2 giá trị — đây chính xác là **toàn bộ dữ liệu băng qua ranh giới** giữa Bước 2/3 và Bước 4:

| Biến | Kiểu | Từ đâu ra |
|---|---|---|
| `quad` | `np.ndarray` shape `(4,2)` float32, thứ tự `[TL, TR, BR, BL]`, hoặc `None` | Bước 2 (`marker_pts`, đã qua padding — `Buoc2.md` mục 7.3) hoặc Bước 3 (`contour_pts` — `Buoc3.md` mục 3f/3g) |
| `quad_strategy` | `str`: `"corner-markers"` / `"page-contour"` / `"none"` | Chuỗi nhãn chẩn đoán, không dùng cho phép toán |

**`_detect_page_quad()`** (hàm #2, `omr_preprocess.py:79-90`) tự nó là hàm điều phối: thử `omr_marker_utils._detect_page_corners_from_black_square_markers(gray_src)` (Bước 2) trước; nếu trả `None`, thử `_find_page_quad_by_contour(gray_src)` (Bước 3). Đây là lý do Bước 2 và Bước 3 **không phải 2 lệnh gọi tách biệt từ bên ngoài** như cách 3 file tài liệu trình bày riêng — cả hai được gọi từ bên trong đúng 1 dòng code này.

**Chi tiết dễ bỏ sót bên trong `_detect_page_quad()` — hoán đổi thứ tự điểm:** Bước 2 (`omr_marker_utils.py`) trả `marker_pts` theo thứ tự `[tl,tr,bl,br]`. Trước khi dùng, `_detect_page_quad()` (dòng 83) phải hoán đổi thành `[tl,tr,br,bl]`:

```python
arr = np.array([arr[0], arr[1], arr[3], arr[2]], dtype=np.float32)
#                tl      tr      br      bl        ← đổi chỗ phần tử 2 và 3
```

Sau đó `_order_quad_points()` (hàm #5 ở mục 1) chạy tiếp — tự sắp lại 4 điểm theo tiêu chí hình học (tổng `x+y` nhỏ nhất = TL, v.v. — `Buoc3.md` mục 3h) bất kể thứ tự đầu vào đúng hay sai, nên đây là lớp tự vệ kép. Kết quả Bước 3 thì đã tự gọi `_order_quad_points()` ngay bên trong nó, không cần hoán đổi thêm.

### Dòng 112 — điểm rẽ nhánh duy nhất

```python
if quad is not None:
```

Từ đây tách thành 2 khả năng: **quad tìm được** (dòng 113-126 chạy) hoặc **quad là `None`** (toàn bộ khối này bị bỏ qua, nhảy thẳng xuống dòng 128 với các giá trị mặc định đã gán ở dòng 100-108).

**Vì sao cần warp khi quad tìm được — vấn đề phối cảnh:** khi chụp điện thoại, tờ giấy hiếm khi vuông góc với ống kính — trông như hình thang trong ảnh:

```
Thực tế (nhìn từ trên):       Trong ảnh chụp nghiêng:

┌─────────────────┐             ╱──────────────╲
│                 │            ╱                ╲
│   Tờ giấy A4   │    →      ╱    Tờ giấy A4   ╲
│                 │          ╱                    ╲
└─────────────────┘         ╱────────────────────╲
```

Nếu không sửa: tọa độ bong bóng MCQ trong ảnh nghiêng bị lệch so với ảnh thẳng. Hệ thống tính vị trí bong bóng theo tỉ lệ % cố định (ví dụ "câu 1 ở 40% chiều cao") — ảnh bị thang thì câu 1 thực ra ở 38% hay 42% → chấm sai. Khối code dòng 113-126 giải quyết việc này bằng cách **kéo 4 góc hình thang (`quad`) về 4 góc hình chữ nhật chuẩn (`dst`)**.

### Dòng 113-121 — tạo 4 điểm đích cố định

```python
dst = np.array([
    [0.0, 0.0],
    [float(a4_warp_w - 1), 0.0],
    [float(a4_warp_w - 1), float(a4_warp_h - 1)],
    [0.0, float(a4_warp_h - 1)],
], dtype=np.float32)
```

**Input:** 2 hằng số `a4_warp_w=2480`, `a4_warp_h=3508` (đã truyền vào hàm từ đầu — lý do chọn con số này giải thích ở mục 5). **Output:** `dst` — 4 góc của 1 hình chữ nhật lý tưởng, **không đổi giữa các lần gọi hàm** (khác với `quad`, vốn khác nhau ở mỗi ảnh). Thứ tự `dst` phải khớp thứ tự `quad` (cùng là `[TL,TR,BR,BL]`) — đây là lý do `_order_quad_points()` ở bước trước quan trọng: nếu `quad` sai thứ tự mà `dst` đúng thứ tự, ảnh kết quả sẽ bị lật/xoay sai.

### Dòng 122 — tính ma trận biến đổi

```python
matrix = cv2.getPerspectiveTransform(quad.astype(np.float32), dst)
```

**Input:** `quad` (từ dòng 111, tức từ Bước 2/3) làm `src`, và `dst` (vừa tạo ở dòng 113-121). **Output:** ma trận 3×3 (hàm #3 ở mục 1). **Cách tính:** giải hệ 8 phương trình tuyến tính từ 4 cặp điểm `src→dst` — chi tiết đầy đủ (công thức, ví dụ số, vì sao cần ma trận 3×3) ở **mục 4** bên dưới.

### Dòng 123 — áp dụng ma trận lên ảnh

```python
working = cv2.warpPerspective(img_bgr, matrix, (a4_warp_w, a4_warp_h))
```

**Input:** `img_bgr` (ảnh **màu gốc**, không phải `gray_src`) và `matrix` (vừa tính). **Output:** `working` được **ghi đè** — không còn là ảnh gốc (dòng 108) nữa mà là ảnh đã "duỗi thẳng", kích thước cố định 2480×3508.

**Lưu ý dễ nhầm:** `quad` được tìm trên ảnh xám (`gray_src`, dòng 110) nhưng warp lại áp dụng lên ảnh màu gốc (`img_bgr`). Việc này hợp lệ vì `cv2.cvtColor(BGR2GRAY)` không đổi kích thước hay tọa độ pixel — chỉ gộp 3 kênh màu thành 1 kênh xám tại đúng vị trí đó, nên tọa độ góc tìm được trên ảnh xám và ảnh màu là cùng 1 hệ tọa độ. **Cách `warpPerspective` lấp từng pixel** — chi tiết ở mục 4.

### Dòng 124-126 — ghi lại 3 giá trị còn phụ thuộc `quad`/`quad_strategy`

```python
strategy = f"coordinate-global-a4:{quad_strategy}"                  # dòng 124
global_warp_used = True                                             # dòng 125
info["detected_quad"] = _norm_quad_from_points(quad, src_w, src_h)  # dòng 126
```

- **Dòng 124:** `quad_strategy` (từ dòng 111 — `"corner-markers"` hoặc `"page-contour"`) chỉ là chuỗi văn bản, được nhúng thẳng vào `strategy` — không tham gia phép tính nào. Đây là cách `"coordinate-global-a4:corner-markers"` xuất hiện trong JSON kết quả cuối cùng.
- **Dòng 125:** cờ xác nhận nhánh warp đã chạy (khác với giá trị mặc định `False` ở dòng 102).
- **Dòng 126:** `_norm_quad_from_points()` (hàm #6 ở mục 1) nhận `quad` (tọa độ pixel tuyệt đối) và kích thước ảnh gốc `src_w, src_h`, chia mỗi tọa độ cho kích thước tương ứng để ra giá trị trong `[0,1]` — chỉ để lưu vào `info` làm **metadata debug** (ví dụ hiển thị lại vị trí quad đã phát hiện trên ảnh gốc), **không** dùng lại cho phép toán warp (warp đã xong ở dòng 123, dùng tọa độ pixel gốc).

### Dòng 128 — chọn phương pháp nội suy

```python
interp = cv2.INTER_AREA if working.shape[1] >= width_img else cv2.INTER_CUBIC
```

**Input:** `working.shape[1]` (chiều rộng hiện tại của `working` — 2480px nếu vừa warp xong, hoặc kích thước ảnh gốc bất kỳ nếu quad thất bại) so với `width_img=1000`. **Output:** hằng số cho biết cách nội suy. **Logic:** nếu ảnh hiện tại rộng hơn đích (trường hợp thường gặp: 2480→1000, đang **thu nhỏ**) → `INTER_AREA` (lấy trung bình vùng pixel bị gộp, mượt hơn khi thu nhỏ); nếu ảnh hiện tại nhỏ hơn đích (ảnh gốc quá nhỏ, đang **phóng to**) → `INTER_CUBIC` (nội suy làm mịn). Lý do chọn đúng 1000×1400 giải thích ở mục 6.

### Dòng 129 — resize về kích thước chuẩn

```python
resized = cv2.resize(working, (width_img, height_img), interpolation=interp)
```

**Input:** `working` (2480×3508 nếu đã warp, hoặc kích thước gốc bất kỳ nếu chưa warp — dòng này chạy **vô điều kiện**, bất kể nhánh dòng 112 có chạy hay không) + `interp` (dòng 128). **Output:** `resized` — luôn đúng 1000×1400, bất kể input trước đó là gì. Đây là lý do mọi ảnh đầu vào (bất kỳ kích thước camera nào, warp thành công hay không) đều cho ra cùng 1 kích thước chuẩn cho các bước sau (hàm #7 ở mục 1).

### Dòng 130 — trả kết quả

```python
return resized, strategy, global_warp_used, info
```

4 giá trị trả về — ý nghĩa từng biến ở bảng mục 8 (Đầu ra). Đến đây mạch trace kết thúc.

---

## 3. Bảng dữ liệu Bước 2/3 → Bước 4 (tóm lại mục 2 ở dạng nhìn nhanh)

| Đến từ | Biến | Dùng ở dòng nào trong Bước 4 |
|---|---|---|
| Bước 2/3 (`_detect_page_quad`) | `quad` | Dòng 122 (làm `src`), dòng 126 (chuẩn hóa lưu debug) |
| Bước 2/3 (`_detect_page_quad`) | `quad_strategy` | Dòng 124 (nhúng vào `strategy`) |

Không còn dữ liệu nào khác đi qua ranh giới này.

---

## 4. Chi tiết toán học — Perspective Transform (dòng 122-123 tính như thế nào)

### 4a. Bắt đầu từ vấn đề — ta muốn điều gì?

Ta có ảnh chụp nghiêng. Tờ giấy trong ảnh trông như hình thang. Ta muốn tạo ra ảnh mới trong đó tờ giấy trông thẳng như hình chữ nhật.

Ta đã biết 4 góc của hình thang trong ảnh gốc (`quad`, từ Bước 2/3 — mục 2). Gọi chúng là P1, P2, P3, P4. Ta muốn: P1 → `(0,0)`, P2 → `(2479,0)`, P3 → `(2479,3507)`, P4 → `(0,3507)` — đây chính là 2 danh sách `src` (=`quad`) và `dst` (dòng 113-121).

### 4b. Công thức biến đổi — 8 số chưa biết

Để biến đổi mỗi điểm `(x, y)` trong ảnh gốc thành `(x_out, y_out)` trong ảnh thẳng:

```
x_out = (a×x + b×y + c) / (g×x + h×y + 1)
y_out = (d×x + e×y + f) / (g×x + h×y + 1)
```

Có **8 số chưa biết**: `a, b, c, d, e, f, g, h`. Mỗi điểm đã biết `(x,y) → (x_out, y_out)` cho 2 phương trình. 4 điểm × 2 = **8 phương trình**, giải ra đúng 8 ẩn.

Ví dụ: nếu P1 = `(120, 85)` phải biến thành `(0, 0)`:

```
0 = (a×120 + b×85 + c) / (g×120 + h×85 + 1)   ← phương trình 1
0 = (d×120 + e×85 + f) / (g×120 + h×85 + 1)   ← phương trình 2
```

`cv2.getPerspectiveTransform(src, dst)` (dòng 122) giải hệ 8 phương trình này và trả về 8 số `a..h`.

### 4c. Ma trận 3×3 — chỉ là cách lưu 8 số đó lại

```
    ┌ a  b  c ┐
M = │ d  e  f │   ← 8 số vừa tìm được + 1 cố định
    └ g  h  1 ┘
```

Ma trận này không phải kết quả cuối — nó là **công cụ** để dùng tiếp ở dòng 123. Mỗi lần muốn biến đổi một điểm `(x, y)`, nhân nó với M để ra `(x_out, y_out)` mà không cần giải lại 8 phương trình.

**Tại sao 3×3 trong khi ảnh chỉ có 2 chiều?** Công thức biến đổi có phép **chia** (`/ (g×x + h×y + 1)`) — không thể viết dưới dạng nhân ma trận thông thường với vector 2D. Thêm chiều thứ 3 (luôn = 1) vào vector điểm để giải quyết:

```
Biểu diễn điểm (x, y) thành vector 3 chiều: [x, y, 1]

Nhân với M:
┌ x' ┐   ┌ a  b  c ┐   ┌ x ┐
│ y' │ = │ d  e  f │ × │ y │
└ w' ┘   └ g  h  1 ┘   └ 1 ┘

Chuyển về 2D:
x_out = x' / w'
y_out = y' / w'
```

Phép chia `/ w'` ở bước cuối chính là phần mà công thức ban đầu có `/ (g×x + h×y + 1)`. Hai cách viết tương đương nhau — ma trận 3×3 chỉ là cách viết gọn hơn.

### 4d. warpPerspective (dòng 123) — áp dụng M lên toàn bộ ảnh

Có M rồi, `cv2.warpPerspective` tạo ra **ảnh đích** kích thước 2480×3508, ban đầu trống rỗng, rồi lấp đầy từng pixel bằng cách đi **ngược**: với mỗi pixel trong ảnh đích, tìm pixel tương ứng trong ảnh gốc.

```
Với mỗi pixel (x_out, y_out) trong ảnh ĐÍCH (2480×3508):
    1. Nhân ngược với M_inverse: tìm (x, y) trong ảnh GỐC
    2. Lấy màu pixel tại (x, y) trong ảnh gốc
    3. Tô màu đó vào (x_out, y_out) của ảnh đích

→ Làm lần lượt cho cả 2480×3508 ≈ 8.7 triệu pixel
```

**Tại sao đi ngược (đích → gốc) thay vì thuận (gốc → đích)?** Nếu đi thuận: pixel nguồn `(x,y)` → tính ra pixel đích `(x_out, y_out)` — nhưng `x_out, y_out` không phải số nguyên nên không khớp chính xác vào ô pixel nào, nhiều ô trong ảnh đích sẽ không được tô. Đi ngược đảm bảo **mọi pixel đích đều có giá trị**.

```
Ảnh gốc (hình thang):             Ảnh đích (hình chữ nhật):

  ╱──────────╲                    ┌──────────────┐
 ╱  A   B   C ╲      warp →      │  A   B   C   │
╱   D   E   F  ╲                 │  D   E   F   │
╲──────────────╱                 └──────────────┘

Mỗi ô trong đích ← tìm ngược vị trí tương ứng trong hình thang
                    rồi lấy màu từ đó
```

---

## 5. Kích thước trung gian 2480×3508 (dùng ở dòng 113-123) — tại sao?

### Tính từ chuẩn A4 tại 300 DPI

```
Khổ giấy A4: 210mm × 297mm

Chuyển sang pixel tại 300 DPI:
  chiều rộng = 210mm ÷ 25.4mm/inch × 300 pixel/inch = 2480 px
  chiều cao  = 297mm ÷ 25.4mm/inch × 300 pixel/inch = 3508 px
```

### Tại sao warp lên kích thước cao (2480×3508) trước khi resize xuống?

Warp là phép biến đổi có thể làm mất thông tin — khi kéo giãn vùng bị nén trong ảnh nghiêng, ta cần đủ pixel để nội suy.

```
Ảnh gốc điện thoại: 4000×3000 px (12MP)
→ Warp lên 2480×3508 (gần với gốc) → ít mất thông tin
→ Resize xuống 1000×1400 (thu nhỏ rõ ràng)

Nếu warp thẳng lên 1000×1400:
→ Mỗi pixel đích đại diện cho vùng lớn hơn nhiều trong ảnh gốc
→ Nội suy kém chính xác hơn
```

300 DPI cũng là chuẩn scan tài liệu — tại độ phân giải này, bong bóng MCQ nhỏ nhất (~4mm đường kính) vẫn chiếm ~47px, đủ để phân tích fill ratio chính xác.

---

## 6. Resize về 1000×1400 (dòng 128-129) — tại sao?

### Tỉ lệ khớp A4

```
1000 : 1400 = 5 : 7 ≈ 0.714
210  : 297  ≈ 0.707  (tỉ lệ A4 thực)

Sai số < 1% → tỉ lệ được giữ gần đúng, không bị méo nội dung
```

### Tại sao chọn 1000×1400 mà không phải lớn hơn hay nhỏ hơn?

```
Quá nhỏ (ví dụ 500×700):
  Bong bóng MCQ nhỏ chiếm ~3–4px → fill ratio không ổn định
  Fiducial marker có thể chỉ còn 2–3px → khó phát hiện

Quá lớn (ví dụ 2000×2800):
  Mọi phép tính (connected components, template matching) chạy chậm hơn 4×
  Bộ nhớ tăng 4× trong khi độ chính xác không tăng đáng kể

1000×1400:
  Bong bóng MCQ chiếm ~7–12px → đủ để tính fill ratio ổn định
  Fiducial marker chiếm ~8–15px → detection tốt
  Toàn bộ pipeline xử lý trong < 1 giây trên CPU thông thường
```

**Quan trọng:** Mọi tọa độ trong toàn bộ pipeline sau Bước 4 (anchor points, ROI, MCQ grid, SID) đều được định nghĩa và tính toán trong hệ **1000×1400 px**. Đây là "hệ tọa độ chuẩn" của hệ thống.

---

## 7. Khi Bước 2 và Bước 3 đều thất bại (nhánh `quad is None` ở dòng 112)

Bước 2/3 có thể không tìm được quad (ảnh quá mờ, thiếu sáng, góc chụp cực đoan). Trong trường hợp đó, khối dòng 113-126 không chạy — hệ thống không crash, chỉ đơn giản giữ nguyên các giá trị mặc định đã gán ở dòng 100-108 (`working=img_bgr`, `strategy="resize-only"`, `global_warp_used=False`), rồi vẫn chạy tiếp dòng 128-129 để resize ảnh gốc về 1000×1400 mà không warp. Nơi gọi hàm (`omr_service.py`) ghi thêm cờ cảnh báo:

```python
if not global_warp_used:
    warning_codes.append("COORD_GLOBAL_WARP_FALLBACK")
```

Cờ này báo cho caller biết ảnh đầu ra có thể bị lệch phối cảnh → kết quả chấm điểm kém tin cậy.

---

## 8. Bước 4 có thật sự cần thiết không?

**Về mặt kỹ thuật, Bước 4 không phải phụ thuộc cứng.** Pipeline vẫn chạy xong hoàn chỉnh dù warp thất bại — mục 7 ở trên chính là bằng chứng: không warp được thì resize thẳng, ghi cờ cảnh báo, pipeline không crash.

**Còn có 1 lớp "tự sửa" khác ở tầng sau, độc lập với Bước 4:** hàm `_resolve_coordinate_anchors` (`be/app/services/omr/omr_layout.py:119-270`, gọi từ `omr_service.py:1066`) tìm marker fiducial đen **ngay trong ảnh 1000×1400 đã resize** (dù ảnh đó có được warp hay chỉ resize thẳng) để xây các mốc neo (`anchor`) cho ROI — vùng số báo danh, mã đề, lưới câu hỏi MCQ. Đây là 1 trong 2 hàm nhắc ở cuối mục 1 — không thuộc Bước 4, chạy ở bước sau.

**Nhưng lớp tự sửa đó chỉ thực sự thích nghi khi tìm được marker fiducial.** Nếu không tìm được, nó rơi về `_default_anchor_percent` (`omr_layout.py:87-116`) — tọa độ **% cố định, hardcode theo template** (ví dụ `mcq_left_top = (0.20, 0.575)`), ngầm giả định trang đã được nắn thẳng tương đối chuẩn. Nói cách khác: nếu **cả Bước 4 lẫn marker fiducial ở tầng sau đều thất bại**, ROI rơi vào "đoán mù" theo % cố định trên 1 ảnh có thể vẫn còn nghiêng — sai số phối cảnh cộng dồn vào đúng lúc hệ thống không còn cách nào bù lại.

`global_warp_used`/`warp_strategy` (kết quả của Bước 4) không được dùng để rẽ nhánh logic chấm điểm ở bất kỳ đâu — chỉ dùng làm **điểm tự tin** (`warp_layout_score`, `omr_service.py:1939-1944`, cộng +0.50 nếu warp thành công) và metadata/log.

**Thừa nhận trung thực:** không có test hay eval nào trong repo đo trực tiếp mức ảnh hưởng của Bước 4 đến độ chính xác chấm điểm cuối cùng — đã kiểm tra `be/tests/` và `eval/run_eval.py`, không có kết quả nào phân tách theo `global_warp_used` hay `warp_strategy`. Kết luận "Bước 4 quan trọng" (mục 2, vấn đề phối cảnh làm lệch tọa độ ROI cố định) là suy luận hợp lý dựa trên cách tọa độ được định nghĩa trong hệ chuẩn 1000×1400 (mục 6), **không phải số liệu đo đạc thực tế**.

**Kết luận:** Bước 4 là 1 bước **tối ưu độ chính xác có cơ sở lý luận**, không phải yêu cầu bắt buộc để pipeline chạy được — nhưng mức độ quan trọng thực tế của nó (chênh lệch bao nhiêu % accuracy giữa có warp và không warp) hiện chưa được đo bằng dữ liệu trong repo này.

---

## 9. Đầu ra (Output)

```python
return resized, strategy, global_warp_used, info
```

| Biến | Kiểu | Nội dung |
|------|------|---------|
| `resized` | `np.ndarray` (1400, 1000, 3) BGR | Ảnh chuẩn 1000×1400, dùng cho mọi bước sau |
| `strategy` | `str` | Phương pháp đã dùng, ghi vào log |
| `global_warp_used` | `bool` | False → ghi cờ COORD_GLOBAL_WARP_FALLBACK |
| `info` | `dict` | Metadata: tọa độ quad phát hiện, kích thước nguồn/đích |

---

## 10. Sơ đồ luồng đầy đủ (tóm tắt trực quan mục 2)

```
img_bgr (bất kỳ kích thước)
        │
        ▼
  gray_src = cvtColor(img_bgr)     ← dòng 110
  quad, quad_strategy = _detect_page_quad(gray_src)   ← dòng 111, Bước 2 + Bước 3
  ├─ Tìm được quad (dòng 112 đúng) →
  │     dst = 4 góc chuẩn                ← dòng 113-121
  │     matrix = getPerspectiveTransform ← dòng 122
  │     working = warpPerspective        ← dòng 123, ảnh 2480×3508
  │     strategy = "coordinate-global-a4:..."   ← dòng 124
  │     global_warp_used = True                  ← dòng 125
  │     info["detected_quad"] = ...              ← dòng 126
  └─ Không tìm được (dòng 112 sai) →
        working = img_bgr (giữ nguyên từ dòng 108)
        strategy = "resize-only" (giữ nguyên từ dòng 101)
        → COORD_GLOBAL_WARP_FALLBACK (ghi ở omr_service.py)
        │
        ▼
  resized = cv2.resize(working, (1000, 1400))   ← dòng 128-129, LUÔN chạy
        │
        ▼
  resized: ảnh BGR 1000×1400  ← đầu vào của mọi bước sau
```

---

## 11. Tóm tắt các giá trị quan trọng

| Hằng số | Giá trị | Lý do |
|---------|---------|-------|
| `A4_WARP_W` | 2480 px | A4 210mm @ 300 DPI |
| `A4_WARP_H` | 3508 px | A4 297mm @ 300 DPI |
| `WIDTH_IMG` | 1000 px | Kích thước chuẩn pipeline |
| `HEIGHT_IMG` | 1400 px | Kích thước chuẩn pipeline |
