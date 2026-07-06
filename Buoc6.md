# Bước 6 — Trích xuất và phân loại marker trong ảnh chuẩn

> **Vị trí trong pipeline:** Bước 5 trả ra `binary_inv` (ảnh đen/trắng 1000×1400) và `gray_norm` (ảnh xám đã chuẩn hóa). Bước 6 quét `gray_norm` để tìm tất cả vùng đen hình vuông trong ảnh, rồi phân loại chúng thành 2 loại với vai trò khác nhau ở các bước sau.

**Module:** `be/app/services/omr/omr_marker_utils.py`
**Hàm chính:** `_extract_black_square_markers_from_gray()` → `_extract_black_square_markers()`

---

## 1. Thông tin đầu vào (Input)

| Tham số | Giá trị | Nguồn |
|---------|---------|-------|
| `gray_norm` | Ảnh xám 1000×1400 đã chuẩn hóa | Bước 5b |

---

## 2. Tại sao cần bước này?

Sau khi ảnh được warp và nhị phân hóa, hệ thống cần biết **cái gì đang nằm ở đâu** trên phiếu. Cụ thể, có hai loại vùng đen xuất hiện trong `binary_inv`:

```
Loại 1 — Fiducial marker:              Loại 2 — Bubble marker:
  ██████                                  ○ ○ ○ ○
  ██████   ← ô vuông đen in sẵn          ◉ ○ ○ ○   ← bong bóng học sinh tô
  ██████     trên phiếu                  ○ ○ ○ ○
```

- **Fiducial marker** (neo định vị): được in sẵn khi in phiếu, luôn ở cùng vị trí theo thiết kế. Dùng để xác định tọa độ vùng MSSV, vùng MCQ.
- **Bubble marker** (bong bóng): học sinh tô bằng bút. Dùng để suy luận khoảng cách dòng, số block câu hỏi.

Nếu bỏ qua bước này, hệ thống không biết ROI nằm ở đâu và phải dùng tọa độ cố định từ profile — kém ổn định vì góc chụp luôn có sai lệch nhỏ so với mẫu lý tưởng.

### Tại sao Bước 6 phải xử lý thêm dù Bước 5 đã làm?

Hàm `_extract_black_square_markers_from_gray` được gọi ở **hai nơi** trong pipeline với đầu vào khác nhau:

```python
# omr_service.py:970 — gọi với ảnh RAW, trước khi warp (detect page quad)
markers = _extract_black_square_markers_from_gray(gray, ...)

# omr_service.py:1088 — gọi với gray_norm đã chuẩn, sau Bước 5
markers = _extract_black_square_markers_from_gray(gray_norm, ...)
```

Lần gọi đầu xảy ra trước Bước 5 — lúc đó không có `gray_norm` hay `binary_inv` nào cả. Vì hàm phải hoạt động đúng ở cả hai ngữ cảnh (ảnh raw lẫn ảnh đã chuẩn), nó **luôn tự chuẩn hóa nội bộ** thay vì phụ thuộc vào input đã được xử lý.

---

## 3. 6a — Chuẩn hóa và nhị phân hóa nội bộ

```python
k = max(31, (min(h, w) // 10) | 1)                                 # k = 101 với 1000×1400
bg = cv2.morphologyEx(gray_norm, cv2.MORPH_CLOSE, ellipse(k, k))
norm = cv2.divide(gray_norm, bg, scale=255)
norm = cv2.GaussianBlur(norm, (3, 3), 0)
_, bin_inv = cv2.threshold(norm, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
```

### Dòng 1 — Tính kernel size `k`

```python
k = max(31, (min(h, w) // 10) | 1)   # → 101 với ảnh 1000×1400
```

- `min(h, w) // 10` = 1/10 cạnh ngắn nhất → kernel tỷ lệ với kích thước ảnh.
- `| 1` (bitwise OR với 1): đảm bảo `k` luôn **lẻ** — kernel OpenCV bắt buộc phải là số lẻ.
- `max(31, ...)`: đặt sàn tối thiểu 31 px, tránh kernel quá nhỏ với ảnh nhỏ.

So với Bước 5b dùng `min(h,w) // 9` → k=111: kernel ở đây nhỏ hơn một chút nhưng cùng tác dụng.

### Dòng 2 — Ước lượng nền bằng Morph Close

```python
bg = cv2.morphologyEx(gray_norm, cv2.MORPH_CLOSE, ellipse(101, 101))
```

**Morph Close = Dilate rồi Erode** với kernel hình elip 101×101 px.

Với kernel lớn hơn mọi marker (~20–50 px), phép Close có tác dụng **lấp đầy** các vùng tối nhỏ (marker) bằng giá trị nền xung quanh:

```
Cắt ngang một hàng pixel:
  gray_norm:  [200, 195,  30,  28,  25, 190, 205]
                              ↑
                        marker đen (~30px rộng)

  bg sau Close 101px:
              [200, 198, 196, 194, 193, 191, 205]
               ↑ marker bị "lấp đầy" — chỉ còn thông tin nền
```

Kết quả `bg` là ảnh ước tính nền, bao gồm cả sự thay đổi độ sáng do ánh sáng không đều, bóng đổ, hay góc chụp lệch.

### Dòng 3 — Chia để chuẩn hóa độ sáng

```python
norm = cv2.divide(gray_norm, bg, scale=255)
# norm[i,j] = gray_norm[i,j] / bg[i,j] * 255
```

Phép chia pixel-wise loại bỏ ảnh hưởng của ánh sáng không đều:

```
Góc ảnh sáng (bg = 210):           Góc ảnh tối do bóng (bg = 120):
  pixel nền:   200/210*255 ≈ 243     pixel nền:   100/120*255 ≈ 213
  pixel marker: 30/210*255 ≈  36     pixel marker:  15/120*255 ≈  32
```

Dù độ sáng tuyệt đối khác nhau giữa hai góc, tỷ lệ `marker/nền` gần như bất biến. Sau khi chia, nền được san bằng còn marker vẫn tối tương đương trên toàn ảnh.

### Dòng 4 — Làm mịn nhẹ

```python
norm = cv2.GaussianBlur(norm, (3, 3), 0)
```

Kernel 3×3 nhỏ, chỉ khử nhiễu muối tiêu (salt-and-pepper) 1 pixel. Không làm mờ cạnh marker.

### Dòng 5 — Ngưỡng hóa Otsu

```python
_, bin_inv = cv2.threshold(norm, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
```

**Otsu** tự động tìm ngưỡng T sao cho phương sai giữa nhóm pixel tối và sáng là lớn nhất — không cần chỉ định T cứng.

`THRESH_BINARY_INV`: pixel < T → **255** (foreground trắng), pixel ≥ T → 0 (background đen).
Vì marker đen < nền sáng → marker trở thành trắng trong `bin_inv`.

### So sánh với `binary_inv` từ Bước 5

`bin_inv` nội bộ này **không phải** là `binary_inv` đã có từ Bước 5. Hai ảnh này khác nhau:

| | `binary_inv` từ Bước 5 | `bin_inv` nội bộ Bước 6a |
|---|---|---|
| Nguồn | `_binarize()`, hỗ trợ otsu/adaptive/hybrid | Luôn dùng Otsu thuần |
| Morphology sau threshold | Open 2×2 (xóa nhiễu) | Chưa có — sẽ làm ở Bước 6b |
| Close để vá marker | Không có | Bước 6b sẽ thêm Close 3×3 |
| Mục đích | Đọc bong bóng MCQ toàn pipeline | Chỉ phục vụ tìm marker hình vuông |

`binary_inv` từ Bước 5 đã qua Open nhưng chưa qua Close → marker có thể bị lỗ hổng nhỏ bên trong. Bước 6 tạo `bin_inv` riêng để sau đó Close 3×3 (Bước 6b) vá lỗ hổng đó, phù hợp hơn cho connected components.

Kết quả: `bin_inv` nội bộ — vùng đen của phiếu = 255, nền = 0.

---

## 4. 6b — Tiền xử lý ảnh nhị phân

```python
prep = cv2.morphologyEx(bin_inv, cv2.MORPH_OPEN,  np.ones((2, 2), np.uint8), iterations=1)
prep = cv2.morphologyEx(prep,    cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=1)
```

Hai phép morphology liên tiếp:

| Phép | Kernel | Tác dụng |
|------|--------|----------|
| **Open 2×2** | 2×2 hình vuông | Erode rồi Dilate — xóa nhiễu điểm đơn lẻ nhỏ hơn 2×2 px |
| **Close 3×3** | 3×3 hình vuông | Dilate rồi Erode — lấp các gap/lỗ hổng nhỏ bên trong marker |

### Tại sao cần cả hai, dù Bước 6a đã xử lý?

**Open 2×2 — giải quyết nhiễu sót lại sau Otsu:**

Otsu chỉ chọn ngưỡng T tối ưu toàn cục, không biết pixel nào là "thật". Sau Otsu vẫn còn pixel nhiễu ở cạnh chữ in mờ, góc kẻ ô, vết bụi nhỏ dưới 4 px². Những thứ này sẽ bị connected components bắt làm thành phần riêng và tốn thời gian lọc vô nghĩa. Open 2×2 xóa chúng sạch trước khi vào bước đó.

```
Sau Otsu:                      Sau Open 2×2:
  . . . . . . .                  . . . . . . .
  . █ █ █ . . .                  . █ █ █ . . .
  . █ █ █ . · .   ← nhiễu 1px   . █ █ █ . . .   ← nhiễu bị xóa
  . . . . . . .                  . . . . . . .
```

**Close 3×3 — giải quyết lỗ hổng bên trong marker:**

Ảnh chụp marker in (mực đen đặc) có thể bị phản sáng nhẹ ở tâm. Sau Otsu, vùng đó thành pixel trắng (0) → marker bị **thủng lỗ bên trong**. Nếu không vá, connected components sẽ thấy vành ngoài là một thành phần với fill_ratio thấp (pixel đen / bbox nhỏ hơn thực tế) → **fail tiêu chí fill ≥ 0.40 và bị loại sai**.

```
Marker thực tế:          Sau Otsu (có lỗ):         Sau Close 3×3 (vá lại):
  ████████████             ████████████               ████████████
  ████████████             ████░░░░████               ████████████
  ████████████    →        ████░░░░████    →           ████████████
  ████████████             ████████████               ████████████
```

Open xóa nhiễu ngoài → Close vá lỗ trong → marker trở nên đặc và liền mạch trước khi vào connected components.

---

## 5. 6c — Phát hiện vùng liên thông (Connected Components)

```python
num, labels, stats, centroids = cv2.connectedComponentsWithStats(prep, connectivity=8)
```

### Đầu vào và bài toán cần giải

**Đầu vào:** `prep` — ảnh binary 1000×1400 sau Bước 6b. Chứa hàng nghìn pixel trắng rải rác: marker, bong bóng tô, chữ in, đường kẻ, nhiễu còn sót.

**Bài toán:** Ảnh chỉ là mảng pixel, không có khái niệm "vật thể". Hệ thống cần nhóm các pixel trắng liền nhau thành từng "hòn đảo" riêng biệt để có thể đo và lọc từng cái:

```
prep (pixel trắng = 255):           Bài toán: pixel nào cùng nhóm?

  . . . . . . . . . . . .             . . . . . . . . . . . .
  . █ █ █ . . . . █ █ . .             . A A A . . . . B B . .
  . █ █ █ . . . . █ █ . .     →       . A A A . . . . B B . .
  . . . . . . █ . . . . .             . . . . . . C . . . . .
  . . . . . . . . . . . .             . . . . . . . . . . . .
               ↑ cùng là pixel trắng nhưng không liền → 3 nhóm riêng
```

### Thuật toán — Two-Pass Labeling

Thuật toán quét ảnh hai lần:

**Pass 1:** Quét từng pixel trái→phải, trên→dưới. Nếu pixel trắng:
- Không có hàng xóm trắng nào trước đó → gán nhãn mới
- Có hàng xóm trắng → kế thừa nhãn (và ghi nhận hai nhãn là "cùng nhóm" nếu hàng xóm khác nhãn)

**Pass 2:** Dùng Union-Find để hợp nhất các nhóm được ghi nhận là "cùng nhóm". Gán lại nhãn nhất quán.

### `connectivity=8` — tại sao không phải 4?

```
connectivity=4 (chỉ ngang/dọc):    connectivity=8 (bao gồm chéo):
  . N .                               N N N
  N X N                               N X N
  . N .                               N N N
```

Marker bị chụp hơi nghiêng → pixel góc của marker nằm ở vị trí chéo so với pixel cạnh. Dùng connectivity=4 thì marker dễ bị tách thành nhiều mảnh. Connectivity=8 đảm bảo marker luôn được nhận là một khối liền.

### Bốn giá trị trả về

**`num`** — tổng số thành phần kể cả background. Thực tế có `num - 1` vật thể (thành phần 0 là nền đen).

**`labels`** — mảng 2D cùng kích thước `prep` (1000×1400), mỗi pixel chứa số hiệu thành phần của nó:

```
prep:                    labels:
  . . █ █ █ . . .          0 0 1 1 1 0 0 0
  . . █ █ █ . . .    →     0 0 1 1 1 0 0 0
  . . . . . █ █ .          0 0 0 0 0 2 2 0
```

Dùng ở Bước 6d để crop từng thành phần riêng lẻ: `labels[y:y+bh, x:x+bw] == idx`.

**`stats[idx]`** — array shape `(num, 5)`, mỗi hàng là `[x, y, w, h, area]`:
- `x, y`: tọa độ góc trên-trái bounding box
- `w, h`: chiều rộng và cao bounding box
- `area`: số pixel trắng thuộc thành phần — **không phải** diện tích bbox

```python
# Bước 6d dùng trực tiếp:
x, y, bw, bh, area = stats[idx]
fill_ratio = area / (bw * bh)   # < 1.0 nếu thành phần không đặc hoàn toàn
```

**`centroids[idx]`** — tọa độ tâm hình học `(cx, cy)` tính theo trọng tâm pixel, không phải tâm bbox.

Có hai cách tính "tâm" của một thành phần:

```python
# Cách đơn giản — tâm bounding box:
cx_bbox = x + bw / 2
cy_bbox = y + bh / 2

# Cách chính xác — tâm hình học (centroid):
cx = Σ(x_i cho mọi pixel i thuộc thành phần) / area
cy = Σ(y_i cho mọi pixel i thuộc thành phần) / area
```

Khi marker đặc và đối xứng, hai cách cho kết quả như nhau. Khi marker bị mờ không đều (ảnh chụp nghiêng, ánh sáng lệch), tâm hình học phản ánh đúng nơi mực đen tập trung nhất:

```
Marker mờ phía phải (ít pixel hơn):
  █████████░░    bw=11, bh=3
  ████████░░░
  ████████░░░

  Tâm bbox:      cx = x + 5.5   (giữa bbox, kéo về phía rỗng)
  Tâm centroid:  cx ≈ x + 4.2   (lệch về phía nhiều pixel hơn — đúng hơn)
```

Tọa độ `(cx, cy)` từ centroid được lưu vào marker dict và dùng ở:

- **Bước 7** — cluster tất cả marker theo `cy` để tìm các hàng bong bóng, từ đó tính `line_h`. Nếu `cy` sai do dùng tâm bbox, khoảng cách dòng tính ra lệch theo.
- **Bước 8** — tìm fiducial marker gần nhất với tọa độ thiết kế bằng cách so sánh `cx, cy` với tọa độ mẫu. Nếu `cx, cy` lệch → chọn nhầm marker → toàn bộ ROI bị dịch chuyển.

```python
cx = float(centroids[idx][0])
cy = float(centroids[idx][1])
# → lưu vào marker dict, dùng ở Bước 7 và 8 để xác định tọa độ ROI
```

### Mục tiêu

Sau bước này, ảnh binary (tập hợp pixel) được biến thành **danh sách vật thể có thể đo đếm được**. Bước 6d sau đó dùng chính xác các con số từ `stats` và `centroids`:

| Tiêu chí 6d | Dùng từ connected components |
|-------------|------------------------------|
| Diện tích | `area` từ `stats` |
| Kích thước | `bw`, `bh` từ `stats` |
| Aspect ratio | `bw / bh` từ `stats` |
| Fill ratio | `area / (bw * bh)` từ `stats` |
| Hình dạng (contour) | crop theo `labels[y:y+bh, x:x+bw] == idx` |
| Tọa độ cuối cùng | `cx`, `cy` từ `centroids` |

---

## 6. 6d — Lọc qua 6 tiêu chí

Với mỗi thành phần, hệ thống lọc qua 6 điều kiện theo thứ tự. **Không qua một điều kiện nào thì bỏ ngay.**

### Tiêu chí 1 — Diện tích hợp lý

```python
min_area = max(12, int(total_area * 0.00002))   # ≈ 20 px²
max_area = max(min_area + 1, int(total_area * 0.012))  # ≈ 12000 px²

if area < min_area or area > max_area: continue
```

| Lý do loại quá nhỏ | Lý do loại quá lớn |
|-------------------|--------------------|
| Bụi, nhiễu ảnh (1–5 px²) | Vùng chữ in, khung phiếu (>12000 px²) |

### Tiêu chí 2 — Kích thước tối thiểu

```python
if bw < 4 or bh < 4: continue
```

Loại các vùng quá hẹp (1–3 pixel) — không thể là marker.

### Tiêu chí 3 — Tỷ lệ chiều (Aspect Ratio)

```python
aspect = bw / bh
if not (0.62 <= aspect <= 1.55): continue
```

Chỉ nhận hình **gần vuông**. Aspect ratio = 1.0 là hình vuông hoàn hảo. Cho phép sai lệch ±38%:

```
0.62  0.80  1.00  1.25  1.55
  ↑                       ↑
  quá cao                 quá rộng
  (hình chữ I)            (hình chữ nhật ngang)
```

### Tiêu chí 4 — Fill Ratio

```python
fill_ratio = area / (bw * bh)
if fill_ratio < 0.40: continue
```

`area` là số pixel trắng **thực sự thuộc thành phần** (đếm từng pixel). `bw * bh` là diện tích bounding box — luôn ≥ area. `fill_ratio` đo mức độ "đặc" của hình dạng bên trong bbox:

```
Bounding box 5×5 = 25 px²:
  ┌─────────┐
  │ █ █ █ █ │  area = 20 pixel trắng
  │ █ █ █ █ │
  │ █ █ █ █ │  fill_ratio = 20/25 = 0.80
  │ . . . . │  ← 5 pixel đen trong bbox, không thuộc thành phần
  └─────────┘
```

Mỗi loại vật thể có fill_ratio đặc trưng do hình dạng khác nhau:

```
Fiducial marker (ô vuông in đặc):     fill ≈ 0.88–0.95
  ████████
  ████████    ← hầu hết bbox đều là pixel đen
  ████████

Bong bóng tô tay (hơi rỗng giữa):    fill ≈ 0.50–0.75
  ████████
  ████░░██    ← giữa còn khoảng trắng do tô không đều
  ████████

Chữ "H" in trên phiếu:               fill ≈ 0.30–0.45
  █░░░░░█
  ███████     ← bbox lớn nhưng nhiều khoảng trắng bên trong
  █░░░░░█

Đường kẻ ngang mỏng:                 fill ≈ 0.05–0.15
  . . . . .
  █████████   ← bbox cao nhưng chỉ 1–2 hàng pixel trắng
  . . . . .
```

`fill_ratio` không chỉ dùng một lần — nó là thước đo xuyên suốt toàn Bước 6:

| Chỗ dùng | Ngưỡng | Mục đích |
|----------|--------|----------|
| Tiêu chí 4 (lọc) | `fill ≥ 0.40` | Loại đường kẻ, chữ mỏng |
| Tiêu chí 6 (circularity) | `fill < 0.92` | Nếu fill rất cao → không loại dù tròn |
| Phân loại bubble | `fill ≥ 0.58` | Nhận bong bóng tô tay |
| Phân loại fiducial | `fill ≥ 0.86` | Nhận marker in sẵn |
| Sort cuối cùng | `(area, fill)` | Ưu tiên marker đặc và lớn lên đầu |

### Tiêu chí 5 — Hình dạng đa giác (approxPolyDP + Solidity)

Tiêu chí 5 thực ra gồm **4 bước con** liên tiếp trong code:

```python
# 5.0 — Trích xuất contour
comp_mask = (labels[y:y+bh, x:x+bw] == idx).astype(np.uint8) * 255
cnts, _ = cv2.findContours(comp_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnt = max(cnts, key=cv2.contourArea)

# 5.1 — Kiểm tra tính nhất quán contour vs pixel
contour_area = float(cv2.contourArea(cnt))
if contour_area < float(area) * 0.55: continue

# 5.2 — approxPolyDP: đếm đỉnh
peri = float(cv2.arcLength(cnt, True))
approx = cv2.approxPolyDP(cnt, 0.08 * peri, True)
vertex_count = int(len(approx))
if vertex_count < 4 or vertex_count > 8: continue

# 5.3 — Solidity: đo mức độ lồi
hull = cv2.convexHull(cnt)
hull_area = float(cv2.contourArea(hull))
solidity = contour_area / max(1.0, hull_area)
if solidity < 0.82: continue
```

#### Bước 5.0 — Trích xuất contour từ component mask

**Đầu vào:** `labels` (mảng nhãn từ connected components) + bbox của thành phần `(x, y, bw, bh, idx)`.

`labels[y:y+bh, x:x+bw] == idx` cắt ra mảng boolean trong vùng bbox, đánh dấu `True` tại các pixel thuộc đúng thành phần `idx`. Nhân `* 255` ra binary mask.

`findContours` dò đường biên: đi dọc ranh giới giữa pixel trắng và đen, thu thập tọa độ từng điểm biên thành đa giác. `RETR_EXTERNAL` chỉ lấy viền ngoài cùng (bỏ lỗ hổng bên trong). `CHAIN_APPROX_SIMPLE` nén đường thẳng — không lưu mọi điểm trên cạnh, chỉ lưu hai đầu mút:

```
Component mask:          Contour (chuỗi điểm biên, đã nén):
  . . . . . . .            (1,1)→(5,1)
  . █ █ █ █ █ .            (5,1)→(5,5)    ← chỉ lưu 4 góc,
  . █ █ █ █ █ .    →       (5,5)→(1,5)       không lưu từng pixel
  . █ █ █ █ █ .            (1,5)→(1,1)
  . █ █ █ █ █ .
  . . . . . . .
```

`cnt = max(cnts, key=cv2.contourArea)` — lấy contour diện tích lớn nhất phòng trường hợp mask có nhiều mảnh nhỏ rời rạc.

#### Bước 5.1 — Kiểm tra tính nhất quán contour vs pixel

```python
contour_area = float(cv2.contourArea(cnt))
if contour_area < float(area) * 0.55: continue
```

`cv2.contourArea` tính diện tích vùng **bao quanh bởi contour** bằng công thức Shoelace (tích phân theo đường biên), không đếm pixel.

`area` từ connected components là số pixel trắng thực tế. Với hình đặc, hai con số này gần bằng nhau. Nếu chúng lệch nhau nhiều, tức là thành phần có hình dạng phức tạp:

```
Hình đặc:    contour_area ≈ area × 0.95   → tỷ lệ ≈ 0.95  ✓
Hình rỗng:   contour_area ≈ area × 0.40   → tỷ lệ ≈ 0.40  ✗
             (contour bao vùng rỗng bên trong, area chỉ tính pixel thật)
```

Ngưỡng 0.55 loại các thành phần mà connected components vô tình gộp nhiều mảnh có lỗ hổng lớn bên trong.

#### Bước 5.2 — approxPolyDP: đếm số đỉnh

```python
peri = float(cv2.arcLength(cnt, True))
approx = cv2.approxPolyDP(cnt, 0.08 * peri, True)
vertex_count = int(len(approx))
if vertex_count < 4 or vertex_count > 8: continue
```

**Douglas-Peucker**: rút gọn contour hàng trăm điểm thành đa giác ít đỉnh nhất mà vẫn đúng hình dạng tổng thể. `epsilon = 0.08 * peri` — ngưỡng sai lệch cho phép. Với marker 30×30 px (peri ≈ 120 px): epsilon = 9.6 px. Bất kỳ điểm nào trên contour mà cách đường nối hai đỉnh liền kề < 9.6 px thì bị bỏ qua (đường thẳng đã đủ gần):

```
Hình vuông đặc:     contour gốc ~100 điểm  →  4 đỉnh   ✓  (4 góc)
Vuông góc hơi mờ:   contour gốc ~100 điểm  →  6 đỉnh   ✓  (góc bo nhẹ)
Hình tròn:          contour gốc ~100 điểm  →  12+ đỉnh  ✗  (không thể đơn giản hóa)
Hình tam giác:      contour gốc ~60 điểm   →  3 đỉnh   ✗  (< 4)
```

Khoảng 4–8 đỉnh bao phủ: marker sắc nét (4), marker mờ nhẹ ở góc (5–6), marker chụp hơi nghiêng (7–8).

#### Bước 5.3 — Solidity: đo mức độ lồi

```python
hull = cv2.convexHull(cnt)
hull_area = float(cv2.contourArea(hull))
solidity = contour_area / max(1.0, hull_area)
if solidity < 0.82: continue
```

**Convex hull** = hình lồi nhỏ nhất bao quanh contour — giống căng dây chun quanh hình dạng. Hull luôn ≥ contour về diện tích.

`solidity = contour_area / hull_area` đo hình có lõm không:

```
Marker vuông:           Chữ L:               Chữ U:
  ████████               ███░                 █░░█
  ████████               ███░                 █░░█
  ████████               ████                 ████
  ████████               ████

  hull = chính nó        hull lấp góc trống   hull lấp khoảng giữa
  solidity ≈ 0.95 ✓      solidity ≈ 0.60 ✗    solidity ≈ 0.55 ✗
```

#### Tại sao cần cả approxPolyDP lẫn Solidity?

Hai phép đo bổ sung lẫn nhau — approxPolyDP chỉ đếm đỉnh, không biết hình lõm hay lồi:

| Hình dạng | approxPolyDP | Solidity | Kết quả |
|-----------|-------------|----------|---------|
| Marker vuông đặc | 4 đỉnh ✓ | 0.95 ✓ | Qua |
| Marker mờ góc | 6 đỉnh ✓ | 0.92 ✓ | Qua |
| Hình tròn (bong bóng) | >8 đỉnh ✗ | — | Loại sớm ở vertex |
| Chữ L | 4 đỉnh ✓ | 0.60 ✗ | approxPolyDP không bắt, solidity bắt |
| Chữ U | 6 đỉnh ✓ | 0.55 ✗ | approxPolyDP không bắt, solidity bắt |
| Hình sao | 8 đỉnh ✓ | 0.50 ✗ | approxPolyDP không bắt, solidity bắt |

**Đầu ra nếu qua tiêu chí 5:** `vertex_count` và `solidity` lưu vào marker dict; `contour_area` và `peri` truyền tiếp sang tiêu chí 6 (circularity).

### Tiêu chí 6 — Circularity (loại bong bóng tròn)

```python
circularity = (4 * π * contour_area) / (perimeter²)
if circularity > 0.90 and fill_ratio < 0.92 and vertex_count > 5: continue
```

`circularity` = 1.0 cho hình tròn hoàn hảo; hình vuông ≈ 0.785. Ngưỡng 0.90 loại bong bóng gần tròn, trừ khi fill rất cao (≥ 0.92) — tức là marker in sẵn có thể bị phát hiện hơi tròn do độ mờ ảnh.

### Kết quả sau lọc

Mỗi marker vượt qua 6 tiêu chí được lưu vào dict:

```python
{
    "cx": 125.3, "cy": 88.7,    # tọa độ tâm
    "x": 112, "y": 76,          # góc trên-trái bounding box
    "w": 28, "h": 26,           # kích thước bounding box
    "area": 672,                 # diện tích pixel
    "fill": 0.92,               # fill ratio
    "size": 27.0,               # (w+h)/2 — kích thước trung bình
    "vertices": 4,               # số đỉnh sau approxPolyDP
    "solidity": 0.94,           # độ đặc
    "circularity": 0.79,        # độ tròn
}
```

Sắp xếp giảm dần theo `(area, fill)`, giữ tối đa 320 marker.

---

## 7. 6e — Phân loại 2 loại marker

Tập `markers` chứa tất cả ứng viên. Ở các bước sau, hệ thống phân loại ngầm khi dùng:

| Loại | Điều kiện lọc | Vai trò |
|------|--------------|---------|
| **Fiducial marker** | fill ≥ 0.86 AND size ≥ 11 px AND circularity ≤ 0.90 | Điểm neo xác định tọa độ SID/MCQ ROI |
| **Bubble marker** | fill ≥ 0.58 AND size ≥ 6 px | Đo khoảng cách dòng (line_h), đếm số block |

**Tại sao ngưỡng khác nhau?**

Fiducial marker được in sẵn bằng máy → đặc, sắc nét → fill cao (~0.90), size lớn. Bong bóng tô tay → mật độ thấp hơn, size nhỏ hơn. Bằng cách đặt ngưỡng fill ≥ 0.86 và size ≥ 11, hệ thống phân biệt được 2 loại ngay cả khi bong bóng được tô rất đậm.

```
fill = 0.90, size = 25 → Fiducial ✓  (ô vuông in sẵn)
fill = 0.72, size = 9  → Bubble  ✓  (bong bóng tô đậm)
fill = 0.55, size = 7  → Bubble  ✓  (bong bóng tô nhạt)
fill = 0.35, size = 5  → Loại ✗     (nhiễu / đường kẻ)
```

---

## 8. Đầu ra (Output)

```python
markers  →  list[dict]  →  truyền vào Bước 7 (_infer_mcq_geometry_from_markers)
                            và Bước 8 (_resolve_coordinate_anchors)
```

| Biến | Kiểu | Dùng ở đâu |
|------|------|-----------|
| `markers` | `list[dict]` — toàn bộ marker tìm được | Bước 7: tính line_h và block_count |
| `fiducial_markers` | subset của markers (fill ≥ 0.86, size ≥ 11) | Bước 8: xác định anchor tọa độ ROI |
| `bubble_markers` | subset (fill ≥ 0.58) | Bước 7: cluster theo Y để đo khoảng cách dòng |

> Lưu ý: Việc phân loại thành fiducial / bubble không xảy ra trong một hàm duy nhất mà được lọc ngay khi dùng trong Bước 7 (`omr_mcq.py`) và Bước 8 (`omr_layout.py`).

Tất cả tọa độ trong `markers` đều thuộc hệ **1000×1400 px** — hệ tọa độ chuẩn của toàn pipeline từ đây trở đi.

---

## 9. Sơ đồ luồng

```
gray_norm (ảnh xám 1000×1400 đã chuẩn hóa)
        │
        ▼
  Chuẩn hóa nội bộ (Morph Close k=101 + divide + Blur + Otsu)
        │
        ▼  bin_inv_internal (nhị phân nội bộ)
        │
  Tiền xử lý: Morph Open 2×2 → xóa nhiễu nhỏ
              Morph Close 3×3 → vá lỗ hổng trong marker
        │
        ▼  prep
        │
  connectedComponentsWithStats (connectivity=8)
        │
        ▼  num thành phần, stats, centroids
        │
  Lọc từng thành phần qua 6 tiêu chí:
    [1] Diện tích ∈ [~20, ~12000] px²
    [2] Kích thước ≥ 4×4 px
    [3] Aspect ratio 0.62 – 1.55
    [4] Fill ratio ≥ 0.40
    [5] approxPolyDP 4–8 đỉnh, Solidity ≥ 0.82
    [6] Circularity < 0.90 (loại bong bóng tròn)
        │
        ▼  markers (≤ 320 marker, sort theo area×fill)
        │
        ├──→ Bước 7: lọc bubble (fill ≥ 0.58) → tính line_h
        └──→ Bước 8: lọc fiducial (fill ≥ 0.86, size ≥ 11) → xác định anchor ROI
```
