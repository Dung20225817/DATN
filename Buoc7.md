# Bước 7 — Suy luận hình học lưới MCQ

> **Vị trí trong pipeline:** Bước 6 trả ra `markers` (list ≤320 dict, mỗi dict chứa cx, cy, fill, size, area...). Bước 7 đọc toàn bộ danh sách đó và suy luận ra **cấu trúc hình học của phiếu**: dòng cách nhau bao nhiêu px, có mấy block MCQ, dòng đầu/cuối ở đâu — mà không cần biết trước tọa độ cố định.

**Module:** `be/app/services/omr/omr_mcq.py`
**Hàm chính:** `_infer_mcq_geometry_from_markers(markers, img_w, img_h, rows_per_block, block_count_hint)`

---

## 1. Tại sao cần bước này?

Sau Bước 6, hệ thống có danh sách marker với tọa độ, nhưng không biết:
- Dòng bong bóng câu 1 nằm ở Y nào?
- Khoảng cách giữa hai dòng là bao nhiêu pixel?
- Phiếu có 2 hay 3 block MCQ?
- Mỗi block nằm ở vùng X nào?

Nếu không suy luận được những thông tin này, hệ thống buộc phải dùng tọa độ cứng từ profile (ví dụ: "dòng 1 luôn ở Y=600"). Tọa độ cứng thất bại ngay khi ảnh chụp hơi nghiêng, zoom khác, hay phiếu in khác cỡ.

Bước 7 giải quyết bằng cách **đọc cấu trúc từ marker thực tế trên ảnh** — thích nghi với từng ảnh cụ thể.

---

## 2. Đầu vào (Input)

| Tham số | Kiểu | Nguồn | Mặc định |
|---------|------|-------|---------|
| `markers` | `list[dict]` | Bước 6 | — |
| `img_w` | `int` | 1000 | — |
| `img_h` | `int` | 1400 | — |
| `rows_per_block` | `Optional[int]` | profile cấu hình phiếu | `20` |
| `block_count_hint` | `Optional[int]` | profile cấu hình phiếu | `1` |

Mỗi marker trong `markers` là dict với ít nhất: `cx`, `cy`, `fill`, `size`, `area`.

---

## 3. 7a — Xác định chế độ: short_form vs long_form

```python
rows_hint = max(1, rows_per_block or 20)    # số hàng/block, mặc định 20
blocks_hint = max(1, block_count_hint or 1)  # số block, mặc định 1
long_form_mode = (rows_hint >= 25 or blocks_hint >= 4)
min_y_ratio = 0.30 if long_form_mode else 0.44
```

Hai chế độ khác nhau ở **vùng tìm kiếm marker**:

| Chế độ | Điều kiện | min_y_ratio | Vùng tìm marker |
|--------|-----------|-------------|-----------------|
| short_form | ≤20 câu/block, ≤3 block | 0.44 | Từ 44% → 98% ảnh |
| long_form | ≥25 câu/block hoặc ≥4 block | 0.30 | Từ 30% → 98% ảnh |

**Tại sao khác nhau?** Phiếu long_form có MCQ bắt đầu cao hơn trên trang (nhiều câu hơn → cần không gian hơn → bắt đầu sớm hơn). Nếu dùng 44% cho long_form sẽ bỏ mất các hàng đầu.

---

## 4. 7b — Cluster marker thành hàng

```python
rows = _cluster_marker_rows(
    markers,
    min_x=img_w * 0.10,        # bỏ 10% vùng ngoài cùng trái
    max_x=img_w * 0.90,        # bỏ 10% vùng ngoài cùng phải
    min_y=img_h * min_y_ratio, # bỏ phần trên (header, SID)
    max_y=img_h * 0.98,        # bỏ 2% cạnh dưới
    y_tol=6.0,
)
```

### Tại sao lọc 10% hai bên trái/phải?

Giá trị 10% **không đọc từ profile hay bước nào trước** — đây là heuristic cứng được thiết kế để loại **4 corner fiducial marker** đã dùng ở Bước 2.

Sau Bước 4 warp, ảnh 1000×1400 đã được căn thẳng. Nhưng `markers` từ Bước 6 chứa **tất cả** marker tìm được trên ảnh đó, bao gồm cả 4 corner marker nằm rất gần 4 góc:

```
Ảnh warped 1000×1400 — markers từ Bước 6:

  ●──────────────────────────────────────●
  │ corner (~50px từ cạnh)               │  ← lọt vào markers list
  │                                      │
  │    ●─────────────────────────●       │  ← hàng fiducial MCQ thật
  │                                      │
  │    ○ ○ ○ ○  ○ ○ ○ ○  ○ ○ ○ ○     │  ← bong bóng MCQ
  │    ○ ○ ○ ○  ○ ○ ○ ○  ○ ○ ○ ○     │
  │                                      │
  ●──────────────────────────────────────●
```

Nếu không lọc, 2 corner marker trên cùng (Y gần nhau, span ≈ 950px > 34% img_w) sẽ bị `_cluster_marker_rows` gom thành 1 hàng và sau đó phân loại nhầm thành **fiducial_row** → `block_count` và `fid_top_y` tính sai hoàn toàn.

**Tại sao chọn đúng 10% (100px)?**

Corner marker trên phiếu chuẩn A4 nằm cách cạnh ~20–60px (2–6% của 1000px). 10% = 100px tạo buffer an toàn: đủ xa để loại corner, đủ gần để không cắt vào vùng MCQ (MCQ thường bắt đầu từ x ≥ 12–15% từ cạnh trái).

| Tham số | Giá trị | Nguồn | Mục đích lọc |
|---------|---------|-------|-------------|
| `min_x` | 10% img_w | cứng | Loại corner marker trái |
| `max_x` | 90% img_w | cứng | Loại corner marker phải |
| `min_y` | 30%/44% img_h | tính từ `long_form_mode` | Loại header/SID vùng trên |
| `max_y` | 98% img_h | cứng | Loại corner marker dưới cùng |

`min_y` linh hoạt hơn vì bố cục dọc (ngắn/dài form) biến thiên nhiều; `min_x/max_x` cứng vì bố cục ngang ít thay đổi.

### "Hàng" ở đây là gì?

Trên phiếu thi, mỗi câu hỏi MCQ có các bong bóng A, B, C, D nằm **ngang hàng nhau** — tức là cùng Y, khác X. Câu hỏi khác nhau thì có Y khác nhau:

```
         Block 1              Block 2
      A   B   C   D        A   B   C   D
Câu 1 ○   ○   ○   ○        ○   ○   ○   ○    ← tất cả ở Y ≈ 620px  = hàng 1
Câu 2 ○   ○   ○   ○        ○   ○   ○   ○    ← tất cả ở Y ≈ 642px  = hàng 2
Câu 3 ○   ○   ○   ○        ○   ○   ○   ○    ← tất cả ở Y ≈ 663px  = hàng 3
```

Vậy **"hàng" = 1 dòng câu hỏi** = tập hợp tất cả bong bóng có Y xấp xỉ bằng nhau (A, B, C, D của câu đó, trên tất cả các block). Các câu khác nhau → Y khác nhau → hàng khác nhau. Y của hàng là trung bình Y của tất cả bong bóng trong dòng câu hỏi đó.

### Bài toán cần giải

`markers` từ Bước 6 là **danh sách phẳng** — không có thông tin nào cho biết marker nào cùng câu với marker nào:

```
Đầu vào (phẳng, không thứ tự):           Đầu ra (đã nhóm theo hàng câu):

  {cx:95,  cy:620}  ← câu 1 block 1 A     Hàng 1 (câu 1): cy≈620
  {cx:400, cy:642}  ← câu 2 block 1 A       markers: [A1b1, B1b1, C1b1, D1b1,
  {cx:125, cy:621}  ← câu 1 block 1 B                 A1b2, B1b2, C1b2, D1b2]
  {cx:700, cy:620}  ← câu 1 block 2 A   →
  {cx:430, cy:641}  ← câu 2 block 1 B     Hàng 2 (câu 2): cy≈642
  ...                                        markers: [A2b1, B2b1, ...]
```

Mục tiêu: chuyển danh sách điểm rời rạc thành danh sách hàng có đủ thống kê (cy, span, count) để các bước 7c–7f làm việc.

### Thuật toán 4 bước

**Bước 1 — Lọc marker ngoài cửa sổ:**

```python
pool = []
for marker in markers:
    cx, cy = marker.get("cx"), marker.get("cy")
    if cx < min_x or cx > max_x or cy < min_y or cy > max_y:
        continue
    pool.append(marker)
```

Loại mọi marker ngoài hình chữ nhật `[min_x, max_x] × [min_y, max_y]`. Sau bước này chỉ còn marker trong vùng MCQ.

**Bước 2 — Sắp xếp theo cy tăng dần:**

```python
pool.sort(key=lambda m: m.cy)
```

Đây là bước **bắt buộc** để greedy hoạt động đúng. Nếu không sort, marker từ nhiều câu khác nhau xen kẽ nhau trong danh sách → greedy gom nhầm:

```
Không sort (xen kẽ):    Sau sort (liền mạch):
  cy=620  câu 1           cy=620  ┐
  cy=642  câu 2           cy=621  ├─ câu 1 liền nhau
  cy=621  câu 1           cy=620  ┘
  cy=641  câu 2           cy=641  ┐
                          cy=642  ├─ câu 2 liền nhau
                          cy=641  ┘
```

**Bước 3 — Gom nhóm greedy:**

```python
cur = [pool[0]]
for marker in pool[1:]:
    cur_center = mean(m.cy for m in cur)   # mean Y của nhóm hiện tại
    if abs(marker.cy - cur_center) <= 6.0:
        cur.append(marker)      # cùng câu → gom vào
    else:
        rows.append(cur)        # câu mới
        cur = [marker]
rows.append(cur)                # lưu câu cuối
```

Cốt lõi: so sánh cy của marker tiếp theo với **mean Y của nhóm hiện tại** (không phải cy cố định của marker đầu). Mỗi lần thêm marker vào nhóm, `cur_center` được tính lại.

Ví dụ đầy đủ (6 marker đã sort, 2 câu, mỗi câu 3 marker):

```
pool: [cy=619.8, cy=620.1, cy=620.5, cy=641.5, cy=641.8, cy=642.3]

B1: cur=[619.8],  cur_center=619.8
B2: xét 620.1 → |620.1-619.8|=0.3 ≤ 6 → thêm vào
    cur=[619.8,620.1], cur_center=619.95
B3: xét 620.5 → |620.5-619.95|=0.55 ≤ 6 → thêm vào
    cur=[619.8,620.1,620.5], cur_center=620.13
B4: xét 641.5 → |641.5-620.13|=21.4 > 6 → HẾT HÀNG
    rows.append([619.8,620.1,620.5]), cur=[641.5]
B5: xét 641.8 → |641.8-641.5|=0.3 ≤ 6 → thêm vào
B6: xét 642.3 → |642.3-641.65|=0.65 ≤ 6 → thêm vào
Kết: rows.append([641.5,641.8,642.3])

→ Hàng 1: cy_mean=620.1,  Hàng 2: cy_mean=641.87
→ line_h = 641.87 - 620.1 = 21.77 px
```

**Tại sao dùng running mean thay vì cy cố định của marker đầu?**

Nếu dùng cy cố định và marker đầu lệch 3px, marker cuối hàng có thể bị từ chối oan:

```
Marker đầu câu ở cy=619.0 (lệch thật sự so với đường giữa):
  marker thứ 5 cy=624.9 → |624.9-619.0|=5.9 ≤ 6 → OK (sát ngưỡng)
  marker thứ 6 cy=625.2 → |625.2-619.0|=6.2 > 6 → BỊ LOẠI ✗

Dùng running mean (sau thêm 5 marker, mean=622.1):
  marker thứ 6 cy=625.2 → |625.2-622.1|=3.1 ≤ 6 → OK ✓
```

Running mean "bám theo" trọng tâm thực của hàng, không bị neo vào điểm đầu có thể lệch.

**Tại sao tolerance 6px không nuốt marker của câu kế tiếp?**

`line_h` ≥ 20px (khoảng cách giữa 2 câu). Khoảng lệch trong cùng 1 câu thường ≤ 3px (do ảnh nghiêng nhẹ). Nên 6px đủ gom, nhỏ hơn nhiều so với 20px để không nhầm sang câu khác.

**Bước 4 — Đóng gói kết quả:**

```python
out = []
for group in rows:
    xs = [m.cx for m in group]
    ys = [m.cy for m in group]
    out.append({
        "cy":      mean(ys),   # Y trung bình của hàng câu
        "x_min":   min(xs),    # X marker trái nhất trong hàng
        "x_max":   max(xs),    # X marker phải nhất trong hàng
        "count":   len(group), # số marker (= số bong bóng được phát hiện)
        "markers": group,      # list marker gốc, dùng ở 7c
    })
```

### Đầu ra của `_cluster_marker_rows`

`list[dict]`, mỗi dict = 1 dòng câu hỏi, ví dụ phiếu 3 câu, 2 block, 4 lựa chọn:

```python
[
  {"cy": 620.1, "x_min": 95.0, "x_max": 765.0, "count": 8, "markers": [...]},  # câu 1
  {"cy": 641.9, "x_min": 95.0, "x_max": 765.0, "count": 8, "markers": [...]},  # câu 2
  {"cy": 663.2, "x_min": 95.0, "x_max": 765.0, "count": 8, "markers": [...]},  # câu 3
]
          ↑ dùng để tính line_h    ↑↑↑ dùng để tính span = x_max - x_min
```

Danh sách này là đầu vào trực tiếp cho **Bước 7c** phân loại fiducial_rows / bubble_rows.

---

## 5. 7c — Phân loại hàng: fiducial_rows vs bubble_rows

Mỗi hàng vừa cluster được kiểm tra **hai tiêu chí độc lập** để phân loại:

### Fiducial row (hàng marker vuông in sẵn)

```python
dark_large = [m for m in row.markers
              if m.fill >= 0.86 and m.area >= 360 and m.size >= 15]

if (len(dark_large) >= 2
        and row.span >= img_w * 0.34   # trải rộng ≥ 34% chiều rộng
        and row.count <= 8):           # không phải hàng bong bóng dày đặc
    fid_rows.append(row)
```

| Tiêu chí | Ngưỡng | Lý do |
|----------|--------|-------|
| fill ≥ 0.86 | "đen lớn" | Marker in sẵn đặc hơn bong bóng tô |
| area ≥ 360 px² | "lớn" | Fiducial lớn hơn bong bóng (~15×15 = 225 px²) |
| size ≥ 15 px | "lớn" | (w+h)/2 ≥ 15 px |
| span ≥ 34% img_w | trải rộng | Hàng fiducial trải từ trái sang phải phiếu |
| count ≤ 8 | ít marker | Fiducial chỉ có 2–5 cái/hàng, không phải 20–80 |

### Bubble row (hàng bong bóng tô)

```python
if row.count >= 8 and row.span >= img_w * 0.50:
    bubble_rows.append(row)
```

| Tiêu chí | Ngưỡng | Lý do |
|----------|--------|-------|
| count ≥ 8 | nhiều marker | Mỗi hàng bong bóng có 4 lựa chọn × N block = 8–20+ |
| span ≥ 50% img_w | trải rộng | Bong bóng trải đều qua nhiều block |

**Lưu ý:** Một hàng có thể **không thuộc loại nào** (bị loại bỏ, không dùng) hoặc **thuộc cả hai** (hiếm, không xảy ra nếu phiếu in đúng chuẩn).

```
Ví dụ phân loại thực tế (ảnh 1000×1400):

  Hàng Y=500: 3 marker, span=650px, fill=[0.91,0.89,0.92]  → fiducial_row ✓
  Hàng Y=600: 16 marker, span=820px, fill=[0.62,0.58,...]  → bubble_row ✓
  Hàng Y=620: 16 marker, span=820px                         → bubble_row ✓
  Hàng Y=640: 16 marker, span=820px                         → bubble_row ✓
  Hàng Y=100: 2 marker, span=200px                          → không loại nào (tiêu đề?)
```

---

## 6. 7d — Tính geometry từ fiducial_rows

Chỉ chạy khi `fid_rows` không rỗng.

### Fiducial marker là gì và nằm ở đâu?

Fiducial marker là **ô vuông đen in sẵn** trên phiếu — không phải bong bóng để tô. Chúng xuất hiện thành hàng ngang ở trên và dưới vùng MCQ, mỗi marker đứng ở **cạnh trái của một block**:

```
     Block 1          Block 2          Block 3
        │                │                │
  ■─────────────────■─────────────────■        ← fiducial row trên (fid_top)
  │                 │                 │
  │  ○ ○ ○ ○       │  ○ ○ ○ ○       │  ○ ○ ○ ○   câu 1
  │  ○ ○ ○ ○       │  ○ ○ ○ ○       │  ○ ○ ○ ○   câu 2
  │    ...          │    ...          │    ...
  │                 │                 │
  ■─────────────────■─────────────────■        ← fiducial row dưới (fid_bottom)
  │                 │                 │
 X=95             X=395             X=695
fid_left_x                         fid_right_x
```

3 marker → 3 block. Mỗi marker đứng ở **cạnh trái** block tương ứng — không có marker ở cạnh phải block cuối (phải ngoại suy).

### Mục tiêu tổng thể của 7d

Từ fiducial_rows, bước 7d rút ra 4 nhóm thông tin:

| Thông tin | Dùng để làm gì |
|-----------|---------------|
| `block_count` | Biết phiếu có mấy cột MCQ → chia đúng số block khi đọc đáp án |
| `fid_top_y`, `fid_bottom_y` | Anchor dọc: biết vùng MCQ bắt đầu/kết thúc ở Y nào |
| `fid_left_x`, `fid_right_x` | Anchor ngang: biết block đầu/marker cuối ở X nào |
| `fid_block_right_x` | Cạnh phải block cuối — cần ngoại suy vì không có marker đánh dấu |

**Tại sao dùng fiducial thay vì chỉ dùng bubble?**

Fiducial in sẵn trên phiếu, luôn tồn tại dù sinh viên bỏ trống câu. Fiducial to và đặc hơn → Bước 6 phát hiện chính xác hơn. Đếm cột fiducial cho `block_count` trực tiếp, không cần thuật toán phức tạp như bucket của 7f.

### 6.1 Sắp xếp và lấy fid_top / fid_bottom

```python
fid_rows = sorted(fid_rows, key=lambda r: r.cy)
fid_top    = fid_rows[0]    # hàng fiducial cao nhất (cy nhỏ nhất)
fid_bottom = fid_rows[-1]   # hàng fiducial thấp nhất (cy lớn nhất)

out["fid_top_y"]    = fid_top.cy
out["fid_bottom_y"] = fid_bottom.cy
```

**Tại sao cần `fid_top_y` và `fid_bottom_y`?** Hai giá trị này là anchor dọc — nói với Bước 8 rằng toàn bộ vùng MCQ nằm trong khoảng Y đó. Bước 8 dùng chúng để thu hẹp vùng tìm kiếm bong bóng, tránh đọc nhầm vùng header hay chữ ký phía trên/dưới.

### 6.2 Đếm block từ số cột fiducial (`_dedupe_x`)

**Vấn đề:** 1 marker vật lý đôi khi bị Connected Components tách thành 2 component riêng do có vùng sáng nhỏ cắt ngang (mực nhòe, ảnh chụp xấu):

```
Marker vật lý:       Sau threshold:

  ████████████         ████████████
  ████████████         ████  ██████   ← vết sáng cắt ngang
  ████████████         ████████████
  ████████████         ████████████

→ Component A: cx=98.2,  Component B: cx=102.1  (cách nhau 3.9px)
→ fid_top.xs = [98.2, 102.1, 398.7, 701.3]  ← 4 giá trị cho 3 marker vật lý
→ nếu đếm thẳng: block_count=4  (SAI)
```

`_dedupe_x` giải quyết bằng cách gộp các X quá gần nhau:

```python
def _dedupe_x(xs, min_gap):
    out = []
    for val in sorted(xs):
        if not out or abs(val - out[-1]) >= min_gap:
            out.append(val)   # giữ lại
        # ngược lại: bỏ qua (quá gần với giá trị vừa giữ)
    return out

fid_top_xs = _dedupe_x(fid_top.xs, min_gap=img_w * 0.08)  # min_gap = 80px
out["block_count"] = len(fid_top_xs)
```

Chạy tay với ví dụ trên:

```
xs sau sort: [98.2, 102.1, 398.7, 701.3]

val=98.2  → out rỗng → thêm vào          out=[98.2]
val=102.1 → |102.1-98.2|=3.9 < 80 → BỎ
val=398.7 → |398.7-98.2|=300.5 ≥ 80 → thêm  out=[98.2, 398.7]
val=701.3 → |701.3-398.7|=302.6 ≥ 80 → thêm out=[98.2, 398.7, 701.3]

→ block_count = 3  ✓
```

**Tại sao min_gap = 80px?** Cùng 1 marker bị tách chỉ cách nhau 3–20px. Khoảng cách giữa 2 block khác nhau là 200–400px. 80px nằm giữa: đủ lớn để gộp cùng marker, đủ nhỏ để không nhầm 2 block khác nhau.

### 6.3 Chọn "best row" làm reference

```python
best_fid_row = max(fid_rows, key=lambda r: r.count)
best_fid_xs  = _dedupe_x(best_fid_row.xs, min_gap=img_w * 0.08)
if len(best_fid_xs) >= len(fid_top_xs):
    out["block_count"] = len(best_fid_xs)
```

**Tại sao không cố định dùng fid_top?** Hàng trên cùng dễ bị cắt xén nhất nếu phiếu lệch khi scan — marker góc trái/phải lọt ra ngoài 10% lề đã lọc → mất marker → đếm thiếu block:

```
fid_top (hàng trên):    ■ . ■ ■   ← marker trái bị cắt, count=2 → block_count=2 (SAI)
fid_bottom (hàng dưới): ■ ■ ■ ■   ← đủ 3 marker, count=3         → block_count=3 (ĐÚNG)

→ best_fid_row = fid_bottom (count lớn hơn) → block_count = 3
```

### 6.4 Tính fid_left_x và fid_right_x

```python
expected  = int(out["block_count"])
full_rows = [r for r in fid_rows if r.count >= expected]  # chỉ hàng đủ marker

out["fid_left_x"]  = np.median([r.x_min for r in full_rows])
out["fid_right_x"] = np.median([r.x_max for r in full_rows])
```

**Tại sao chỉ dùng "full_rows"?** Hàng bị cắt xén thiếu marker có `x_min` lớn hơn thực tế (marker trái bị mất) hoặc `x_max` nhỏ hơn thực tế (marker phải bị mất) → đưa vào tính sẽ kéo lệch kết quả.

**Tại sao median?** Nếu 1 hàng có marker bị lệch vị trí do nhòe:

```
x_min của 3 hàng: [95.1, 94.8, 115.0]  ← hàng 3 lệch 20px

mean   = (95.1+94.8+115.0)/3 = 101.6  ← bị kéo theo
median = 95.1                          ← không bị ảnh hưởng
```

### 6.5 Ngoại suy right edge của block cuối

Fiducial marker đánh dấu **cạnh trái** của mỗi block — không có marker ở cạnh phải block cuối. Cần ngoại suy bằng cách giả định các block **rộng bằng nhau**:

```python
n_fid = len(best_fid_xs)
if n_fid >= 2 and fid_right_x > fid_left_x:
    fid_block_w = (fid_right_x - fid_left_x) / (n_fid - 1)
    out["fid_block_right_x"] = fid_right_x + fid_block_w
    out["fid_block_width"]   = fid_block_w
```

Ví dụ với 3 block:

```
Fiducial tại X:  [95,  395,  695]
                   │    │     │
                block1 block2 block3
                  fid_left_x=95   fid_right_x=695

fid_block_w      = (695 - 95) / (3-1) = 300px   ← bề rộng 1 block
fid_block_right_x = 695 + 300 = 995px            ← cạnh phải block 3 (ngoại suy)

→ Block 1: X ∈ [95,  395)
→ Block 2: X ∈ [395, 695)
→ Block 3: X ∈ [695, 995]
```

---

## 7. 7e — Tính line_h và top/bottom_center_y từ bubble_rows

Chỉ chạy khi `bubble_rows` không rỗng.

### 7.1 Tính line_h

```python
bubble_rows = sorted(bubble_rows, key=lambda r: r.cy)

diffs = []
for i in range(len(bubble_rows) - 1):
    dy = bubble_rows[i+1].cy - bubble_rows[i].cy
    if 12.0 <= dy <= 65.0:   # chỉ nhận khoảng hợp lệ
        diffs.append(dy)

out["line_h"] = np.median(diffs)
```

**Tại sao lọc [12, 65] px?**
- < 12px: hai hàng quá gần, có thể cùng 1 dòng bị cluster nhầm thành 2
- > 65px: khoảng trắng giữa block hoặc marker bị bỏ sót → không phản ánh line_h thật

**Tại sao median?**

```
Ví dụ 5 khoảng dòng liên tiếp: [22.1, 21.8, 22.3, 45.0, 21.9]
                                                     ↑
                               khoảng trắng giữa block (bị lọt qua điều kiện ≤65)

  mean   = (22.1+21.8+22.3+45+21.9) / 5 = 26.6 px  ← bị kéo lệch
  median = 22.1                                      ← không bị ảnh hưởng
```

### 7.2 Tính left_x, right_x

```python
sample_rows = bubble_rows[:4] + bubble_rows[-4:]   # 4 hàng đầu + 4 hàng cuối

out["left_x"]  = np.median([r.x_min for r in sample_rows])
out["right_x"] = np.median([r.x_max for r in sample_rows])
```

Lấy mẫu từ đầu và cuối (bỏ hàng giữa) vì hàng giữa có thể bị căn lệch hơn do nhiều bong bóng tô chồng lấn.

### 7.3 top_center_y và bottom_center_y

```python
out["top_center_y"]    = bubble_rows[0].cy   # Y tâm hàng bong bóng đầu tiên
out["bottom_center_y"] = bubble_rows[-1].cy  # Y tâm hàng bong bóng cuối cùng
```

`top_center_y` là tọa độ Y của **câu MCQ đầu tiên**. Tất cả câu tiếp theo được tính: `cy_câu_n = top_center_y + (n-1) * line_h`.

---

## 8. 7f — Cluster bubble columns thành block_bands

### block_bands là gì và tại sao cần?

Bước 8 cần biết **mỗi block MCQ chiếm vùng X nào** để cắt đúng phần ảnh chứa bong bóng của từng block:

```
X:  0    100   200   300   400   500   600   700   800   900
         │                      │                      │
         │   block 1            │   khoảng trống       │   block 2
         │ bong bóng ở đây      │                      │ bong bóng ở đây
         │ x_min=85  x_max=310  │                      │ x_min=540 x_max=765
```

`block_bands` là list X-range của từng block:
```python
block_bands = [
    {"x_min": 85,  "x_max": 310},   # block 1
    {"x_min": 540, "x_max": 765},   # block 2
]
```

**Tại sao không dùng kết quả fiducial từ 7d?** Fiducial cho vị trí marker in sẵn (cạnh trái block). `block_bands` cho chỗ bong bóng thực sự nằm — hai thứ không bằng nhau, bong bóng thường lùi vào trong so với marker. Hơn nữa nếu phiếu không có fiducial, 7d không chạy, `block_bands` từ 7f là nguồn duy nhất.

### Bài toán cần giải

X-coordinates của bong bóng trên các hàng câu **cụm thành nhóm** — mỗi nhóm là 1 cột lựa chọn (A, B, C, D). Khoảng cách trong cùng block nhỏ (~25px), khoảng cách giữa 2 block lớn (~380px):

```
X bong bóng hàng câu 1 (2 block, 4 lựa chọn):

  85  110  135  160      540  565  590  615
  A    B    C    D        A    B    C    D
  └────── block 1 ───┘   └──── block 2 ────┘
  khoảng ≈25px           gap=380px >> 25px
```

Mục tiêu: tìm chỗ có gap lớn → đó là ranh giới block.

### Đầu vào

| Tham số | Giá trị |
|---------|---------|
| `bubble_rows` | List hàng bong bóng từ 7c (đã sort theo cy) |
| `fid_top_xs` | List X của fiducial hàng trên từ 7d (có thể rỗng) |
| `img_w` | 1000px |

### Thuật toán 4 giai đoạn

**Giai đoạn 1 — Thu thập X từ sample rows:**

**Mục tiêu của giai đoạn này:** Tìm hiểu *vị trí X nào là cột bong bóng thật*. Đặc điểm nhận biết cột bong bóng thật: nó xuất hiện **nhất quán qua nhiều hàng câu** — câu 1 có cột A ở X≈85, câu 2 cũng vậy, câu 3 cũng vậy. Vị trí nhiễu (bong bóng tô lệch, vết mực ngẫu nhiên) chỉ xuất hiện 1–2 hàng rồi mất. Giai đoạn này thu thập bằng chứng, giai đoạn 2 mới lọc lấy các X đủ nhất quán.

```python
band_rows = bubble_rows[:4]                    # chỉ dùng 4 hàng đầu làm mẫu
bucket_w  = max(8.0, img_w * 0.010)            # kích thước 1 bucket = 10px
x_buckets = defaultdict(lambda: {"sum":0, "count":0, "rows":set()})

for ridx, row in enumerate(band_rows):
    row_xs = _dedupe_x(row.xs, min_gap=img_w * 0.018)  # loại X trùng trong hàng

    row_xs = [x for x in row_xs                         # loại X trùng với fiducial
              if min(abs(x - fx) for fx in fid_top_xs) > img_w * 0.030]

    for x in row_xs:
        key = round(x / bucket_w)    # x=87 → key=round(87/10)=9
        x_buckets[key]["sum"]   += x
        x_buckets[key]["count"] += 1
        x_buckets[key]["rows"].add(ridx)
```

**Tại sao chỉ 4 hàng đầu?**

Vị trí cột A, B, C, D là cố định theo thiết kế phiếu — câu 1 hay câu 20 đều có A ở cùng X. Không cần dùng tất cả hàng để biết cột ở đâu, 4 hàng là đủ.

Dùng nhiều hơn thực ra gây hại: sinh viên tô không đều (câu này tô A, câu kia tô C, câu nọ bỏ trống) → các cột xuất hiện với tần suất khác nhau → khó phân biệt cột thật với cột ít sinh viên tô. 4 hàng đầu cho phân phối đều hơn.

**Bucket là gì và tại sao cần?**

Bong bóng cùng cột A qua 4 hàng không nằm đúng cùng X pixel — do ảnh chụp hơi nghiêng hoặc bóp méo nhẹ:

```
Câu 1 cột A: cx = 85.2
Câu 2 cột A: cx = 86.1   ← lệch 0.9px
Câu 3 cột A: cx = 84.8
Câu 4 cột A: cx = 85.5
```

Nếu so sánh X chính xác, 4 giá trị này trông như 4 vị trí khác nhau → mỗi cái chỉ xuất hiện 1 lần → bị lọc bỏ ở Giai đoạn 2. Mất toàn bộ cột A.

Bucket chia trục X thành các ô 10px. Mọi X trong cùng ô quy về cùng 1 ID:

```
x=85.2 → key=round(85.2/10)=9   ┐
x=86.1 → key=round(86.1/10)=9   ├─ cùng bucket 9  → bucket 9 thấy 4 hàng ✓
x=84.8 → key=round(84.8/10)=9   │
x=85.5 → key=round(85.5/10)=9   ┘
x=112  → key=round(112/10)=11     ── bucket 11 (cột B)
```

Sau đó lấy `sum/count` của bucket làm tọa độ X trung bình đại diện cho cột đó — chính xác hơn bất kỳ giá trị đơn lẻ nào.

**Tại sao ghi nhận `rows` (set) thay vì chỉ đếm số lần?**

Nếu chỉ đếm `count += 1`, một hàng duy nhất có nhiều bong bóng gần nhau (sinh viên tô nhầm 2 ô) có thể tạo bucket count=2 trông giống "xuất hiện nhiều":

```
Hàng câu 2: sinh viên tô cả A lẫn B sát nhau
  cx=85.2 → bucket 9, count=1, rows={1}
  cx=86.9 → bucket 9, count=2, rows={1}  ← count tăng nhưng vẫn 1 hàng duy nhất!
```

Dùng `set.add(ridx)` chỉ ghi nhận "hàng nào từng thấy bucket này", không quan tâm hàng đó thấy bao nhiêu lần:

```
ridx=1 đã có trong rows → rows.add(1) không thay đổi → rows={1}  (vẫn 1 hàng)
```

`len(bucket["rows"])` = số hàng **khác nhau** từng thấy vị trí X đó — đây mới là độ tin cậy thực sự.

**Tại sao dedupe X trong từng hàng trước?**

Tương tự vấn đề fiducial ở 7d: 1 bong bóng vật lý có thể bị Connected Components phát hiện thành 2 component gần nhau. Không dedupe → 2 component cùng hàng → cùng bucket → `rows` vẫn chỉ thêm 1 lần (vì set), nhưng `count` tăng 2 và `sum` tính sai → tâm bucket bị lệch.

**Tại sao loại X trùng với fiducial (khoảng cách < 3% img_w)?**

Fiducial marker nằm ở cạnh trái mỗi block, đúng trong vùng X của bong bóng đầu tiên. Nếu không loại, X của fiducial tạo bucket giả — xuất hiện đủ hàng (vì fiducial luôn ở đó) → Giai đoạn 2 giữ lại → Giai đoạn 3 thấy thêm 1 "cột" ảo ở cạnh trái mỗi block → block_bands bị lệch và block_count bị cộng thêm.

**Kết quả sau Giai đoạn 1** — ví dụ 2 block, 4 lựa chọn, 4 hàng mẫu:

```python
x_buckets = {
    9:  {"sum": 341.6, "count": 4, "rows": {0,1,2,3}},  # cột A block 1 → GIỮ
    11: {"sum": 444.0, "count": 4, "rows": {0,1,2,3}},  # cột B block 1 → GIỮ
    14: {"sum": 540.8, "count": 4, "rows": {0,1,2,3}},  # cột C block 1 → GIỮ
    16: {"sum": 641.2, "count": 4, "rows": {0,1,2,3}},  # cột D block 1 → GIỮ
    54: {"sum": 2163,  "count": 4, "rows": {0,1,2,3}},  # cột A block 2 → GIỮ
    17: {"sum": 173.0, "count": 1, "rows": {1}},         # nhiễu → BỎ ở G2
}
```

Giai đoạn 2 chỉ cần lọc `len(rows) >= 2` rồi lấy `sum/count` làm X đại diện cho mỗi cột.

**Giai đoạn 2 — Lọc bucket đáng tin cậy:**

```python
min_row_hits = max(2, ceil(len(band_rows) * 0.5))  # ≥ 50% số hàng mẫu = 2

merged_xs = []
for bucket in x_buckets.values():
    if len(bucket["rows"]) >= min_row_hits:         # xuất hiện ≥ 2 hàng
        merged_xs.append(bucket["sum"] / bucket["count"])  # tâm bucket

merged_xs = sorted(merged_xs)
```

Chỉ giữ bucket xuất hiện trong ít nhất 2 hàng — đây là cột bong bóng thật. Bucket chỉ 1 hàng là nhiễu (bong bóng tô lệch, vết mực ngẫu nhiên):

```
Bucket 9  (X≈85):  rows={0,1,2,3} → 4 hàng ≥ 2 → GIỮ ✓  (cột A thật)
Bucket 17 (X≈173): rows={1}       → 1 hàng < 2 → BỎ  ✗  (nhiễu)
Bucket 55 (X≈540): rows={0,1,2,3} → 4 hàng ≥ 2 → GIỮ ✓  (cột A block 2)
```

**Giai đoạn 3 — Tìm ranh giới giữa các block:**

```python
diffs = [merged_xs[i+1] - merged_xs[i] for i in range(len(merged_xs)-1)]
small_diffs = [d for d in diffs if 8.0 <= d <= 80.0]  # khoảng trong 1 block

split_gap = max(58.0, np.median(small_diffs) * 2.1)

groups  = []
current = [merged_xs[0]]
for val in merged_xs[1:]:
    if (val - current[-1]) > split_gap:
        groups.append(current)   # kết thúc block
        current = [val]
    else:
        current.append(val)
groups.append(current)
```

**Cách xác định split_gap — ví dụ số:**

```
merged_xs: [85, 110, 135, 160,   540, 565, 590, 615]
diffs:          25   25   25  380    25   25   25

small_diffs (8–80px): [25, 25, 25, 25, 25, 25]
median(small_diffs) = 25px  ← khoảng trong 1 block

split_gap = max(58, 25×2.1) = max(58, 52.5) = 58px

Duyệt:
  85→110:  gap=25 ≤ 58 → cùng group (block 1)
  110→135: gap=25 ≤ 58 → cùng group
  135→160: gap=25 ≤ 58 → cùng group
  160→540: gap=380 > 58 → HẾT BLOCK 1 → bắt đầu block 2
  540→565: gap=25 ≤ 58 → cùng group (block 2)
  ...
```

**Tại sao `split_gap = max(58, median × 2.1)` chứ không phải cứng 1 giá trị?**

Khoảng cách giữa 2 bong bóng trong 1 block thay đổi theo loại phiếu (phiếu bong bóng nhỏ ≈15px, phiếu lớn ≈40px). Khoảng trống giữa 2 block luôn lớn hơn đáng kể. Nhân `2.1×` đảm bảo split_gap luôn nằm giữa 2 vùng đó:

```
Phiếu bong bóng nhỏ: khoảng trong block=15px, gap giữa block=120px
  split_gap = max(58, 15×2.1=31.5) = 58px → 58 < 120 → tách đúng ✓

Phiếu bong bóng lớn: khoảng trong block=40px, gap giữa block=200px
  split_gap = max(58, 40×2.1=84) = 84px → 84 < 200 → tách đúng ✓

Nếu không có sàn 58px, phiếu bong bóng nhỏ (15px):
  split_gap = 31.5px → khoảng B→C trong block là 33px > 31.5 → tách nhầm thành 2 block! ✗
```

**Giai đoạn 4 — Tạo block_bands:**

```python
band_groups = [g for g in groups if len(g) >= 3]   # ≥ 3 cột = block hợp lệ

out["block_bands"] = [
    {"x_min": min(g), "x_max": max(g)}
    for g in band_groups
]
out["block_count"] = len(band_groups)
```

`len(g) >= 3`: loại group chỉ có 1–2 điểm (nhiễu sót). Block MCQ tối thiểu có 3 lựa chọn (A, B, C).

**Ví dụ đầy đủ 2 block, 4 lựa chọn:**

```
merged_xs: [85, 110, 135, 160, 540, 565, 590, 615]

groups sau tách:
  group 1: [85, 110, 135, 160]   len=4 ≥ 3 → hợp lệ
  group 2: [540, 565, 590, 615]  len=4 ≥ 3 → hợp lệ

block_bands = [
    {"x_min": 85,  "x_max": 160},   # block 1
    {"x_min": 540, "x_max": 615},   # block 2
]
block_count = 2
```

---

## 9. Đầu ra (Output)

Hàm trả về `dict` — có thể không chứa tất cả các key nếu không tìm được marker tương ứng:

| Key | Kiểu | Nguồn | Dùng ở đâu |
|-----|------|-------|-----------|
| `line_h` | `float` | median(delta bubble rows) | Bước 8: tính tọa độ Y từng câu |
| `top_center_y` | `float` | bubble_rows[0].cy | Bước 8: Y câu MCQ đầu tiên |
| `bottom_center_y` | `float` | bubble_rows[-1].cy | Bước 8: xây ROI MCQ |
| `block_count` | `float` | fiducial cột hoặc band_groups | Bước 8: số block cần decode |
| `block_bands` | `list[dict]` | bucket algorithm | Bước 8: ROI X từng block |
| `fid_top_y` | `float` | fid_rows[0].cy | Bước 8: anchor Y trên |
| `fid_bottom_y` | `float` | fid_rows[-1].cy | Bước 8: anchor Y dưới |
| `fid_left_x` | `float` | median(fid_rows.x_min) | Bước 8: căn lề trái |
| `fid_right_x` | `float` | median(fid_rows.x_max) | Bước 8: căn lề phải |
| `fid_block_right_x` | `float` | ngoại suy từ fid_block_w | Bước 8: right edge block cuối |
| `fid_block_width` | `float` | (fid_right_x-fid_left_x)/(N-1) | Bước 8: bề rộng 1 block |
| `left_x` | `float` | median(bubble_rows.x_min) | Bước 8: biên ngang trái MCQ |
| `right_x` | `float` | median(bubble_rows.x_max) | Bước 8: biên ngang phải MCQ |
| `rows_per_block_hint` | `int` | input `rows_per_block` | Bước 8: tham chiếu |
| `block_count_hint` | `int` | input `block_count_hint` | Bước 8: tham chiếu |
| `long_form_mode` | `bool` | rows_hint ≥ 25 hoặc blocks_hint ≥ 4 | Bước 8: chọn template anchor |

**Trường hợp trả về dict rỗng `{}`:** Khi `_cluster_marker_rows` không tìm được hàng nào (ảnh không có marker trong vùng tìm kiếm). Bước 8 sẽ fallback hoàn toàn về tọa độ profile.

---

## 10. Sơ đồ luồng

```
markers (list[dict] từ Bước 6)
        │
        ▼
  7a — Xác định chế độ:
        rows_hint, blocks_hint → long_form_mode → min_y_ratio
        │
        ▼
  7b — _cluster_marker_rows (tolerance ±6px)
        Lọc: x∈[10%,90%], y∈[min_y_ratio,98%]
        Gom nhóm theo Y → danh sách "rows"
        │
        ├── rows rỗng → return {}
        │
        ▼
  7c — Phân loại từng hàng:
        ┌─────────────────────┐    ┌────────────────────┐
        │   fiducial_rows     │    │    bubble_rows      │
        │ (fill≥0.86,         │    │ (count≥8,          │
        │  area≥360,size≥15,  │    │  span≥50% img_w)   │
        │  span≥34%, count≤8) │    │                    │
        └──────────┬──────────┘    └─────────┬──────────┘
                   │                         │
        ┌──────────▼──────────┐   ┌──────────▼──────────────┐
        │ 7d — Fiducial geo   │   │ 7e — Bubble geo          │
        │                     │   │                          │
        │ Sort → top/bottom   │   │ Sort → delta-Y → median  │
        │ dedupe_x → block_   │   │ → line_h                 │
        │   count             │   │                          │
        │ best_row reference  │   │ top_center_y             │
        │ median left/right_x │   │ bottom_center_y          │
        │ ngoại suy block_    │   │ left_x, right_x          │
        │   right_x           │   │                          │
        └──────────┬──────────┘   └──────────┬──────────────┘
                   │                         │
                   │              ┌──────────▼──────────────┐
                   │              │ 7f — Block bands         │
                   │              │                          │
                   │              │ Bucket X-positions       │
                   │              │ Lọc ≥50% rows            │
                   │              │ Split tại gap lớn        │
                   │              │ → block_bands[]          │
                   │              └──────────┬──────────────┘
                   │                         │
                   └────────────┬────────────┘
                                ▼
                    out dict (geometry parameters)
                                │
                    ├──→ Bước 8: _resolve_coordinate_anchors()
                    └──→ Bước 8: _build_rois_from_anchors()
```
