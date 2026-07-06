# Bước 2 — Chuẩn hóa ánh sáng & Phát hiện marker góc

> Tài liệu này giải thích **tại sao** từng con số, từng phương pháp được chọn trong Bước 2 của pipeline OMR, không chỉ mô tả **cái gì**. Cấu trúc các mục dưới đây đi **đúng theo thứ tự hàm gọi hàm trong code thật** — đọc từ mục 3 đến mục 8 là đọc đúng theo thứ tự code chạy. Code tham chiếu: `omr_marker_utils.py` và `omr_preprocess.py`.

---

## 1. Mở đầu — điểm chuyển giao từ Bước 1

**Đầu vào (`gray_img`):** là ảnh xám của `img_input` — chính là `img_raw` ở Bước 1 (không qua bước cắt nào trung gian, `img_input = img_raw`) — ở **độ phân giải gốc** (kích thước ảnh chụp/scan ban đầu, **chưa** warp, **chưa** resize về 1000×1400).

**Chuỗi gọi thật trong code** (không phải 1 lệnh gọi riêng tên "Bước 2"):

```
process_omr_exam()  (omr_service.py:999)
  → _warp_to_standard_layout(img_input, ...)          ← 1 lệnh gọi duy nhất
      (omr_preprocess.py:93)
      → gray_src = cv2.cvtColor(img_bgr, COLOR_BGR2GRAY)   (dòng 110)
      → _detect_page_quad(gray_src)                        (dòng 111)   ← BƯỚC 2 BẮT ĐẦU TỪ ĐÂY, xem mục 3
```

> **Ghi chú:** trước đây `gray_src`/`_detect_page_quad(...)` còn bọc trong điều kiện `if not global_warp_used:` — tàn dư của nhánh `manual_quad_norm` (crop 4 góc thủ công) đã bị xóa vì không có UI/driver nào dùng tới. Sau khi xóa nhánh đó, điều kiện luôn đúng nên cũng đã bỏ hẳn — `_warp_to_standard_layout()` giờ chỉ còn đúng 1 đường thẳng, không rẽ nhánh. Chi tiết xem `Buoc1.md` mục 3.5.

**Lưu ý quan trọng:** Bước 2 (phát hiện marker góc), Bước 3 (contour fallback), Bước 4 (warp + resize) — theo sơ đồ khái niệm ở `xu_ly_anh.md` — **không phải 3 lệnh gọi tách biệt** trong `process_omr_exam()`. Cả 3 đều nằm lồng bên trong **một** hàm duy nhất, `_warp_to_standard_layout()`. Tài liệu này chỉ mô tả phần đầu của hàm đó — từ lúc gọi `_detect_page_quad()` cho đến khi có được `quad`.

---

## 2. Bối cảnh: tại sao Bước 2 phức tạp?

Mục tiêu là tìm 4 ô vuông đen in ở 4 góc tờ phiếu để làm điểm anchor cho phép warp phối cảnh. Nghe đơn giản, nhưng ảnh đầu vào có 3 vấn đề thực tế:

1. **Ánh sáng không đều**: điện thoại chụp tay, góc trên có thể sáng hơn góc dưới 40–60 độ sáng. Một ngưỡng đen/trắng toàn cục sẽ "mất" marker ở vùng tối hoặc nhận nhầm bóng đổ ở vùng sáng.
2. **Nhiễu hình học**: ảnh nén JPEG tạo artifact; bút tô tạo blob tròn có thể nhầm là marker; đường kẻ, chữ in có thể tạo cạnh sắc.
3. **Marker corner là minority**: phiếu có hàng trăm bong bóng câu hỏi, chữ số MSSV, đường kẻ — tất cả đều là điểm đen. Marker góc chỉ có 4 cái trong một "đám đông" hàng trăm blob.

Ba vấn đề này chính là lý do phải qua nhiều tầng hàm xử lý (mục 3 → 8) thay vì chỉ đặt 1 ngưỡng đen/trắng đơn giản rồi tìm 4 blob hình vuông.

---

## 3. Hàm điều phối — `_detect_page_quad()`

**Mục đích:** đây là hàm cấp cao nhất mà `_warp_to_standard_layout()` gọi. Nhiệm vụ duy nhất của nó là **quyết định chiến lược nào tìm được 4 góc trang giấy**: thử marker trước (đáng tin nhất), nếu thất bại thử contour (kém tin cậy hơn nhưng còn hơn không), nếu vẫn thất bại thì báo thua.

```python
# omr_preprocess.py:79-90 — toàn bộ hàm
def _detect_page_quad(gray_img):
    marker_pts = omr_marker_utils._detect_page_corners_from_black_square_markers(gray_img)
    if marker_pts is not None:
        arr = ...                                        # sắp lại thứ tự 4 điểm
        return _order_quad_points(arr), "corner-markers"  # ← thử marker THÀNH CÔNG

    contour_pts = _find_page_quad_by_contour(gray_img)
    if contour_pts is not None:
        return contour_pts, "page-contour"                # ← marker thất bại, contour cứu được

    return None, "none"                                    # ← cả 2 đều thất bại
```

**Đầu ra — `(quad, quad_strategy)`:**

- **`quad`** — mảng 4 điểm tọa độ `[TL, TR, BR, BL]` (đã sắp thứ tự qua `_order_quad_points()`), hoặc `None` nếu không tìm được góc nào. Đây là giá trị **dùng thật** để tính ma trận `getPerspectiveTransform` ở Bước 4 — không phải chỉ để hiển thị.
- **`quad_strategy`** — một **chuỗi nhãn chẩn đoán** (diagnostic label), hoàn toàn không ảnh hưởng đến phép toán warp. Nó chỉ ghi lại *quad này tìm được bằng cách nào*, để sau này debug/thống kê biết phiếu đó dễ hay khó xử lý:

| Giá trị `quad_strategy` | Ý nghĩa |
|---|---|
| `"corner-markers"` | Tìm được nhờ 4 marker góc — nhánh đầu tiên, trường hợp tốt nhất, chính xác nhất |
| `"page-contour"` | Nhánh marker thất bại, phải dùng contour fallback (Bước 3) — kém chính xác hơn |
| `"none"` | Cả 2 nhánh đều thất bại, `quad = None` |

`quad_strategy` sau đó được nhúng thẳng vào chuỗi `strategy` của `_warp_to_standard_layout()` (dòng `strategy = f"coordinate-global-a4:{quad_strategy}"`), rồi trả ngược lên thành `warp_strategy` trong JSON kết quả cuối cùng ở Bước 15 — ví dụ `"coordinate-global-a4:page-contour"` cho biết phiếu này phải rơi vào contour fallback thay vì marker.

Toàn bộ phần còn lại của tài liệu này (mục 4 → 8) giải thích **nhánh đầu tiên** — `omr_marker_utils._detect_page_corners_from_black_square_markers(gray_img)` — vì đây là nhánh chạy trước, phức tạp nhất, và là trường hợp mong muốn (chính xác nhất).

---

## 4. Hàm tìm 4 góc từ marker — `_detect_page_corners_from_black_square_markers()`

**Mục đích:** nhận ảnh xám gốc, trả về 4 điểm góc của tờ phiếu — xác định bằng cách tìm 4 marker vuông đen in sẵn gần 4 góc.

**Đầu vào:** `gray_img` (nguyên vẹn từ `_detect_page_quad`, chưa xử lý gì thêm).
**Đầu ra:** mảng 4 điểm `[TL, TR, BR, BL]`, hoặc `None` nếu không tìm đủ 4 marker.

Việc đầu tiên hàm này làm là gọi một hàm con để lấy **danh sách toàn bộ marker ứng viên** trên ảnh — chưa biết cái nào là góc, chỉ mới lọc ra những blob "trông giống marker vuông đen":

```python
# omr_marker_utils.py:184-192 — nửa đầu hàm _detect_page_corners_from_black_square_markers()
h, w = gray_img.shape[:2]

markers = _extract_black_square_markers_from_gray(
    gray_img, min_area_ratio=0.00008, max_area_ratio=0.025, max_markers=260,
)
if len(markers) < 4:
    return None   # không đủ ứng viên để có 4 góc → _detect_page_quad() sẽ thử contour fallback (Bước 3)
```

Lệnh gọi `_extract_black_square_markers_from_gray(...)` này là **toàn bộ nội dung mục 5 và mục 6** bên dưới — hàm đó tự làm 2 việc bên trong nó (chuẩn hóa ánh sáng rồi lọc theo 8 tiêu chí) trước khi trả danh sách marker về đây. Sau khi có danh sách này, hàm `_detect_page_corners_from_black_square_markers()` còn phải chạy tiếp phần thân còn lại của nó (chia 4 vùng, chấm điểm, chọn góc) — phần đó quay lại giải thích ở **mục 7**, sau khi đã hiểu rõ danh sách `markers` được tạo ra như thế nào.

---

## 5. Hàm chuẩn hóa ánh sáng — `_extract_black_square_markers_from_gray()`

**Mục đích:** nhận ảnh xám còn nguyên gradient sáng/tối do ánh đèn không đều, khử gradient đó, nhị phân hóa thành ảnh đen/trắng sạch, rồi gọi tiếp hàm lọc hình dạng.

**Đầu vào:** `gray_img`, cùng 3 tham số `min_area_ratio`, `max_area_ratio`, `max_markers` (được `_detect_page_corners_from_black_square_markers` truyền vào ở mục 4).
**Đầu ra:** danh sách marker đã qua lọc — output này chính là biến `markers` dùng ở mục 4 và mục 7.

```python
# omr_marker_utils.py:109-115 — nửa đầu hàm (phần chuẩn hóa ánh sáng)
h, w = gray_img.shape[:2]
k = max(31, (min(h, w) // 10) | 1)
bg = cv2.morphologyEx(gray_img, cv2.MORPH_CLOSE,
         cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
norm = cv2.divide(gray_img, bg, scale=255)
norm = cv2.GaussianBlur(norm, (3, 3), 0)
_, bin_inv = cv2.threshold(norm, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
```

Năm dòng trên biến `gray_img` (ảnh xám, nhiễu sáng) thành `bin_inv` (ảnh nhị phân sạch, marker = pixel trắng). Bốn mục con dưới đây giải thích từng dòng.

### 5.1. Tại sao dùng Morphological Close để ước tính background?

**Ý tưởng cốt lõi**: muốn tách "nền sáng" ra khỏi "mực đen". Nếu có thể ước tính được ảnh nền (như thể xóa hết mực), chia ảnh gốc cho nền → mực đen sẽ ra giá trị nhỏ (sẫm), nền trắng ra ~1.0.

**Morphological Close = Dilate rồi Erode — hoạt động như thế nào?**

Hãy tưởng tượng một dòng pixel cắt ngang qua tờ phiếu (0 = đen, 255 = trắng):

```
Nền sáng   Marker đen        Nền (tối hơn do bóng đổ)   Nền sáng
  255  255  255  255   40   35   42   38   255  180  200  220  255
```

Mục tiêu: ước tính đường nền — giá trị sẽ là bao nhiêu nếu không có mực.

**Dilate (giãn):**

Mỗi pixel được thay bằng **giá trị lớn nhất** trong vùng k×k xung quanh nó.

> **Kernel là gì?** Kernel k×k là một ô vuông k×k pixel dùng để "quét" qua từng pixel của ảnh — đặt tâm kernel lên pixel đang xét, nhìn vào toàn bộ vùng k×k xung quanh, lấy max (Dilate) hoặc min (Erode). Kernel 5×5 có 25 pixel, kernel 303×303 có 91,809 pixel. Kernel trượt qua lần lượt từng pixel — với ảnh 12MP thì ~12 triệu lần tính.

```
Kernel 5×5 đặt tâm lên pixel "42":
┌─┬─┬─┬─┬─┐
│ │ │ │ │ │   255  255  255  255  255
├─┼─┼─┼─┼─┤    40   35   42   38  255
│ │ │█│ │ │   255  180  200  220  255
├─┼─┼─┼─┼─┤   255  255  255  255  255
│ │ │ │ │ │   255  255  255  255  255
└─┴─┴─┴─┴─┘
       ↑ pixel đang xét (tâm)
Max của 25 pixel này = 255 → pixel "42" được thay bằng 255
```

```
Trước Dilate:  255  255  255  255   40   35   42   38  255  180  200  220  255
Sau Dilate:    255  255  255  255  255  255  255  255  255  255  255  255  255
```

Vùng marker đen (40, 35, 42, 38) biến mất hoàn toàn — kernel 5×5 chạm vào vùng 255 bên cạnh, nên "giá trị lớn nhất trong vùng 5×5" luôn là 255.

Vùng bóng đổ (180, 200, 220) cũng bị kéo sáng về 255 vì kernel chạm vào vùng 255 lân cận.

> Dilate = pixel sáng "lan" ra xung quanh. Vùng tối nhỏ bị nuốt chửng bởi vùng sáng lân cận.

**Erode (co):**

Mỗi pixel được thay bằng **giá trị nhỏ nhất** trong vùng k×k xung quanh nó.

```
Sau Dilate:  255  255  255  255  255  255  255  255  255  255  255  255  255
Sau Erode:   255  255  255  255  255  255  255  255  255  255  255  255  255
```

Vì sau Dilate mọi vùng đều là 255, Erode không thay đổi gì ở đây. Ở cạnh ảnh hoặc bóng đổ rộng (lớn hơn kernel), Erode khôi phục kích thước vùng sáng ban đầu — nhưng vùng tối nhỏ (marker) đã bị Dilate lấp đầy nên **không được khôi phục**.

**Kết quả:** ảnh background ước tính — tờ giấy trắng không có mực, nhưng vẫn giữ gradient sáng tối do ánh đèn:

```
Ảnh gốc:     255  255  255  255   40   35   42   38  255  180  200  220  255
Sau Close:   255  255  255  255  255  255  255  255  255  210  215  220  255
                                   ↑ marker biến mất    ↑ bóng đổ được giữ
```

**Chia để chuẩn hóa:**

```
norm = (ảnh_gốc / ảnh_background) × 255

Vùng marker:  40 / 255 × 255 =  40  → vẫn tối (có mực)
Vùng nền:    255 / 255 × 255 = 255  → trắng (nền sáng)
Vùng bóng:   180 / 210 × 255 = 218  → sáng lên (bóng đổ đã bị bù)
```

Sau phép chia, bóng đổ từ 180 lên 218 — gần bằng nền bình thường. Mực đen vẫn là 40. Otsu threshold bây giờ dễ dàng phân tách: nền ≈ 210–255, mực ≈ 30–80.

**Tại sao phải Close (Dilate + Erode), không chỉ Dilate?**

Nếu chỉ Dilate: vùng bóng đổ rộng bị kéo sáng hoàn toàn về 255 dù thực tế chỉ là 180–220. Sau khi chia, vùng bóng đổ cho giá trị thấp hơn thực tế → Otsu nhầm thành "mực".

Erode sau Dilate "trả lại" kích thước vùng sáng lớn ban đầu. Chỉ những vùng tối **nhỏ hơn kernel** (marker, chữ, bong bóng) mới bị mất — vùng tối **lớn hơn kernel** (bóng đổ rộng do ánh đèn) vẫn được giữ lại ở mức gần đúng.

Với kernel cỡ 31+ px, Close xóa mọi vùng đen có kích thước ≤ ~15 px (nửa kernel), bao gồm marker (~15–20px), bong bóng, chữ in — chỉ giữ lại gradient sáng nền.

**Tại sao không dùng các phương pháp khác?**

| Phương pháp | Vấn đề |
|-------------|--------|
| Gaussian Blur (kernel lớn) | Blur làm mờ vùng tối, không xóa hẳn. Vùng mực vẫn kéo tối background ước tính → hiệu chuẩn sai |
| Mean Blur | Tương tự Gaussian Blur, cùng vấn đề |
| CLAHE (Contrast Limited Adaptive Histogram Equalization) | Tăng tương phản cục bộ nhưng không ước tính background → threshold sau vẫn nhạy với gradient sáng toàn cục |
| Histogram equalization toàn cục | Chỉ phân phối lại histogram, không loại bỏ gradient ánh sáng |
| **Morphological Close** | **Xóa hẳn mực bằng cách lấp vùng tối → background ước tính sạch** |

**Tại sao kernel hình elip (`MORPH_ELLIPSE`) thay vì hình chữ nhật?**

Kernel chữ nhật tạo artifact ở góc: khi dilate, vùng tối vuông góc được lấp không đều, tạo "vệt" theo chiều ngang và dọc. Kernel elip (hình tròn) đều theo mọi hướng — phù hợp với gradient ánh sáng thực tế, vốn thay đổi mượt mà theo hướng bất kỳ, không theo trục nào cố định.

**`cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))` cụ thể làm gì?**

Hàm này **không xử lý ảnh** — nó chỉ tạo ra **ma trận kernel** (một mảng 0/1 kích thước k×k) mà `cv2.morphologyEx()` dùng làm khuôn quét Dilate/Erode. `cv2.morphologyEx()` bắt buộc cần một ma trận cụ thể, không chỉ một con số kích thước — nếu tự viết `np.ones((k, k))` thay vào đó, kernel sẽ đổi từ ellipse (hình tròn) sang hình vuông đặc, mất đúng đặc tính vừa nêu ở trên.

So sánh 2 kernel k=7 (1 = thuộc kernel, quét max/min tại đó; 0 = không thuộc, bỏ qua):

```
Kernel vuông np.ones((7,7)):        Kernel ellipse getStructuringElement(MORPH_ELLIPSE, (7,7)):
1 1 1 1 1 1 1                        0 0 1 1 1 0 0
1 1 1 1 1 1 1                        1 1 1 1 1 1 1
1 1 1 1 1 1 1                        1 1 1 1 1 1 1
1 1 1 1 1 1 1                        1 1 1 1 1 1 1
1 1 1 1 1 1 1                        1 1 1 1 1 1 1
1 1 1 1 1 1 1                        1 1 1 1 1 1 1
1 1 1 1 1 1 1                        0 0 1 1 1 0 0
```

Kernel vuông có "tầm với" theo đường chéo dài hơn theo trục ngang/dọc (bán kính chéo ≈ k/√2 ≈ 0.71k, trong khi bán kính ngang/dọc chỉ 0.5k) — không đều theo hướng. Kernel ellipse có bán kính gần bằng nhau ở mọi hướng (≈ k/2) — khớp đúng bản chất đẳng hướng (không lệch hướng) của gradient ánh sáng đèn thật, nên không tạo vệt giả theo đường chéo như kernel vuông.

---

### 5.2. Tại sao `k = max(31, (min(h, w) // 10) | 1)`?

Mục tiêu của công thức này: tính ra một kích thước kernel vừa đủ lớn để xóa hoàn toàn marker ra khỏi ảnh background ước tính, vừa tự thích nghi với mọi kích thước ảnh đầu vào.

Ba thành phần của công thức phục vụ ba yêu cầu khác nhau:

---

#### Thành phần 1 — `min(h, w) // 10` : kernel tỷ lệ theo ảnh

**Làm gì:** lấy 10% cạnh nhỏ hơn của ảnh làm kích thước kernel.

**Tại sao cần:** marker in trên giấy có kích thước vật lý cố định (~5mm), nhưng kích thước pixel thay đổi hoàn toàn theo ai chụp và thiết bị nào:

| Cách chụp | Kích thước ảnh | Marker ~5mm |
|-----------|---------------|-------------|
| Điện thoại 12MP | 4032×3024 | ~95px |
| Scan 300 DPI | 2480×3508 | ~59px |
| Ảnh nhỏ/crop | 400×300 | ~10px |

Nếu dùng kernel cố định 51px:
- Ảnh 12MP: marker 95px → kernel 51px < marker → **kernel không đủ lớn để xóa tâm marker** → tâm marker sót lại trong background ước tính → sau khi chia, vùng marker cho giá trị thấp → Otsu nhầm thành "nền tối" thay vì mực đen.
- Ảnh nhỏ 400px: kernel 51px so với ảnh 400px = 12.7% → quá lớn → Close xóa cả bóng đổ ánh sáng → background ước tính phẳng hoàn toàn → mất thông tin gradient ánh sáng → chuẩn hóa kém.

Dùng 10% cạnh ảnh → kernel luôn tỷ lệ với ảnh → marker (cũng tỷ lệ với ảnh) luôn nhỏ hơn kernel.

**Kết quả đạt được:** với mọi thiết bị chụp, kernel đủ lớn để xóa marker nhưng không quá lớn để xóa cả bóng đổ nền.

**Nếu không dùng (dùng giá trị cố định):** hệ thống hoạt động tốt với một loại điện thoại nhưng thất bại với điện thoại khác có độ phân giải cao hơn hoặc thấp hơn.

---

#### Thành phần 2 — `| 1` : ép về số lẻ

**Làm gì:** đảm bảo k là số lẻ bằng phép bitwise OR với 1.

**Tại sao cần:** kernel morphological phải có kích thước lẻ vì cần một điểm tâm xác định — pixel đang xét nằm đúng giữa kernel. Kernel chẵn có tâm lý thuyết ở tọa độ (1.5, 1.5) — không phải pixel nguyên → ảnh output bị lệch nửa pixel.

```
Kernel 3×3 (lẻ):   Kernel 4×4 (chẵn):
┌─┬─┬─┐            ┌─┬─┬─┬─┐
│ │ │ │            │ │ │ │ │
├─┼─┼─┤            ├─┼─┼─┼─┤
│ │█│ │ ← tâm rõ   │ │?│?│ │ ← tâm ở đâu?
├─┼─┼─┤            ├─┼─┼─┼─┤
│ │ │ │            │ │ │ │ │
└─┴─┴─┘            └─┴─┴─┘
```

`| 1` là phép **bitwise OR** (OR từng bit nhị phân) với 1 — trick ép về số lẻ trong 1 phép tính:

```
302 nhị phân:  1 0 0 1 0 1 1 1 0   (chẵn → bit cuối = 0)
  1 nhị phân:  0 0 0 0 0 0 0 0 1
OR từng bit:   1 0 0 1 0 1 1 1 1  = 303  ← cộng thêm 1

303 nhị phân:  1 0 0 1 0 1 1 1 1   (lẻ → bit cuối = 1)
  1 nhị phân:  0 0 0 0 0 0 0 0 1
OR từng bit:   1 0 0 1 0 1 1 1 1  = 303  ← giữ nguyên
```

Quy tắc OR: chỉ cần một trong hai là 1 thì ra 1. Kết quả: số chẵn +1 (thành lẻ), số lẻ giữ nguyên.

**Kết quả đạt được:** k luôn lẻ, OpenCV chấp nhận, ảnh background ước tính căn chỉnh chính xác với ảnh gốc.

**Nếu không dùng:** `min(h,w) // 10` ra số chẵn khoảng 50% trường hợp → OpenCV báo lỗi `(-5:Bad argument) in function 'morphologyEx'` và dừng toàn bộ pipeline.

---

#### Thành phần 3 — `max(31, ...)` : sàn tối thiểu

**Làm gì:** đảm bảo k không bao giờ nhỏ hơn 31px, dù ảnh đầu vào rất nhỏ.

**Tại sao cần:** với ảnh rất nhỏ (ví dụ 200×150px sau crop nặng), `min(h,w) // 10 = 15` — kernel 15px nhỏ hơn marker 10px chỉ ~1.5 lần, bán kính kernel (7px) chưa chắc vượt qua tâm marker (5px) → Close không xóa hết.

Marker góc dù ảnh nhỏ vẫn chiếm ~5% cạnh ảnh (5mm / tổng ~100mm). Kernel 15px trên ảnh 200px = 7.5% — sát ngưỡng, dễ thất bại khi marker in đậm hơn bình thường.

Sàn 31px = 2× kích thước marker tối thiểu thực tế (~15px trên ảnh nhỏ nhất có thể xử lý được) — đảm bảo luôn đủ bán kính để xóa tâm marker.

**Kết quả đạt được:** trên ảnh nhỏ hoặc ảnh crop, pipeline không thất bại thầm lặng (marker còn sót trong background → chuẩn hóa sai → threshold sai → mất marker → warp thất bại).

**Nếu không dùng:** ảnh nhỏ hơn ~300px có thể ra background ước tính bị "nhiễm" marker → phép chia sau đó cho vùng marker giá trị gần bằng nền → Otsu không thể phân tách → không tìm được marker góc → toàn bộ warp phải dùng fallback contour (kém chính xác hơn).

---

**Kích thước kernel thực tế theo loại ảnh:**

| Ảnh đầu vào | Kích thước | k thực tế | Marker ~5mm | Bán kính k | Bán kính marker | Kết quả |
|-------------|-----------|-----------|-------------|-----------|----------------|---------|
| Điện thoại 12MP | 4032×3024 | **303** | ~95px | 151px | 47px | 151 > 47 ✓ |
| Ảnh trung bình | 1200×900 | **91** | ~29px | 45px | 14px | 45 > 14 ✓ |
| Ảnh nhỏ crop | 400×300 | **31** | ~10px | 15px | 5px | 15 > 5 ✓ |
| Ảnh rất nhỏ | 200×150 | **31** (sàn) | ~5px | 15px | 2px | 15 > 2 ✓ |

Bán kính kernel luôn lớn hơn bán kính marker → Dilate luôn "vươn qua" tâm marker ra vùng nền trắng bên ngoài → marker bị xóa hoàn toàn khỏi background ước tính.

---

### 5.3. Tại sao chia (`divide`) thay vì trừ (`subtract`)?

```python
norm = cv2.divide(gray_img, bg, scale=255)
# tương đương: norm = (gray / bg) × 255
```

**`norm` là gì:** tỷ lệ sáng của pixel so với giấy trắng tại cùng vị trí đó — hay nói cách khác, *nếu giấy ở đây đạt 255, pixel này sẽ là bao nhiêu?*

- Giấy trắng: gray ≈ bg → norm ≈ 255
- Mực đen: gray ≪ bg → norm ≈ 64 (mực phản chiếu ~25% ánh sáng, giấy ~95%)

**Vấn đề cần giải quyết:**

Khi ánh đèn chiếu không đều (điện thoại chụp nghiêng), cùng một tờ phiếu có hai vùng sáng khác nhau. Reflectance của mực là tính chất vật lý cố định — mực luôn phản chiếu ~25% ánh sáng. Nhưng pixel value thay đổi theo cường độ đèn:

```
Vùng sáng (Light=200):  gray_mực = 0.25 × 200 = 50,  bg = 0.95 × 200 = 190
Vùng tối  (Light=80):   gray_mực = 0.25 × 80  = 20,  bg = 0.95 × 80  = 76
```

Histogram ảnh gốc có 3 đỉnh — đỉnh mực (~20–50), đỉnh nền tối (~76), đỉnh nền sáng (~190):

```
Histogram ảnh gốc (chưa chuẩn hóa):
    │  ↑ mực    ↑ nền tối   ↑ nền sáng
    │ (~20–50)   (~76)       (~190)
    │  ╭──╮    ╭──╮          ╭───╮
    │  │  │    │  │          │   │
    └──────────────────────────────── giá trị pixel
```

Đỉnh mực và đỉnh nền tối **quá gần nhau** → Otsu không tìm được thung lũng rõ → ngưỡng sai.

**Tại sao phép chia giải quyết được:**

Giải thích bằng công thức vật lý. Ảnh gốc và background đều phụ thuộc vào cùng một Light:

```
gray  = R_vật_liệu × Light
bg    = R_giấy     × Light

gray / bg = (R_vật_liệu × Light) / (R_giấy × Light)
          = R_vật_liệu / R_giấy        ← Light bị triệt tiêu hoàn toàn
```

Kết quả chỉ còn tỷ lệ reflectance — cố định theo vật liệu, không phụ thuộc đèn:

```
norm_mực  = R_mực  / R_giấy = 0.25 / 0.95 ≈ 0.26 → × 255 ≈ 67  (mọi vùng)
norm_nền  = R_giấy / R_giấy = 0.95 / 0.95 = 1.00 → × 255 = 255  (mọi vùng)
```

Histogram sau divide chỉ còn 2 đỉnh — nền dồn về ~255, mực dồn về ~67 → Otsu tìm được thung lũng.

**Tại sao phép trừ không giải quyết được (kể cả lấy giá trị tuyệt đối):**

```
gray - bg = (R_vật_liệu × Light) − (R_giấy × Light)
          = (R_vật_liệu − R_giấy) × Light    ← Light vẫn còn

|gray - bg| = |R_vật_liệu − R_giấy| × Light  ← Light vẫn còn
```

Lấy giá trị tuyệt đối không giúp được gì vì Light vẫn nhân vào kết quả:

| Vùng | Light | gray_mực | bg | \|gray−bg\| | gray/bg×255 |
|------|-------|---------|-----|------------|-------------|
| Sáng | 200 | 50 | 190 | **140** | **67** |
| Tối | 80 | 20 | 76 | **56** | **67** |

Sau `|trừ|`: mực ở vùng sáng ra 140, vùng tối ra 56 — chênh 2.5 lần, Otsu vẫn thấy hai cụm mực khác nhau.
Sau `chia`: mực ở cả hai vùng ra 67 — một cụm duy nhất, Otsu phân tách dễ dàng.

> Lưu ý: trong thực tế norm_mực vẫn dao động nhẹ (không hoàn toàn bằng 67 ở mọi vùng) vì bg chỉ là ước tính từ Close, không phải giá trị bg thật lý tưởng. Nhưng dao động đó nhỏ hơn nhiều so với phép trừ — đủ để Otsu hoạt động ổn định.

**Kết quả đạt được:** histogram 2 đỉnh rõ ràng → Otsu chọn đúng ngưỡng → ảnh nhị phân sạch cho bước Connected Components.

**Nếu dùng trừ:** Light vẫn trong kết quả → mực ở vùng sáng và vùng tối ra giá trị khác nhau → Otsu tìm ngưỡng sai → nền tối bị nhận là mực, mực nhạt bị nhận là nền → Connected Components ra hàng trăm blob rác.

---

### 5.4. Tại sao Otsu thay vì ngưỡng cố định?

**Làm gì:** chuyển ảnh xám chuẩn hóa `gray_norm` thành ảnh nhị phân (0 hoặc 255) bằng cách tự động tìm ngưỡng T tối ưu — không cần con người chọn.

**Thuật toán Otsu hoạt động thế nào:**

Otsu quét qua toàn bộ 256 giá trị T từ 0 → 255. Với mỗi T, nó chia tất cả pixel thành 2 nhóm:
- Nhóm tối: pixel < T (mực in, marker)
- Nhóm sáng: pixel ≥ T (nền giấy)

OpenCV tính **phương sai liên nhóm** (inter-class variance) và **tối đa hóa** nó:

```
σ²_between(T) = w₁ × w₂ × (μ₁ − μ₂)²
```

Trong đó `w₁`, `w₂` = tỷ lệ pixel hai nhóm; `μ₁`, `μ₂` = mean hai nhóm. Otsu chọn T* làm **tối đa hóa** σ²_between(T) — tức là T* mà tại đó hai nhóm cách xa nhau nhất (khoảng cách giữa hai mean lớn nhất, có trọng số).

Lý do OpenCV dùng công thức này thay vì công thức gốc Otsu 1979 (tối thiểu hóa intra-class variance): chỉ cần tính mean hai nhóm, không cần tính variance từng nhóm → nhanh hơn đáng kể trên ảnh 12MP.

Hai công thức cho cùng T* vì `σ²_total = σ²_within + σ²_between`, và `σ²_total` là phương sai toàn ảnh — không đổi theo T. Khi `σ²_between` tăng thì `σ²_within` giảm bằng đó → tối đa hóa inter-class ≡ tối thiểu hóa intra-class. Chi tiết xem `math.md`.

**Trực giác:** histogram của `gray_norm` sau bước `divide` thường có 2 đỉnh rõ ràng:

```
Số pixel
    │              ╭──╮
    │             ╭╯  ╰──╮         ← đỉnh nền (240–255)
    │      ╭──╮  ╯        ╰
    │     ╱    ╲╱           ╰──
    │ ──╯                       ╰──
    └───────────────────────────── giá trị (0–255)
          40–80         240–255
       (mực marker)    (nền giấy)
           ↑ T* ≈ 150–170 (thung lũng giữa 2 đỉnh)
```

Otsu tìm "thung lũng" giữa 2 đỉnh này — đây là điểm mà xác suất nhận nhầm (pixel nền bị cắt vào nhóm mực, hoặc pixel mực bị cắt vào nhóm nền) là thấp nhất.

**Vấn đề: T* thay đổi theo từng ảnh**

| Loại ảnh | Nền sau divide | Mực sau divide | T* Otsu chọn |
|----------|----------------|----------------|--------------|
| Giấy trắng + mực đậm | ~252 | ~40 | ~170 |
| Giấy ngà vàng + mực trung bình | ~238 | ~65 | ~155 |
| Mực phai, in cũ | ~248 | ~120 | ~190 |
| Ảnh chụp ánh sáng yếu | ~220 | ~80 | ~145 |

Bốn trường hợp trên → T* dao động từ 145 đến 190. Không có giá trị cố định nào phù hợp tất cả.

**Kết quả đạt được:** ảnh nhị phân sạch — marker đen rõ nét trên nền trắng, không có vùng xám trung gian gây nhập nhằng cho bước Connected Components.

**Nếu không dùng (dùng ngưỡng cố định T = 127):**

Trường hợp "mực phai, in cũ" — T* thực tế cần 190, nhưng ta cắt ở 127:
- Pixel mực có giá trị 120–189 (nhiều pixel mực nhạt) đều < 127? Không — hầu hết 120–189 > 127
- Kết quả: chỉ những pixel mực đậm nhất (< 127) mới được nhận diện
- Marker bị "rỗng ruột" — chỉ còn viền ngoài → Connected Components cho ra blob hình khung, không phải hình vuông đặc → fill_ratio thấp → bị loại ở bước lọc → 0 marker tìm được → warp phải dùng fallback contour (kém chính xác hơn)

---

**Tại sao thêm Gaussian Blur 3×3 trước Otsu?**

**Làm gì:** làm mờ nhẹ ảnh bằng cách trung bình có trọng số 9 pixel lân cận (kernel Gaussian 3×3).

**Tại sao cần:** nén JPEG tạo ra "noise muối tiêu" — các pixel trắng lẻ nằm trong vùng mực đen (và ngược lại). Những pixel lẻ này không phải nền thật, nhưng chúng tạo thêm đỉnh nhỏ trong histogram:

```
Histogram có noise:            Histogram sau Gaussian 3×3:
    │              ╭──╮            │              ╭──╮
    │        ↑    ╭╯  ╰──╮         │             ╭╯  ╰──╮
    │     spike  ╯        ╰         │      ╭──╮  ╯        ╰
    │      ╭──╮╱           ╰       │     ╱    ╲╱           ╰
    │     ╱    ╲                   │ ──╯
    └───────────────────────        └────────────────────────
             ↑ Otsu có thể chọn nhầm         ↑ Otsu chọn đúng
               spike làm T*                  thung lũng thật
```

Gaussian 3×3 kéo giá trị mỗi pixel về trung bình lân cận → pixel lẻ bị hòa tan vào màu xung quanh → histogram mượt → Otsu tìm đúng thung lũng thật.

**Tại sao 3×3, không phải 5×5?**

Kernel 5×5 bắt đầu làm mờ cạnh marker — góc marker 95px bị "bóng mờ" rộng 2px → fill_ratio tính ở bước sau hơi thấp hơn thực tế. Kernel 3×3 chỉ dịch mỗi pixel ~0.5 gray level ở cạnh rõ → không ảnh hưởng đến hình dạng marker đủ để bước lọc nhận sai.

Đến đây, `_extract_black_square_markers_from_gray()` đã có `bin_inv` (ảnh nhị phân sạch). Việc còn lại của hàm này chỉ là 1 dòng: gọi hàm lọc theo 8 tiêu chí — chuyển sang mục 6.

---

## 6. Hàm lọc theo 8 tiêu chí — `_extract_black_square_markers()`

**Mục đích:** nhận ảnh nhị phân (`bin_inv` từ mục 5), gom các pixel đen liền nhau thành từng "blob", rồi lọc qua 8 tiêu chí hình dạng để chỉ giữ lại những blob trông giống marker vuông — loại bỏ bong bóng MCQ, chữ số, đường kẻ.

**Đầu vào:** `binary_img` (chính là `bin_inv` mục 5 truyền sang), cùng `min_area_ratio`, `max_area_ratio`, `min_fill_ratio=0.40`, `max_markers`.
**Đầu ra:** danh sách marker đã lọc — đây là giá trị mà `_extract_black_square_markers_from_gray()` (mục 5) trả lại cho `_detect_page_corners_from_black_square_markers()` (mục 4), rồi được dùng tiếp ở mục 7.

### Bối cảnh — ảnh nhị phân chưa "biết" marker ở đâu

Kết quả của mục 5 là một ảnh nhị phân: mỗi pixel chỉ là 0 (đen) hoặc 255 (trắng). Toàn bộ những thứ đậm màu trên tờ phiếu đều thành pixel đen — hệ thống chưa biết cái nào là marker, cái nào là bong bóng, cái nào là chữ số:

```
Ảnh binary sau Otsu (0 = đen, _ = trắng):

  ████             ████        ← cả hai đều là pixel đen
                               ← marker góc TL và TR trông giống nhau
  ○  ○  ○  ○  ○  ○  ○  ○      ← bong bóng MCQ (cũng là pixel đen)
  ○  ○  ○  ○  ○  ○  ○  ○
  1  2  3  4  5  6             ← chữ số MSSV (cũng là pixel đen)
  ══════════════════════        ← đường kẻ (cũng là pixel đen)
  ○  ○  ○  ○  ○  ○  ○  ○

  ████             ████
```

Hàm này làm hai việc:
1. **Connected Components** — gom các pixel đen liền nhau thành "blob" có địa chỉ, diện tích, tọa độ
2. **Lọc 8 tiêu chí** — trong hàng nghìn blob đó, giữ lại chỉ những blob trông giống marker

```python
# omr_marker_utils.py:21-24 — đầu hàm _extract_black_square_markers()
prep = cv2.morphologyEx(binary_img, cv2.MORPH_OPEN, np.ones((2,2)), iterations=1)
prep = cv2.morphologyEx(prep, cv2.MORPH_CLOSE, np.ones((3,3)), iterations=1)
num, labels, stats, centroids = cv2.connectedComponentsWithStats(prep, connectivity=8)
```

### 6.1. Connected Components tính toán thế nào — và output là gì?

**CC không nhận diện loại blob.** Nó chỉ làm một việc: gán số thứ tự cho từng vùng pixel đen liền nhau, rồi đo kích thước của từng vùng đó. Kết quả là một danh sách "vùng số 1 ở đâu, to bao nhiêu; vùng số 2 ở đâu, to bao nhiêu…" — không hề biết vùng nào là marker hay bong bóng. Việc nhận diện xảy ra ở **mục 6.3**, không phải ở đây.

**Cách thuật toán lan nhãn — Flood Fill:**

Hình dung đổ sơn lên ảnh:
1. Quét từ góc trên-trái, tìm pixel đen đầu tiên chưa có màu → đổ màu số 1 lên đó
2. Màu "chảy" lan sang tất cả pixel đen liền kề (kể cả chéo góc) — như nước ngấm vào vải
3. Khi màu không chảy được nữa (xung quanh toàn trắng hoặc đã có màu) → vùng số 1 kết thúc
4. Tiếp tục quét, tìm pixel đen chưa có màu → đổ màu số 2, lan tiếp
5. Lặp lại cho đến khi không còn pixel đen nào chưa có màu

```
Ảnh gốc (■=đen, □=trắng):    Sau CC (mỗi chữ số = một vùng):

□ □ □ □ □ □ □                 0 0 0 0 0 0 0
□ ■ ■ □ □ □ □                 0 1 1 0 0 0 0   ← vùng 1: màu lan từ (1,1) sang (1,2)
□ ■ ■ □ □ □ □                 0 1 1 0 0 0 0   ← rồi xuống (2,1) và (2,2)
□ □ □ □ ■ ■ □                 0 0 0 0 2 2 0   ← vùng 2: tách biệt, không kề vùng 1
□ □ □ □ ■ ■ □                 0 0 0 0 2 2 0
□ □ ■ □ □ □ □                 0 0 3 0 0 0 0   ← vùng 3: pixel đơn lẻ
□ □ □ □ □ □ □                 0 0 0 0 0 0 0
```

> **Lưu ý:** "đổ sơn/flood fill" ở trên là **ẩn dụ trực giác** để hình dung kết quả (pixel liền nhau → cùng 1 số), không phải cách OpenCV cài đặt thật bên trong. Thuật toán thật là **quét 2 lượt (two-pass) + Union-Find**: lượt 1 quét ảnh theo thứ tự tuyến tính (trái→phải, trên→dưới), gán nhãn tạm dựa trên lân cận đã quét qua, đồng thời ghi lại các nhãn "tương đương" khi 2 nhánh riêng biệt hóa ra nối với nhau (ví dụ hình chữ U); lượt 2 hợp nhất các nhãn tương đương thành nhãn cuối cùng. Cách này quét ảnh theo thứ tự bộ nhớ tuyến tính (cache-friendly), nhanh hơn nhiều so với flood-fill đệ quy thật sự (vốn cần ngăn xếp/hàng đợi phình to với blob lớn). Kết quả cuối giống hệt nhau — chỉ khác cách tính ra.

**Output:** `cv2.connectedComponentsWithStats` trả về đồng thời 4 giá trị:

```python
num, labels, stats, centroids = cv2.connectedComponentsWithStats(prep, connectivity=8)
#    ^^^^^^  ^^^^^  ^^^^^^^^
#    |       |      |
#    |       |      stats[i] = [x, y, w, h, area] của vùng i
#    |       labels = ảnh cùng kích thước, mỗi pixel ghi số vùng của nó
#    num = tổng số vùng tìm được (kể cả vùng 0 = nền trắng)
```

**`labels` — kiểu dữ liệu và cấu trúc cụ thể:** là một `numpy.ndarray` 2 chiều, **cùng kích thước H×W với ảnh đầu vào** — nhưng khác `dtype`: ảnh đầu vào (`prep`) là `uint8` (0–255), còn `labels` là `int32` (mặc định `ltype=cv2.CV_32S`). Lý do không dùng lại `uint8`: một tờ phiếu có hàng trăm bong bóng/marker/chữ số có thể sinh ra **hơn 255 vùng liên thông** — nếu dùng `uint8` (tối đa 255), nhãn vùng thứ 256 sẽ tràn số về 0, trùng với nhãn nền → sai hoàn toàn. `int32` chứa được tới ~2 tỷ, dư sức cho bất kỳ số blob nào. (`stats` cũng là `int32`; `centroids` là `float64` vì tọa độ tâm là trung bình cộng nhiều pixel, thường lẻ.)

Có thể hiểu `labels` là một "bản đồ chỉ mục" — cùng hình dạng với ảnh gốc, nhưng thay vì lưu độ sáng, mỗi ô lưu **số hiệu vùng** mà pixel đó thuộc về. Tra `labels[y, x]` biết ngay pixel đó thuộc blob nào. Đây chính là cơ chế Tiêu chí 5 (mục 6.3) dùng để cắt riêng 1 blob: `comp_mask = (labels[y:y+bh, x:x+bw] == int(idx))` — so sánh từng ô trong bounding box với đúng số hiệu `idx`.

| Thông số | Ý nghĩa | Dùng ở tiêu chí |
|----------|---------|-----------------|
| `x, y, w, h` | Bounding box nhỏ nhất bao quanh vùng | aspect ratio (tiêu chí 3) |
| `area` | Số pixel đen trong vùng (đếm từ `labels`) | area_ratio (tiêu chí 1), fill_ratio (tiêu chí 4) |
| `centroid` | Tọa độ tâm: trung bình `x` và `y` của tất cả pixel trong vùng | scoring ở mục 7 |

Một tờ phiếu điển hình cho ra **200–2000 vùng** — tất cả đều chỉ là "vùng số N, bounding box là (x,y,w,h), diện tích bao nhiêu". CC không biết vùng nào là marker. Mục 6.3 mới đọc các số đó và lọc.

**Lưu ý thứ tự thực thi:** trong code, làm sạch ảnh (mục 6.2) chạy trước, rồi CC mới chạy trên ảnh đã làm sạch. Tài liệu đặt CC (khái niệm trung tâm) trước, sau đó mới giải thích bước chuẩn bị — thứ tự giải thích khác thứ tự code chạy, chỉ ở đây thôi.

**Vì sao không dùng các kỹ thuật phát hiện hình dạng khác?**

- **Hough Circle Transform** (tìm hình tròn): Marker là hình **vuông**, không tròn → không phù hợp ngay từ đầu.
- **Hough Line Transform** (tìm đường thẳng): Chỉ tìm được cạnh marker, không trả về blob vùng kín → cần bước xử lý thêm.
- **Template matching**: Cần biết trước kích thước marker (pixel) → phụ thuộc độ phân giải ảnh chụp (rất khác nhau); chậm với nhiều scale khác nhau.
- **SIFT/ORB feature matching**: Marker là hình vuông đơn giản → không đủ distinctive feature cho SIFT; OpenCV free tier không có SIFT (patent), phải dùng ORB → kém chính xác hơn.
- **Connected Components** (được chọn): Hoàn toàn không phụ thuộc kích thước tuyệt đối → scale-invariant tự nhiên (dùng tỷ lệ diện tích); nhanh O(n); trả về đầy đủ bounding box/diện tích/centroid để tính các tiêu chí lọc.

**`connectivity=8`** (8-connected) thay vì 4-connected: marker chụp điện thoại có góc đôi khi bị "đứt" 1 pixel (JPEG artifact). 8-connected: pixel chéo cũng được coi là liền nhau → marker không bị tách thành nhiều mảnh.

**"8-connected" nghĩa là gì, cụ thể?** Đây là định nghĩa: *2 pixel đen được coi là "liền nhau" khi nào?* Mỗi pixel có 8 lân cận trong khối 3×3 quanh nó:

```
TL  T  TR
 L  ■   R
BL  B  BR
```

- **4-connected**: chỉ 4 lân cận cạnh (T, B, L, R) tính là liền kề — 4 lân cận chéo (TL, TR, BL, BR) KHÔNG tính.
- **8-connected**: tính cả 8 (4 cạnh + 4 chéo).

Ví dụ 2 pixel chỉ chạm nhau ở góc:
```
■ □
□ ■
```
4-connected → 2 vùng riêng biệt (không có cạnh chung). 8-connected → 1 vùng duy nhất (chạm góc chéo vẫn tính liền kề). Đây chính là lý do 8-connected "cứu" được marker bị JPEG làm đứt 1 pixel ở góc.

**Vì sao "8" mà lúc quét (mục trên) chỉ nhìn 4 lân cận đã quét qua (trái, trên, trên-trái, trên-phải), không nhìn đủ 8 hướng?** Không mâu thuẫn — "8-connected" là *định nghĩa quan hệ liền kề* trên toàn ảnh, còn "chỉ nhìn 4 lân cận đã quét" là *mẹo cài đặt* để hiện thực đúng định nghĩa đó. Khi quét trái→phải, trên→dưới, 4 lân cận **phải, dưới, dưới-trái, dưới-phải** chưa được quét tới (chưa có nhãn). Quan hệ liền kề là hai chiều (A kề B thì B cũng kề A) nên mỗi cặp liền kề chỉ cần phát hiện **một lần**, từ phía pixel quét tới sau — khi thuật toán quét tới pixel dưới-phải của một cặp, nó tự nhìn ngược lên-trái và bắt được quan hệ đó, không bỏ sót. Với 4-connected, chỉ cần nhìn lại 2 lân cận đã quét (trái, trên) là đủ.

**Nếu dùng connectivity=2 hoặc 16 thì sao?** `cv2.connectedComponentsWithStats()` **chỉ chấp nhận 4 hoặc 8** — không có giá trị nào khác hợp lệ (truyền số khác sẽ báo lỗi tham số). Lý do nằm ở hình học lưới pixel vuông: mỗi pixel chỉ có đúng 8 lân cận trực tiếp (khối 3×3 quanh nó, trừ tâm) — không thể có "16 lân cận trực tiếp" trừ khi mở rộng bán kính ra xa hơn 1 pixel, nhưng đó là khái niệm "connectivity mở rộng" hoàn toàn khác, không phải thứ thuật toán connected-component tiêu chuẩn xử lý. "4" là tập con của "8" (bỏ 4 góc chéo) — không có khái niệm "2-connected" cho ảnh 2D (2 lân cận chỉ có ý nghĩa trong tín hiệu 1 chiều, không áp dụng cho lưới pixel 2 chiều).

---

### 6.2. Morphological Open và Close trước khi phân tích

```python
prep = cv2.morphologyEx(binary_img, cv2.MORPH_OPEN, np.ones((2,2)))   # xóa noise nhỏ
prep = cv2.morphologyEx(prep,       cv2.MORPH_CLOSE, np.ones((3,3)))  # lấp lỗ nhỏ
```

**Tại sao cần làm sạch trước khi chạy CC?**

Ảnh binary sau Otsu có hai loại nhiễu nhỏ làm CC cho kết quả sai:

```
Loại 1 — Noise dot:           Loại 2 — Lỗ trong marker:
□ □ □ □ □ □ □                 ■ ■ ■ ■ ■ ■
□ ■ □ □ □ □ □   ← 1px lẻ     ■ □ □ □ □ ■   ← pixel nền lọt vào trong
□ □ □ □ □ □ □                 ■ □ □ □ □ ■      do mực in không đều
□ □ □ ■ □ □ □   ← 1px lẻ     ■ ■ ■ ■ ■ ■
```

- **Noise dot** → CC tạo thêm hàng trăm blob kích thước 1–2 px. Dù tiêu chí lọc loại được chúng, chúng làm tốn bộ nhớ và thời gian xử lý
- **Lỗ trong marker** → làm `fill_ratio` tính ở tiêu chí 4 thấp hơn thực tế → marker có thể bị loại nhầm

---

**Erode — thao tác cơ bản thu nhỏ vùng mực:**

Trượt một kernel (cửa sổ nhỏ) qua từng pixel. Tại mỗi vị trí, nếu **tất cả** pixel dưới kernel đều là mực → pixel trung tâm giữ nguyên là mực. Nếu **bất kỳ** pixel nào là nền → trung tâm thành nền.

Kết quả: vùng mực bị "bào" mỏng từ mép vào trong. Vùng nhỏ hơn kernel biến mất hoàn toàn vì không có vị trí nào kernel khớp hoàn toàn trên vùng đó.

```
Erode với kernel 2×2:

Noise dot 1×1:      Sau Erode:        Marker 4×4:         Sau Erode:
□ □ □ □             □ □ □ □           ■ ■ ■ ■             ■ ■ ■ □
□ ■ □ □     →       □ □ □ □           ■ ■ ■ ■      →      ■ ■ ■ □
□ □ □ □             □ □ □ □           ■ ■ ■ ■             ■ ■ ■ □
                                      ■ ■ ■ ■             □ □ □ □
(biến mất vì không                    (mép bị cắt 1px,
có vị trí nào 2×2                     phần lõi còn nguyên)
toàn là mực)
```

---

**Dilate — thao tác cơ bản mở rộng vùng mực:**

Ngược lại với Erode: nếu **bất kỳ** pixel nào dưới kernel là mực → trung tâm thành mực.

Kết quả: vùng mực "phình" ra xung quanh. Lỗ nhỏ bên trong vùng mực bị lấp kín vì chúng có pixel mực lân cận.

```
Dilate với kernel 3×3:

Marker có lỗ:       Sau Dilate:
■ ■ ■ ■ ■           ■ ■ ■ ■ ■
■ □ □ □ ■    →      ■ ■ ■ ■ ■     ← lỗ 3×3 bên trong bị lấp
■ □ □ □ ■           ■ ■ ■ ■ ■        vì các pixel mực lân cận lan vào
■ □ □ □ ■           ■ ■ ■ ■ ■
■ ■ ■ ■ ■           ■ ■ ■ ■ ■
```

---

**MORPH_OPEN (2×2) = Erode trước → Dilate sau:**

- **Erode**: noise dot 1px biến mất; marker 30×30 shrink thành ~28×28
- **Dilate**: marker ~28×28 phình lại ~30×30; noise dot đã biến mất rồi, không có gì để phình lại

```
Noise dot:    Sau Erode:    Sau Dilate (= kết quả Open):
□ □ □ □       □ □ □ □       □ □ □ □
□ ■ □ □  →   □ □ □ □  →   □ □ □ □    ← biến mất hoàn toàn
□ □ □ □       □ □ □ □       □ □ □ □

Marker 4×4:   Sau Erode:    Sau Dilate (= kết quả Open):
■ ■ ■ ■       ■ ■ ■ □       ■ ■ ■ ■
■ ■ ■ ■  →   ■ ■ ■ □  →   ■ ■ ■ ■    ← trở lại gần như ban đầu
■ ■ ■ ■       ■ ■ ■ □       ■ ■ ■ ■
■ ■ ■ ■       □ □ □ □       ■ ■ ■ □
```

---

**MORPH_CLOSE (3×3) = Dilate trước → Erode sau:**

- **Dilate**: lỗ trắng nhỏ trong marker bị lấp; marker phình to thêm ~1px mọi phía
- **Erode**: marker co lại kích thước ban đầu; lỗ đã bị lấp rồi, pixel xung quanh nó toàn là mực nên Erode không mở lại

```
Marker có lỗ 2×2:   Sau Dilate:         Sau Erode (= kết quả Close):
■ ■ ■ ■ ■           ■ ■ ■ ■ ■           ■ ■ ■ ■ ■
■ □ □ ■ ■           ■ ■ ■ ■ ■           ■ ■ ■ ■ ■
■ □ □ ■ ■    →      ■ ■ ■ ■ ■    →      ■ ■ ■ ■ ■    ← lỗ đã bị lấp
■ ■ ■ ■ ■           ■ ■ ■ ■ ■           ■ ■ ■ ■ ■
■ ■ ■ ■ ■           ■ ■ ■ ■ ■           ■ ■ ■ ■ ■
```

---

**Tại sao Open trước, Close sau?**

Nếu đảo thứ tự (Close trước, Open sau):
- Close làm noise dot 1px phình thành vùng 3×3
- Open 2×2 cần tối thiểu 2×2 để giữ lại → vùng 3×3 sống sót → noise không bị xóa

Open trước loại sạch noise, sau đó Close chỉ còn làm việc với các blob hợp lệ.

**Tại sao kernel 2×2 cho Open, 3×3 cho Close?**

- Open 2×2: nhỏ vừa đủ chỉ xóa noise 1–2 px, marker 30×30 gần như không bị ảnh hưởng (shrink 1px rồi dilate lại)
- Close 3×3: đủ lớn để lấp lỗ JPEG 2–3 px bên trong marker, nhưng không đủ lớn để hợp nhất hai marker riêng biệt cạnh nhau

---

### 6.3. Các tiêu chí lọc và lý do chọn ngưỡng

Sau CC, tờ phiếu điển hình có 200–2000 blob. 8 tiêu chí dưới đây áp dụng lần lượt — blob nào không qua được tiêu chí nào thì bị loại ngay, không kiểm tra tiếp:

| # | Tiêu chí | Loại bỏ |
|---|----------|---------|
| 1 | Diện tích nằm trong khoảng [min, max] | Noise nhỏ; vùng nền cực lớn |
| 2 | Chiều rộng và cao ≥ 4 px | Blob quá nhỏ |
| 3 | Tỷ lệ chiều rộng/cao trong [0.62, 1.55] | Đường kẻ ngang/dọc |
| 4 | Mật độ pixel mực / bounding box ≥ 0.40 | Chữ in thưa, noise |
| 5 | Diện tích đường viền ≥ 55% diện tích CC | Blob có đường viền quá gãy khúc |
| 6 | Số đỉnh sau giản lược trong [4, 8] | Hình tròn (bong bóng); đường thẳng |
| 7 | Độ lồi (solidity) ≥ 0.82 | Hình lõm: chữ L, sao, zigzag |
| 8 | Không vừa tròn vừa thưa vừa nhiều cạnh | Bong bóng tô mực |

---

#### Tiêu chí 1 — Diện tích (`area`)

```python
min_area = max(12, int(total_area * 0.00002))
max_area = max(min_area + 1, int(total_area * 0.02))
```

`area` = số pixel mực trong blob (lấy từ CC stats). `total_area = h × w` — diện tích tính bằng pixel của **chính ảnh đang xử lý ngay lúc đó** (`gray_img`, ở độ phân giải chụp/scan gốc, chưa warp/resize — xem mục 1). Đây **không phải** một con số cố định, càng không phải "kích thước A4" — A4 chỉ là 1 khổ giấy trong nhiều khổ có thể dùng, và dù dùng A4 thì số pixel thật sự vẫn phụ thuộc hoàn toàn vào DPI lúc scan hoặc độ phân giải camera lúc chụp (xem bảng ở mục 5.2: cùng là "ảnh" nhưng điện thoại 12MP ra 4032×3024, scan 300 DPI ra 2480×3508 — hai con số khác hẳn nhau).

Nói bằng lời: thay vì đặt sẵn "marker phải rộng từ X đến Y pixel" (chỉ đúng với đúng 1 độ phân giải), hệ thống đặt "marker phải chiếm từ 0.002% đến 2% tổng diện tích ảnh" — một **tỷ lệ**, đúng với mọi độ phân giải. Mỗi lần xử lý ảnh mới, `total_area` được tính lại từ chính ảnh đó, rồi `min_area`/`max_area` tự động co giãn theo pixel thật.

Áp dụng đúng 2 ảnh ví dụ đã dùng ở mục 5.2:

| Nguồn ảnh | Kích thước | `total_area` | `min_area` (sàn 0.002%) | `max_area` (trần 2%) | Marker góc thực tế |
|---|---|---|---|---|---|
| Điện thoại 12MP | 4032×3024 | ~12.2M px | ~243 px (~16×16) | ~243,855 px (~494×494) | ~95×95 ≈ 9,000 px |
| Scan A4 @ 300 DPI | 2480×3508 | ~8.7M px | ~173 px (~13×13) | ~174,000 px (~417×417) | ~59×59 ≈ 3,500 px |

Cùng một công thức, áp cho 2 ảnh chênh nhau ~40% số pixel, marker thật vẫn nằm thoải mái giữa `min_area` và `max_area` ở **cả hai** trường hợp. Đây chính là điều một ngưỡng pixel cố định không bao giờ làm được — đặt cứng "area phải từ 500 đến 50000" sẽ đúng với ảnh này nhưng sai với ảnh kia.

`max(12, ...)` là sàn cứng bổ sung: nếu ảnh quá nhỏ (thumbnail vài trăm pixel) khiến 0.002% tính ra chưa tới 1 px, marker vẫn phải có ít nhất 12 px mới được xét tiếp.

**Hàm `_detect_page_corners_from_black_square_markers` dùng `min_area_ratio=0.00008`** (4× lớn hơn):
Marker góc của tờ phiếu to hơn đáng kể so với các ô MCQ. Nâng sàn 4× để loại các marker nhỏ không phải corner.

---

#### Tiêu chí 2 — Kích thước tối thiểu tuyệt đối

```python
if bw < 4 or bh < 4:
    continue
```

`bw`, `bh` = chiều rộng, chiều cao của bounding box (hình chữ nhật nhỏ nhất bao quanh blob).

Kiểm tra này là sàn cứng bổ sung cho tiêu chí 1: một blob 3×1 px có area = 3 px sẽ qua tiêu chí 1 (nếu ảnh đủ nhỏ), nhưng không thể là marker hữu dụng — tọa độ corner chỉ chính xác đến ±2 px nếu blob quá nhỏ.

---

#### Tiêu chí 3 — Tỷ lệ chiều (aspect ratio = chiều rộng / chiều cao)

```python
aspect = float(bw) / max(1.0, float(bh))
if not (0.62 <= aspect <= 1.55):
    continue
```

`aspect` = bw / bh. Hình vuông hoàn hảo → aspect = 1.0.

```
aspect = 0.3      aspect = 0.7      aspect = 1.0      aspect = 1.8      aspect = 4.0
┌─┐               ┌──┐              ┌───┐             ┌──────┐          ┌────────────┐
│ │               │  │              │   │             │      │          │            │
│ │               │  │              │   │             └──────┘          └────────────┘
│ │               └──┘              └───┘
└─┘
 LOẠI             GIỮ              GIỮ               GIỮ                 LOẠI
(đường kẻ dọc)  (marker nghiêng)  (marker thẳng)   (marker nghiêng)   (đường kẻ ngang)
```

Ngưỡng [0.62, 1.55] cho phép marker bị bóp méo do góc chụp nghiêng ~30° (tạo aspect tới 1.4–1.5), nhưng loại được đường kẻ ngang/dọc và chữ có tỷ lệ cực đoan.

**`max(1.0, float(bh))` ở mẫu số để làm gì?** Đây **không phải** một phần của việc lọc hình dạng — thuần túy là **guard chống chia cho 0**. `bh` (chiều cao bounding box) trên thực tế luôn ≥ 1 (không blob nào cao 0 pixel), nên guard này gần như không bao giờ thực sự kích hoạt — nó chỉ phòng hờ trường hợp `bh = 0` không làm chương trình sập.

Nếu bỏ guard này (`aspect = float(bw) / float(bh)`), khi `bh = 0` Python sẽ ném `ZeroDivisionError` → **toàn bộ pipeline dừng đột ngột** cho ảnh đó, không "loại" được blob này một cách có kiểm soát. Có guard: mẫu số bị ép về tối thiểu `1.0`, `aspect` tính ra một số hữu hạn (thường rất lớn nếu `bh` thực sự nhỏ bất thường) → vượt xa `[0.62, 1.55]` → vẫn bị `continue` loại **bình thường**, pipeline không bị gián đoạn. Tức là guard này đổi kết cục từ "crash" sang "loại có kiểm soát" — không phải để "dễ loại hơn" hay "khó loại hơn". Mẫu code này giống hệt `max(1.0, hull_area)` ở Tiêu chí 7 bên dưới — cùng mục đích đảm bảo phép chia luôn có mẫu số hợp lệ.

---

#### Tiêu chí 4 — Mật độ pixel mực (fill ratio)

```python
fill_ratio = float(area) / float(bw * bh)
if fill_ratio < 0.40:
    continue
```

`fill_ratio` = số pixel mực / diện tích bounding box. Đo xem bên trong bounding box có "đặc" không.

```
Marker vuông:        Chữ "H":             Bong bóng tô:        Noise thưa:
■ ■ ■ ■ ■           ■ □ □ □ ■            □ ■ ■ ■ □            □ ■ □ □ □
■ ■ ■ ■ ■           ■ ■ ■ ■ ■            ■ ■ ■ ■ ■            □ □ □ ■ □
■ ■ ■ ■ ■           ■ □ □ □ ■            ■ ■ ■ ■ ■            □ □ □ □ □
■ ■ ■ ■ ■           ■ □ □ □ ■            ■ ■ ■ ■ ■            ■ □ □ □ □
■ ■ ■ ■ ■           ■ □ □ □ ■            □ ■ ■ ■ □            □ □ □ □ ■
fill ≈ 0.90         fill ≈ 0.40          fill ≈ 0.72           fill ≈ 0.16
GIỮ                 Ranh giới            GIỮ (bị loại ở tiêu chí 8)   LOẠI
```

Ngưỡng 0.40 thay vì 0.60: marker in bằng máy thông thường khi gần hết mực có fill giảm xuống ~0.45, nâng ngưỡng lên 0.60 sẽ bỏ sót chúng.

---

#### Tiêu chí 5 — Diện tích đường viền (contour area check)

```python
comp_mask = (labels[y:y + bh, x:x + bw] == int(idx)).astype(np.uint8) * 255
cnts, _ = cv2.findContours(comp_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnt = max(cnts, key=cv2.contourArea)
contour_area = float(cv2.contourArea(cnt))
if contour_area < float(area) * 0.55:
    continue
```

**Nguồn gốc các biến:**
- `labels` — ảnh nhãn từ CC (mục 6.1): mỗi pixel ghi số thứ tự blob nó thuộc về
- `y, x, bh, bw` — tọa độ góc trên-trái và kích thước bounding box (lấy từ `stats` của CC)
- `area` — số pixel mực của blob này, cũng từ `stats` của CC
- `comp_mask` — ảnh mới tạo tại chỗ: cắt vùng bounding box, chỉ blob `idx` = trắng (255), còn lại = đen (0)

---

**Tính toán từng bước với blob 5×5:**

> Lưu ý trước khi vào ví dụ: blob 5×5 chọn ở đây là một **hình vuông đặc hoàn toàn** (mọi ô trong bounding box đều là pixel mực) — nên `area` (đếm pixel, = 25) tình cờ trùng với `5×5` (nhân chiều dài×chiều rộng). Đây **không phải công thức chung** — `area` luôn là số pixel mực **đếm thật** từ CC stats, chỉ bằng `bw×bh` khi blob đặc hoàn toàn không lỗ không khuyết. Với blob có lỗ hoặc khuyết (như chữ "H" ở Tiêu chí 4: fill ≈ 0.40, tức `area` chỉ bằng ~40% `bw×bh`), hai con số này khác nhau rõ rệt. Ví dụ đơn giản (hình vuông đặc) được chọn ở đây chỉ để tính toán Shoelace cho dễ theo dõi.

**Tạo mask riêng cho blob:**

```
labels (ảnh gốc):      comp_mask (cắt theo bbox):
0 0 0 0 0 0 0          255 255 255 255 255
0 2 2 2 2 2 0   →      255 255 255 255 255    (chỉ giữ pixel thuộc blob #2)
0 2 2 2 2 2 0          255 255 255 255 255
0 2 2 2 2 2 0          255 255 255 255 255
0 2 2 2 2 2 0          255 255 255 255 255
0 0 0 0 0 0 0
```

**`findContours` dò theo biên ngoài (`RETR_EXTERNAL` = chỉ lấy biên ngoài cùng):**

Trả về list tọa độ (x, y) của các điểm nằm trên biên. `CHAIN_APPROX_SIMPLE` (giản lược đơn giản) chỉ giữ lại các điểm góc, bỏ các điểm thẳng hàng ở giữa:

```
Biên đầy đủ (25 điểm):          Sau CHAIN_APPROX_SIMPLE (4 điểm):
· · · · ·                        A · · · B
·       ·         →              ·       ·
·       ·                        ·       ·
·       ·                        ·       ·
· · · · ·                        D · · · C

                    A=(0,0)  B=(4,0)  C=(4,4)  D=(0,4)
```

`cnt` — biến lưu danh sách 4 điểm góc đó — được chọn từ tất cả contour tìm được bằng cách lấy cái có diện tích lớn nhất (`max(..., key=cv2.contourArea)`), phòng trường hợp blob có nhiều mảnh vỡ nhỏ kèm theo.

**`cnts` (số nhiều) và `cnt` (số ít) khác nhau thế nào — vì sao có thể có "nhiều mảnh vỡ" trong khi `comp_mask` chỉ chứa đúng 1 blob (`idx`)?**

`cnts` là **danh sách các hình dạng đường viền khép kín khác nhau** tìm được trong `comp_mask` — không phải nhiều cách mô tả của cùng 1 hình. Trường hợp phổ biến nhất: `cnts` chỉ có đúng 1 phần tử (đúng 1 blob → đúng 1 đường viền), nên `max(cnts, ...)` chỉ đơn giản lấy ra phần tử duy nhất đó.

Nhưng đôi khi `cnts` có **nhiều hơn 1 phần tử**, dù `comp_mask` chỉ chứa 1 label `idx` duy nhất — vì `cv2.connectedComponentsWithStats(..., connectivity=8)` và `cv2.findContours()` xét "liền nhau" theo 2 cách khác nhau:

```
■ ■ □ □
■ ■ □ □
□ □ ■ ■
□ □ ■ ■
```

- CC (8-connectivity): pixel ở góc dưới-phải hình vuông trên-trái và pixel ở góc trên-trái hình vuông dưới-phải **chạm chéo nhau** → tính là **liền kề** → cả 4 ô vuông nhỏ này được gán chung **1 nhãn `idx`** duy nhất, `comp_mask` chứa toàn bộ 2 hình vuông.
- `findContours` dò biên theo **cạnh pixel** (không phải góc chéo). Tại điểm chỉ chạm góc đó, không có cạnh pixel chung để đi vòng liền mạch qua cả 2 hình vuông như 1 đường viền duy nhất → thuật toán trả về **2 đường viền khép kín tách biệt**, mỗi hình vuông 1 đường viền riêng.

Kết quả: `cnts = [contour_hình_vuông_1, contour_hình_vuông_2]` — 2 phần tử, 2 hình dạng **thật sự khác nhau**, dù CC coi chúng là "1 blob". `max(cnts, key=cv2.contourArea)` chọn ra đúng 1 hình to nhất (giả định là hình dạng chính của marker), **loại bỏ hẳn** hình còn lại (coi như mảnh vỡ/nhiễu) — không phải "lấy đại diện cho cả danh sách".

**`contourArea` tính diện tích đa giác bằng công thức Shoelace (diện tích hình đa giác từ tọa độ đỉnh):**

```
Area = ½ × |Σ (xᵢ × yᵢ₊₁ − xᵢ₊₁ × yᵢ)|

Với 4 đỉnh A(0,0) → B(4,0) → C(4,4) → D(0,4):

= ½ × |(0×0 − 4×0) + (4×4 − 4×0) + (4×4 − 0×4) + (0×0 − 0×4)|
= ½ × |  0         +   16         +   16         +   0        |
= ½ × 32 = 16
```

`contour_area` = **16**, trong khi `area` (đếm pixel) = **25**.

---

**Tại sao `contour_area` (16) nhỏ hơn `area` (25)?**

Mỗi pixel là một ô vuông 1×1. Pixel `(col, row)` chiếm diện tích từ `(col, row)` đến `(col+1, row+1)`:

```
    0    1    2    3    4    5   ← tọa độ liên tục (cạnh ô pixel)
  0 ┌────┬────┬────┬────┬────┐
    │(0,0│(1,0│(2,0│(3,0│(4,0│
  1 ├────┼────┼────┼────┼────┤
    │    │    │    │    │    │
  2 ├────┼────┼────┼────┼────┤
    │    │    │    │    │    │
  3 ├────┼────┼────┼────┼────┤
    │    │    │    │    │    │
  4 ├────┼────┼────┼────┼────┤
    │(0,4│(1,4│(2,4│(3,4│(4,4│
  5 └────┴────┴────┴────┴────┘
```

**Diện tích thật** của 5×5 pixel = từ góc `(0,0)` đến góc `(5,5)` = 5×5 = **25**.

**`findContours`** trả về **tọa độ nguyên của pixel** (góc trên-trái của từng ô), không phải cạnh ngoài cùng. Pixel góc dưới-phải là pixel `(4,4)` — tọa độ `(4,4)`, không phải `(5,5)`. Nên contour dừng ở `(4,4)`:

```
Contour: (0,0) → (4,0) → (4,4) → (0,4)   ← tọa độ pixel, không phải cạnh ngoài
         ↑                  ↑
     góc trên-trái      góc trên-trái
     của pixel (0,0)    của pixel (4,4)
     = đúng             = lệch 1 đơn vị so với cạnh ngoài (5,5)
```

Shoelace trên 4 điểm đó đo hình vuông từ `(0,0)` đến `(4,4)` = 4×4 = **16**.

Phần bị bỏ sót là dải 1 pixel dọc theo mép phải và mép dưới — vì contour không đến `(5,5)`:

```
■ ■ ■ ■ ■│
■ ■ ■ ■ ■│← cột này (x=4→5) không được Shoelace đếm
■ ■ ■ ■ ■│
■ ■ ■ ■ ■│
■ ■ ■ ■ ■│
─ ─ ─ ─ ─┘
↑ hàng này (y=4→5) cũng không được đếm

Shoelace: 4×4 = 16    Bị bỏ sót: 5+5−1 = 9    Tổng: 16+9 = 25 ✓
```

Không có gì bí ẩn — đây là lệch 1 đơn vị ở mép vì `findContours` dùng tọa độ pixel thay vì tọa độ cạnh ngoài của pixel đó.

Mối quan hệ tổng quát cho blob vuông n×n đặc:

```
contour_area / area = (n−1)² / n²

n = 5  →  16/25  = 0.64   (blob nhỏ: chênh lệch lớn hơn)
n = 10 →  81/100 = 0.81
n = 30 →  841/900 = 0.93  (blob lớn: hai giá trị gần nhau)
```

Blob càng lớn → tỉ lệ càng tiến về 1.0.

---

**Ngưỡng 0.55 — bắt gì?**

Với blob vuông đặc, tỉ lệ tối thiểu khi qua tiêu chí 2 (bw ≥ 4, bh ≥ 4) là:

```
n = 4  →  9/16 = 0.5625  ← vừa qua ngưỡng 0.55
n = 3  →  4/9  = 0.44   ← đã bị loại ở tiêu chí 2 rồi
```

Ngưỡng 0.55 là tầng phòng thủ cuối cho các blob bất quy tắc lọt qua tiêu chí 1–4: blob quá mảnh (que 1×n pixel: contour shoelace ra ~0), blob do nhiều mảnh JPEG ghép lại mà bounding box lớn bất thường so với diện tích contour thực. Với marker vuông hợp lệ (n ≥ 5), tỉ lệ luôn ≥ 0.64 — qua thoải mái.

---

#### Tiêu chí 6 — Số đỉnh sau giản lược (vertex count)

```python
peri = float(cv2.arcLength(cnt, True))          # chu vi đường viền
approx = cv2.approxPolyDP(cnt, 0.08 * peri, True)
vertex_count = int(len(approx))
if vertex_count < 4 or vertex_count > 8:
    continue
```

**Vấn đề:** `cnt` từ `findContours` có hàng trăm điểm — marker 30×30 có ~120 điểm biên, bong bóng đường kính 20 px có ~63 điểm. Số điểm thô thay đổi theo kích thước blob nên không thể dùng để phân biệt hình dạng.

**`cv2.arcLength(cnt, True)` tính `peri` như thế nào?**

Cộng dồn khoảng cách Euclid giữa từng cặp điểm liên tiếp trên đường viền, rồi cộng thêm đoạn nối điểm cuối về điểm đầu (vì `closed=True` — đường viền là một vòng khép kín):

```
peri = Σ √((xᵢ₊₁ − xᵢ)² + (yᵢ₊₁ − yᵢ)²)   với i chạy từ điểm đầu đến điểm cuối, và vòng lại điểm đầu
```

Với contour có hàng trăm điểm sát nhau (mỗi bước đi 1 pixel dọc biên), tổng này xấp xỉ đúng chu vi hình học thật của blob.

**Giải pháp — `approxPolyDP` (thuật toán Douglas-Peucker, giản lược đa giác):**

Bỏ các điểm "không cần thiết" — những điểm gần như thẳng hàng với hai điểm bên cạnh. Chỉ giữ những điểm là góc ngoặt thật sự.

`epsilon = 0.08 × peri` (ngưỡng dung sai — _tolerance_): nếu khoảng cách từ một điểm đến đường thẳng nối hai điểm bên cạnh nó < epsilon → điểm đó bị xóa.

**Công thức tính khoảng cách từ 1 điểm đến 1 đường thẳng** (dùng để so với `epsilon`): với điểm cần xét `P0=(x0,y0)`, đường thẳng đi qua 2 điểm `P1=(x1,y1)` và `P2=(x2,y2)`:

```
d = | (y2−y1)·x0 − (x2−x1)·y0 + x2·y1 − y2·x1 |  ÷  √((y2−y1)² + (x2−x1)²)
```

**Công thức này từ đâu ra — suy luận từng bước từ diện tích tam giác:**

Diện tích tam giác 3 đỉnh A, B, C theo công thức định thức quen thuộc (chính là Shoelace ở Tiêu chí 5, áp cho trường hợp 3 điểm):

```
S = ½ |xA(yB − yC) + xB(yC − yA) + xC(yA − yB)|
```

Diện tích tam giác còn tính được theo cách khác, quen thuộc hơn: **½ × đáy × chiều cao**. Nếu chọn cạnh P1-P2 làm đáy (độ dài `|P1P2| = √((x2−x1)² + (y2−y1)²)`), thì **chiều cao hạ từ đỉnh P0 xuống đáy đó chính là khoảng cách vuông góc `d` đang cần tìm** — vì "chiều cao" trong hình học chính là khoảng cách từ 1 đỉnh đến đường thẳng chứa cạnh đối diện:

```
S = ½ × |P1P2| × d   ⟹   d = 2S ÷ |P1P2|
```

Thay `A=P0, B=P1, C=P2` vào công thức định thức ở trên để tính `S`, rồi thế vào `d = 2S ÷ |P1P2|`:

```
d = |x0(y1 − y2) + x1(y2 − y0) + x2(y0 − y1)|  ÷  √((x2−x1)² + (y2−y1)²)
```

Khai triển biểu thức trong dấu trị tuyệt đối:

```
x0(y1−y2) + x1(y2−y0) + x2(y0−y1)
= x0y1 − x0y2 + x1y2 − x1y0 + x2y0 − x2y1
```

Nhóm lại theo `x0` và `y0` (2 biến của điểm P0 đang xét — vì đây là "hàm số" của điểm cần đo khoảng cách, các điểm P1, P2 cố định):

```
= x0(y1 − y2) + y0(x2 − x1) + (x1y2 − x2y1)
= −(y2−y1)·x0 + (x2−x1)·y0 + (x1y2 − x2y1)
```

Lấy trị tuyệt đối thì đổi dấu toàn bộ biểu thức bên trong không làm thay đổi kết quả (`|−k| = |k|`), nhân (−1) vào cả 3 số hạng:

```
= (y2−y1)·x0 − (x2−x1)·y0 + (x2y1 − x1y2)
= (y2−y1)·x0 − (x2−x1)·y0 + x2y1 − y2x1        (vì x1y2 = y2x1, chỉ đổi thứ tự nhân)
```

→ Đúng bằng biểu thức đã viết ở công thức `d` phía trên. Vậy đây **không phải 2 công thức khác nhau tình cờ giống nhau** — nó là **cùng một công thức diện tích tam giác định thức**, chỉ khai triển và nhóm lại số hạng để lộ rõ 2 biến `x0, y0` của điểm đang xét, sau đó chia cho độ dài đáy để "diện tích" biến thành "chiều cao" = khoảng cách vuông góc cần đo.

**Thuật toán chạy đệ quy trên đoạn [điểm_đầu → điểm_cuối]:**
1. Kẻ đường thẳng nối điểm đầu và điểm cuối
2. Tìm điểm ở giữa cách đường thẳng đó xa nhất (áp công thức `d` ở trên cho từng điểm giữa, lấy max)
3. Nếu khoảng cách xa nhất > epsilon → giữ điểm đó, chia đoạn thành 2 và lặp lại từ bước 1
4. Nếu khoảng cách xa nhất ≤ epsilon → xóa tất cả điểm ở giữa

**Ví dụ tính số cụ thể — giả sử `epsilon = 3px`:**

*Điểm nhiễu trên cạnh thẳng* — 3 điểm biên `P1=(0,0)`, điểm giữa `M=(5,1)` (lệch 1px do răng cưa JPEG), `P2=(10,0)`:
```
d = |(0−0)×5 − (10−0)×1 + 10×0 − 0×0| / √(0² + 10²)
  = |0 − 10 + 0 − 0| / 10 = 10/10 = 1px

1px < epsilon (3px) → M bị xóa, chỉ còn P1, P2 (đúng là cạnh thẳng)
```

*Điểm ở góc vuông thật* — `P1=(0,0)`, góc `CORNER=(10,0)`, `P2=(10,10)` (đường chéo nối P1-P2):
```
d = |(10−0)×10 − (10−0)×0 + 10×0 − 10×0| / √(10² + 10²)
  = |100 − 0 + 0 − 0| / √200 ≈ 100/14.14 ≈ 7.07px

7.07px > epsilon (3px) → CORNER được giữ lại, chia đoạn thành [P1..CORNER] và [CORNER..P2]
```

Hai ví dụ trên cho thấy chính xác cách `epsilon` phân biệt "nhiễu nhỏ nên xóa" (d=1px) với "góc thật nên giữ" (d=7.07px) — cùng 1 công thức, chỉ khác giá trị `d` tính ra.

```
Kết quả trên marker vuông thật: 4 góc + 0–2 điểm JPEG noise sót lại = 4–6 đỉnh
```

Điều kiện 4–8 đỉnh:
- **< 4**: đường thẳng (2), tam giác (3) → không phải hình 4 cạnh
- **4–8**: tứ giác đến bát giác → giữ lại
- **> 8**: hình tròn hoặc quá phức tạp → loại

`epsilon = 8%` thay vì nhỏ hơn: marker bị camera distort vẫn ra 4–5 đỉnh thay vì 15+ đỉnh nếu epsilon nhỏ bắt mọi pixel noise ở góc.

---

#### Tiêu chí 7 — Độ lồi (solidity)

```python
hull = cv2.convexHull(cnt)
hull_area = float(cv2.contourArea(hull))
solidity = contour_area / max(1.0, hull_area)
if solidity < 0.82:
    continue
```

**Tiêu chí 6 bỏ sót gì?** Chữ L có 6 đỉnh sau giản lược (4 góc ngoài + 2 góc trong) → lọt qua điều kiện [4, 8] của tiêu chí 6. Tiêu chí 7 dùng convex hull để bắt những hình lõm như vậy.

**"Convex hull" (bao lồi)** = đa giác lồi nhỏ nhất bao quanh toàn bộ blob. Hình dung: đặt blob lên bảng, căng sợi dây chun quanh tất cả các điểm rồi thả ra — hình dây chun là convex hull.

`cv2.convexHull(cnt)` trả về tập con các điểm biên tạo thành bao lồi. `cv2.contourArea(hull)` tính diện tích bao đó bằng Shoelace (giống tiêu chí 5).

**Vì sao dùng `contourArea` chứ không dùng `area` (đếm pixel từ CC stats) cho `hull_area`?** Hai lý do:
1. `hull` là một **đa giác mới tự tạo ra** từ tọa độ đỉnh (`cv2.convexHull`) — nó chưa từng được "tô" thành vùng pixel nào trên ảnh, nên không hề có `area` (đếm pixel) sẵn có cho nó như `cnt` gốc. Cách duy nhất tính diện tích của nó là hình học từ tọa độ đỉnh — đúng việc `cv2.contourArea()` làm.
2. Quan trọng hơn: `contour_area` (tử số, tái sử dụng từ Tiêu chí 5) vốn đã tính bằng Shoelace, và Shoelace luôn **nhỏ hơn có hệ thống** so với đếm pixel thật (xem Tiêu chí 5: blob 5×5 đặc có `contour_area=16` nhưng `area=25`). Nếu mẫu số `hull_area` tính bằng cách khác (đếm pixel) trong khi tử số tính bằng Shoelace, độ lệch hệ thống này sẽ làm `solidity` bị bơm sai lệch — không còn phản ánh đúng "hình có lõm hay không" nữa. Dùng `cv2.contourArea()` cho cả 2 (cùng 1 hàm, cùng 1 công thức, chỉ khác tập điểm đầu vào) khiến độ lệch hệ thống đó triệt tiêu khi chia, chỉ còn lại đúng phần khác biệt do lõm/lồi gây ra.

```
Marker vuông 4×4:                  Chữ L (thiếu góc trên-phải):

■ ■ ■ ■                            ■ ■ □ □
■ ■ ■ ■                            ■ ■ □ □
■ ■ ■ ■                            ■ ■ ■ ■
■ ■ ■ ■                            ■ ■ ■ ■

Hull = chính nó:                   Hull lấp góc trống:
■ ■ ■ ■                            ■ ■ ■ ■   ← hull mở rộng bao phủ vùng □
■ ■ ■ ■                            ■ ■ ■ ■
■ ■ ■ ■                            ■ ■ ■ ■
■ ■ ■ ■                            ■ ■ ■ ■

contour_area (Shoelace 4×4) = 9    contour_area (Shoelace chữ L) ≈ 7.5
hull_area    (Shoelace 4×4) = 9    hull_area    (Shoelace 4×4)  = 9

solidity = 9/9 = 1.0  ✓ GIỮ       solidity = 7.5/9 ≈ 0.83  ← sát ngưỡng
```

Hình có phần lõm vào → hull lớn hơn contour → solidity < 1. Hình lồi hoàn toàn (vuông, tròn, chữ nhật) → hull = contour → solidity = 1.

Ngưỡng 0.82: marker in thực tế bị bo góc nhẹ do lens → hull hơi lớn hơn contour → solidity ~0.90–0.98. Ngưỡng 0.82 chấp nhận điều đó mà vẫn loại chữ L (0.83 sát ngưỡng), hình sao (< 0.6), zigzag.

---

#### Tiêu chí 8 — Độ tròn (circularity)

```python
circularity = (4.0 * math.pi * contour_area) / (peri * peri)
if circularity > 0.90 and fill_ratio < 0.92 and vertex_count > 5:
    continue
```

**Tiêu chí 6 + 7 bỏ sót gì?** Bong bóng tô mực đặc (học sinh bôi đen hoàn toàn) có:
- `vertex_count` sau giản lược: 6–8 → lọt qua tiêu chí 6
- `solidity` ≈ 1.0 (hình tròn đặc hoàn toàn lồi, hull = chính nó) → lọt qua tiêu chí 7

Tiêu chí 8 phân biệt tròn với vuông qua mối quan hệ giữa diện tích và chu vi.

**Tại sao `4π × diện_tích / chu_vi²` đo được độ tròn?**

Với bất kỳ hình phẳng nào, chu vi càng "lãng phí" (đi vòng vèo nhiều mà không bao được nhiều diện tích) thì tỉ lệ này càng thấp. Hình tròn là hình **hiệu quả nhất**: cùng một độ dài chu vi, hình tròn bao được diện tích lớn nhất có thể — đây là **bất đẳng thức đẳng chu vi** (_isoperimetric inequality_):

```
Diện_tích ≤ P² / (4π)          — luôn đúng với mọi hình phẳng
→  4π × Diện_tích / P²  ≤ 1.0  — dấu = chỉ xảy ra với hình tròn
```

**Kiểm tra bằng số:**

Hình tròn bán kính r (diện tích = πr², chu vi = 2πr):
```
4π × πr² / (2πr)² = 4π²r² / 4π²r² = 1.0
```

Hình vuông cạnh s (diện tích = s², chu vi = 4s):
```
4π × s² / (4s)² = 4πs² / 16s² = π/4 ≈ 0.785
```

Hình chữ nhật 1×4 (chu vi lãng phí hơn vuông):
```
4π × 4 / (10)² = 16π / 100 ≈ 0.503
```

| Hình | Circularity | Ghi chú |
|------|-------------|---------|
| Tròn hoàn hảo | 1.000 | Giới hạn lý thuyết trên |
| Bong bóng tô bút bi | 0.85–0.95 | Không hoàn toàn tròn vì tô tay |
| Marker vuông sạch | ~0.785 | = π/4 |
| Marker bị distort | 0.70–0.82 | Méo do góc chụp |
| Chữ nhật dài | < 0.70 | Chu vi lãng phí |

**Tại sao cần 3 điều kiện AND, không chỉ `circularity > 0.90`?**

Marker góc nhỏ in laser đôi khi có circularity 0.88–0.92 do bo góc nhẹ. Chỉ dùng `circularity > 0.90` sẽ loại nhầm marker hợp lệ đó.

Ba điều kiện phải đúng cùng lúc mới loại:
- `circularity > 0.90` — rất gần tròn
- `fill_ratio < 0.92` — tô không đặc hoàn toàn (marker in laser fill ≥ 0.92 → giữ lại dù trông tròn)
- `vertex_count > 5` — nhiều đỉnh (marker nhỏ bị bo chỉ có 4–5 đỉnh → giữ lại)

Chỉ bong bóng tô mực đáp ứng cả 3 cùng lúc: circularity cao + fill trung bình (~0.70) + nhiều đỉnh.

Đến đây, `_extract_black_square_markers()` trả về danh sách marker cuối cùng (đã qua cả 8 tiêu chí) cho `_extract_black_square_markers_from_gray()` (mục 5), rồi hàm đó trả tiếp lên cho `_detect_page_corners_from_black_square_markers()` (mục 4) — biến `markers` ở mục 4 chính là kết quả này. Quay lại mục 4/7 để xem phần còn lại xử lý danh sách này ra sao.

---

## 7. Quay lại hàm tìm 4 góc — chọn 4 marker tốt nhất

Ở mục 4, hàm `_detect_page_corners_from_black_square_markers()` đã gọi và nhận về `markers` (danh sách ~5–20 ứng viên qua mục 5+6). Bây giờ hàm này chạy tiếp phần thân còn lại — chia 4 vùng, chấm điểm, chọn ra đúng 4 cái là marker góc thật sự của tờ phiếu:

```python
# omr_marker_utils.py:194-224 — nửa sau hàm _detect_page_corners_from_black_square_markers()

# Định nghĩa hàm _pick() — chọn marker tốt nhất trong 1 vùng
def _pick(cands, tx, ty):
    """Trong danh sách cands, chọn marker có score cao nhất.
       tx, ty = tọa độ góc ảnh mà vùng này hướng về."""
    best, best_score = None, -1e9
    for m in cands:
        dx = float(m["cx"]) - float(tx)   # khoảng cách x từ centroid marker đến góc ảnh
        dy = float(m["cy"]) - float(ty)   # khoảng cách y từ centroid marker đến góc ảnh
        dist_norm = (dx*dx + dy*dy) / max(1.0, float(w * h))   # bình phương khoảng cách, chuẩn hóa
        area_norm = float(m["area"])      / max(1.0, float(w * h))   # diện tích chuẩn hóa
        score = area_norm * 1000.0 + float(m["fill"]) * 40.0 - dist_norm * 230.0
        if score > best_score:
            best_score, best = score, m
    return best

# Chia ảnh thành 4 vùng góc 30%×30%, lọc ứng viên theo vùng và gọi _pick() cho từng vùng
corner_w = int(w * 0.30)   # ngưỡng x: 30% chiều rộng ảnh
corner_h = int(h * 0.30)   # ngưỡng y: 30% chiều cao ảnh

tl = _pick([m for m in markers if m["cx"] <= corner_w          and m["cy"] <= corner_h         ], 0.0,    0.0   )
tr = _pick([m for m in markers if m["cx"] >= (w - corner_w)    and m["cy"] <= corner_h         ], float(w), 0.0  )
bl = _pick([m for m in markers if m["cx"] <= corner_w          and m["cy"] >= (h - corner_h)   ], 0.0,    float(h))
br = _pick([m for m in markers if m["cx"] >= (w - corner_w)    and m["cy"] >= (h - corner_h)   ], float(w), float(h))

# Thêm padding vào tọa độ 4 góc trước khi trả về (dùng cho warp ở Bước 4)
pad = int(max(3, min(10, 0.25 * np.median([tl["size"], tr["size"], bl["size"], br["size"]]))))
tl_pt = [max(0, int(tl["x"]) - pad), max(0, int(tl["y"]) - pad)]   # tr_pt, bl_pt, br_pt tính tương tự
```

### 7.1. Hàm `_pick()` — scoring để chọn marker tốt nhất

**Lưu ý trước khi đọc mục này:** đây **không phải bước "tìm marker"** nữa — việc đó đã xong ở mục 5-6 (`markers` là kết quả đã qua lọc 8 tiêu chí hình dạng). Vấn đề ở mục 7.1 khác hẳn: trong list `markers` đó, **vẫn có thể còn nhiều hơn 1 ứng viên rơi vào cùng 1 vùng góc 30%×30%** — vì bong bóng tô đặc hình gần-vuông, hay chữ số đậm, vẫn có thể vượt qua cả 8 tiêu chí hình dạng ở mục 6. `_pick()` là **1 lớp lọc nữa**, chạy sau, chỉ để trả lời: trong các ứng viên còn sót lại ở 1 vùng góc, cái nào giống marker thật nhất?

**Vấn đề cần giải:** Sau khi lọc vùng, mỗi vùng vẫn còn nhiều ứng viên — marker góc thật + bong bóng lọt vùng + text lớn ở góc + noise. Cần chọn đúng 1 cái. Không thể dùng một rule cứng ("chọn cái to nhất") vì:

```
Vùng TL ví dụ:

  ████ marker thật (40×40, fill=0.91, sát góc)
  ●    bong bóng tô đặc (20×20, fill=0.85, giữa vùng)
  ▓▓   text đậm (50×15, fill=0.70, gần góc)

→ "Chọn to nhất"  → text thắng (area 750px > 1600px? không, nhưng đôi khi text block lớn)
→ "Chọn fill cao" → marker ≈ bong bóng (0.91 vs 0.85), phân biệt không rõ
→ "Chọn gần góc"  → noise pixel ngay sát góc có dist = 0
```

Giải pháp: kết hợp 3 tín hiệu, mỗi cái bù vào điểm yếu của cái kia.

---

**Các biến đầu vào:**

`tx`, `ty` **không phải giá trị tính toán ra** — chỉ là **tọa độ cố định của 4 góc khung ảnh**, biết trước ngay từ `h, w = gray_img.shape[:2]` lúc đầu hàm:

```
(0,0)────────────(w,0)
  │                 │
  │   ảnh w × h     │
  │                 │
(0,h)────────────(w,h)
```

```
_pick(cands_TL, tx=0,   ty=0  )   ← góc trên-trái
_pick(cands_TR, tx=w,   ty=0  )   ← góc trên-phải
_pick(cands_BL, tx=0,   ty=h  )   ← góc dưới-trái
_pick(cands_BR, tx=w,   ty=h  )   ← góc dưới-phải
```

**"`tx, ty` là góc ảnh mà vùng này hướng về" nghĩa là gì?** Mỗi vùng góc (TL/TR/BL/BR) có 1 "điểm neo lý tưởng" là đúng góc khung ảnh của nó — vì marker góc thật **luôn được in sát mép giấy vật lý** (đây là ràng buộc thiết kế của tờ phiếu, biết trước chứ không đo đạc). Khi đánh giá 1 ứng viên trong vùng TL, ta so vị trí thật của nó với điểm lý tưởng `(0,0)` — càng gần điểm đó càng đáng tin là marker thật.

`dx`, `dy` = độ lệch giữa **vị trí thật của ứng viên** (`cx, cy` — centroid tính từ CC ở mục 6.1) và **vị trí lý tưởng của góc ảnh** (`tx, ty`):
```python
dx = m["cx"] - tx
dy = m["cy"] - ty
```

Ứng viên nằm sát góc → `dx, dy` nhỏ. Ứng viên nằm sâu vào giữa trang (như bong bóng MCQ) → `dx, dy` lớn. Đây là nguyên liệu để tính Tín hiệu 3 (`dist_norm`) bên dưới. Cả hai đều dương (ứng viên luôn nằm trong ảnh, góc nằm ở biên).

---

**Tín hiệu 1 — `area_norm` (diện tích chuẩn hóa):**

```python
area_norm = m["area"] / (w × h)
```

Ý đơn giản đằng sau: **marker in sẵn luôn to hơn đáng kể so với bong bóng/chữ số** trên cùng tờ phiếu — đây là 1 trong 3 đặc điểm để phân biệt marker thật với rác còn sót trong vùng góc.

**Tại sao không dùng thẳng `m["area"]` (số pixel) mà phải chia cho `w×h`?** Vì `area` phụ thuộc hoàn toàn vào độ phân giải ảnh chụp, không phải bản chất của marker. Cùng 1 tờ phiếu:

| Cách chụp | Kích thước ảnh | `area` marker (px²) |
|---|---|---|
| Điện thoại 12MP | 4032×3024 | ~9000 |
| Scan giấy cũ, chất lượng thấp | 800×600 | ~350 |

So sánh `area` tuyệt đối giữa 2 marker từ 2 ảnh khác độ phân giải là vô nghĩa (9000 > 350 không có nghĩa marker đầu "to hơn" về mặt thiết kế). Nhưng **tỷ lệ** `area / (w×h)` (marker chiếm bao nhiêu % diện tích ảnh) gần như **không đổi** dù độ phân giải nào, vì marker luôn in theo cùng 1 tỷ lệ cố định so với khổ giấy A4. Chia cho `w×h` biến `area_norm` thành con số **so sánh được** giữa các ảnh khác nhau — đúng là cùng ý tưởng normalize đã dùng ở kernel Close (mục 5.2) và ở tiêu chí area_ratio (mục 6.3 tiêu chí 1).

`area_norm` đại diện cho: **marker này to hay nhỏ so với toàn ảnh?** Trong vùng 30%, marker thật thường là blob to nhất.

---

**Tín hiệu 2 — `fill` (fill_ratio từ tiêu chí 4, mục 6.3):**

`fill` đã được tính ở mục 6.3 (tiêu chí 4) và lưu trong dict marker: `m["fill"]` = `area / (bw × bh)`.

`fill` đại diện cho: **blob này đặc hay rỗng?**

| Loại blob | Fill thực tế |
|---|---|
| Marker in laser | 0.85 – 0.95 |
| Bong bóng tô bút bi đặc | 0.70 – 0.82 |
| Text đậm | 0.50 – 0.70 |
| Đường kẻ / noise | < 0.50 |

---

**Tín hiệu 3 — `dist_norm` (khoảng cách đến góc, chuẩn hóa):**

```python
dist_norm = (dx*dx + dy*dy) / (w * h)
```

**Từng phần của công thức đến từ đâu:**

- **`dx² + dy²`** — đây chính là bình phương khoảng cách Euclid thật từ centroid ứng viên đến góc ảnh lý tưởng, theo định lý Pythagoras: `khoảng_cách² = dx² + dy²` (tam giác vuông với 2 cạnh góc vuông là `dx`, `dy`). Nói cách khác, đây là con số đo trực tiếp "ứng viên này cách góc ảnh bao xa".
- **Vì sao dùng `dx²+dy²` mà không lấy căn `√(dx²+dy²)` cho ra đúng khoảng cách?** Vì công thức này chỉ dùng để **xếp hạng** (ai gần góc hơn thì thắng), không dùng con số đó cho việc gì khác. Bình phương và căn bậc hai luôn cho **cùng thứ tự xếp hạng** (nếu A gần hơn B thì `dist²_A < dist²_B` cũng đúng như `dist_A < dist_B`) — nên bỏ hẳn bước `sqrt()` cho nhanh, không mất tính đúng đắn.
- **Vì sao chia cho `w×h`?** Cùng lý do như `area_norm` ở trên: lệch 50px là "rất gần góc" trên ảnh 4K, nhưng là "lệch nặng" trên ảnh 500px. Chia cho diện tích ảnh đưa khoảng cách về **tỷ lệ tương đối**, so sánh được giữa các độ phân giải khác nhau.

`dist_norm` đại diện cho: **blob này có thực sự nằm sát góc ảnh không?** Marker góc được in **sát góc phiếu** → centroid gần (0,0) hoặc (w,h) tùy vùng → `dist_norm` nhỏ. Bong bóng và chữ nằm sâu trong trang (dù vẫn lọt vào vùng 30% do phiếu chụp nghiêng) → `dist_norm` lớn hơn hẳn.

**Mục tiêu dùng nó để làm gì:** dùng làm **hình phạt (penalty)**, không phải điểm cộng. Trong công thức score bên dưới nó bị **trừ đi** (`− dist_norm × 230`) — mục đích là hạ điểm những ứng viên "lọt được vào vùng góc 30%" nhưng thực ra không nằm sát mép giấy, để chúng thua marker thật (vốn luôn nằm sát mép).

---

**Công thức score:**

```
score = area_norm × 1000  +  fill × 40  −  dist_norm × 230
```

**Trọng số 1000, 40, 230 từ đâu?**

Ba tín hiệu có **đơn vị và khoảng giá trị khác nhau hoàn toàn**. Trọng số được chọn để **đưa chúng về cùng thang điểm** (~0–40 mỗi tín hiệu), tránh một tín hiệu át hoàn toàn hai cái còn lại:

| Tín hiệu | Khoảng giá trị thực | × Trọng số | Đóng góp vào score |
|---|---|---|---|
| `area_norm` | [0.00008, 0.025] | × 1000 | [0.08, **25**] |
| `fill` | [0.40, 1.0] | × 40 | [16, **40**] |
| `dist_norm` | [0, ~0.18] | × 230 | [0, **~41**] penalty |

Tất cả đều trong khoảng 0–40. Không tín hiệu nào thắng mặc định — marker phải **tốt trên cả ba mặt** mới thắng.

---

**Ví dụ tính toán — ảnh 3500×2500 (w×h = 8.75M):**

```
Marker góc thật (40×40 px):
  area_norm = 1600 / 8.75M = 0.000183  → × 1000 = 0.18
  fill      = 0.91                      → ×  40  = 36.4
  dx=25, dy=20 → dist_norm = (625+400)/8.75M = 0.000117 → × 230 = 0.027
  score = 0.18 + 36.4 − 0.03 = 36.55

Bong bóng tô đặc trong vùng TL (20×20 px, ở vị trí cx=350, cy=280):
  area_norm = 400 / 8.75M = 0.0000457  → × 1000 = 0.046
  fill      = 0.78                      → ×  40  = 31.2
  dx=350, dy=280 → dist_norm = (122500+78400)/8.75M = 0.023 → × 230 = 5.3
  score = 0.046 + 31.2 − 5.3 = 25.95

Text block ở góc (100×12 px, fill=0.60, cx=80, cy=40):
  area_norm = 1200 / 8.75M = 0.000137  → × 1000 = 0.14
  fill      = 0.60                      → ×  40  = 24.0
  dx=80, dy=40 → dist_norm = (6400+1600)/8.75M = 0.000914 → × 230 = 0.21
  score = 0.14 + 24.0 − 0.21 = 23.93

→ Marker thắng: 36.55 > 25.95 > 23.93 ✓
→ Marker thắng chủ yếu vì fill cao hơn (36.4 vs 31.2 và 24.0)
→ dist giúp phân biệt thêm giữa bong bóng (−5.3) và marker (−0.03)
→ area đóng góp nhỏ (~0.04–0.18) — chỉ là tiebreaker khi fill và dist gần bằng nhau
```

---

### 7.2. Chia 4 vùng góc và lọc ứng viên

`corner_w` và `corner_h` tạo ra 4 vùng hình chữ nhật ở 4 góc ảnh. Mỗi vùng chiếm 30%×30% diện tích ảnh:

```
┌──────────────────────────────────┐
│  ████      │        │      ████  │
│  TL vùng   │        │   TR vùng  │
│  cx≤30%    │        │   cx≥70%   │
│  cy≤30%    │        │   cy≤30%   │
│────────────┤        ├────────────│
│            │        │            │
│────────────┤        ├────────────│
│  ████      │        │      ████  │
│  BL vùng   │        │   BR vùng  │
│  cx≤30%    │        │   cx≥70%   │
│  cy≥70%    │        │   cy≥70%   │
└──────────────────────────────────┘
```

Mỗi ứng viên marker được lọc bằng điều kiện `cx` và `cy` (centroid từ CC) để chỉ vào đúng 1 vùng. Sau đó `_pick()` (mục 7.1) chọn marker tốt nhất trong vùng đó.

**Tại sao 30%, không phải 25% hay 40%?**

Trước hết cần nhớ: `gray_img` ở bước này là ảnh gốc **chưa warp** (`gray_src` — nguyên khung hình camera chụp được, xem mục 1), không phải ảnh chỉ chứa riêng tờ giấy. Trong khung hình đó luôn có 2 yếu tố kéo marker thật ra khỏi đúng góc `(0,0)/(w,0)/(0,h)/(w,h)` của ảnh:

1. **Tờ giấy hiếm khi lấp đầy 100% khung hình** — luôn còn 1 khoảng lề (mép bàn, nền xung quanh) giữa mép giấy và mép ảnh.
2. **Tờ giấy hiếm khi được chụp thẳng hàng tuyệt đối** — cầm tay chụp luôn nghiêng 1 góc nào đó.

Cả 2 yếu tố cộng lại khiến marker góc thật (nằm sát mép **tờ giấy**) không nằm sát mép **khung ảnh** — nó bị dịch vào trong một khoảng nào đó, tùy ảnh chụp ra sao. `corner_w`/`corner_h` chính là bề rộng của "vùng an toàn" quanh mỗi góc ảnh, phải đủ lớn để luôn chứa được marker dù nó bị dịch vào trong bao nhiêu, nhưng đủ nhỏ để không lấn vào nội dung ở giữa trang. Đây là 1 ngưỡng **chưa thấy bằng chứng canh chỉnh bằng dữ liệu đo thật trong code hiện tại** (không suy ra được từ 1 công thức đóng, và cũng không có test/dataset nào trong repo đo lại con số này — xem "Lưu ý quan trọng" cuối mục) — chỉ có thể lý luận định tính về 2 đầu cực đoan:

**Nếu chọn quá nhỏ (ví dụ 25%):** với ảnh nghiêng nhiều + lề nền rộng, marker thật có thể bị dịch vào trong sâu hơn 25% tính từ mép ảnh. Khi đó điều kiện lọc `m["cx"] <= corner_w` (mục 7.2) loại luôn cả **marker thật** ra khỏi danh sách ứng viên của vùng đó — `_pick()` không còn gì để chọn, trả về `None` cho góc này → cả 4 góc phải đủ cả (dòng `if tl is None or ... : return None`) nên chỉ 1 góc thất bại là toàn bộ nhánh marker thất bại, phải rơi về contour fallback (kém chính xác hơn — xem mục 3).

**Nếu chọn quá lớn (ví dụ 40%):** hai vấn đề riêng biệt:
- *Vùng giữa trang bị thu hẹp quá mức:* phần "trung lập" (không thuộc góc nào) chỉ còn `100% − 2×40% = 20%` mỗi chiều. Phần lớn diện tích ảnh giờ rơi vào 1 trong 4 "vùng góc" — nên nội dung thật sự nằm ở giữa/rìa trang (khối bong bóng MCQ, ô số báo danh, logo tiêu đề) rất dễ vô tình lọt vào 1 vùng góc và cạnh tranh điểm số với marker thật ở `_pick()`.
- *Tiến sát ngưỡng chồng lấn:* 2 vùng đối diện theo 1 trục (ví dụ trái dùng `x ≤ corner_w`, phải dùng `x ≥ w − corner_w`) chỉ thật sự **chồng lên nhau** khi `corner_w > w − corner_w`, tức tỷ lệ vượt quá 50%. Ở 40% chưa chồng lấn thật (còn cách ngưỡng vỡ 10 điểm %), nhưng không còn biên an toàn nào nếu gặp ảnh có tỷ lệ khung hình lệch (ví dụ ảnh bị crop lệch tâm) — chỉ cần thêm vài % nữa là 2 vùng bắt đầu dẫm lên nhau, một blob ở giữa-lệch có thể được xét ở cả 2 vùng đối diện cùng lúc.

**30%** là điểm cân bằng giữa 2 rủi ro trên: đủ rộng để chịu được mức nghiêng + lề thường gặp khi chụp tay (rủi ro "quá nhỏ"), nhưng vẫn cách xa ngưỡng chồng lấn 50% và giữ được vùng giữa trang tương đối "sạch" (rủi ro "quá lớn").

**Lưu ý quan trọng — 30% lấy từ đâu ra?** Đã kiểm tra lại (git history, test suite, config) và câu trả lời trung thực là: **đây cũng là giá trị hand-pick, không có cơ sở đo đạc**, cùng pattern như `pad` ở mục 7.3:

- `git log -S "corner_w" --all` chỉ ra đúng **1 commit duy nhất** từng đụng đến hằng số này — `ed6dc2d "update all"`, cùng commit gộp đưa cả file `omr_marker_utils.py` vào repo lần đầu. Không có commit nào sau đó tinh chỉnh giá trị 0.30 theo dữ liệu.
- Không có test nào trong `be/tests/` hay `eval/` đo tỷ lệ marker bị bỏ sót (do vùng quá hẹp) hay bị lẫn nhầm (do vùng quá rộng) theo ngưỡng này — cũng không có bộ ảnh chụp nghiêng nhiều góc độ nào được dùng để kiểm chứng.
- Không config/comment nào trong code, và không có tài liệu/kỹ thuật OMR bên ngoài nào được tham chiếu làm cơ sở cho đúng con số 30%.

Hai đoạn lý luận "quá nhỏ"/"quá lớn" ở trên vẫn **đúng về mặt logic** (tự suy ra được từ chính công thức lọc `cx <= corner_w` và ngưỡng chồng lấn hình học 50% trong code) — nhưng đó là suy luận định tính về 2 đầu cực đoan, không phải kết quả đo bằng số liệu thật, nên **không chứng minh được 30% là số tối ưu**, chỉ chứng minh nó nằm trong khoảng "không cực đoan".

**Vì sao vẫn chấp nhận được dù chưa kiểm chứng:** khác với `pad` (được warp về kích thước đích cố định "pha loãng" sai số), ở đây cơ chế an toàn là **fallback sang contour detection** (mục 3, `_find_page_quad_by_contour()`). Nếu ngưỡng 30% khiến 1 góc không tìm được marker nào lọt vùng, `_detect_page_corners_from_black_square_markers()` trả `None`, và `_detect_page_quad()` tự chuyển sang chiến lược kém chính xác hơn nhưng vẫn chạy được, thay vì lỗi cứng toàn pipeline. Đây là lý do hệ thống "chịu được" một hằng số chưa kiểm chứng — không phải bằng chứng cho thấy 30% là lựa chọn tối ưu.

**Hướng cải thiện nếu cần độ tin cậy cao hơn (chưa triển khai):** đo tỷ lệ thành công marker-detection trên 1 tập ảnh chụp thật với nhiều góc nghiêng khác nhau, để chọn ngưỡng có cơ sở thống kê thay vì hand-pick.

---

### 7.3. Padding khi lấy tọa độ góc

**Vì sao cần bước này — "tìm được 4 vị trí marker rồi thì thôi chứ"?**

`_pick()` (mục 7.1) mới chỉ trả lời **"marker nào"** ở mỗi góc — chọn ra đúng 1 ứng viên trong số nhiều blob cùng vùng. Nhưng "tìm được 4 vị trí marker" **không phải là đích đến cuối cùng** của mục 2: nhắc lại chuỗi gọi ở mục 1 — toàn bộ mục 2 chỉ để phục vụ `_warp_to_standard_layout()` tính ra `quad` (4 điểm tọa độ) đưa vào `cv2.getPerspectiveTransform` ở **Bước 4**, nhằm "duỗi thẳng" cả tờ phiếu (đang bị nghiêng/méo phối cảnh) về đúng hình chữ nhật chuẩn. Việc còn thiếu là: trên mỗi marker vừa chọn được, **điểm chính xác nào** (pixel nào) sẽ được dùng làm 1 trong 4 góc của `quad` đó? Đây mới là câu hỏi mục 7.3 trả lời.

**Vì sao không dùng thẳng tọa độ bounding box của marker (`tl["x"], tl["y"]` từ CC) làm góc quad luôn cho xong?**

Vì tọa độ đó luôn bị lệch **vào phía trong** một chút so với góc thật của marker in trên giấy, do 2 nguyên nhân cộng dồn:
1. **Binarization/morphology ăn mòn biên:** Otsu (mục 5.4) có thể bỏ sót vài pixel biên mờ (anti-alias/JPEG), và `MORPH_OPEN`/`MORPH_CLOSE` ở mục 6.2 vốn dùng Erode nên luôn "bào mỏng" biên blob đi 1-2px trước khi trả lại kích thước gần đúng — kết quả CC nhận diện marker hơi **nhỏ hơn** thực tế.
2. **Lệch tọa độ do rời rạc hóa (đã thấy ở Tiêu chí 5, mục 6.3):** `findContours`/CC trả về tọa độ theo **chỉ số pixel**, không phải cạnh ngoài cùng của pixel đó — nên bounding box luôn "thiếu" đúng 1 hàng/cột ở mép ngoài so với diện tích hình học thật (ví dụ blob 5×5 đặc: bounding box đo được dừng ở tọa độ (4,4) chứ không phải (5,5)).

Cả 2 nguyên nhân đều đẩy tọa độ phát hiện được **vào trong**, chưa bao giờ ra ngoài.

**"Lệch vào trong" nghĩa là gì, cụ thể bằng số:**

Giả sử marker in thật trên giấy chiếm đúng 10×10 pixel, từ tọa độ `(100,100)` đến `(109,109)` — góc trên-trái **thật** của nó là điểm `(100,100)`, đây là điểm lẽ ra ta muốn dùng.

```
Marker in thật (10×10):          Marker sau threshold/morphology
                                  (biên mờ bị coi là nền, Erode bào bớt):

(100,100)                        (100,100)
   ┌──────────┐                     ┌┈┈┈┈┈┈┈┈┈┈┐   ← vùng chấm: pixel biên
   │██████████│                     ┊██████████┊     bị "mất" khỏi mắt CC
   │██████████│                     ┊  ┌──────┐┊
   │██████████│         →           ┊  │██████│┊  ← chỉ vùng lõi giữa
   │██████████│                     ┊  │██████│┊     mới được tính là
   │██████████│                     ┊  └──────┘┊     "marker" (6×6)
   └──────────┘                     └┈┈┈┈┈┈┈┈┈┈┘
              (109,109)                        (109,109)
                                     ↑
                              CC trả bounding box
                              bắt đầu từ (102,102)
                              — không phải (100,100)
```

`tl["x"], tl["y"]` (từ CC) trong ví dụ này ra `(102, 102)` — **lùi vào phía trong 2px** so với góc thật `(100,100)`, tức là gần tâm marker hơn, xa mép ảnh/mép giấy hơn so với điểm lẽ ra phải có. Đó chính là ý nghĩa của "lệch vào trong": điểm code tính được nằm sâu hơn vào bên trong so với vị trí góc thật mà nó đại diện.

Nếu dùng thẳng `(102,102)` làm góc `quad` cho warp ở Bước 4, khung tham chiếu dùng để "duỗi thẳng" tờ phiếu sẽ hơi nhỏ hơn tờ phiếu thật ở mọi góc → ảnh kết quả sau warp bị cắt mất 1 viền mỏng sát mép — có thể mất đúng phần nội dung sát mép nhất (bong bóng ngoài cùng, ô số báo danh nằm sát viền).

**`pad` là gì:** là một số pixel được **trừ ngược lại** vào tọa độ đã lệch vào trong, để kéo nó ra lại gần đúng vị trí góc thật:

```python
pad = int(max(3, min(10, 0.25 * np.median([tl["size"], tr["size"], bl["size"], br["size"]]))))
tl_pt = [max(0, int(tl["x"]) - pad), max(0, int(tl["y"]) - pad)]
```

Với ví dụ trên, nếu `pad = 2`: `tl_pt = (102 - 2, 102 - 2) = (100, 100)` — đúng khớp lại góc thật. Nói cách khác, `pad` là "khoảng bù" ước tính cho phần bị threshold/morphology ăn mất, cộng/trừ theo đúng hướng đi ra xa tâm marker (ra phía góc ảnh gần nhất): trừ đi cho `x` và `y` của TL (đi về phía góc `(0,0)`), cộng thêm cho `x` của TR/BR và `y` của BL/BR (đi về phía các góc `(w,0)/(0,h)/(w,h)` tương ứng).

**Độ lớn của pad — suy luận định tính, không phải số đo thực nghiệm:**
- Pad = 25% kích thước marker (median của 4 marker): marker 20px → pad 5px; marker 40px → pad 10px. Trực giác đằng sau việc lấy theo % kích thước marker (thay vì 1 số cố định) là: marker càng lớn (ảnh độ phân giải cao) thì các bước ăn mòn ở trên *có thể* cũng lệch nhiều pixel tuyệt đối hơn — đây là 1 giả định hợp lý, không phải kết quả đo được.
- `max(3)`: sàn tối thiểu 3px, ý định là để phiếu nhỏ/ảnh nhỏ vẫn có padding, không để pad tụt về 0-1px (gần như vô nghĩa).
- `min(10)`: trần tối đa 10px, ý định là tránh lấy quá nhiều nền — pad quá lớn thì quad phình ra ngoài cả mép marker vào vùng nền trắng ngoài phiếu.

**Lưu ý quan trọng — 3 con số 0.25 / 3 / 10 này lấy từ đâu ra?** Đã kiểm tra lại (git history, test suite, config) và câu trả lời trung thực là: **đây là giá trị tự chọn (hand-pick), không có cơ sở đo đạc hay tinh chỉnh bằng dữ liệu thật đứng sau nó**:

- Toàn bộ file `omr_marker_utils.py` (gồm cả công thức này) xuất hiện trọn vẹn trong đúng **1 commit gộp** (`ed6dc2d "update all"`) — là file mới hoàn toàn ngay từ đầu, không có chuỗi commit tinh chỉnh dần theo thời gian, không commit nào nhắc đến "padding", "corner accuracy" hay "warp calibration".
- Không có test nào trong `be/tests/` hay script nào trong `eval/` đo lỗi hình học của `pad`/`quad`/warp (so với vị trí marker thật biết trước) — `eval/run_eval.py` chỉ đo accuracy chấm điểm cuối cùng, không đo sai số pixel của bước này.
- Không có config/constants file hay comment nào trong code giải thích vì sao chọn đúng 0.25, 3, 10 chứ không phải số khác.

Nói cách khác, phần lý giải "vì marker lớn thì ăn mòn nhiều hơn", "sàn 3px để không vô nghĩa", "trần 10px để không lấy dư nền" ở trên là **suy luận hợp lý được dựng ra để diễn giải công thức đã có sẵn trong code**, không phải công thức được suy ra từ số liệu đo đạc hay mô hình toán học nào.

**Vì sao vẫn chấp nhận được dù không có cơ sở chặt chẽ:** `_warp_to_standard_layout()` (Bước 4) luôn warp `quad` về 1 kích thước đích **cố định** (khổ A4 ~2480×3508, hằng số `A4_WARP_W/H` ở `omr_service.py`) bất kể `quad` lệch bao nhiêu pixel. Vì đích đến cố định, sai số vài pixel ở `pad` chủ yếu biến thành co giãn/lệch tỷ lệ rất nhẹ trên toàn ảnh, chứ không gây lỗi cục bộ nghiêm trọng — đây là lý do hợp lý để pipeline "chịu được" một hằng số chưa qua kiểm chứng, nhưng không biến 0.25/3/10 thành công thức có cơ sở khoa học.

**Hướng cải thiện nếu cần độ chính xác cao hơn (chưa triển khai):** có thể đo lỗi thực tế bằng cách so `quad` phát hiện được với tọa độ marker in sẵn biết trước trên ảnh mẫu, hoặc thay bounding-box + pad bằng `cv2.cornerSubPix()` để tinh chỉnh góc ở độ chính xác sub-pixel có cơ sở toán học rõ ràng hơn.

Nếu bất kỳ vùng nào trong 4 vùng không chọn được marker (`tl`/`tr`/`bl`/`br` có cái là `None`) — hàm `_detect_page_corners_from_black_square_markers()` trả về `None` ngay (không tính padding), báo hiệu cho `_detect_page_quad()` (mục 3) biết cần thử contour fallback.

---

## 8. Quay lại hàm điều phối — kết quả đi đâu?

4 điểm `tl_pt, tr_pt, bl_pt, br_pt` vừa tính ở mục 7.3 chính là giá trị mà `_detect_page_corners_from_black_square_markers()` (mục 4) trả về — biến `marker_pts` mà `_detect_page_quad()` (mục 3) nhận lại. Nối tiếp chuỗi gọi:

```
_detect_page_corners_from_black_square_markers()  → trả về marker_pts (4 điểm góc)
  → _detect_page_quad()  (omr_preprocess.py:79)    → đóng gói thành (quad, "corner-markers")
    → _warp_to_standard_layout()  (omr_preprocess.py:93)
        dùng quad để cv2.warpPerspective + cv2.resize   ← Bước 4
      → trả về img_std, warp_strategy, global_warp_used, warp_info
        → process_omr_exam() nhận lại kết quả này        ← quay về pipeline chính
```

Nếu không tìm đủ 4 marker ở mục 7 (một hoặc nhiều vùng góc không có ứng viên nào qua được bộ lọc 8 tiêu chí ở mục 6.3) → `marker_pts = None` → `_detect_page_quad()` (mục 3) chuyển sang thử contour fallback (Bước 3, hàm `_find_page_quad_by_contour()`) trước khi đành chấp nhận `quad_strategy = "none"` (không warp).

---

## 9. Tóm tắt: chuỗi quyết định kỹ thuật

```
Ảnh gốc (ánh sáng không đều)
    │
    ▼ Morphological Close (kernel thích nghi) → ước tính background        [mục 5.1-5.2]
    │   [Tại sao: xóa mực khỏi background tốt hơn Gaussian Blur]
    │
    ▼ Divide normalization → chuẩn hóa độ sáng                             [mục 5.3]
    │   [Tại sao: mô hình nhân, nhất quán hơn phép trừ]
    │
    ▼ Gaussian Blur 3×3 → làm mượt histogram
    │
    ▼ Otsu threshold → nhị phân tự động                                    [mục 5.4]
    │   [Tại sao: không cần tune theo từng loại phiếu]
    │
    ▼ Connected Components (8-connectivity)                                [mục 6.1]
    │   [Tại sao: scale-invariant, nhanh, không cần template]
    │
    ▼ 8 tiêu chí lọc (diện tích, tỷ lệ chiều, fill, polygon, solidity, circularity)  [mục 6.3]
    │   [Mỗi tiêu chí loại 1 loại nhiễu cụ thể]
    │
    ▼ Chia 4 vùng góc 30%                                                  [mục 7.2]
    │   [Tại sao 30%: chịu được nghiêng ~17° mà không overlap]
    │
    ▼ Scoring (area×1000 + fill×40 − dist×230)                             [mục 7.1]
        [area dominant: marker góc luôn to nhất]
        [fill phụ: loại blob rỗng]
        [dist phụ: trong 2 marker gần bằng, chọn gần góc hơn]
```

---

## 10. Giới hạn của phương pháp

1. **Phiếu bị gấp nếp ở góc**: marker góc bị biến dạng → solidity giảm, có thể bị lọc. Giải pháp dự phòng: Bước 3 (contour detection).

2. **Ngón tay che góc**: marker bị che một phần → blob bị tách → connected component nhỏ hơn → bị lọc diện tích. Không có giải pháp hoàn hảo — chỉ cần tìm được 4 góc.

3. **Ảnh quá tối toàn bộ** (ánh sáng < 10% bình thường): normalization cải thiện nhưng không hoàn hảo khi SNR quá thấp. Thực tế: điện thoại hiện đại auto-expose, hiếm gặp.

4. **In laser nhạt** (fill ~0.40–0.45): vượt qua được tiêu chí fill ≥ 0.40 nhưng sát ngưỡng. Nếu máy in thực sự hết mực, có thể bỏ sót.
