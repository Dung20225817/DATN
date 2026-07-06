# Bước 1 — Tải ảnh

> Tài liệu này giải thích **tại sao** từng lựa chọn kỹ thuật được dùng trong Bước 1 của pipeline OMR, không chỉ mô tả **cái gì**. Code tham chiếu: `omr_service.py`, `app/api/omr/grading.py`.

---

## 1. Mục tiêu

Bước 1 chỉ có một nhiệm vụ: đưa file ảnh học sinh nộp (chụp bằng điện thoại hoặc scan) từ dạng **byte thô trên đĩa** thành một **ma trận pixel trong bộ nhớ** (`numpy.ndarray`) mà toàn bộ pipeline phía sau (Bước 2 → Bước 15) có thể xử lý bằng OpenCV.

Nghe đơn giản, nhưng đây là bước "cửa vào" — mọi giả định về định dạng dữ liệu cho các bước sau (thứ tự kênh màu, số kênh, kiểu dữ liệu pixel) đều được quyết định ngay tại bước này. Nếu giả định sai ở đây, toàn bộ các phép tính threshold, morphology, template matching ở các bước sau đều sai theo mà không có cách nào "sửa lại" ở giữa pipeline.

---

## 2. Sơ đồ luồng tổng quan

Trước khi đi vào chi tiết từng lựa chọn kỹ thuật, đây là toàn bộ trình tự chạy thật của Bước 1 — mỗi bước bên dưới sẽ được giải thích sâu ở mục tương ứng trong phần 3:

```
1. Client gửi multipart/form-data chứa file ảnh (JPEG/PNG)
        │
        ▼
2. FastAPI nhận qua UploadFile, lưu tạm xuống đĩa bằng shutil.copyfileobj()
   → file_location = BASE_OMR_DIR / <tên file gốc>
        │
        ▼
3. Gọi process_omr_exam(image_path=file_location, ...) trong threadpool
   (run_in_threadpool — vì cv2.imread + toàn bộ xử lý ảnh là tác vụ blocking CPU-bound,
    không được chạy trực tiếp trong event loop async của FastAPI)
        │
        ▼
4. img_raw = cv2.imread(image_path)                         ───→ chi tiết: mục 3.1–3.3
   → Đọc byte ảnh từ đĩa, giải mã theo định dạng file (JPEG/PNG/BMP...)
   → Trả về numpy.ndarray shape (H, W, 3), dtype=uint8, thứ tự kênh BGR
        │
        ▼
5. Kiểm tra img_raw is None → nếu đọc thất bại, trả lỗi ngay  ───→ chi tiết: mục 3.4
        │
        ▼
6. img_input = img_raw (bí danh trực tiếp — không qua bước cắt trung gian nào)
        │
        ▼
7. _warp_to_standard_layout(img_input, ...) — MỘT lệnh gọi duy nhất,      ───→ chi tiết: mục 3.5–3.6
   tự gộp bên trong cả Bước 2 (dò marker góc), Bước 3 (contour fallback),
   Bước 4 (warp + resize) → đây là điểm bàn giao sang Bước 2
```

---

## 3. Chi tiết từng lựa chọn kỹ thuật

### 3.1. Đọc ảnh — `cv2.imread()`

```python
# omr_service.py → process_omr_exam()  (dòng 976)
img_raw = cv2.imread(image_path)
if img_raw is None:
    return {"error": "Khong the doc file anh"}
```

Đây là điểm vào duy nhất của pipeline đọc ảnh chấm bài — không truyền flag thứ hai cho `cv2.imread()`, nên OpenCV dùng mặc định `cv2.IMREAD_COLOR`.

> **Ghi chú:** trước đây còn một điểm vào thứ hai — hàm `suggest_omr_crop_quad()` (tính năng "gợi ý crop 4 góc" cho giao diện cũ) — cũng gọi `cv2.imread()` riêng. Hàm này đã bị xóa (dead code, không còn được frontend gọi từ khi tính năng "Smart Camera Scanner" thay thế luồng gợi ý crop phía server bằng phát hiện marker thời gian thực ngay trên trình duyệt).

**Tại sao dùng `cv2.imread()` thay vì đọc bằng PIL/Pillow?**

Toàn bộ pipeline phía sau (Bước 2 – 15) được viết bằng OpenCV: `cv2.morphologyEx`, `cv2.connectedComponentsWithStats`, `cv2.warpPerspective`, `cv2.matchTemplate`... Tất cả các hàm này đều nhận vào `numpy.ndarray` theo đúng layout mà `cv2.imread()` trả về (BGR, `uint8`, `HxWxC`).

Nếu dùng `PIL.Image.open()` thay thế, ảnh trả về là đối tượng `PIL.Image` ở hệ màu **RGB**, phải convert thủ công (`np.array(img)[:, :, ::-1]`) trước khi dùng được với OpenCV. Dùng thẳng `cv2.imread()` tránh được bước chuyển đổi thừa và tránh nguy cơ quên đảo kênh màu — một lỗi rất dễ xảy ra và khó phát hiện (ảnh vẫn hiển thị "nhìn được" nhưng kênh R/B bị hoán đổi, làm sai lệch toàn bộ phép nhị phân hóa dựa trên độ sáng ở Bước 5).

---

### 3.2. Tại sao ảnh đọc ra là **BGR** chứ không phải RGB?

Đây là một quyết định lịch sử của OpenCV (thư viện ra đời khi Windows dùng thứ tự byte BGR trong bitmap DIB), không phải lựa chọn của hệ thống này — nhưng nó chi phối toàn bộ các bước sau:

- **Bước 4/5** (`cv2.cvtColor(img_std, cv2.COLOR_BGR2GRAY)`): công thức chuyển xám `Y = 0.299R + 0.587G + 0.114B` phải biết chính xác kênh nào là R, kênh nào là B. Nếu dữ liệu bị hiểu nhầm là RGB trong khi thực chất là BGR, trọng số 0.114 (vốn dành cho Blue) sẽ bị áp nhầm cho Red và ngược lại → ảnh xám bị lệch độ sáng, nhất là ở vùng mực đỏ/xanh (nếu phiếu có màu).
- **Bước 15 / `omr_visualize.py`**: vẽ overlay kết quả (khung ROI, khoanh đáp án) dùng màu `(B, G, R)` — ví dụ `(45, 45, 45)` cho vòng tròn xám ở Bước 2 markers. Nếu lộn kênh, màu vẽ ra sẽ sai (đỏ thành xanh dương).

Vì toàn hệ thống nhất quán dùng BGR từ đầu đến cuối, không có bước nào cần đảo kênh màu — chỉ cần nhớ quy ước này khi debug bằng cách lưu ảnh trung gian ra file để xem bằng mắt.

---

### 3.3. Tại sao không truyền flag `cv2.IMREAD_GRAYSCALE` ngay từ bước này?

Ảnh màu gốc (BGR) vẫn cần được giữ nguyên qua Bước 4 (warp phối cảnh) vì:

1. **Bước 15** cần vẽ overlay kết quả bằng màu (đỏ cho câu sai, xanh cho câu đúng, v.v.) lên đúng ảnh học sinh đã nộp — nếu đọc thẳng grayscale ở Bước 1 thì không thể phục hồi lại màu gốc để vẽ.
2. Việc chuyển xám (`cv2.cvtColor(..., COLOR_BGR2GRAY)`) chỉ thực sự cần thiết **sau khi đã warp** (Bước 5), vì chuyển xám sớm rồi mới warp sẽ không thay đổi kết quả toán học (warp là phép biến đổi hình học, không phụ thuộc số kênh màu) nhưng sẽ làm mất khả năng vẽ overlay màu ở bước cuối.

Do đó ảnh màu được giữ nguyên xuyên suốt Bước 1 → 4, và chỉ chuyển xám tại Bước 5 khi bắt đầu cần nhị phân hóa.

---

### 3.4. Xử lý lỗi — tại sao kiểm tra `img_raw is None` thay vì bắt exception?

`cv2.imread()` **không throw exception** khi đọc thất bại (khác với PIL sẽ raise `UnidentifiedImageError`). Đây là hành vi đặc trưng của OpenCV: nếu file không tồn tại, file bị hỏng (corrupt), hoặc định dạng không được hỗ trợ (ví dụ `.heic` chưa có codec, hoặc file `.pdf` bị đặt nhầm đuôi `.jpg`), hàm âm thầm trả về `None` thay vì báo lỗi.

```python
img_raw = cv2.imread(image_path)
if img_raw is None:
    return {"error": "Khong the doc file anh"}
```

Nếu bỏ qua kiểm tra này, dòng code tiếp theo (`img_raw.shape[:2]` hoặc bất kỳ phép toán nào trên `None`) sẽ ném `AttributeError: 'NoneType' object has no attribute 'shape'` — một lỗi runtime khó hiểu ở tầng dưới, không cho người dùng cuối biết nguyên nhân thực sự (file hỏng) mà chỉ thấy lỗi server 500 chung chung.

Kiểm tra `is None` tường minh cho phép trả về thông báo lỗi có ý nghĩa (`"Khong the doc file anh"`) ngay tại điểm vào, dừng pipeline sớm thay vì lan lỗi xuống các bước morphology/threshold phía sau.

---

### 3.5. Điểm nối sang Bước 2 — lệnh gọi `_warp_to_standard_layout()`

> **Ghi chú:** trước đây có 2 nhánh bỏ qua Bước 2 ở đây:
> 1. `_apply_optional_rect_crop()` (crop hình chữ nhật `crop_x/y/w/h`) — đã xóa ở phiên trước, không endpoint/frontend/test nào từng gửi giá trị thật.
> 2. Nhánh `manual_quad_norm` (4 điểm góc thủ công `crop_tl_x...crop_bl_y`, hoặc `profile.strategy.crop_quad`) — cũng đã xóa: tuy được nối dây đúng, đầy đủ tới cả `/grade` và `/grade-batch`, nhưng điều tra xác nhận **không component frontend nào từng gửi 8 tham số này**, và **không UI nào (kể cả trình chỉnh sửa profile) từng ghi `strategy.crop_quad`** — mọi profile lưu qua UI đều có `crop_quad: null`. Đây là hạ tầng dự phòng được cố ý giữ lại khi xóa `suggest_omr_crop_quad()`, nhưng chưa từng có UI nào dùng tới nên đã xóa nốt.
>
> Sau khi xóa cả 2, `img_input` chỉ còn là **bí danh trực tiếp của `img_raw`** (`img_input = img_raw`), và `_warp_to_standard_layout()` chỉ còn đúng 1 đường duy nhất: dò marker góc tự động.

**Đoạn code nào:** lệnh gọi tại `omr_service.py:999-1005`, và toàn bộ định nghĩa hàm tại `omr_preprocess.py:93-130`.

```python
# omr_service.py:999-1005 — nơi process_omr_exam() gọi hàm
img_std, warp_strategy, global_warp_used, warp_info = _warp_to_standard_layout(
    img_input,
    width_img=WIDTH_IMG,
    height_img=HEIGHT_IMG,
    a4_warp_w=A4_WARP_W,
    a4_warp_h=A4_WARP_H,
)
```

```python
# omr_preprocess.py:93-130 — TOÀN BỘ hàm (trước đây tài liệu này chỉ trích 3 dòng đầu, gây khó đối chiếu với code thật)
def _warp_to_standard_layout(img_bgr, width_img, height_img, a4_warp_w, a4_warp_h):
    src_h, src_w = img_bgr.shape[:2]
    strategy = "resize-only"           # giá trị mặc định nếu không tìm được quad
    global_warp_used = False           # sẽ thành True nếu tìm được quad bên dưới
    info = {
        "source": {"width": src_w, "height": src_h},
        "target": {"width": a4_warp_w, "height": a4_warp_h},
    }

    working = img_bgr                  # img_bgr chính là img_input (= img_raw) truyền vào từ trên

    gray_src = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)   # ← chuyển xám ảnh GỐC (chưa warp)
    quad, quad_strategy = _detect_page_quad(gray_src)      # ← ĐIỂM NỐI SANG BƯỚC 2

    if quad is not None:                                   # tìm được 4 góc (marker hoặc contour)
        dst = [[0,0], [a4_warp_w-1,0], [a4_warp_w-1,a4_warp_h-1], [0,a4_warp_h-1]]  # ← A4_WARP_W/H dùng ở đây
        matrix = cv2.getPerspectiveTransform(quad, dst)
        working = cv2.warpPerspective(img_bgr, matrix, (a4_warp_w, a4_warp_h))      # ← và ở đây
        strategy = f"coordinate-global-a4:{quad_strategy}"
        global_warp_used = True
        info["detected_quad"] = _norm_quad_from_points(quad, src_w, src_h)

    interp = cv2.INTER_AREA if working.shape[1] >= width_img else cv2.INTER_CUBIC
    resized = cv2.resize(working, (width_img, height_img), interpolation=interp)    # ← WIDTH_IMG/HEIGHT_IMG dùng ở đây
    return resized, strategy, global_warp_used, info
```

> **Ghi chú dọn dẹp:** trước đây dòng `gray_src.../quad...` còn bọc trong `if not global_warp_used:` — tàn dư từ nhánh `manual_quad_norm` đã xóa (nhánh đó từng có thể set `global_warp_used = True` trước điều kiện này, khiến điều kiện có ý nghĩa thật). Sau khi xóa nhánh đó, điều kiện luôn đúng nên đã bỏ hẳn — hành vi chạy không đổi.

**Làm gì (đọc theo đúng thứ tự code ở trên):**
1. `img_input` (= `img_raw`) được truyền vào với tên tham số `img_bgr`.
2. Chuyển `img_bgr` sang ảnh xám (`gray_src`), gọi `_detect_page_quad(gray_src)` — **đây chính là dòng code bắt đầu Bước 2** (dò 4 marker góc, hoặc contour fallback Bước 3 nếu marker thất bại). Hàm này trả về 2 giá trị `(quad, quad_strategy)`: `quad` là 4 điểm góc thật dùng để warp (hoặc `None`); `quad_strategy` chỉ là nhãn chẩn đoán ghi lại tìm được bằng cách nào (`"corner-markers"`/`"page-contour"`/`"none"`) — **xem giải thích đầy đủ tại `Buoc2.md`, mục "Vị trí trong pipeline"**.
3. Nếu `_detect_page_quad()` trả về `quad` (không phải `None`) — dùng `quad` tính ma trận `getPerspectiveTransform` rồi `warpPerspective` ảnh gốc vào khung kích thước `(a4_warp_w, a4_warp_h)` — **đây là Bước 4**, và đây chính là chỗ 2 hằng số `A4_WARP_W`, `A4_WARP_H` (giải thích ở mục 3.6) được dùng thật. `quad_strategy` cũng được nhúng vào chuỗi `strategy = f"coordinate-global-a4:{quad_strategy}"` ngay tại đây.
4. Nếu `quad is None` (cả Bước 2 và Bước 3 đều thất bại) — `working` giữ nguyên là `img_bgr` gốc, `strategy` giữ giá trị mặc định `"resize-only"` — không warp, chỉ resize thẳng ảnh gốc.
5. Dù có warp hay không, bước cuối luôn chạy: `cv2.resize(working, (width_img, height_img))` — **đây chính là chỗ 2 hằng số `WIDTH_IMG`, `HEIGHT_IMG`** (cũng giải thích ở mục 3.6) **được dùng thật**, đưa ảnh về đúng 1000×1400px trước khi trả về cho `process_omr_exam()`.

**Mục tiêu:** Đây là điểm bàn giao giữa Bước 1 (tải ảnh) và Bước 2 (dò marker góc) — nhưng bàn giao **qua một hàm duy nhất** chứ không phải một lời gọi hàm tên "buoc2()" riêng biệt. Lý do gộp chung: việc dò 4 góc (Bước 2/3) và việc warp bằng 4 góc đó (Bước 4) phụ thuộc chặt vào nhau trong cùng một luồng dữ liệu (`quad` vừa tìm được phải dùng ngay để tính ma trận `getPerspectiveTransform` và warp) — tách thành nhiều lời gọi riêng ở tầng `process_omr_exam()` sẽ chỉ làm tăng số tham số phải truyền qua lại mà không tách được trách nhiệm rõ ràng hơn.

---

### 3.6. Vì sao 4 hằng số kích thước trong lệnh gọi trên lại cố định như vậy?

```python
WIDTH_IMG = 1000
HEIGHT_IMG = 1400
A4_WARP_W = 2480
A4_WARP_H = 3508
```

**Lưu ý quan trọng trước tiên — đây KHÔNG phải kích thước đo tờ giấy vật lý.** `2480` và `3508` là **số lượng pixel**, không phải mm. Hệ thống không "đo" tờ phiếu thật — nó tự dựng ra một khung ảnh trống làm **đích cho phép warp**, và chọn kích thước khung đó dựa trên câu hỏi: *"nếu tờ A4 chuẩn được scan phẳng, thẳng góc, ở độ phân giải scan tài liệu tiêu chuẩn thì nó sẽ chiếm bao nhiêu pixel?"*

**DPI (Dots Per Inch) là gì?** Là số pixel dùng để biểu diễn mỗi inch chiều dài thật. Cùng một tờ A4 (kích thước vật lý cố định 210mm × 297mm), nhưng số pixel đại diện cho nó thay đổi tùy độ phân giải scan:

| Scan ở độ phân giải | Số pixel cho cạnh 210mm |
|---|---|
| 72 DPI (màn hình cũ) | 210 ÷ 25.4 × 72 ≈ 595 px |
| 150 DPI (scan thường) | 210 ÷ 25.4 × 150 ≈ 1240 px |
| **300 DPI (chuẩn scan tài liệu)** | 210 ÷ 25.4 × 300 ≈ **2480 px** ← hệ thống chọn giá trị này |
| 600 DPI (scan chất lượng cao) | 210 ÷ 25.4 × 600 ≈ 4960 px |

`25.4` là hằng số đổi inch → mm (1 inch = 25.4mm). Áp dụng cho cả 2 cạnh A4 ở 300 DPI:

```
A4_WARP_W = 210mm ÷ 25.4mm/inch × 300 px/inch = 2480 px   (cạnh ngắn)
A4_WARP_H = 297mm ÷ 25.4mm/inch × 300 px/inch = 3508 px   (cạnh dài)
```

**`A4_WARP_W/H` dùng để làm gì:** đây là kích thước khung ảnh đích của `cv2.warpPerspective(img_bgr, matrix, (a4_warp_w, a4_warp_h))` — nơi ảnh nghiêng/méo do góc chụp được "nắn thẳng" vào. Chọn theo tỷ lệ A4 thật ở 300 DPI vì hai lý do:
1. **Giữ đúng tỷ lệ khung hình (aspect ratio) 210:297** — không bóp méo ngang/dọc so với tờ giấy thật.
2. **Gần với độ phân giải ảnh gốc** (ảnh điện thoại thường 3000–4000px một cạnh) → phép nội suy trong lúc warp ít mất chi tiết nhất. Nếu warp thẳng xuống 1000×1400 ngay từ đầu, mỗi pixel đích phải đại diện một vùng lớn hơn nhiều ở ảnh gốc → nội suy kém chính xác hơn, cạnh marker/bong bóng bị mờ hơn. 300 DPI cũng đủ để bong bóng MCQ nhỏ nhất (~4mm đường kính) vẫn chiếm ~47px — đủ để phân tích fill ratio chính xác ở các bước sau.

**`WIDTH_IMG = 1000`, `HEIGHT_IMG = 1400` — kích thước làm việc chuẩn cho toàn bộ pipeline sau Bước 4:**

Sau khi warp lên 2480×3508 (giữ chi tiết), ảnh được resize xuống còn 1000×1400 — đây mới là "hệ tọa độ chuẩn" mà mọi ROI, anchor, ngưỡng pixel (`line_h` clamp [12,44]px, kích thước marker tối thiểu...) ở Bước 5-15 đều tính toán dựa trên đó. Lý do chọn đúng 1000×1400, không lớn hơn hay nhỏ hơn:

| Kích thước | Vấn đề |
|---|---|
| Nhỏ hơn (vd 500×700) | Bong bóng MCQ chỉ còn ~3–4px → fill ratio không ổn định; marker fiducial khó phát hiện |
| Lớn hơn (vd 2000×2800) | Mọi phép tính (connected components, template matching) chậm hơn ~4 lần, không tăng độ chính xác đáng kể |
| 1000×1400 | Bong bóng ~7–12px, marker ~8–15px — đủ ổn định; toàn bộ pipeline xử lý dưới 1 giây trên CPU thường |

Tỉ lệ 1000:1400 ≈ 0.714 gần đúng tỉ lệ A4 thật (210:297 ≈ 0.707, sai số < 1%) nên resize không làm méo nội dung đáng kể.

**Tại sao khai báo thành hằng số toàn cục (`omr_service.py:41-44`) thay vì tham số cục bộ?** Vì mọi hàm từ Bước 4 (warp) đến Bước 12 (decode MCQ) đều phải dùng chung một hệ quy chiếu pixel — khai báo 1 lần duy nhất ở đầu module đảm bảo không hàm nào vô tình tính toán trên một kích thước khác, tránh lệch tọa độ ROI giữa các bước.

> Xem đầy đủ công thức, ví dụ tính toán và sơ đồ luồng warp → resize tại `Buoc4.md`, mục "4. Kích thước trung gian 2480×3508 — tại sao?" và "5. Resize về 1000×1400 — tại sao?".

---

## 4. Vị trí code (tra cứu nhanh)

| Thành phần | File | Vị trí |
|---|---|---|
| Nhận file upload, lưu tạm xuống đĩa | `app/api/omr/grading.py` | dòng 109–113 (endpoint chấm 1 ảnh), dòng 308–313 (endpoint chấm nhiều ảnh) |
| Gọi `process_omr_exam` qua threadpool | `app/api/omr/grading.py` | dòng 129–131 |
| Đọc ảnh chính (pipeline chấm bài) | `app/services/omr/omr_service.py` → `process_omr_exam()` | dòng 976–978 |
| Điểm nối sang Bước 2/3/4 (1 lệnh gọi duy nhất) | `app/services/omr/omr_service.py` → `_warp_to_standard_layout()` | dòng 999–1005 |
| Hằng số kích thước chuẩn dùng ở các bước sau | `app/services/omr/omr_service.py` | dòng 41–44 (`WIDTH_IMG`, `HEIGHT_IMG`, `A4_WARP_W`, `A4_WARP_H`) |

---

## 5. Giới hạn hiện tại (chưa xử lý ở Bước 1)

- **Không xử lý EXIF orientation**: nếu ảnh JPEG có cờ EXIF xoay (ví dụ điện thoại chụp dọc nhưng lưu kèm cờ "xoay 90°" thay vì xoay pixel thật), `cv2.imread()` đọc đúng theo pixel đã lưu, **bỏ qua metadata EXIF orientation** — ảnh có thể bị nghiêng 90°/180° khi hiển thị so với khi chụp. Marker góc ở Bước 2 vẫn phát hiện được (vì không phụ thuộc hướng ảnh), nhưng nếu cả 4 góc marker không tìm được, fallback contour ở Bước 3 có thể nhận sai layout do ảnh bị xoay.
- **Không giới hạn kích thước/định dạng đầu vào**: bất kỳ định dạng nào OpenCV hỗ trợ giải mã (JPEG, PNG, BMP, TIFF, WebP tùy build) đều được chấp nhận; không có kiểm tra kích thước tối đa trước khi đọc, nên ảnh rất lớn (ví dụ ảnh gốc 50MP) sẽ được đọc toàn bộ vào RAM trước khi các bước sau mới resize nhỏ lại.
