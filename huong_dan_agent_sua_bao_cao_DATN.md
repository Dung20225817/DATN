# Hướng dẫn agent sửa báo cáo DATN

## 1. Mục tiêu chỉnh sửa

Sửa file `DATN.md` từ dạng báo cáo thiên về sơ đồ sang dạng báo cáo giải thích rõ **quá trình hoạt động của hệ thống, cấu trúc mã nguồn, luồng xử lý thực tế, các thay đổi trong code và kết quả kiểm thử**.

Báo cáo cũ đã có khung chương mục đầy đủ, gồm Chương 1 đến Chương 6, trong đó Chương 4 hiện tập trung nhiều vào thiết kế, kiến trúc, package, sequence diagram, pipeline xử lý ảnh và kiểm thử. Nhiệm vụ chính là **không viết lại toàn bộ**, mà chỉnh mạnh các phần cần giải thích kỹ hơn, đặc biệt là Chương 4 và Chương 5.

File `README_cau_truc_thu_muc_mau_bao_cao.md` được dùng làm mẫu tham khảo về cách viết: trình bày cấu trúc thư mục, giải thích vai trò từng module, sau đó nối các module thành luồng hoạt động của hệ thống. Không được bê nguyên nội dung README mẫu vào báo cáo DATN vì README mẫu thuộc hệ thống dự báo chứng khoán, không phải hệ thống OMR.

---

## 2. Nguyên tắc sửa báo cáo

### 2.1. Giảm lạm dụng sơ đồ

Báo cáo hiện có nhiều sơ đồ như use case, kiến trúc three-tier, frontend package, backend package, ERD, sequence chấm một phiếu, sequence chấm hàng loạt, pipeline OMR, MCQ Map Search Rescue.

Agent cần xử lý theo nguyên tắc:

- Giữ lại các sơ đồ thật sự cần thiết.
- Không thêm sơ đồ mới nếu có thể giải thích bằng bảng hoặc mô tả luồng.
- Mỗi sơ đồ được giữ lại phải có đoạn giải thích rõ ràng ngay sau sơ đồ.
- Các sơ đồ trùng ý hoặc quá chi tiết nên chuyển thành bảng mô tả module, bảng input/output hoặc đoạn giải thích request flow.

### 2.2. Tăng giải thích quá trình hoạt động

Báo cáo cần trả lời được các câu hỏi sau:

- Người dùng thao tác gì trên frontend?
- Frontend gọi API nào?
- Backend nhận request ở router/file nào?
- Service nào xử lý nghiệp vụ?
- Module xử lý ảnh nào được gọi?
- Dữ liệu nào được đọc từ PostgreSQL?
- File nào được lưu vào `storage/uploads`?
- JSON trả về frontend gồm những thông tin gì?
- Kết quả được lưu lại như thế nào để xem thống kê hoặc export?

### 2.3. Tăng giải thích phần sửa code

Chương 5 không chỉ nêu “giải pháp” mà cần viết theo dạng:

1. Vấn đề gặp phải.
2. Nguyên nhân kỹ thuật.
3. Phần code/module đã sửa.
4. Cách sửa hoạt động như thế nào.
5. Kết quả sau khi sửa.
6. Test case hoặc bằng chứng kiểm thử liên quan.

---

## 3. Phạm vi chỉnh sửa ưu tiên

## 3.1. Chương 1 - Giới thiệu đề tài

Chỉ chỉnh nhẹ nếu cần.

Yêu cầu:

- Giữ nội dung đặt vấn đề, mục tiêu, phạm vi.
- Không thêm quá nhiều kỹ thuật chi tiết ở Chương 1.
- Đảm bảo phần định hướng giải pháp nói đúng hệ thống hiện tại: React frontend, FastAPI backend, OpenCV/NumPy xử lý ảnh, PostgreSQL lưu nghiệp vụ, file system lưu ảnh/template/crop.

Không nên:

- Mô tả quá sâu về code.
- Khẳng định đã dùng CNN/OCR trong luồng chấm chính nếu thực tế chưa dùng.

---

## 3.2. Chương 2 - Khảo sát và phân tích yêu cầu

Chương 2 có thể giữ phần use case nhưng nên kiểm tra lại tính nhất quán giữa use case và code.

Yêu cầu chỉnh:

- Giữ use case tổng quan nếu đã có.
- Với các use case quan trọng như đăng nhập, tạo bài thi, chấm điểm, xem thống kê, cần bổ sung hoặc giữ bảng:
  - Input.
  - Điều kiện hợp lệ.
  - API/module liên quan.
  - Kết quả đầu ra.
- Không nên để Chương 2 có quá nhiều biểu đồ phân rã nếu nội dung có thể diễn giải bằng bảng.

Nên bổ sung bảng dạng sau nếu chưa rõ:

```markdown
| Mã UC | Use case | Module triển khai | Dữ liệu chính | Kết quả |
|---|---|---|---|---|
| UC-01 | Đăng nhập | `auth.py`, `LoginPage.tsx` | email, password | Lưu user vào localStorage |
| UC-04 | Tạo bài thi | `assignments.py` | uid, title, profile | Bản ghi `omr_assignment` |
| UC-08 | Chấm 1 phiếu | `grading.py`, `omr_service.py` | image, uid, aid | JSON điểm, MSSV, mã đề, overlay |
```

---

## 3.3. Chương 3 - Công nghệ sử dụng

Chương 3 nên giải thích công nghệ theo lý do lựa chọn, không chỉ liệt kê định nghĩa.

Yêu cầu chỉnh:

- Với mỗi công nghệ, viết theo cấu trúc:
  - Công nghệ dùng để làm gì trong hệ thống.
  - Vì sao chọn công nghệ đó.
  - Công nghệ thay thế là gì.
  - Vì sao không chọn công nghệ thay thế.

Các công nghệ cần có:

- OpenCV.
- NumPy.
- FastAPI.
- PostgreSQL + SQLAlchemy.
- React + TypeScript + Vite.
- WebRTC MediaDevices API.
- PyTorch/OCR dependencies chỉ nên ghi là phục vụ hướng mở rộng nếu chưa dùng trong pipeline chính.

Không nên:

- Viết như giáo trình định nghĩa công nghệ.
- Nói hệ thống đã dùng học sâu để chấm bong bóng nếu thực tế pipeline chính dựa trên OpenCV/NumPy, marker, ROI và mật độ tô.

---

# 4. Hướng dẫn sửa Chương 4

Chương 4 là phần cần sửa mạnh nhất.

## 4.1. Đổi trọng tâm Chương 4

Chương 4 hiện có nhiều nội dung thiết kế. Cần chuyển trọng tâm thành:

> Thiết kế kiến trúc + cấu trúc mã nguồn + luồng hoạt động thực tế + pipeline xử lý ảnh + lưu trữ + kiểm thử.

Không chỉ nói “hệ thống có các package nào”, mà phải giải thích “request đi qua package/file nào và tạo ra kết quả gì”.

---

## 4.2. Thêm mục cấu trúc mã nguồn

Thêm một mục mới trong Chương 4, ví dụ:

```markdown
### 4.1.5 Cấu trúc mã nguồn hệ thống
```

Nội dung nên trình bày theo phong cách README mẫu: cây thư mục trước, giải thích từng thư mục sau.

Cây thư mục đề xuất:

```text
omr-grading-system/
│
├── fe/
│   ├── src/
│   │   ├── pages/
│   │   ├── features/omr/
│   │   │   ├── components/
│   │   │   ├── styles/
│   │   │   └── types/
│   │   ├── api/
│   │   └── auth/
│
├── be/
│   ├── main.py
│   ├── app/
│   │   ├── api/
│   │   │   ├── auth.py
│   │   │   └── omr/
│   │   │       ├── grading.py
│   │   │       ├── assignments.py
│   │   │       ├── profiles.py
│   │   │       └── grade_results.py
│   │   ├── services/
│   │   │   └── omr/
│   │   │       └── omr_service.py
│   │   ├── models/
│   │   ├── database/
│   │   └── core/
│
├── storage/
│   ├── uploads/
│   │   ├── omr/
│   │   ├── omr_templates/
│   │   ├── omr_data/profiles/
│   │   └── answer_keys/omr/
│   └── logs/
│
└── README.md
```

Sau cây thư mục, thêm bảng giải thích:

```markdown
| Thành phần | Vai trò |
|---|---|
| `fe/src/features/omr/` | Chứa giao diện và logic chính của chức năng chấm phiếu OMR |
| `fe/src/api/` | Gửi request từ frontend tới backend |
| `be/app/api/omr/grading.py` | Nhận request chấm một phiếu và chấm hàng loạt |
| `be/app/services/omr/omr_service.py` | Thực hiện pipeline xử lý ảnh, decode MSSV, mã đề, đáp án và tính điểm |
| `storage/uploads/omr/` | Lưu ảnh upload, ảnh overlay, crop, JSON confidence và ZIP batch |
| `storage/uploads/omr_data/profiles/` | Lưu Form Profile cấu hình ROI và tham số xử lý ảnh |
| PostgreSQL | Lưu user, bài thi, đáp án, template và lịch sử chấm |
```

---

## 4.3. Viết lại phần kiến trúc ba tầng

Giữ sơ đồ Three-Tier Architecture, nhưng sau sơ đồ cần giải thích bằng văn bản:

- Presentation Tier: React SPA, camera scanner, upload ảnh, hiển thị kết quả.
- Application Tier: FastAPI router, service nghiệp vụ, pipeline OpenCV/NumPy, MCQ Rescue.
- Data Tier: PostgreSQL và file system.

Cần nhấn mạnh:

- Frontend không xử lý chấm điểm chính, chỉ xử lý giao diện, camera scanner và gửi ảnh.
- Backend là nơi xử lý ảnh và tính điểm.
- PostgreSQL không lưu ảnh lớn trực tiếp, chỉ lưu metadata và đường dẫn file.
- `storage/uploads` lưu ảnh gốc, ảnh overlay, crop và file runtime.

---

## 4.4. Viết thêm luồng request chấm một phiếu

Thêm mục mới, ví dụ:

```markdown
### 4.3.x Luồng xử lý chấm một phiếu theo mã nguồn
```

Nội dung bắt buộc có:

```text
Giáo viên chọn/chụp ảnh
   ↓
Frontend kiểm tra bài thi và bộ đáp án
   ↓
Frontend tạo FormData gồm ảnh, uid, aid, profile_id hoặc thông tin liên quan
   ↓
Gửi POST /api/omr/grade
   ↓
Backend router `grading.py` nhận request
   ↓
Backend validate uid, bài thi, đáp án và profile
   ↓
Ảnh upload được lưu vào storage/uploads
   ↓
Gọi `process_omr_exam()` trong `omr_service.py`
   ↓
Pipeline OpenCV đọc ảnh, crop/warp, nhị phân hóa, tìm marker, suy luận ROI
   ↓
Decode MSSV, mã đề, đáp án MCQ
   ↓
Nếu nhiều câu không chắc chắn thì kích hoạt MCQ Map Search Rescue
   ↓
So sánh với answer_sets và tính điểm
   ↓
Lưu ảnh overlay, crop và JSON confidence
   ↓
Lưu bản ghi vào `omr_grade_result`, cập nhật `last_result`/`graded_count` nếu cần
   ↓
Trả JSON kết quả về frontend
   ↓
Frontend hiển thị điểm, MSSV, mã đề, ảnh overlay và chi tiết từng câu
```

Sau luồng này, thêm bảng:

```markdown
| Bước | File/module liên quan | Dữ liệu vào | Dữ liệu ra |
|---|---|---|---|
| Chọn ảnh | `MultichoicePage.tsx` | File ảnh/camera frame | `FormData` |
| Nhận request | `grading.py` | Multipart request | File tạm + tham số chấm |
| Xử lý ảnh | `omr_service.py` | Đường dẫn ảnh + profile | `OMRResult` |
| Lưu kết quả | `grade_results.py`, DB model | `OMRResult` | Bản ghi `omr_grade_result` |
| Hiển thị | React component | JSON response | Điểm, overlay, bảng đáp án |
```

---

## 4.5. Viết thêm luồng chấm hàng loạt

Nếu báo cáo đang có sequence diagram batch, có thể giữ hoặc rút gọn. Nhưng bắt buộc phải có mô tả bằng text:

- Người dùng chọn tối đa 50 ảnh.
- Frontend gửi `POST /api/omr/grade-batch`.
- Backend kiểm tra số lượng file.
- Lặp qua từng ảnh.
- Mỗi ảnh chạy chung pipeline với chấm đơn.
- Ảnh thành công được lưu kết quả.
- Ảnh lỗi được ghi vào danh sách lỗi.
- Backend trả về `success_count`, `failed_count`, `results`, `zip_url` nếu có.

Bổ sung bảng:

```markdown
| Trường kết quả batch | Ý nghĩa |
|---|---|
| `success_count` | Số ảnh chấm thành công |
| `failed_count` | Số ảnh lỗi |
| `results` | Danh sách kết quả từng ảnh |
| `zip_url` | Đường dẫn file ZIP chứa overlay |
```

---

## 4.6. Viết lại pipeline xử lý ảnh OMR

Pipeline 15 bước nên giữ vì đây là lõi kỹ thuật. Tuy nhiên mỗi bước cần được giải thích ngắn theo input/output.

Bảng bắt buộc:

```markdown
| Bước | Mục đích | Input | Output |
|---|---|---|---|
| Đọc ảnh | Nạp ảnh vào OpenCV | File ảnh | Ma trận ảnh BGR |
| Crop/warp | Chuẩn hóa phối cảnh phiếu | Ảnh gốc + marker/crop | Ảnh chuẩn 1000x1400 |
| Binarize | Tách vùng tô đen | Ảnh xám | Ảnh nhị phân |
| Detect marker | Xác định vị trí phiếu | Ảnh nhị phân | Tọa độ marker |
| Build ROI | Tạo vùng MSSV, mã đề, MCQ | Profile + marker | Danh sách ROI |
| Decode | Nhận diện MSSV, mã đề, đáp án | ROI | Kết quả đọc phiếu |
| Rescue | Hiệu chỉnh lưới khi lệch | MCQ baseline | MCQ result tốt hơn nếu có |
| Scoring | Tính điểm | Kết quả đọc + answer key | Điểm và chi tiết đúng/sai |
| Save overlay | Tạo bằng chứng trực quan | Ảnh chuẩn + kết quả | Ảnh overlay/crop/JSON |
```

---

## 4.7. Làm rõ Form Profile và ROI

Phần Form Profile cần giải thích theo hướng “vì sao cần” và “hoạt động ra sao”.

Cần có các ý:

- Form Profile là cấu hình giúp hệ thống hỗ trợ nhiều mẫu phiếu.
- Profile chứa vị trí marker, vùng MSSV, vùng mã đề, vùng câu hỏi, vùng họ tên, tham số decode.
- Backend đọc profile tại thời điểm chấm bài.
- ROI được quy đổi theo ảnh đã warp, không phải ảnh gốc.
- Khi thêm mẫu phiếu mới, có thể thêm profile mới thay vì sửa pipeline chính.

Bảng nên thêm:

```markdown
| Thành phần profile | Ý nghĩa |
|---|---|
| `corner_markers` | Vị trí 4 marker dùng để căn chỉnh phiếu |
| `crop_quad` | Tứ giác crop nếu cần cắt thủ công |
| `sid_roi` | Vùng đọc MSSV |
| `exam_code_roi` | Vùng đọc mã đề |
| `mcq_roi` | Vùng chứa câu hỏi trắc nghiệm |
| `mcq_decode` | Nhóm ngưỡng và tham số nhận diện bong bóng |
| `handwriting_fields` | Vùng crop chữ viết tay như họ tên |
```

---

## 4.8. Làm rõ thiết kế cơ sở dữ liệu

Không chỉ đặt ERD. Sau ERD, cần giải thích bảng nào được dùng trong bước nào của hệ thống.

Bảng bắt buộc:

```markdown
| Bảng | Vai trò trong luồng hoạt động |
|---|---|
| `users` | Xác định giáo viên đang dùng hệ thống |
| `omr_assignment` | Lưu bài thi, số câu, tổng điểm, nhiều bộ đáp án theo mã đề |
| `omr_test` | Lưu template/profile phiếu và metadata cấu hình |
| `omr_grade_result` | Lưu từng lượt chấm, điểm, MSSV, mã đề, đường dẫn overlay/crop và JSON kết quả |
```

Cần giải thích rõ:

- `answer_sets` dùng để lưu nhiều mã đề.
- `last_result` chỉ là kết quả gần nhất/tương thích giao diện cũ.
- `omr_grade_result` mới là bảng lưu lịch sử từng lượt chấm.
- Ảnh, crop, ZIP không nên lưu trực tiếp vào SQL, chỉ lưu đường dẫn.

---

## 4.9. Viết lại phần kiểm thử

Bảng kiểm thử hiện có thể giữ, nhưng cần liên kết với phần triển khai.

Yêu cầu:

- Mỗi test case nên có mã, chức năng, đầu vào, kỳ vọng, kết quả thực tế.
- Ưu tiên test các luồng chính:
  - Đăng nhập.
  - Tạo bài thi.
  - Thêm mã đề.
  - Chấm một ảnh.
  - Chấm ảnh có câu không rõ.
  - Cấu hình ROI.
  - Batch.
  - Camera.
  - Export.

Nếu có thể, thêm cột “module liên quan”:

```markdown
| Mã TC | Chức năng | Module liên quan | Đầu vào | Kết quả kỳ vọng | Kết quả thực tế |
|---|---|---|---|---|---|
```

---

# 5. Hướng dẫn sửa Chương 5

Chương 5 phải thể hiện đóng góp kỹ thuật và quá trình sửa code.

## 5.1. Cấu trúc bắt buộc cho từng đóng góp

Mỗi mục 5.x nên viết theo cấu trúc:

```markdown
### 5.x Tên giải pháp

#### 5.x.1 Vấn đề gặp phải
Mô tả lỗi hoặc hạn chế khi chạy hệ thống thực tế.

#### 5.x.2 Nguyên nhân kỹ thuật
Giải thích vì sao lỗi xảy ra: ảnh nghiêng, ROI lệch, ánh sáng yếu, tô mờ, marker thiếu, lưới MCQ sai, v.v.

#### 5.x.3 Cách sửa trong code
Nêu module/file đã sửa, hàm chính, logic chính.

#### 5.x.4 Kết quả sau khi sửa
Nêu kết quả đạt được và test case liên quan.
```

---

## 5.2. MCQ Map Search Rescue

Cần viết rõ:

- Vấn đề: lưới MCQ bị lệch làm nhiều câu không chắc chắn.
- Nguyên nhân: ảnh chụp thực tế không khớp hoàn toàn ROI cố định.
- Cách sửa: thêm cơ chế thử nhiều biến thể `line_scale`, `top_shift`, đánh giá ứng viên và chọn kết quả tốt hơn baseline.
- Module: `be/app/services/omr/omr_service.py`.
- Điều kiện kích hoạt: `uncertain_count` vượt ngưỡng.
- Kết quả: giảm số câu không chắc chắn nếu tìm được ứng viên tốt hơn.

Không nên chỉ viết chung chung “hệ thống tự động hiệu chỉnh”. Phải giải thích hiệu chỉnh bằng cách nào.

---

## 5.3. Thuật toán tính điểm bong bóng tổng hợp

Cần viết rõ:

- Vấn đề: chỉ dùng pixel đen dễ sai khi tô mờ, ảnh nhiễu hoặc nền không đều.
- Nguyên nhân: threshold nhị phân không phản ánh đầy đủ độ đậm thực tế.
- Cách sửa: kết hợp `density` từ ảnh nhị phân và `darkness` từ ảnh xám.
- Giải thích các đại lượng:
  - `density`.
  - `darkness`.
  - `margin`.
  - `best_score`.
  - `left_penalty` nếu có.
- Kết quả: nhận diện ổn định hơn với bong bóng tô nhẹ.

---

## 5.4. Smart Camera Scanner

Cần viết rõ:

- Vấn đề: giáo viên cần chụp nhanh bằng điện thoại, không muốn cài app native.
- Nguyên nhân kỹ thuật: trình duyệt cần HTTPS để dùng camera; phải xác định khi nào phiếu đủ marker và đủ sáng.
- Cách sửa: dùng WebRTC MediaDevices API, đọc video frame qua canvas, tính độ sáng trung tâm và tỷ lệ pixel tối tại 4 marker.
- Điều kiện locked:
  - Đủ 4 marker tối.
  - Vùng giấy đủ sáng.
  - Marker có tương phản so với nền.
  - Duy trì nhiều frame liên tiếp.
- Kết quả: tự động chụp khi phiếu nằm đúng khung.

---

# 6. Các bảng nên thêm vào báo cáo

## 6.1. Bảng input/output chức năng

```markdown
| Chức năng | Input | Xử lý chính | Output |
|---|---|---|---|
| Đăng nhập | Email, password | Kiểm tra tài khoản | Lưu user vào localStorage |
| Tạo bài thi | Title, profile, số câu, tổng điểm | Lưu `omr_assignment` | Bài thi mới |
| Nhập đáp án | Mã đề, danh sách đáp án | Lưu `answer_sets` | Bộ đáp án theo mã đề |
| Chấm một ảnh | Ảnh, uid, aid, profile | Pipeline OMR | Điểm, MSSV, mã đề, overlay |
| Chấm hàng loạt | Danh sách ảnh | Lặp pipeline từng ảnh | Danh sách kết quả, ZIP overlay |
| Xem thống kê | Bài thi, lịch sử chấm | Tính trung bình, cao nhất, thấp nhất | Bảng/biểu đồ thống kê |
| Xuất kết quả | Danh sách bản ghi | Tạo file Excel/PDF | File tải về |
```

## 6.2. Bảng API chính

```markdown
| API | Method | Chức năng | Input chính | Output chính |
|---|---|---|---|---|
| `/api/omr/grade` | POST | Chấm một phiếu | image, uid, aid | OMRResult |
| `/api/omr/grade-batch` | POST | Chấm nhiều phiếu | images, uid, aid | Batch result |
| `/api/omr/assignments/{uid}` | GET | Lấy danh sách bài thi | uid | Danh sách bài thi |
| `/api/omr/assignments` | POST | Tạo bài thi | title, profile | Assignment mới |
| `/api/omr/suggest-crop` | POST | Gợi ý vùng crop | image | Tọa độ 4 góc |
```

## 6.3. Bảng file runtime

```markdown
| Loại file | Vị trí lưu | Mục đích |
|---|---|---|
| Ảnh upload | `storage/uploads/omr/` | Lưu ảnh gốc người dùng gửi |
| Ảnh overlay | `storage/uploads/omr/` | Hiển thị kết quả chấm trực quan |
| Crop MSSV | `storage/uploads/omr/` | Kiểm tra vùng nhận diện MSSV |
| Crop MCQ | `storage/uploads/omr/` | Kiểm tra vùng câu hỏi |
| Confidence JSON | `storage/uploads/omr/` | Lưu thông tin độ tin cậy từng bong bóng |
| Template | `storage/uploads/omr_templates/` | Lưu ảnh/PDF mẫu phiếu |
| Profile JSON | `storage/uploads/omr_data/profiles/` | Lưu cấu hình ROI và tham số xử lý |
```

---

# 7. Quy tắc viết lại văn phong

Agent cần viết theo văn phong báo cáo kỹ thuật, dễ hiểu, tránh quá quảng cáo.

Nên dùng:

- “Hệ thống thực hiện...”
- “Backend nhận request tại...”
- “Module này có nhiệm vụ...”
- “Kết quả được lưu vào...”
- “Cơ chế này được kích hoạt khi...”

Không nên dùng quá nhiều:

- “Tối ưu vượt trội”.
- “Hiện đại nhất”.
- “Thông minh hoàn toàn”.
- “AI tự động” nếu không có AI/model thật trong luồng đó.

---

# 8. Checklist đầu ra cho agent

Sau khi sửa báo cáo, agent phải đảm bảo các mục sau đã có:

- [ ] Có mục cấu trúc mã nguồn hệ thống.
- [ ] Có bảng giải thích từng thư mục/module.
- [ ] Có luồng chấm một phiếu theo mã nguồn.
- [ ] Có luồng chấm hàng loạt theo mã nguồn.
- [ ] Có bảng input/output cho các chức năng chính.
- [ ] Có bảng API chính.
- [ ] Có bảng file runtime/storage.
- [ ] Có giải thích Form Profile và ROI.
- [ ] Có giải thích bảng database theo vai trò trong luồng hoạt động.
- [ ] Có giải thích pipeline OMR theo input/output từng bước.
- [ ] Chương 5 viết theo dạng vấn đề → nguyên nhân → sửa code → kết quả.
- [ ] Không mô tả sai rằng CNN/OCR đã được dùng trong pipeline chấm chính nếu chưa triển khai.
- [ ] Không thêm quá nhiều sơ đồ mới.
- [ ] Mỗi sơ đồ còn lại đều có đoạn giải thích sau sơ đồ.
- [ ] Bảng kiểm thử có liên hệ với module/chức năng đã triển khai.

---

# 9. Prompt gợi ý để giao cho agent

Có thể dùng prompt sau để yêu cầu agent sửa báo cáo:

```text
Bạn là agent hỗ trợ chỉnh sửa báo cáo đồ án tốt nghiệp.

Tôi có file DATN.md là báo cáo cũ của hệ thống chấm điểm phiếu trắc nghiệm bằng hình ảnh. Báo cáo hiện bị nhận xét là có quá nhiều sơ đồ, chưa giải thích đủ quá trình hoạt động thực tế và phần sửa code. Tôi có file README_cau_truc_thu_muc_mau_bao_cao.md làm mẫu về cách trình bày cấu trúc thư mục, giải thích từng module và mô tả luồng hoạt động.

Nhiệm vụ của bạn:
1. Đọc DATN.md và README mẫu.
2. Không viết lại toàn bộ báo cáo từ đầu.
3. Tập trung sửa mạnh Chương 4 và Chương 5.
4. Bổ sung mục cấu trúc mã nguồn hệ thống theo cây thư mục và bảng giải thích module.
5. Bổ sung luồng chấm một phiếu và chấm hàng loạt theo mã nguồn: frontend → API → service → OpenCV pipeline → database/storage → response.
6. Bổ sung bảng input/output chức năng, bảng API chính, bảng file runtime và bảng vai trò database.
7. Giữ các sơ đồ quan trọng, nhưng giảm sơ đồ không cần thiết. Mỗi sơ đồ giữ lại phải có đoạn giải thích rõ.
8. Viết Chương 5 theo cấu trúc: vấn đề gặp phải → nguyên nhân kỹ thuật → cách sửa trong code → kết quả sau khi sửa → test case liên quan.
9. Không được mô tả sai rằng hệ thống đã dùng CNN/OCR trong pipeline chấm chính nếu thực tế pipeline chính đang dùng OpenCV/NumPy, marker, ROI và mật độ tô.
10. Trả về file DATN.md đã chỉnh sửa và một bản tóm tắt các phần đã thay đổi.

Ưu tiên làm báo cáo rõ ràng, thực tế, có tính kỹ thuật và thể hiện được công sức triển khai/sửa code.
```

---

# 10. Kết luận

Hướng sửa phù hợp nhất là biến Chương 4 thành phần mô tả **hệ thống được tổ chức và vận hành như thế nào trong code**, còn Chương 5 thành phần mô tả **những vấn đề kỹ thuật đã gặp và cách sửa trong code**. README mẫu chỉ dùng để học cách trình bày cấu trúc thư mục và luồng module, không dùng để thay thế nội dung chuyên môn của hệ thống OMR.
