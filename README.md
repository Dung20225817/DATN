# Hệ Thống Chấm Điểm Essay Viết Tay

**Stack**: FastAPI + React + PostgreSQL + VietOCR + Llama3  
**Python**: 3.10 | **Node**: 18+

---

## Mục Lục

1. [Giới thiệu](#giới-thiệu)
2. [Tính năng](#tính-năng)
3. [Kiến trúc & Thư mục](#kiến-trúc--thư-mục)
4. [Database](#database)
5. [Cài đặt](#cài-đặt)
6. [Luồng hoạt động](#luồng-hoạt-động)
7. [Cách tính điểm](#cách-tính-điểm)
8. [API Endpoints](#api-endpoints)
9. [Hướng dẫn sử dụng](#hướng-dẫn-sử-dụng)
10. [Troubleshooting](#troubleshooting)

---

## Giới thiệu

Hệ thống tự động chấm điểm bài tập viết tay tiếng Việt gồm 2 tính năng chính:

| Tính năng | Mô tả |
|----------|-------|
| **Chấm essay viết tay** | Upload ảnh bài làm → OCR → AI chấm điểm theo đáp án |
| **Chấm trắc nghiệm (OMR)** | Upload ảnh phiếu → nhận diện tô đáp án → chấm tự động |

---

## Tính năng

### Chấm Essay Viết Tay (HandwrittenQuestionPage)

- Upload đáp án mẫu định dạng `.docx` hoặc `.pdf`
- Upload 1 hoặc nhiều ảnh bài làm viết tay
- **Chọn model OCR/VLM** từ 6 engine khác nhau (dropdown)
- **Hai luồng xử lý khác nhau**:
  - **VLM mode** (Qwen2VL, InternVL, LLaVA): Dùng toàn bộ ảnh gốc → AI vision evaluate trực tiếp
  - **VietOCR mode**: CRAFT line detection + Linear Regression baseline + Perspective deskew → crop từng dòng → VietOCR nhận diện
- Hỗ trợ cả ảnh có đường kẻ lẫn ảnh không có đường kẻ (blank paper)
- AI làm sạch text (sửa lỗi OCR) bằng **Llama3** qua Ollama
- Tự động tách câu hỏi (fallback "Câu 1" khi không tìm thấy marker)
- Chấm điểm từng câu theo **tiêu chí chi tiết** (Nội dung, Hình thức, ...)
- Hiển thị:
  - Điểm tổng + thanh tiến độ màu sắc theo ngưỡng
  - **Breakdown từng tiêu chí**: điểm/max, ✅ ý đạt, ❌ ý thiếu
  - Dòng OCR gốc thu được từ ảnh (có thể mở/đóng)
  - Đáp án đã dùng để so sánh (có thể mở/đóng)
  - Nhận xét tổng hợp từ AI

### Chấm Trắc Nghiệm (MultichoicePage)

- Nhận diện phiếu trắc nghiệm bằng OpenCV
- Phát hiện vùng bài làm, biến đổi phối cảnh (warp perspective)
- Đếm pixel tô để xác định đáp án học sinh chọn
- So sánh với đáp án → tính điểm + vẽ kết quả lên ảnh

---

## Phiên Bản Mới — Nâng Cấp (v2.1)

### Thêm mới

- **CRAFT Text Detection**: Phát hiện text region chính xác (polygon 4 góc)
- **Linear Regression Baseline**: Gộp region thành dòng, xử lý chữ nghiêng
- **Perspective Deskew**: Xoay text về nằm ngang (0°) trước OCR
- **VLM Models**: Qwen2VL, InternVL, LLaVA — dùng full image
- **Debug Crop List**: Lưu ảnh crop từng dòng vào uploads/croplist/

### Loại bỏ

- **GOT-OCR 2.0**: API chậm, không batch
- **EasyOCR**: Không tối ưu chữ viết tay Việt
- **TrOCR**: Kết quả kém trên chữ viết tay
- **DBNet (mmocr)**: Yêu cầu Visual C++ Build Tools

### Kiến Trúc: Cũ vs Mới

**Cũ**: HybridLineDetector (Hough) → axis-aligned crop → VietOCR
- Đơn giản, đủ dùng
- Xử lý chữ nghiêng yếu

**Mới**: CRAFT → Linear Regression → Perspective Deskew → VietOCR
- Chính xác cao, xử lý chữ phức tạp
- Fallback HybridLineDetector nếu CRAFT fail

---

## Kiến trúc & Thư mục

```
OCR_CRNN/
├── be/                              Backend (FastAPI)
│   ├── main.py                      Entry point - khởi server, tạo bảng DB tự động
│   ├── requirements.txt
│   ├── app/
│   │   ├── db_connect.py            Kết nối PostgreSQL, Base SQLAlchemy
│   │   ├── logging_config.py        Cấu hình logging (rotate theo ngày)
│   │   ├── api/
│   │   │   ├── auth.py              Đăng nhập / Đăng ký
│   │   │   ├── handwritten_load_picture.py   API chấm essay viết tay
│   │   │   └── omr_grading.py       API chấm trắc nghiệm
│   │   ├── db/
│   │   │   ├── table.py             Model: User, Picture
│   │   │   └── ocr_tables.py        Model: AnswerKey, EssaySubmission
│   │   └── services/
│   │       ├── handwritten_services.py    Pipeline xử lý chính
│   │       ├── ocr/
│   │       │   ├─ reader.py        OCRReader (EasyOCR), OCRReader2 (VietOCR), OCRReader3 (fine-tuned HF)
│   │       │   ├─ line_detector.py HybridLineDetector (Hough angle≤15° + Projection fallback)
│   │       │   └─ essay_grading/
│   │       │       └─ answer_parser.py  Parse file đáp án DOCX/PDF
│   │       └─ llm/
│   │           └─ grading_engine.py    Llama3GradingEngine (+ regex fallback, criteria_breakdown)
│   ├── logs/                        Log files (app_YYYYMMDD.log)
│   └── uploads/
│       ├── answer_keys/             File đáp án đã upload
│       ├── handwritten/             Ảnh bài essay
│       ├── omr/                     Ảnh phiếu trắc nghiệm
│       └── temp/                    Temp files
│
└── fe/                              Frontend (React + TypeScript + Vite)
    └── src/
        ├── config/api.ts            Cấu hình URL API (mặc định: localhost:8000)
        ├── Page_Components/
        │   ├── LoginPage.tsx
        │   ├── RegisterPage.tsx
        │   ├── HomePage.tsx
        │   ├── HandwrittenQuestionPage.tsx   Trang chấm essay (2 bước)
        │   └── MultichoicePage.tsx           Trang chấm trắc nghiệm
        └── UI_Components/
            ├── TopMenu.tsx
            ├── ImageUploader.tsx
            ├── ImagePreviewList.tsx
            ├── UploadPopup.tsx
            └── ViewImageModal.tsx
```

---

## Database

Sử dụng **PostgreSQL** + **SQLAlchemy ORM**. Bảng được **tạo tự động** khi server khởi động (`Base.metadata.create_all()`).

### Bảng `users`

| Cột | Kiểu | Mô tả |
|-----|------|-------|
| `uuid` | Integer PK | User ID |
| `user_name` | String | Tên đăng nhập |
| `email` | String | Email |
| `phone` | String | Số điện thoại |
| `password` | String | Mật khẩu |

### Bảng `pictures`

| Cột | Kiểu | Mô tả |
|-----|------|-------|
| `pid` | Integer PK | Picture ID |
| `p_name` | String | Tên file |
| `save_time` | Timestamp | Thời gian upload |
| `uuid` | Integer FK → users | Người upload |

### Bảng `answer_keys`

Lưu file đáp án mẫu mà giáo viên upload.

| Cột | Kiểu | Mô tả |
|-----|------|-------|
| `id` | Integer PK | ID đáp án |
| `uuid` | Integer FK → users | Giáo viên upload |
| `file_name` | String | Tên file gốc (vd: `Dap_an_Van.docx`) |
| `file_path` | String | Đường dẫn lưu trên server |
| `rubric_json` | JSON | Tiêu chí chấm điểm chi tiết |
| `sample_answers_json` | JSON | Đáp án mẫu: `{"Câu 1": "...", "Câu 2": "..."}` |
| `keywords_json` | JSON | Từ khóa bắt buộc theo câu |
| `total_points` | Float | Tổng điểm tối đa |
| `num_questions` | Integer | Số câu |
| `questions` | JSON | List tên câu: `["Câu 1", "Câu 2"]` |
| `is_deleted` | Integer | Soft delete: `0` = active, `1` = đã xóa |
| `deleted_at` | DateTime | Thời điểm xóa |
| `created_at` | DateTime | Thời điểm tạo |
| `updated_at` | DateTime | Lần cập nhật cuối |

### Bảng `essay_submissions`

Lưu bài làm và kết quả chấm điểm.

| Cột | Kiểu | Mô tả |
|-----|------|-------|
| `id` | Integer PK | ID bài nộp |
| `uuid` | Integer FK → users | Học sinh |
| `answer_key_id` | Integer FK → answer_keys | Đáp án được dùng để chấm |
| `image_paths` | JSON | Đường dẫn các ảnh bài làm |
| `ocr_text_raw` | Text | Text OCR thô (chưa xử lý) |
| `ocr_text_raw_json` | JSON | OCR chi tiết: `[{line_num, text, confidence}]` |
| `cleaned_text` | Text | Text sau khi Llama3 làm sạch |
| `segmented_answers` | JSON | Đáp án đã tách câu: `{"Câu 1": "...", ...}` |
| `score` | Float | Tổng điểm |
| `question_scores` | JSON | Điểm từng câu: `{"Câu 1": 8.5, "Câu 2": 7.0}` |
| `feedback` | Text | Nhận xét tổng quát |
| `detailed_feedback` | JSON | Feedback chi tiết từng câu |
| `status` | String | `pending` / `processing` / `completed` / `error` |
| `processing_error` | Text | Lỗi nếu status = error |
| `is_deleted` | Integer | Soft delete: `0` = active, `1` = đã xóa |
| `deleted_at` | DateTime | Thời điểm xóa |
| `created_at` | DateTime | Thời điểm nộp bài |
| `updated_at` | DateTime | Lần cập nhật cuối |

---

## Cài đặt

### Backend

```bash
# 1. Vào thư mục backend
cd be

# 2. Kích hoạt virtual environment
.\venv310\Scripts\Activate.ps1           # Windows PowerShell
# source venv310/bin/activate            # Linux/macOS

# 3. Cài dependencies
pip install -r requirements.txt

# 3a. Cài thêm dependencies cho CRAFT + VLM (tùy theo model chọn)
# CRAFT (bắt buộc nếu dùng VietOCR mode):
pip install craft-text-detector

# VLM Models (tùy chọn, chọn 1 trong các cái dưới):
# - Qwen2VL (~4GB VRAM - chính xác cao):
pip install transformers accelerate qwen-vl-utils pillow

# - InternVL (~4GB VRAM - độ phân giải cao):
pip install transformers accelerate pillow

# - LLaVA (~1GB VRAM - nhẹ nhất):
pip install git+https://github.com/haotian-liu/LLaVA.git

# 4. Cấu hình database trong app/db_connect.py
DATABASE_URL = "postgresql://postgres:1111@localhost:5432/postgres"
# Dev (SQLite): DATABASE_URL = "sqlite:///./ocr_grading.db"

# 5. Khởi động server (bảng DB tự tạo khi start)
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --app-dir .
```

### Ollama (tùy chọn — cần cho chấm điểm AI)

```bash
# Tải từ https://ollama.ai rồi cài đặt, sau đó:
ollama pull llama3
ollama serve      # Chạy trên port 11434

# Nếu không cài Ollama → hệ thống tự dùng regex fallback
```

### Frontend

```bash
cd fe
npm install
npm run dev       # Dev server tại http://localhost:5173
```

---

## Luồng hoạt động

### Chấm Essay Viết Tay

```
BƯỚC 1 — Upload đáp án
────────────────────────────────────────────────────
Giáo viên chọn file .docx/.pdf
    │
    ▼
POST /api/handwritten/upload-answer-key
    │
    ├─ Lưu file vào uploads/answer_keys/
    │
    ├─ answer_parser.py phân tích file:
    │   ├─ Format đầy đủ (nếu có section RUBRIC / SAMPLE_ANSWER / KEYWORDS):
    │   │    ├─ extract_rubric()         → tiêu chí điểm từng câu
    │   │    ├─ extract_sample_answers() → đáp án mẫu
    │   │    └─ extract_keywords()       → từ khóa bắt buộc
    │   │
    │   └─ Format linh hoạt (fallback — khi không có RUBRIC):
    │        └─ extract_flexible()  → tìm pattern "Câu X"
    │             → điểm mặc định 10đ/câu, không có rubric chi tiết
    │
    └─ Lưu vào bảng answer_keys → trả về answer_key_id
         (frontend tự động chọn đáp án vừa upload)


BƯỚC 2 — Upload ảnh & chấm điểm
────────────────────────────────────────────────────
Giáo viên chọn ảnh bài làm → nhấn "Gửi ảnh"
    │
    ▼
POST /api/handwritten/upload
    │
    ├─ [1] Lưu ảnh vào uploads/handwritten/{uid}/
    │
    ├─ [2] CRAFT Line Detection + Linear Regression + Perspective Deskew
    │       
    │       **CRAFT (Character Region Awareness For Text)**
    │       ├─ Input: Ảnh bất kỳ kích cỡ
    │       ├─ Xuất xứ: Naver Labs (Korea), CVPR 2019
    │       ├─ Architecture: ResNet backbone + 2 detection head:
    │       │    • Character map: tính xác suất từng pixel thuộc ký tự
    │       │    • Affinity map: tính mối liên kết giữa ký tự liền kề (ngang)
    │       ├─ Workflow:
    │       │    1. Feature extraction qua ResNet18
    │       │    2. Tính character map + affinity map
    │       │    3. Threshold & component labeling (từng ký tự riêng)
    │       │    4. Dùng affinity để ghép ký tự thành từ/region
    │       │    5. Vẽ polygon bao quanh mỗi region (4 góc chính xác)
    │       └─ Output: list polygon `[(x1,y1,x2,y2,...), ...]`
    │
    │       **Linear Regression Baseline**
    │       ├─ Gộp các polygon CRAFT vào dòng text bằng:
    │       │    1. Sắp xếp polygon top-to-bottom theo y_center
    │       │    2. Với polygon mới, fit linear regression trên center của
    │       │       các polygon đã gộp: y = slope×x + b
    │       │    3. Tính tolerance = max(avg_height, new_box_height) × 0.6
    │       │    4. Nếu |y_actual - y_predicted| < tolerance → cùng dòng
    │       │    5. Sau gộp: sắp xếp các box trong dòng theo x1 (left→right)
    │       │       để VietOCR đọc đúng thứ tự chữ
    │       └─ Output: List[Tuple[List[boxes], float(slope)]]
    │
    │       **Perspective Deskew (Xoay chữ nghiêng)**
    │       ├─ Tính góc nghiêng: angle = degrees(arctan(slope))
    │       ├─ Nếu |angle| < 1.0° → crop thẳng (bỏ qua noise từ ít box)
    │       ├─ Nếu |angle| ≥ 1.0° → Rotate ảnh quanh tâm crop:
    │       │    - OpenCV getRotationMatrix2D(center, +angle, 1.0)
    │       │    - Góc dương = xoay ngược chiều kim đồng hồ (CCW)
    │       │    - BORDER_REPLICATE tránh viền đen
    │       └─ Crop từ ảnh đã xoay → text nằm ngang (0°)
    │
    │       ├─ Ưu tiên fallback: Nếu CRAFT fail → HybridLineDetector
    │       │    (Hough angle≤15° + Horizontal Projection)
    │       └─ Lưu crops debug vào uploads/croplist/{stem}/ cho kiểm tra
    │
    ├─ [3] OCR từng dòng crop — chọn qua tham số ocr_model
    │       
    │       **VietOCR models** (Sequence-to-sequence CRNN):
    │       ├─ vietocr_finetuned (mặc định):
    │       │    • Model: DungHugging/vietocr-handwritten-finetune
    │       │    • Nguồn: HuggingFace hub, fine-tune trên chữ viết tay
    │       │    • Input: (H=32, W=variable) fixed height
    │       │    • Backbone: ResNet hoặc MobileNet
    │       │    • Decoder: GRU + Attention + CTC loss
    │       ├─ vietocr_transformer:
    │       │    • Architecture: vgg_transformer + BeamSearch (K=3)
    │       │    • Chậm hơn seq2seq nhưng chính xác hơn
    │       │    • Phù hợp chữ viết tay phức tạp
    │       └─ vietocr_seq2seq:
    │            • Architecture: vgg_seq2seq + fast decode
    │            • Nhanh, phù hợp chữ viết sáng sủa
    │
    │       **VLM models** (Vision Language Model — full image mode):
    │       ├─ qwen2vl (Toàn ảnh gốc ~4GB VRAM):
    │       │    • Qwen2-VL-32B (HuggingFace quan_3 quantized)
    │       │    • Nhận diện không qua line detection
    │       │    • Tốt với bài viết tay phức tạp, hình vẽ, bảng
    │       ├─ internvl (Toàn ảnh gốc ~4GB VRAM):
    │       │    • InternVL2-8B (high resolution support)
    │       │    • Độ phân giải cao, chi tiết
    │       └─ llava (Toàn ảnh gốc ~1GB VRAM):
    │            • LLaVA-1.6 quantized (nhẹ nhất)
    │            • Nhanh, dùng CPU acceptably
    │
    │       → [{text, y, line_num}, ...]
    │
    ├─ [4] Text cleanup (use_llama3=True)
    │       ├─ Llama3 qua Ollama (port 11434):
    │       │    Prompt: sửa lỗi OCR, thêm dấu câu, viết hoa đúng
    │       └─ Fallback (không có Ollama): regex strip extra spaces
    │
    ├─ [5] QuestionSplitter — Tách câu hỏi
    │       Nhận diện đầu câu theo các pattern:
    │         "Câu X" / "Câu X:" / "X." / "Bài X" / "Question X"
    │       Fallback: nếu không tìm thấy marker nào →
    │         toàn bộ text gộp vào "Câu 1" tự động
    │       → [{id: "Câu 1", content: "toàn bộ text câu đó", ocr_lines: [...]}, ...]
    │
    ├─ [6] Llama3GradingEngine — Chấm điểm (nếu có answer_key_id)
    │       ├─ Llama3 so sánh bài làm với đáp án mẫu:
    │       │    Phân tích ý chính → kiểm tra ý đạt → điểm 0-10 + feedback
    │       └─ Fallback: keyword matching + length ratio
    │
    └─ [7] Trả kết quả về frontend
            {
              "results": [{
                "file": "essay_1.jpg",
                "questions": [
                  {
                    "id": "Câu 1",
                    "content": "Nội dung dã xử lý...",
                    "ocr_lines": ["dòng 1 thô", "dòng 2 thô", ...],
                    "score": 8.5,
                    "max_score": 10,
                    "feedback": "...",
                    "criteria_breakdown": {
                      "Nội dung": {"score": 3.33, "max_points": 3.33,
                                   "matched": ["Ý 1", "Ý 2"], "missing": []},
                      "Hình thức": {"score": 5.0, "max_points": 6.67,
                                    "matched": ["Lập luận chặt chẽ"], "missing": ["Bố cục rõ"]}
                    },
                    "answer_key_used": { ... }   // rubric đã dùng để chấm
                  }
                ],
                "total_score": 77.5
              }],
              "processing_log": ["✅ Detected 12 lines", "✅ OCR completed", ...]
            }
```

### Chấm Trắc Nghiệm (OMR)

```
POST /api/omr/grade
    │
    ├─ Grayscale → Binary Inverse thresholding
    ├─ Tìm contour phiếu → Warp Perspective (top-down view)
    ├─ Chia lưới ô (Hàng = câu, Cột = lựa chọn A/B/C/D)
    ├─ cv2.countNonZero mỗi ô → ô có nhiều pixel trắng nhất = đáp án chọn
    └─ So sánh đáp án chọn với answer key → tính điểm
       + Vẽ vòng tròn xanh (đúng) / đỏ (sai) lên ảnh gốc (inverse warp)
```

---

## Cách tính điểm

### Khi có Llama3 + rubric

Llama3 so sánh bài làm với từng ý trong đáp án theo tiêu chí. Kết quả gồm `criteria_breakdown` chi tiết:

```
Điểm tiêu chí = Σ điểm các ý đạt được trong tiêu chí đó
Điểm câu X   = Σ điểm tất cả tiêu chí
```

Ví dụ (10 điểm, 2 tiêu chí):

| Tiêu chí | Ý đạt | Ý thiếu | Điểm |
|----------|-------|---------|------|
| Nội dung (3.33đ) | 2/2 | 0 | 3.33 |
| Hình thức (6.67đ) | 3/4 | 1 | 5.00 |
| **Tổng** | | | **8.33** |

### Khi không có Llama3 (regex fallback)

```
keyword_score = (Số từ trong bài khớp với đáp án / Tổng từ trong đáp án) × 10

length_ratio  = min(độ_dài_bài_làm / độ_dài_đáp_án, 1.0)

Điểm câu X   = keyword_score × (0.7 + 0.3 × length_ratio)
```

Tức là: **70%** từ khóa khớp + **30%** bù trừ từ độ dài bài làm.

### Tổng điểm

```
Tổng điểm = Σ điểm từng câu
```

Hiển thị dạng `X / 100`, màu sắc theo ngưỡng:

| Điểm | Màu |
|------|-----|
| ≥ 80 | Xanh lá |
| 60–79 | Cam |
| < 60 | Đỏ |

### Format file đáp án

**Format đầy đủ** — chấm chính xác nhất (Llama3 dùng rubric để đánh giá):

```
RUBRIC
Câu 1 - Nghị luận văn học (10 điểm)
Nội dung (6 điểm):
  - Ý 1 [2 điểm]
  - Ý 2 [2 điểm]
  - Ý 3 [2 điểm]
Hình thức (4 điểm):
  - Lập luận chặt chẽ [2 điểm]
  - Bố cục rõ ràng [2 điểm]

SAMPLE_ANSWER - MẪU ĐÁP ÁN
Câu 1: Toàn bộ nội dung đáp án mẫu câu 1...

Câu 2: Toàn bộ nội dung đáp án mẫu câu 2...

KEYWORDS - TỪ KHÓA PHẢI CÓ
Câu 1:
  - từ khóa 1
  - từ khóa 2
```

**Format đơn giản** — hệ thống tự nhận diện, điểm mặc định **10đ/câu**:

```
Câu 1: Nội dung đáp án câu 1...

Câu 2: Nội dung đáp án câu 2...
```

---

## API Endpoints

### Xác thực

| Method | Endpoint | Mô tả |
|--------|----------|-------|
| POST | `/api/login` | Đăng nhập |
| POST | `/api/register` | Đăng ký tài khoản |

### Quản lý đáp án

| Method | Endpoint | Mô tả |
|--------|----------|-------|
| POST | `/api/handwritten/upload-answer-key` | Upload file đáp án (docx/pdf) |
| GET | `/api/handwritten/answer-keys/{uid}` | Danh sách đáp án của user |
| GET | `/api/handwritten/answer-key/{id}?uid={uid}` | Chi tiết 1 đáp án |
| DELETE | `/api/handwritten/answer-key/{id}?uid={uid}` | Xóa đáp án (soft delete) |

### Chấm điểm

| Method | Endpoint | Mô tả |
|--------|----------|-------|
| POST | `/api/handwritten/upload` | Upload ảnh bài essay + chấm điểm |
| POST | `/api/omr/grade` | Chấm phiếu trắc nghiệm |

#### Tham số `POST /api/handwritten/upload`

| Tham số | Kiểu | Mặc định | Mô tả |
|---------|------|----------|-------|
| `uid` | int | bắt buộc | User ID |
| `essay_images` | File[] | bắt buộc | Ảnh bài làm (jpg/png) |
| `answer_key_id` | int | null | ID đáp án để chấm, bỏ qua nếu chỉ muốn OCR |
| `ocr_model` | string | `vietocr_finetuned` | Model OCR (xem bảng bên dưới) |
| `use_llama3` | bool | true | Dùng Llama3 để cleanup text và chấm điểm |
| `use_gpu` | bool | false | Dùng GPU NVIDIA (CUDA) |

**Các giá trị `ocr_model` hợp lệ:**

| Giá trị | Engine | Đặc điểm |
|---------|--------|-----------|
| `vietocr_finetuned` | OCRReader3 (HuggingFace) | Chính xác nhất cho chữ viết tay, tải ~500MB lần đầu |
| `vietocr_transformer` | OCRReader2 vgg_transformer | Chính xác cao, chậm hơn |
| `vietocr_seq2seq` | OCRReader2 vgg_seq2seq | Nhanh, phù hợp chữ in |
| `easyocr` | EasyOCR | Đa ngôn ngữ, không cần cài VietOCR |

---

## Hướng dẫn sử dụng

### Chấm essay viết tay (2 bước)

**BƯỚC 1 — Upload đáp án:**
1. Vào trang **Chấm Essay Viết Tay**
2. Chuẩn bị file đáp án `.docx` hoặc `.pdf` (xem format ở trên)
3. Chọn file tại ô **BƯỚC 1: Tải đáp án**
4. Sau khi upload thành công → đáp án hiện trong dropdown và tự động được chọn

**BƯỚC 2 — Upload bài làm:**
1. Sau khi chọn đáp án, ô **BƯỚC 2** được kích hoạt (viền xanh lá)
2. Click ô upload → chọn ảnh bài làm viết tay (1 hoặc nhiều ảnh)
3. Xem trước, xóa ảnh không cần nếu muốn

**BƯỚC 3 — Tùy chọn & Gửi:**
1. Điều chỉnh mục **Tùy chọn nâng cao** nếu cần:
   - **Model OCR** (dropdown): chọn engine phù hợp
     - *Fine-tuned VietOCR*: tốt nhất cho chữ viết tay, cần tải ~500MB lần đầu
     - *VietOCR vgg_transformer*: chính xác, không cần tải thêm
     - *VietOCR vgg_seq2seq*: nhanh, dùng cho chữ in
     - *EasyOCR*: đa ngôn ngữ
   - *Llama3 AI*: cần Ollama đang chạy trên port 11434
   - *GPU*: cần card NVIDIA với CUDA
2. Nhấn **Gửi ảnh** → đợi 5–30 giây
3. Xem kết quả:
   - Tổng điểm + thanh tiến độ màu
   - Điểm từng tiêu chí (Nội dung, Hình thức, ...) với ✅/❌ từng ý
   - Mở **"Dòng OCR gốc"** để xem từng dòng đọc được từ ảnh
   - Mở **"Đáp án dùng để so sánh"** để xem rubric đã dùng
   - Nhận xét tổng hợp từ AI

### Chấm trắc nghiệm

1. Vào trang **Chấm Trắc Nghiệm**
2. Upload ảnh phiếu trắc nghiệm đã tô
3. Nhập đáp án chuẩn
4. Nhấn chấm → xem kết quả với ảnh đánh dấu đúng/sai

---

## Troubleshooting

| Lỗi | Nguyên nhân | Giải pháp |
|-----|-------------|-----------|
| `relation "answer_keys" does not exist` | Bảng chưa tạo | Restart server — tự tạo bảng khi khởi động |
| Upload đáp án → 400 Bad Request | File không chứa "Câu X" nào | Thêm `Câu 1: ...` vào file |
| Bước 2 bị mờ, không click được | Chưa chọn đáp án ở bước 1 | Chọn từ dropdown hoặc upload đáp án mới |
| OCR nhận diện sai nhiều | Ảnh chất lượng thấp / chữ nhòe | Dùng ảnh 300+ DPI, chụp thẳng góc, đủ sáng |
| `Connection refused: 11434` | Ollama chưa chạy | Chạy `ollama serve` hoặc bỏ chọn "Sử dụng Llama3" |
| Fine-tuned model chậm lần đầu | Đang tải weights ~500MB từ HuggingFace | Chờ tải xong, lần sau sẽ dùng cache |
| OCR nhận 0 câu trả về | Ảnh không có chữ "Câu X" làm tiêu đề | Bình thường — hệ thống tự gộp vào "Câu 1" |
| `criteria_breakdown` rỗng | Đáp án không có rubric chi tiết | Upload file đáp án theo format đầy đủ (có RUBRIC) |
| CRAFT ImportError: model_urls | torchvision removal | Patch trong code |
| CRAFT ValueError: inhomogeneous | NumPy 2.x issue | Use SafeNP proxy |
| opencv downgrade | craft-text-detector deps | Force install 4.12.0.88 |
| Qwen2VL OutOfMemory | Large model | Use LLaVA or VietOCR mode |

---

## Known Issues & Setup Notes

### CRAFT Compatibility

CRAFT yêu cầu special handling cho NumPy 2.x:
- Patch `torchvision.models.vgg.model_urls` (removed in 0.13+)
- Use SafeNP proxy wrapper để handle inhomogeneous array creation
- Force install correct opencv version

```bash
pip install craft-text-detector
pip install opencv-python==4.12.0.88 --force-reinstall --no-deps
```

### Perspective Deskew

Dễ tối ưu hóa với:
- Linear regression fit trên region centers
- Threshold 1.0° để tránh noise
- Xoay quanh tâm vùng crop

### Yêu cầu hệ thống

```
Python    : 3.10+
RAM       : 4GB+ (8GB+ nếu dùng Llama3)
Disk      : 2GB+ (OCR model weights)
Database  : PostgreSQL 12+ (hoặc SQLite cho dev)
GPU       : Tùy chọn — NVIDIA CUDA, tăng tốc OCR đáng kể
Ollama    : Tùy chọn — cần cho chấm điểm AI chính xác
```

### API Docs (Swagger)

```
http://localhost:8000/docs     ← Swagger UI
http://localhost:8000/redoc    ← ReDoc
```

