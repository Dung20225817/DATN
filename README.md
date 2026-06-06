# OMR Grading System

Ứng dụng web full-stack để tạo bài kiểm tra trắc nghiệm, quét phiếu OMR và xuất kết quả chấm điểm.

## Phạm vi hiện tại

Project hiện được tổ chức theo hướng **OMR-only**:

- tạo và quản lý bài kiểm tra;
- quản lý nhiều mã đề trong cùng một bài;
- chấm một ảnh hoặc nhiều ảnh theo lô;
- nhận diện MSSV, mã đề và đáp án tô;
- lưu lịch sử chấm và xuất Excel/PDF.

## Công nghệ chính

| Tầng | Công nghệ |
|---|---|
| Frontend | React, TypeScript, Vite |
| Backend | FastAPI, SQLAlchemy |
| Xử lý ảnh | OpenCV, NumPy |
| Cơ sở dữ liệu | PostgreSQL |

## Cấu trúc thư mục

```txt
OCR_CRNN/
├── be/
│   ├── main.py
│   └── app/
│       ├── api/
│       │   ├── auth.py
│       │   └── omr/
│       │       ├── assignments.py
│       │       ├── grading.py
│       │       ├── profiles.py
│       │       ├── shared.py
│       │       └── templates.py
│       ├── core/
│       │   ├── logging.py
│       │   └── paths.py
│       ├── db/
│       │   ├── session.py
│       │   └── models/
│       │       ├── user.py
│       │       └── omr.py
│       └── services/
│           └── omr/
│               ├── omr_service.py
│               ├── answer_keys.py
│               ├── omr_preprocess.py
│               ├── omr_layout.py
│               ├── omr_marker_utils.py
│               ├── omr_mcq.py
│               ├── omr_numeric.py
│               ├── omr_scoring.py
│               ├── omr_handwriting.py
│               ├── omr_visualize.py
│               ├── omr_labels.py
│               └── omr_utils.py
├── fe/
│   └── src/
│       ├── App.tsx
│       ├── pages/
│       │   ├── HomePage.tsx
│       │   ├── LoginPage.tsx
│       │   └── RegisterPage.tsx
│       ├── components/
│       │   ├── TopMenu.tsx
│       │   ├── UserSidebar.tsx
│       │   └── ViewImageModal.tsx
│       ├── features/
│       │   └── omr/
│       │       ├── types.ts
│       │       ├── utils.ts
│       │       ├── components/
│       │       │   ├── ExportPanel.tsx
│       │       │   └── StatsPanel.tsx
│       │       ├── pages/
│       │       │   └── MultichoicePage.tsx
│       │       └── styles/
│       │           └── OmrMobileApp.css
│       └── config/
│           └── api.ts
├── docs/
│   └── ARCHITECTURE.md
├── storage/
│   ├── uploads/                         # runtime OMR data, not committed
│   └── logs/                            # runtime logs, not committed
├── deploy/
└── tools/
```

## Luồng hệ thống

1. Frontend gọi API xác thực tại `/api/login` hoặc `/api/register`.
2. Người dùng tạo bài kiểm tra OMR và các bộ đáp án theo mã đề.
3. Ảnh phiếu được gửi tới `/api/omr/grade` hoặc `/api/omr/grade-batch`.
4. Backend tiền xử lý ảnh, xác định ROI, giải mã lưới số và bong bóng đáp án, rồi tính điểm.
5. Kết quả, ảnh overlay và lịch sử chấm được lưu để frontend hiển thị hoặc xuất file.

## Chạy local

### Backend

```bash
cd be
python -m venv venv310
venv310\Scripts\activate
pip install -r requirements.txt
python main.py
```

Backend chạy mặc định tại `http://localhost:8000`.

Có thể ghi đè kết nối DB bằng biến môi trường `DATABASE_URL`; nếu không khai báo, backend vẫn dùng PostgreSQL local như trước.

### Frontend

```bash
cd fe
npm install
npm run dev
```

Frontend chạy mặc định tại `https://localhost:5173`.

## Chạy bằng Docker

Docker Compose chay 3 service:

- `db`: PostgreSQL 16, luu du lieu trong volume `pgdata`.
- `backend`: FastAPI, tu tao bang khi khoi dong neu database trong.
- `frontend`: Caddy phuc vu React build va reverse proxy `/api/*`, `/static/*` ve backend.

```powershell
Copy-Item .env.docker.example .env.docker
docker compose --env-file .env.docker up --build
```

Sau khi chay:

- Frontend: `http://localhost:8080`
- Backend: `http://localhost:8000`
- Health check: `http://localhost:8000/health`

Huong dan chi tiet ve cau hinh, backup, restore, reset du lieu va dung database ben ngoai nam trong [docs/DOCKER_DEPLOYMENT.md](docs/DOCKER_DEPLOYMENT.md).

## Kiểm thử

```bash
# Backend
cd be
..\venv310\Scripts\python.exe -m unittest discover -s tests -v
..\venv310\Scripts\python.exe -m compileall app main.py tests

# Frontend
cd ../fe
npm run lint
npm run build
```

## Endpoint chính

| Method | Endpoint | Mục đích |
|---|---|---|
| `POST` | `/api/login` | Đăng nhập |
| `POST` | `/api/register` | Đăng ký |
| `POST` | `/api/omr/assignments` | Tạo bài kiểm tra |
| `GET` | `/api/omr/assignments/{uid}` | Danh sách bài kiểm tra |
| `PUT` | `/api/omr/assignments/{uid}/{aid}` | Cập nhật bài kiểm tra |
| `DELETE` | `/api/omr/assignments/{uid}/{aid}` | Xóa bài kiểm tra |
| `POST` | `/api/omr/grade` | Chấm một phiếu |
| `POST` | `/api/omr/grade-batch` | Chấm nhiều phiếu |
| `GET` | `/api/omr/form-profiles` | Danh sách profile phiếu |

## Ghi chú kiến trúc

- `api/omr/shared.py` và `features/omr/pages/MultichoicePage.tsx` vẫn là hai điểm còn lớn nhất trong codebase; bước tách tiếp theo nên tập trung vào shared helper phía backend và camera/grading hooks phía frontend. Phần thống kê và xuất file đã được tách thành component riêng.
- File runtime sinh ra nằm trong `storage/uploads/` và không được commit.
- Tài liệu kiến trúc chi tiết hơn nằm trong `docs/ARCHITECTURE.md`.
