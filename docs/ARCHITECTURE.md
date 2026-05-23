# Kiến trúc hiện tại

## 1. Nguyên tắc tổ chức

Codebase được làm sạch theo hướng:

1. **Một domain chính**: OMR.
2. **Tách hạ tầng khỏi nghiệp vụ**:
   - `core/` chứa cấu hình nền tảng;
   - `db/` chứa session và model;
   - `services/omr/` chứa pipeline xử lý ảnh.
3. **Một runtime root duy nhất**:
   - `storage/uploads/` chứa file mà backend đang dùng;
   - `storage/logs/` chứa log.
4. **Frontend chia theo mục đích sử dụng**:
   - `pages/` cho màn hình cấp ứng dụng;
   - `components/` cho UI dùng chung;
   - `features/omr/` cho phần nghiệp vụ OMR.

## 2. Backend

```txt
app/
├── api/
│   ├── auth.py
│   └── omr/
│       ├── assignments.py
│       ├── grading.py
│       ├── profiles.py
│       ├── shared.py
│       └── templates.py
├── core/
│   ├── logging.py
│   └── paths.py
├── db/
│   ├── session.py
│   └── models/
│       ├── user.py
│       └── omr.py
└── services/
    └── omr/
```

### Vai trò từng lớp

| Lớp | Vai trò |
|---|---|
| `api/` | Nhận request, validate input, trả response |
| `db/` | Quản lý kết nối SQLAlchemy và model |
| `services/omr/` | Chứa logic nghiệp vụ xử lý phiếu |
| `core/` | Chứa tiện ích nền tảng như logging và runtime paths |

### Runtime storage

Tất cả đường dẫn runtime được khai báo tập trung trong `app/core/paths.py` và được resolve từ vị trí source code, không phụ thuộc thư mục hiện hành của terminal. Nhờ đó chạy backend từ repo root hay từ `be/` đều dùng cùng một nơi:

```txt
storage/
├── uploads/
│   ├── answer_keys/omr/
│   ├── omr/
│   ├── omr_data/
│   ├── omr_templates/
│   └── temp/
└── logs/
```

### Luồng xử lý OMR

```txt
HTTP request
  -> api/omr/{templates,profiles,assignments,grading}.py
  -> services/omr/omr_service.py
  -> preprocess/layout/mcq/numeric/scoring/visualize
  -> lưu file + cập nhật database
  -> JSON response
```

## 3. Frontend

```txt
src/
├── pages/
├── components/
├── features/
│   └── omr/
│       ├── types.ts
│       ├── utils.ts
│       ├── components/
│       │   ├── ExportPanel.tsx
│       │   └── StatsPanel.tsx
│       ├── pages/
│       └── styles/
└── config/
```

### Quy ước

| Vùng | Dùng cho |
|---|---|
| `pages/` | Login, Register, Home |
| `components/` | Thành phần giao diện tái sử dụng |
| `features/omr/` | Code chỉ phục vụ nghiệp vụ OMR |
| `config/` | Cấu hình dùng chung như API endpoint |

## 4. Nợ kỹ thuật còn lại

### Ưu tiên cao

1. Tách tiếp `be/app/api/omr/shared.py` theo nhóm helper:
   - profile helpers
   - answer-key helpers
   - assignment serializers
2. Tách tiếp `fe/src/features/omr/pages/MultichoicePage.tsx` thành:
   - hooks quản lý camera / assignments / grading;
   - components cho các tab còn lại.

`StatsPanel` và `ExportPanel` đã được tách ra khỏi page chính.

### Ưu tiên trung bình

1. Thay xác thực plain text bằng hash mật khẩu và token thật.
2. Thêm fixture ảnh ổn định cho test grading / batch grading tự động.
3. Tách bundle frontend lớn, đặc biệt phần export PDF/Excel, bằng lazy loading khi cần.

## 5. Ranh giới nên giữ

- `services/omr/` không nên import từ frontend hoặc API.
- `api/` không nên tự xử lý ảnh trực tiếp; chỉ gọi service.
- Component dùng chung không nên phụ thuộc ngược vào `features/omr/`.
