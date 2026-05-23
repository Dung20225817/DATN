# Hướng dẫn chạy hệ thống

## 1. Chuẩn bị trước

Cần có sẵn:

- Python 3.10
- Node.js
- PostgreSQL

Backend hiện mặc định kết nối tới PostgreSQL theo cấu hình:

```txt
postgresql://postgres:1111@localhost:5432/postgres
```

Vì vậy cách đơn giản nhất là dùng:

- user: `postgres`
- password: `1111`
- database: `postgres`
- port: `5432`

Nếu máy bạn dùng cấu hình PostgreSQL khác, có thể đặt biến môi trường `DATABASE_URL` trước khi chạy backend.

## 2. Chạy backend

Mở terminal tại thư mục project, chạy:

```powershell
cd be
python -m venv ..\venv310
..\venv310\Scripts\activate
pip install -r requirements.txt
python main.py
```

Khi chạy thành công, backend sẽ có tại:

```txt
http://localhost:8000
```

Có thể kiểm tra nhanh bằng cách mở:

```txt
http://localhost:8000/health
```

Nếu thấy:

```json
{"status":"healthy"}
```

thì backend đã chạy đúng.

## 3. Chạy frontend

Mở **terminal thứ hai** tại thư mục project, chạy:

```powershell
cd fe
copy .env.example .env.local
npm install
npm run dev
```

Frontend sẽ chạy tại:

```txt
https://localhost:5173
```

Lần đầu mở trình duyệt có thể hiện cảnh báo chứng chỉ tự ký. Chọn tiếp tục truy cập là được.

## 4. Cách sử dụng nhanh

1. Mở `https://localhost:5173`
2. Đăng ký tài khoản mới hoặc đăng nhập
3. Vào phần OMR
4. Tạo bài kiểm tra mới
5. Thêm mã đề và đáp án
6. Tải ảnh phiếu làm bài lên hoặc dùng camera để chấm
7. Xem kết quả trong tab thống kê
8. Xuất PDF hoặc Excel nếu cần

## 5. Những lần chạy sau

Nếu đã cài thư viện rồi, mỗi lần mở lại project chỉ cần:

### Terminal 1

```powershell
cd be
..\venv310\Scripts\activate
python main.py
```

### Terminal 2

```powershell
cd fe
npm run dev
```

## 6. Nếu bị lỗi thường gặp

### Lỗi kết nối database

Kiểm tra PostgreSQL đã bật chưa, và cấu hình có đúng:

```txt
postgres / 1111 / localhost:5432 / postgres
```

Nếu dùng cấu hình khác, chạy backend bằng:

```powershell
$env:DATABASE_URL="postgresql://USER:PASSWORD@localhost:5432/DATABASE"
python main.py
```

### Frontend không gọi được backend

Kiểm tra file `fe/.env.local` có:

```txt
VITE_API_URL=http://localhost:8000
```

Sau khi sửa file `.env.local`, hãy tắt và chạy lại frontend.

### Camera không hoạt động

- Dùng trình duyệt hỗ trợ camera
- Cho phép quyền camera
- Ưu tiên mở bằng `https://localhost:5173`

