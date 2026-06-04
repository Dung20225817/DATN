# ĐẠI HỌC BÁCH KHOA HÀ NỘI
# ĐỒ ÁN TỐT NGHIỆP

## Xây dựng hệ thống chấm điểm phiếu trắc nghiệm bằng hình ảnh

**Sinh viên thực hiện:** Phạm Quốc Dũng  
**Email:** Dung.pq225817@sis.hust.edu.vn  
**Ngành:** Công nghệ thông tin Việt Nhật  
**Giảng viên hướng dẫn:** TS. Trần Việt Trung  
**Trường:** Công nghệ Thông tin và Truyền thông  

**HÀ NỘI, 06/2025**

---

## LỜI CẢM ƠN

[Nội dung lời cảm ơn ghi tại đây...]

---

## LỜI CAM KẾT

**Họ và tên sinh viên:** Phạm Quốc Dũng  
**Điện thoại liên lạc:** 0343624070  
**Email:** Dung.pq225817@sis.hust.edu.vn  
**Lớp:** Công nghệ thông tin Việt Nhật 04 - K67  
**Hệ đào tạo:** Đại học chính quy  

Tôi - Phạm Quốc Dũng xin cam kết rằng Đồ án Tốt nghiệp (ĐATN) là công trình nghiên cứu do chính tôi thực hiện dưới sự hướng dẫn của TS. Trần Việt Trung. Các kết quả trình bày trong ĐATN là trung thực và là sản phẩm do tôi tự nghiên cứu, không sao chép từ bất kỳ công trình nào khác.

Tất cả các tài liệu tham khảo trong ĐATN bao gồm hình ảnh, bảng biểu, số liệu và trích dẫn đều được trích dẫn rõ ràng và đầy đủ trong danh mục tài liệu tham khảo. Tôi hoàn toàn chịu trách nhiệm trước nhà trường nếu có bất kỳ hành vi sao chép nào vi phạm quy định.

Hà Nội, ngày ... tháng ... năm 2025  
**Tác giả ĐATN** *(Ký và ghi rõ họ tên)* ---

## TÓM TẮT NỘI DUNG ĐỒ ÁN

Đồ án này trình bày quá trình xây dựng hệ thống web tự động chấm điểm phiếu trả lời trắc nghiệm (OMR) bằng hình ảnh. Phiên bản hiện tại sử dụng pipeline xử lý ảnh có tính xác định dựa trên OpenCV/NumPy, marker định vị, ROI và phân tích mật độ tô; các thư viện học sâu/OCR được giữ trong backend để phục vụ hướng mở rộng nhận dạng chữ viết tay trong các phiên bản sau.

---

## MỤC LỤC
* LỜI CẢM ƠN (i)
* LỜI CAM KẾT (ii)
* TÓM TẮT NỘI DUNG ĐỒ ÁN (iii)
* DANH MỤC HÌNH VẼ (vi)
* DANH MỤC BẢNG BIỂU (vii)
* DANH MỤC THUẬT NGỮ VÀ TỪ VIẾT TẮT (viii)
* CHƯƠNG 1. GIỚI THIỆU ĐỀ TÀI (1)
  * 1.1 Đặt vấn đề (1)
  * 1.2 Mục tiêu và phạm vi đề tài (1)
  * 1.3 Định hướng giải pháp (2)
  * 1.4 Bố cục đồ án (2)
* CHƯƠNG 2. KHẢO SÁT VÀ PHÂN TÍCH YÊU CẦU (4)
  * 2.1 Khảo sát hiện trạng (4)
  * 2.2 Tổng quan chức năng (5)
    * 2.2.1 Biểu đồ use case tổng quan (5)
  * 2.3 Đặc tả chức năng (7)
    * 2.3.1 Đặc tả use case Đăng nhập (7)
    * 2.3.2 Đặc tả use case Tạo bài thi (8)
    * 2.3.3 Đặc tả use case Chấm điểm (10)
    * 2.3.4 Đặc tả use case Xem thống kê (13)
  * 2.4 Yêu cầu phi chức năng (14)
    * 2.4.1 Tính bảo mật (14)
    * 2.4.2 Khả năng lưu trữ (15)
    * 2.4.3 Tính dễ dùng (15)
    * 2.4.4 Khả năng mở rộng (16)
    * 2.4.5 Hiệu năng (16)
* CHƯƠNG 3. CÔNG NGHỆ SỬ DỤNG (18)
  * 3.1 Xử lý ảnh số - OpenCV 4.12.0.88 (18)
  * 3.2 Tính toán số và học sâu - NumPy, PyTorch và OCR dependencies (19)
  * 3.3 Backend API - FastAPI 0.121.3 (19)
  * 3.4 Cơ sở dữ liệu - PostgreSQL + SQLAlchemy 2.0.44 (20)
  * 3.5 Frontend - React 19 + TypeScript 5.9.3 + Vite 7.2.2 (20)
  * 3.6 WebRTC MediaDevices API (21)
* CHƯƠNG 4. THIẾT KẾ, TRIỂN KHAI VÀ ĐÁNH GIÁ HỆ THỐNG (22)
  * 4.1 Thiết kế kiến trúc (22)
    * 4.1.1 Lựa chọn kiến trúc phần mềm (22)
    * 4.1.2 Mô tả kiến trúc ba tầng cho hệ thống (23)
    * 4.1.3 Thiết kế tổng quan Frontend (24)
    * 4.1.4 Thiết kế tổng quan Backend (25)
    * 4.1.5 Cấu trúc mã nguồn hệ thống (25)
    * 4.1.6 Biểu đồ package chi tiết (25)
  * 4.2 Thiết kế chi tiết (25)
    * 4.2.1 Thiết kế giao diện (25)
    * 4.2.2 Thiết kế lớp (25)
    * 4.2.3 Luồng chấm một phiếu theo mã nguồn (26)
    * 4.2.4 Luồng chấm hàng loạt theo mã nguồn (26)
    * 4.2.5 Thiết kế cơ sở dữ liệu (27)
  * 4.3 Xây dựng ứng dụng (28)
    * 4.3.1 Thư viện và công cụ sử dụng (28)
    * 4.3.2 Pipeline xử lý ảnh OMR – 15 bước (30)
    * 4.3.3 Cấu hình Form Profile và ROI (31)
    * 4.3.4 Cơ chế MCQ Map Search Rescue (31)
    * 4.3.5 Kết quả đạt được (31)
  * 4.4 Kiểm thử (31)
  * 4.5 Triển khai hệ thống (32)
* CHƯƠNG 5. CÁC GIẢI PHÁP VÀ ĐÓNG GÓP NỔI BẬT (32)
  * 5.1 Cơ chế MCQ Map Search Rescue - Tự động hiệu chỉnh lưới câu hỏi (32)
    * 5.1.1 Vấn đề gặp phải (32)
    * 5.1.2 Nguyên nhân kỹ thuật (32)
    * 5.1.3 Cách sửa trong code (32)
    * 5.1.4 Kết quả sau khi sửa (33)
  * 5.2 Thuật toán tính điểm bong bóng tổng hợp (33)
    * 5.2.1 Vấn đề gặp phải (33)
    * 5.2.2 Nguyên nhân kỹ thuật (34)
    * 5.2.3 Cách sửa trong code (34)
    * 5.2.4 Kết quả sau khi sửa (34)
  * 5.3 Smart Camera Scanner trên trình duyệt (34)
    * 5.3.1 Vấn đề gặp phải (34)
    * 5.3.2 Nguyên nhân kỹ thuật (35)
    * 5.3.3 Cách sửa trong code (35)
    * 5.3.4 Kết quả sau khi sửa (36)
* CHƯƠNG 6. KẾT LUẬN VÀ HƯỚNG PHÁT TRIỂN (37)
  * 6.1 Kết luận và hướng phát triển (37)
    * 6.1.1 Kết luận (37)
      * 6.1.1.1 Kết quả kỹ thuật (37)
      * 6.1.1.2 Tính thực tiễn (37)
    * 6.1.2 Hướng phát triển (38)
* TÀI LIỆU THAM KHẢO (40)

---

## DANH MỤC HÌNH VẼ
* **Hình 2.1:** Biểu đồ use case tổng quan (Trang 5)
* **Hình 2.2:** Biểu đồ use case phân rã Giáo viên
* **Hình 2.3:** Biểu đồ use case phân rã Khách vãng lai
* **Hình 2.4:** Quy trình nghiệp vụ chấm bài
* **Hình 2.5:** Quy trình nghiệp vụ tổng thể liên kết các use case
* **Hình 4.1:** Kiến trúc Three-Tier Architecture (Trang 23)
* **Hình 4.2:** Thiết kế tổng quan Frontend (Trang 24)
* **Hình 4.3:** Thiết kế tổng quan Backend (Trang 25)
* **Hình 4.4:** Sơ đồ ERD (Trang 27)
* **Hình 4.5a:** Biểu đồ package chi tiết Frontend
* **Hình 4.5b:** Biểu đồ package chi tiết Backend
* **Hình 4.5c:** Phụ thuộc giữa Frontend, Backend và Storage
* **Hình 4.6:** Luồng sequence chấm một phiếu
* **Hình 4.7:** Luồng sequence chấm hàng loạt
* **Hình 4.8:** Pipeline xử lý ảnh OMR
* **Hình 4.9:** Luồng MCQ Map Search Rescue
* **Hình 5.1:** Luồng MCQ Map Search Rescue

---

## DANH MỤC BẢNG BIỂU
* **Bảng 2.6:** Phân rã use case mức chi tiết
* **Bảng 4.1:** Vai trò thư mục và module chính
* **Bảng 4.2:** Input/output các chức năng chính
* **Bảng 4.3:** API chính của hệ thống
* **Bảng 4.4:** Các màn hình chính và điều kiện hiển thị
* **Bảng 4.5:** State machine của MultichoicePage
* **Bảng 4.6:** Luồng chấm một phiếu theo mã nguồn
* **Bảng 4.7:** Trường kết quả chấm hàng loạt
* **Bảng 4.8:** Mô tả các cột cơ sở dữ liệu
* **Bảng 4.9:** Vai trò bảng dữ liệu trong luồng hoạt động
* **Bảng 4.10:** Thư viện và công cụ sử dụng
* **Bảng 4.11:** Pipeline xử lý ảnh theo input/output
* **Bảng 4.12:** Thành phần Form Profile và ROI
* **Bảng 4.13:** Tham số `mcq_decode` và ý nghĩa
* **Bảng 4.14:** Kết quả kiểm thử chức năng
* **Bảng 4.15:** Cấu hình triển khai hệ thống
* **Bảng 4.16:** File runtime và storage

---

## DANH MỤC THUẬT NGỮ VÀ TỪ VIẾT TẮT
* **API:** Giao diện lập trình ứng dụng (Application Programming Interface)
* **CNN:** Mạng nơ-ron tích chập (Convolutional Neural Network)
* **OMR:** Nhận diện dấu quang học (Optical Mark Recognition)

---

## CHƯƠNG 1. GIỚI THIỆU ĐỀ TÀI

### 1.1 Đặt vấn đề
Trong các cơ sở giáo dục Việt Nam, hình thức thi trắc nghiệm được sử dụng rộng rãi từ cấp trung học đến đại học. Quy trình chấm bài thủ công hoặc bằng máy quét chuyên dụng đắt tiền gây ra nhiều hạn chế:
* **Thời gian:** Chấm thủ công 100 bài thi mất 1-2 giờ.
* **Chi phí:** Máy chấm OMR chuyên dụng có giá từ 50-200 triệu đồng.
* **Tính di động:** Không phù hợp với môi trường thiếu cơ sở hạ tầng.
* **Sai số người chấm:** Chấm thủ công dễ phát sinh sai sót khi làm việc nhiều giờ.

### 1.2 Mục tiêu và phạm vi đề tài
Đề tài hướng đến xây dựng một hệ thống web ứng dụng di động cho phép:
1. Tự động hóa hoàn toàn: Giáo viên chỉ cần chụp ảnh phiếu bằng điện thoại, hệ thống xử lý và trả kết quả trong vòng vài giây.
2. Không phụ thuộc phần cứng chuyên dụng: Chạy trên smartphone phổ thông, không cần máy quét.
3. Hỗ trợ nhiều loại phiếu: Kiến trúc Form Profile linh hoạt cho phép cấu hình theo nhiều mẫu phiếu khác nhau.
4. Xử lý ảnh thực tế: Chịu được ảnh chụp nghiêng, thiếu sáng, tô không rõ thông qua các thuật toán xử lý ảnh tiên tiến.
5. Quản lý theo bài thi: Giáo viên tạo bài thi, quản lý nhiều mã đề, theo dõi lịch sử chấm và xuất báo cáo.

Trong phạm vi của đề tài, hệ thống được thiết kế dưới dạng một ứng dụng web hoạt động ổn định trên môi trường mạng. Chức năng cốt lõi của hệ thống tập trung vào việc tự động chấm điểm các mẫu phiếu trắc nghiệm nhiều lựa chọn (hỗ trợ các dạng 4 đáp án A, B, C, D). Bên cạnh đó, ứng dụng được tích hợp khả năng nhận diện mã số sinh viên (MSSV) thông qua lưới ô tô số kết hợp với phân tích chữ viết tay, đồng thời tự động trích xuất mã đề thi từ lưới ô tô số. Để tối ưu hóa quá trình nhập liệu, hệ thống cung cấp tính năng quét trực tiếp (live scanner) bằng camera hoạt động ngay trên trình duyệt web. Cuối cùng, toàn bộ kết quả chấm điểm có thể được trích xuất dễ dàng dưới các định dạng tiêu chuẩn như Excel và PDF, phục vụ hiệu quả cho công tác thống kê và báo cáo.

### 1.3 Định hướng giải pháp
Để đáp ứng các mục tiêu đã đặt ra và giải quyết triệt để những hạn chế của các phương pháp chấm điểm truyền thống, hệ thống được thiết kế theo kiến trúc ba tầng (Three-Tier Architecture) và phát triển dựa trên các công nghệ hiện đại, mang lại hiệu năng cao cùng khả năng mở rộng linh hoạt:
* **Phần giao diện người dùng (Frontend):** Được phát triển bằng React 19 kết hợp với TypeScript, cung cấp giao diện web động được tối ưu hóa mạnh mẽ cho thiết bị di động (mobile-first). Đặc biệt, tính năng Smart Camera Scanner sử dụng WebRTC Media Devices API cho phép truy cập luồng video và phát hiện phiếu trắc nghiệm theo thời gian thực ngay trên trình duyệt mà không cần cài đặt ứng dụng native.
* **Phần xử lý logic (Backend):** Sử dụng FastAPI (Python) làm REST API server. Với cơ chế xử lý bất đồng bộ (Asynchronous) dựa trên nền tảng ASGI, FastAPI đảm bảo khả năng đáp ứng đồng thời nhiều yêu cầu xử lý ảnh với độ trễ thấp, đồng thời dễ dàng tích hợp với các thư viện trí tuệ nhân tạo đặc thù của Python.
* **Thị giác máy tính (Computer Vision):** Thư viện OpenCV đóng vai trò cốt lõi trong việc thực thi pipeline xử lý ảnh 15 bước. Các kỹ thuật như phát hiện cạnh Canny, biến đổi phối cảnh (Perspective Warp) và phân tích mật độ pixel được áp dụng để trích xuất và chuẩn hóa vùng dữ liệu phiếu từ các bức ảnh chụp trong điều kiện ánh sáng thực tế.
* **Cơ sở dữ liệu (Database):** Hệ thống sử dụng PostgreSQL làm cơ sở dữ liệu chính thông qua SQLAlchemy ORM. PostgreSQL được lựa chọn nhờ khả năng hỗ trợ lưu trữ và truy vấn dữ liệu kiểu JSON một cách tự nhiên, cực kỳ phù hợp để lưu trữ cấu trúc linh hoạt của các bộ đáp án và lịch sử chấm điểm phức tạp của người dùng.

### 1.4 Bố cục đồ án
Báo cáo đồ án tốt nghiệp được tổ chức thành 6 chương với các nội dung chính như sau:
* **Chương 1:** Giới thiệu đề tài. Trình bày lý do chọn đề tài, mục tiêu, phạm vi nghiên cứu và định hướng giải pháp công nghệ cho hệ thống.
* **Chương 2:** Khảo sát và phân tích yêu cầu. Trình bày kết quả khảo sát hiện trạng chấm điểm tại các cơ sở giáo dục và phân tích các hạn chế của những giải pháp hiện có. Xác định các yêu cầu của hệ thống thông qua biểu đồ use case, đặc tả chức năng chi tiết và các yêu cầu phi chức năng về hiệu năng, bảo mật, cũng như khả năng mở rộng.
* **Chương 3:** Công nghệ sử dụng. Trình bày các công nghệ được chọn, các lựa chọn thay thế tương đương và lý do lựa chọn theo yêu cầu kỹ thuật của hệ thống.
* **Chương 4:** Thiết kế, triển khai và đánh giá hệ thống. Đây là phần cốt lõi của báo cáo, bao gồm việc thiết kế kiến trúc ba tầng, thiết kế chi tiết giao diện và cơ sở dữ liệu. Chương này cũng mô tả toàn bộ pipeline xử lý ảnh 15 bước, danh sách các công cụ triển khai, kết quả đạt được cùng quy trình kiểm thử hệ thống.
* **Chương 5:** Các giải pháp và đóng góp nổi bật. Đi sâu phân tích cơ chế MCQ Map Search Rescue, thuật toán tính điểm bong bóng tổng hợp, và tính năng Smart Camera Scanner hoạt động trực tiếp trên trình duyệt.
* **Chương 6:** Kết luận và hướng phát triển. Tóm tắt những kết quả đã đạt được, đánh giá các điểm mạnh, điểm còn hạn chế của hệ thống và đề xuất các định hướng phát triển, mở rộng trong tương lai.

---

## CHƯƠNG 2. KHẢO SÁT VÀ PHÂN TÍCH YÊU CẦU

Dựa trên những định hướng đó, trong chương 2 em sẽ trình bày chi tiết quá trình khảo sát thực tiễn, làm cơ sở để xác định các yêu cầu chức năng và phi chức năng của hệ thống. Các yêu cầu chức năng sẽ được mô tả qua biểu đồ use case và đặc tả chi tiết những use case quan trọng, làm nền tảng cho việc thiết kế và triển khai hệ thống.

### 2.1 Khảo sát hiện trạng
Qua khảo sát thực tế tại một số trường trung học và đại học, quá trình chấm điểm phiếu trắc nghiệm hiện đang được thực hiện theo ba hình thức chính:
* **Chấm thủ công hoàn toàn:** Giáo viên sử dụng đáp án được in sẵn trên giấy (dạng đục lỗ hoặc đối chiếu trực tiếp) để dò từng câu với bài làm của học sinh. Đây là hình thức phổ biến nhất hiện nay nhưng tiêu tốn rất nhiều thời gian (trung bình mất 1-2 phút cho một bài thi 40 câu hỏi).
* **Dùng phần mềm kết hợp máy quét (Scanner):** Một số trường đại học và cơ sở giáo dục lớn trang bị máy quét OMR chuyên dụng. Mặc dù phương pháp này cho tốc độ và độ chính xác cao, tuy nhiên, thiết bị đắt tiền và yêu cầu bảo trì định kỳ gây ra rào cản lớn về chi phí.
* **Dùng ứng dụng của bên thứ ba:** Gần đây, một số ứng dụng trên thiết bị di động cho phép chụp và chấm phiếu. Dù mang lại sự tiện lợi, các ứng dụng này thường yêu cầu trả phí thuê bao hàng tháng hoặc giới hạn số lượng bài chấm đối với tài khoản miễn phí.

Bên cạnh đó, các giáo viên tham gia khảo sát cũng phản ánh một số hạn chế chung của các phương pháp hiện tại:
* Chi phí đầu tư cho thiết bị và phần mềm bản quyền quá cao đối với các trường học ở khu vực ngoại ô hoặc vùng sâu vùng xa.
* Các phương pháp thủ công và một số phần mềm đơn giản không hỗ trợ tốt việc quản lý và chấm đồng thời nhiều mã đề trong cùng một buổi thi, dễ dẫn đến nhầm lẫn trong khâu phân loại phiếu.
* Khó khăn trong việc xuất và trích xuất dữ liệu điểm số để tích hợp với hệ thống quản lý điểm nội bộ của nhà trường.
* Thiếu một giao diện thân thiện, tối ưu hóa cho trải nghiệm trên các thiết bị di động, hạn chế tính linh hoạt cho giáo viên.

### 2.2 Tổng quan chức năng

#### 2.2.1 Biểu đồ use case tổng quan
*(Tham chiếu đến Hình 2.1: Biểu đồ use case tổng quan)*

Biểu đồ use case tổng quan mô tả các tương tác chính giữa người dùng và Hệ thống Quản lý Chấm thi OMR. Hệ thống được thiết kế hướng tới đối tượng sử dụng chính là Giáo viên, người trực tiếp thực hiện các nghiệp vụ tạo lập và quản lý dữ liệu chấm thi. Các chức năng của hệ thống được chia thành 3 nhóm (package) nghiệp vụ cốt lõi như sau:
* **Nhóm Xác thực (Authentication):** Đảm bảo tính bảo mật và cá nhân hóa dữ liệu cho từng người dùng. Giáo viên tương tác với hệ thống thông qua hai chức năng cơ bản là Đăng ký tài khoản (dành cho người dùng mới) và Đăng nhập (để truy cập vào không gian làm việc cá nhân).
* **Nhóm Quản lý phiếu OMR (OMR Template Management):** Cung cấp các công cụ để giáo viên quản lý các định dạng phiếu trắc nghiệm. Giáo viên có thể thực hiện các thao tác: Xem danh sách phiếu, Xem ảnh mẫu phiếu và Xóa phiếu OMR. Đối với chức năng Tạo phiếu OMR mới, hệ thống ràng buộc bắt buộc (`<<include>>`) người dùng phải thực hiện thao tác Nhập đáp án mẫu và Lưu thông tin phiếu OMR để hoàn tất quy trình khởi tạo một mẫu phiếu hợp lệ.
* **Nhóm Quản lý bài thi (Assignment Management):** Tập trung vào việc tổ chức dữ liệu phù hợp với các đợt kiểm tra/thi thực tế dựa trên các mẫu phiếu đã có. Giáo viên có thể Xem danh sách bài thi đã tạo và Xóa bài thi khi không còn nhu cầu sử dụng. Khi thực hiện chức năng Tạo bài thi mới, hệ thống yêu cầu bắt buộc (`<<include>>`) phải Gán profile phiếu mẫu (Form Profile) để xác định định dạng phiếu sẽ dùng cho bài thi đó. Trong quá trình Cập nhật bài thi, giáo viên sẽ thực hiện chức năng đi kèm (`<<include>>`) là Cấu hình bộ đáp án theo mã đề, cho phép một bài thi hỗ trợ nhiều mã đề khác nhau, đáp ứng nhu cầu trộn đề trong thực tế.

#### 2.2.2 Biểu đồ use case phân rã Giáo viên

**Hình 2.2: Biểu đồ use case phân rã Giáo viên**

```mermaid
graph TD
    Teacher["Giáo viên\n(đã đăng nhập, có uid)"]

    subgraph "Mẫu phiếu"
        T1["Xem Form Profile"]
        T2["Xem/tải ảnh mẫu phiếu"]
    end

    subgraph "Quản lý bài thi"
        A1["Tạo bài thi"]
        A2["Cập nhật tiêu đề/ngày"]
        A3["Xóa bài thi"]
        A4["Chọn Form Profile"]
    end

    subgraph "Quản lý đáp án"
        B1["Thêm mã đề 3 chữ số"]
        B2["Nhập đáp án A/B/C/D/E"]
        B3["Xóa mã đề"]
        B4["Chọn mã đề đang dùng"]
    end

    subgraph "Chấm điểm"
        C1["Smart Camera Scanner"]
        C2["Upload ảnh từ thư viện"]
        C3["Chấm 1 phiếu"]
        C4["Chấm hàng loạt tối đa 50 ảnh"]
    end

    subgraph "Kết quả"
        D1["Xem thống kê"]
        D2["Lọc theo MSSV/mã đề"]
        D3["Xem chi tiết bản ghi"]
        D4["Xuất Excel/PDF"]
    end

    Teacher --> T1
    Teacher --> T2
    Teacher --> A1
    Teacher --> A2
    Teacher --> A3
    Teacher --> A4
    Teacher --> B1
    Teacher --> B2
    Teacher --> B3
    Teacher --> B4
    Teacher --> C1
    Teacher --> C2
    Teacher --> C3
    Teacher --> C4
    Teacher --> D1
    Teacher --> D2
    Teacher --> D3
    Teacher --> D4
```

Như vậy, báo cáo vẫn có thể phân rã use case theo hai tác nhân "Giáo viên" và "Khách vãng lai", nhưng cần hiểu đây là phân rã theo quyền truy cập. Về mặt code, các màn hình này được gộp chung trong cùng route và cùng component; hệ thống không có cơ chế `ProtectedRoute` riêng cho từng tác nhân.

#### 2.2.3 Quy trình nghiệp vụ chấm bài

**Hình 2.3: Quy trình nghiệp vụ chấm bài**

```mermaid
flowchart TD
    A([Bắt đầu]) --> B[Giáo viên đăng nhập]
    B --> C[Tạo hoặc chọn bài thi]
    C --> D[Nhập đáp án theo từng mã đề]
    D --> E{Chọn phương thức lấy ảnh}
    E -->|Camera live| F[Smart Camera Scanner dò 4 marker]
    E -->|Upload| G[Chọn 1 hoặc nhiều ảnh]
    F --> H{Đủ marker và tương phản?}
    H -->|Chưa| F
    H -->|Đã khóa| I[Tự động chụp]
    G --> I
    I --> J[Gọi /api/omr/grade hoặc /grade-batch]
    J --> K[Pipeline xử lý ảnh OMR]
    K --> L{Nhiều câu không chắc chắn?}
    L -->|Có| M[MCQ Map Search Rescue]
    L -->|Không| N[Tính điểm]
    M --> N
    N --> O[Lưu kết quả vào omr_grade_result và cập nhật last_result]
    O --> P[Xem thống kê và ảnh overlay]
    P --> Q[Xuất Excel/PDF nếu cần]
    Q --> R([Kết thúc])
```

#### 2.2.4 Quy trình nghiệp vụ tổng thể liên kết các use case

**Hình 2.4: Quy trình nghiệp vụ tổng thể liên kết các use case**

```mermaid
flowchart LR
    Start([Truy cập hệ thống])
    Start --> Auth{Đã có tài khoản?}
    Auth -->|Chưa| Register[UC-02 Đăng ký]
    Auth -->|Có| Login[UC-01 Đăng nhập]
    Register --> Login
    Login --> Home[Trang chủ OMR]
    Home --> Profile[UC-03 Xem/chọn Form Profile]
    Profile --> Assignment[UC-04 Tạo bài thi]
    Assignment --> Answer[UC-05 Quản lý mã đề và đáp án]
    Answer --> Capture{Chọn nguồn ảnh}
    Capture --> Camera[UC-06 Smart Camera Scanner]
    Capture --> Upload[UC-07 Upload ảnh]
    Camera --> GradeOne[UC-08 Chấm 1 phiếu]
    Upload --> GradeOne
    Upload --> GradeBatch[UC-09 Chấm hàng loạt]
    GradeOne --> Save[UC-10 Lưu lịch sử chấm]
    GradeBatch --> Save
    Save --> Stats[UC-11 Xem/lọc thống kê]
    Stats --> Detail[UC-12 Xem chi tiết bản ghi]
    Stats --> Export[UC-13 Xuất Excel/PDF]
    Export --> End([Kết thúc])
```

#### 2.2.5 Phân rã use case mức chi tiết

**Bảng 2.5: Phân rã use case mức chi tiết**

| Mã UC | Use case mức cao | Use case con | Module triển khai | Dữ liệu chính | Kết quả |
|---|---|---|---|---|---|
| UC-01 | Xác thực | Đăng nhập | `auth.py`, `LoginPage.tsx` | email, password | Lưu thông tin user vào `localStorage` |
| UC-02 | Xác thực | Đăng ký | `auth.py`, `RegisterPage.tsx` | user_name, email, phone, password | Bản ghi mới trong `users` |
| UC-03 | Mẫu phiếu | Xem Form Profile | `profiles.py`, `MultichoicePage.tsx` | profile JSON| Danh sách mẫu phiếu |
| UC-04 | Quản lý bài thi | Tạo/sửa/xóa bài thi | `assignments.py` | uid, aid, title | Bản ghi `omr_assignment` |
| UC-05 | Quản lý đáp án | Cấu hình nhiều mã đề | `answer_sets`, `MultichoicePage.tsx` | code, answers | JSON `answer_sets` |
| UC-06 | Chụp ảnh | Smart Camera Scanner | `evaluateAlignment()` | video frame, scanner_hint | Ảnh JPEG tự chụp |
| UC-07 | Chụp ảnh | Upload ảnh | `MultichoicePage.tsx` | File ảnh | FormData gửi backend |
| UC-08 | Chấm điểm | Chấm 1 phiếu | `grading.py`, `omr_service.py` | image, uid, aid | JSON điểm và ảnh overlay |
| UC-09 | Chấm điểm | Chấm hàng loạt | `grading.py`, `omr_service.py` | tối đa 50 ảnh | Danh sách kết quả và zip overlay |
| UC-10 | Kết quả | Lưu lịch sử | `grading.py`, `grade_results.py`, `assignments.py` | GradeRecord, `OMRGradeResult` | Thêm bản ghi `omr_grade_result`, cập nhật `last_result`, `graded_count` |
| UC-11 | Kết quả | Xem/lọc thống kê | `StatsPanel.tsx` | records, keyword | Danh sách bản ghi phù hợp |
| UC-12 | Kết quả | Xem chi tiết bản ghi | route `/multichoice/record-detail/:testId/:recordId` | recordId | Chi tiết ảnh overlay và câu trả lời |
| UC-13 | Kết quả | Xuất Excel/PDF | `ExportPanel.tsx` | records | File `.xlsx` hoặc `.pdf` |

### 2.3 Đặc tả chức năng
#### 2.3.1 Đặc tả use case Đăng nhập
* **Tên use case:** Đăng nhập
* **Tác nhân:** Giáo viên
* **Tiền điều kiện:** Người dùng chưa đăng nhập
* **Luồng sự kiện chính (Thành công):**
  1. Người dùng: Nhập email và mật khẩu.
  2. Hệ thống: Kiểm tra xem người dùng đã nhập các trường bắt buộc và hợp lệ hay chưa.
  3. Hệ thống: Tra cứu tài khoản, xác minh mật khẩu.
  4. Hệ thống: Thông báo đăng nhập thành công và chuyển hướng đến trang chủ.
* **Luồng sự kiện thay thế:**
  * 4a. Hệ thống thông báo: Vui lòng điền đầy đủ thông tin bắt buộc.
  * 4b. Hệ thống thông báo: Các thông tin cung cấp chưa hợp lệ.
* **Hậu điều kiện:** Người dùng đã đăng nhập vào hệ thống.

**Bảng dữ liệu đầu vào cho chức năng Đăng nhập:**
| STT | Trường dữ liệu | Bắt buộc | Điều kiện hợp lệ | Ví dụ |
|---|---|---|---|---|
| 1 | Email | Có | Phải là duy nhất, đúng định dạng email | giaovien@truong.edu.vn |
| 2 | Mật khẩu | Có | Tối thiểu 6 kí tự | 123456 |

#### 2.3.2 Đặc tả use case Tạo bài thi
* **Tên use case:** Tạo bài thi
* **Tác nhân:** Giáo viên
* **Tiền điều kiện:** Giáo viên đã đăng nhập vào hệ thống
* **Luồng sự kiện chính (Thành công):**
  1. Người dùng: Nhấn nút “Tạo bài thi mới” trên trang chủ.
  2. Người dùng: Nhập tên bài thi, ngày tổ chức, số câu hỏi, tổng điểm và chọn profile phiếu mẫu.
  3. Người dùng: Nhấn “Xác nhận tạo”.
  4. Hệ thống: Kiểm tra tính hợp lệ của dữ liệu đầu vào.
  5. Hệ thống: Gọi API `POST /api/omr/assignments`, lưu vào cơ sở dữ liệu.
  6. Hệ thống: Hiển thị thẻ bài thi mới và thông báo "Tạo bài thi thành công".
* **Luồng sự kiện thay thế:**
  * 4a. Tên thi trống: hệ thống thông báo yêu cầu nhập tên bài để yêu cầu.
  * 4b. Profile không tồn tại: hệ thống thông báo lỗi tìm kiếm profile.
  * 5a. Lỗi server: hệ thống thông báo lỗi và giữ nguyên dữ liệu tại form.
* **Hậu điều kiện:** Bài thi mới xuất hiện trong danh sách; giáo viên có thể cấu hình đáp án và tiến hành chấm điểm.

**Bảng dữ liệu đầu vào cho chức năng Tạo bài thi:**
| STT | Trường dữ liệu | Bắt buộc | Điều kiện hợp lệ | Ví dụ |
|---|---|---|---|---|
| 1 | Tên bài thi | Có | Chuỗi văn bản không được để trống. | Kiểm tra HK1 Toán 12 |
| 2 | Ngày tổ chức | Không | Định dạng ngày tháng | 15/01/2025 |
| 3 | Số câu hỏi | Không | Số nguyên dương (mặc định 40). | 40 |
| 4 | Tổng điểm | Không | Số nguyên dương (mặc định 10). | 10 |
| 5 | Profile phiếu mẫu | Không | Mã profile phải tồn tại trong hệ thống. | a4-standard-40 |

#### 2.3.3 Đặc tả use case Chấm điểm
* **Tên use case:** Chấm điểm phiếu trả lời
* **Tác nhân:** Giáo viên
* **Tiền điều kiện:** Người dùng đã ở trang chấm điểm; có ít nhất một bộ đáp án hoặc phiếu OMR đã lưu trong hệ thống.
* **Luồng sự kiện chính (Chấm đơn lẻ):**
  1. Người dùng: Tải lên ảnh phiếu trả lời (chụp hoặc chọn từ thư viện).
  2. Người dùng: Chọn bài thi hoặc nhập/upload file đáp án.
  3. Người dùng: Nhấn “Chấm điểm”.
  4. Hệ thống: Gọi API `POST /api/omr/grade`, xử lý ảnh qua pipeline OpenCV/NumPy.
  5. Hệ thống: Trả về ảnh kết quả có overlay màu, mã số sinh viên, điểm số và chi tiết từng câu.
  6. Người dùng: Xem kết quả, tải ảnh kết quả nếu cần.
* **Luồng thay thế (Chấm hàng loạt):**
  * 1'. Người dùng: Chọn nhiều ảnh cùng lúc (tối đa 50 ảnh).
  * 3'. Người dùng: Nhấn “Chấm hàng loạt”.
  * 4'. Hệ thống: Gọi API `POST /api/omr/grade-batch`, xử lý tuần tự từng ảnh.
  * 5'. Hệ thống: Trả về danh sách kết quả và file ZIP chứa toàn bộ ảnh chấm.
* **Luồng ngoại lệ:**
  * 4a. Không nhận diện được phiếu (ảnh mờ, góc lệch quá lớn): hệ thống thông báo lỗi, đề xuất chụp lại.
  * 4b. Không tìm thấy đề phù hợp (chế độ tự động): hệ thống thông báo “Không tìm thấy đề phù hợp trong DB”.
  * 4c. Số đáp án không khớp số câu hỏi: hệ thống thông báo lỗi định dạng.
* **Hậu điều kiện:** Kết quả chấm điểm được hiển thị; nếu chấm theo bài thi thì `graded_count` được cập nhật.

**Bảng dữ liệu đầu vào cho chức năng Chấm điểm:**
| STT | Trường dữ liệu | Bắt buộc | Điều kiện hợp lệ | Ví dụ |
|---|---|---|---|---|
| 1 | Ảnh phiếu trả lời | Có | Định dạng JPG/PNG/WEBP, rõ nét. | phieu_hs001.jpg |
| 2 | Bộ đáp án / Bài thi | Có | Chọn bài thi đã lưu hoặc nhập thủ công. | Kiểm tra HK1 |
| 3 | Số câu hỏi | Không | Số nguyên dương (mặc định 80). | 40 |
| 4 | Số lựa chọn | Không | Từ 2 đến 6 (mặc định 5). | 4 |
| 5 | Số chữ số MSSV | Không | Số nguyên dương (mặc định 6). | 8 |
| 6 | File đáp án | Không | Định dạng .doc/.docx / .pdf/.txt. | dap_an.docx |

#### 2.3.4 Đặc tả use case Xem thống kê
* **Tên use case:** Xem thống kê kết quả bài thi
* **Tác nhân:** Giáo viên
* **Tiền điều kiện:** Giáo viên đã đăng nhập; bài thi có ít nhất một lần chấm điểm trong lịch sử.
* **Luồng sự kiện chính (Thành công):**
  1. Người dùng: Chọn một bài thi trong danh sách, nhấn vào thẻ để xem chi tiết.
  2. Hệ thống: Gọi API `GET /api/omr/assignments/{uid}`, tải dữ liệu bài thi về client.
  3. Hệ thống: Hiển thị tab “Thống kê” với: điểm trung bình, điểm cao nhất/thấp nhất, số bài đã chấm.
  4. Hệ thống: Hiển thị biểu đồ phân bố điểm và bảng danh sách kết quả từng học sinh.
  5. Người dùng: (Tùy chọn) Xuất danh sách kết quả ra file Excel hoặc PDF.
  6. Hệ thống: Tạo và tải file xuất về máy người dùng.
* **Luồng thay thế:**
  * 3a. Bài thi chưa có kết quả chấm (`graded_count = 0`): hiển thị thông báo “Chưa có dữ liệu thống kê”.
  * 5a. Người dùng lọc danh sách theo mã đề hoặc khoảng điểm: hệ thống cập nhật bảng theo bộ lọc.
  * 6a. Hệ thống lỗi tạo file: thông báo lỗi, giữ nguyên trang thống kê.
* **Hậu điều kiện:** Giáo viên nắm được tổng quan chất lượng bài thi; có thể xuất báo cáo kết quả.

**Bảng dữ liệu đầu ra của chức năng Xem thống kê:**
| STT | Thông tin | Mô tả |
|---|---|---|
| 1 | Số bài đã chấm | Tổng số phiếu đã được chấm điểm trong bài thi. |
| 2 | Điểm trung bình | Trung bình cộng điểm số của toàn bộ học sinh. |
| 3 | Điểm cao nhất | Điểm số lớn nhất đạt được trong bài thi. |
| 4 | Điểm thấp nhất | Điểm số nhỏ nhất đạt được trong bài thi. |
| 5 | Phân bố điểm | Biểu đồ cột thể hiện số lượng học sinh theo từng mức điểm. |
| 6 | Danh sách kết quả | Bảng chi tiết: MSSV, họ tên, mã đề, điểm số, đáp án từng câu. |
| 7 | File xuất | File Excel (.xlsx) hoặc PDF tổng hợp kết quả toàn bài thi. |

### 2.4 Yêu cầu phi chức năng

#### 2.4.1 Tính bảo mật
Hệ thống được thiết kế theo nguyên tắc xác thực tại tầng API, kiểm soát truy cập theo người dùng và hạn chế tối đa các vector tấn công phổ biến.
* **Xác thực người dùng:** Mọi thao tác quản lý dữ liệu (tạo bài thi, xem kết quả, xóa phiếu OMR) đều yêu cầu `uid` hợp lệ trong request. Hệ thống kiểm tra `uid` tại từng endpoint thông qua hàm `_validate_uid()`, từ chối yêu cầu nếu tham số không hợp lệ hoặc không tồn tại trong cơ sở dữ liệu.
* **Phân quyền theo sở hữu:** Mỗi bài thi (`OMRAssignment`) và phiếu OMR (`OMRTest`) đều được gắn với `uuid` của giáo viên tạo ra. Các truy vấn cơ sở dữ liệu luôn lọc đồng thời theo `uid` và `aid/omrid`, ngăn người dùng truy cập dữ liệu của nhau.
* **Kiểm soát CORS:** Backend cấu hình CORS chỉ cho phép các nguồn gốc từ localhost, 127.0.0.1 và dải địa chỉ LAN nội bộ (192.168.x.x), hạn chế các yêu cầu cross-origin từ domain lạ.
* **Làm sạch dữ liệu đầu vào:** Tên file upload được lọc qua `os.path.basename()` trước khi ghi xuống đĩa, ngăn chặn tấn công path traversal. Mã đề thi (`omr_code`) bắt buộc đúng định dạng 3 chữ số bằng regex `\d{3}`; mã profile được chuẩn hóa về ký tự an toàn trước khi dùng làm tên file JSON.
* **Hướng phát triển:** Phiên bản hiện tại lưu mật khẩu dạng plain-text và dùng token giả định `fake_jwt_123`. Để triển khai môi trường production, cần bổ sung băm mật khẩu bằng bcrypt và thay thế bằng JWT có thời hạn hiệu lực.

#### 2.4.2 Khả năng lưu trữ
Hệ thống áp dụng chiến lược lưu trữ phân tán giữa cơ sở dữ liệu quan hệ và hệ thống file để cân bằng hiệu năng và dung lượng.
* **Cơ sở dữ liệu quan hệ:** PostgreSQL lưu trữ metadata nghiệp vụ gồm thông tin người dùng, bài thi, template OMR, cấu hình template và kết quả chấm. Dữ liệu bán cấu trúc như `answer_sets`, `last_result`, `omr_answer`, `info_fields` và `result_json` được lưu dưới dạng JSON.
* **Bảng lịch sử chấm độc lập:** Ngoài `last_result` trên `omr_assignment` để tương thích giao diện cũ, hệ thống đã bổ sung bảng `omr_grade_result` để lưu từng lượt chấm với `aid`, `uuid`, `omrid`, MSSV, mã đề, điểm, đường dẫn ảnh kết quả/crop và JSON kết quả đầy đủ.
* **Lưu trữ file:** Ảnh phiếu thi gốc, ảnh kết quả có overlay màu, ảnh crop vùng MSSV/MCQ/họ tên, file JSON confidence của từng bong bóng, file ZIP kết quả batch và ảnh/PDF template được lưu trong `storage/uploads` (`omr`, `omr_templates`, `omr_data`). Static file server tích hợp trong FastAPI phục vụ các file này qua đường dẫn `/static/`.
* **Metadata template trong SQL:** Các trường metadata của template như `template_image`, `info_fields`, `options`, `rows_per_block` và `student_id_digits` được lưu trong bảng `omr_test`.
* **File đáp án tạm:** Thư mục `storage/uploads/answer_keys/omr` chỉ dùng tạm khi giáo viên upload file đáp án Word/PDF/TXT. Sau khi parse, đáp án đã được lưu vào SQL (`OMRTest.omr_answer` hoặc `OMRAssignment.answer_sets`) và file upload được xóa để giảm dữ liệu rác.
* **Dung lượng ước tính:** Mỗi phiếu thi sau xử lý sinh ra khoảng 2-5 MB file ảnh kết quả. Với quy mô 1.000 bài chấm mỗi kỳ thi, hệ thống cần dự phòng khoảng 5-10 GB dung lượng lưu trữ cho mỗi năm học. Kết quả chấm hàng loạt được đóng gói ZIP để giảm số lần truyền tải.

#### 2.4.3 Tính dễ dùng
Hệ thống được thiết kế hướng đến giáo viên không có chuyên môn kỹ thuật, ưu tiên quy trình thao tác tối giản và phản hồi trực quan.
* **Giao diện một trang (SPA):** Toàn bộ luồng chấm điểm từ chọn bài thi, tải ảnh lên đến xem kết quả diễn ra trên một màn hình duy nhất mà không cần tải lại trang, giảm thiểu thao tác điều hướng.
* **Smart Camera Scanner:** Tính năng quét camera tích hợp tự động nhận diện bốn góc phiếu thi theo thời gian thực, khóa khung hình khi điều kiện ánh sáng và góc chụp đạt yêu cầu và tự động chụp mà không cần người dùng nhấn nút. Điều này đặc biệt hữu ích khi chấm số lượng lớn tại lớp học.
* **Gợi ý vùng cắt tự động:** Endpoint `/api/omr/suggest-crop` phân tích ảnh và trả về tọa độ bốn góc đề xuất, giúp giáo viên hiệu chỉnh chính xác mà không cần ước lượng thủ công.
* **Phản hồi lỗi rõ ràng:** API trả về thông báo lỗi bằng tiếng Việt có nội dung cụ thể (ví dụ: “Đề thi này chưa được cung cấp đáp án”), giúp giáo viên tự xử lý mà không cần hỗ trợ kỹ thuật.
* **Quản lý đáp án theo mã đề:** Giao diện chính hiện cho giáo viên cấu hình đáp án trực tiếp bằng các lựa chọn A/B/C/D theo từng mã đề.
* **Xuất kết quả:** Danh sách kết quả có thể xuất ra file Excel (.xlsx) hoặc PDF ngay trên trình duyệt, phục vụ nhu cầu lưu trữ và báo cáo mà không cần phần mềm bổ sung.

#### 2.4.4 Khả năng mở rộng
Kiến trúc hệ thống được thiết kế theo hướng module hóa, cho phép bổ sung tính năng và tăng quy mô mà không cần tái cấu trúc toàn bộ.
* **Tách biệt frontend và backend:** React SPA giao tiếp với FastAPI hoàn toàn qua REST API. Hai tầng có thể triển khai độc lập, cho phép mở rộng backend theo chiều ngang (thêm instance) hoặc thay thế frontend mà không ảnh hưởng đến logic nghiệp vụ.
* **Form Profile linh hoạt:** Cấu hình nhận diện phiếu thi (Form Profile) được lưu dưới dạng file JSON độc lập trong `storage/uploads/omr_data/profiles`. Để hỗ trợ loại phiếu mới, chỉ cần thêm file mẫu và cấu hình profile tương ứng; metadata nghiệp vụ của template được đồng bộ vào SQL để dễ quản lý.
* **Pipeline xử lý ảnh có tham số hóa:** Toàn bộ các tham số xử lý ảnh (ngưỡng nhị phân, tỉ lệ vùng bong bóng, chiến lược ROI, bộ giải mã MCQ) đều có thể ghi đè qua profile mà không cần sửa code nguồn, cho phép tinh chỉnh độ chính xác theo từng loại phiếu mà không ảnh hưởng đến các loại phiếu khác.
* **Hỗ trợ đa mã đề:** Cấu trúc `answer_sets` lưu danh sách bộ đáp án theo mã đề (ví dụ: mã 001, 002, 003), cho phép một bài thi quản lý nhiều đề thi khác nhau. Hệ thống tự động chọn đáp án phù hợp dựa trên mã đề nhận diện từ phiếu.
* **Cơ chế MCQ Map Search Rescue:** Tính năng tự động hiệu chỉnh lưới MCQ được triển khai trong `omr_service.py`, có thể bật/tắt qua cấu hình profile `disable_mcq_rescue`. Cơ chế này thử các biến thể line-height/top-shift khi số câu không chắc chắn vượt ngưỡng.
* **Sẵn sàng tách object storage:** Các file nhị phân lớn được tham chiếu bằng đường dẫn thay vì lưu trực tiếp vào SQL. Khi triển khai lên AWS, lớp lưu trữ này có thể chuyển từ filesystem cục bộ sang S3 mà không thay đổi mô hình dữ liệu chính.

#### 2.4.5 Hiệu năng
Hệ thống áp dụng nhiều kỹ thuật tối ưu để đảm bảo thời gian phản hồi chấp nhận được trong điều kiện tài nguyên phần cứng hạn chế tại trường học.
* **Xử lý bất đồng bộ:** FastAPI sử dụng mô hình ASGI với asyncio. Các tác vụ tính toán nặng như xử lý ảnh OpenCV/NumPy được chạy trong thread pool riêng qua `run_in_threadpool`, tránh chặn vòng lặp sự kiện và cho phép server xử lý đồng thời nhiều request.
* **Thời gian xử lý một phiếu:** Trên phần cứng CPU thông thường (Intel Core i5 thế hệ 10 trở lên), một phiếu thi 40 câu được xử lý hoàn chỉnh trong khoảng 1-3 giây bao gồm nhị phân hóa, phát hiện góc, cắt phối cảnh, nhận diện MSSV và chấm điểm MCQ.
* **Chấm hàng loạt tối ưu:** Endpoint `/grade-batch` chấp nhận tối đa 50 ảnh trong một request, giảm overhead HTTP so với gửi từng ảnh riêng lẻ. Ảnh kết quả được đóng gói ZIP ngay trên server trước khi trả về, giảm số lần tải file.
* **Đọc profile theo request:** Form Profile được resolve tại thời điểm xử lý request; cơ chế `_resolve_profile()` ưu tiên profile đã lưu trên đĩa và chỉ dùng cấu hình mặc định khi không tìm thấy profile tương ứng.
* **Tối ưu phục vụ file tĩnh:** Ảnh kết quả và ảnh mẫu được phục vụ trực tiếp qua `StaticFiles` của Starlette, tận dụng cơ chế buffer và header Content-Type tự động mà không qua lớp xử lý Python, cho throughput tương đương Nginx trong môi trường nội bộ.

---

## CHƯƠNG 3. CÔNG NGHỆ SỬ DỤNG

Chương này không chỉ giới thiệu định nghĩa công nghệ mà còn so sánh với các lựa chọn thay thế tương đương. Tiêu chí lựa chọn xuất phát từ hệ thống thực tế: backend Python xử lý ảnh, ảnh phiếu được xử lý bằng OpenCV/NumPy và lưu kết quả vào PostgreSQL.

### 3.1 Xử lý ảnh số - OpenCV 4.12.0.88

Pipeline OMR cần phát hiện/crop trang giấy, biến đổi phối cảnh, nhị phân hóa, tìm marker đen, suy luận ROI và tính mật độ tô trong từng ô. Đây là các phép toán ảnh ma trận lớn, yêu cầu tốc độ xử lý đủ nhanh trên CPU.

| Công nghệ | Vai trò tương đương | Đánh giá |
|---|---|---|
| **OpenCV** | Xử lý ảnh tổng quát, contour, warp, threshold, morphology | Được chọn vì có đầy đủ hàm cần dùng, lõi C++ nhanh, binding Python ổn định và tương thích trực tiếp với NumPy |
| scikit-image | Xử lý ảnh khoa học | API rõ nhưng không thuận lợi bằng OpenCV cho contour, connected components và xử lý realtime |
| Pillow | Đọc/ghi và biến đổi ảnh cơ bản | Không đủ cho bài toán phát hiện hình học, marker và warp phối cảnh |
| MATLAB Image Processing Toolbox | Xử lý ảnh nghiên cứu | Mạnh nhưng có chi phí license và khó tích hợp vào web backend Python |

Lý do khoa học lựa chọn OpenCV là các thuật toán như `cv2.warpPerspective`, `cv2.connectedComponentsWithStats`, `cv2.findContours`, `cv2.threshold` và `cv2.adaptiveThreshold` đã được tối ưu ở mức native. Phương pháp Otsu phù hợp khi nền ảnh tương đối đều; adaptive threshold được dùng khi ảnh có bóng đổ hoặc ánh sáng không đồng nhất.

### 3.2 Tính toán số và học sâu - NumPy, PyTorch và OCR dependencies

Luồng chấm điểm hiện tại dùng NumPy và OpenCV là chính. PyTorch, TorchVision, EasyOCR, VietOCR và Transformers có trong `be/requirements.txt` để phục vụ hướng mở rộng OCR/chữ viết tay, nhưng code hiện tại chưa đưa mô hình CNN tự huấn luyện vào luồng chấm điểm chính. Vì vậy báo cáo không mô tả `SimpleBubbleCNN` hoặc `HandwrittenDigitCNN` như thành phần đã triển khai.

| Công nghệ | Vai trò tương đương | Đánh giá |
|---|---|---|
| **NumPy** | Tính toán mảng và thống kê pixel | Được chọn vì là nền tảng chuẩn của xử lý ảnh Python và tương thích trực tiếp với OpenCV |
| **PyTorch** | Runtime học sâu/OCR mở rộng | Giữ trong môi trường vì EasyOCR/VietOCR/Transformers phụ thuộc Torch và phù hợp nếu bổ sung OCR sau này |
| TensorFlow/Keras | Runtime học sâu | Không cần thiết trong luồng hiện tại vì chưa có model TensorFlow |
| ONNX Runtime | Inference model tối ưu | Phù hợp khi đã có artifact ONNX; repo hiện chưa có model ONNX |
| Tesseract OCR | OCR truyền thống | Phù hợp chữ in hơn chữ viết tay tiếng Việt, cần binary ngoài Python |

Cơ sở lựa chọn là bài toán OMR có cấu trúc hình học cố định, nên thuật toán deterministic dựa trên marker, ROI, mật độ tô và độ tối ảnh xám dễ kiểm soát hơn model học sâu trong phạm vi đồ án. Module `omr_handwriting.py` hiện crop và tiền xử lý vùng họ tên để lưu dữ liệu kiểm tra hoặc phục vụ huấn luyện/OCR về sau.

### 3.3 Backend API - FastAPI 0.121.3

Backend cần nhận request JSON, upload ảnh multipart, phục vụ file tĩnh và chạy tác vụ xử lý ảnh CPU-bound trong threadpool.

| Công nghệ | Vai trò tương đương | Đánh giá |
|---|---|---|
| **FastAPI** | REST API Python trên ASGI | Được chọn vì hỗ trợ `UploadFile`, dependency injection, Pydantic validation, OpenAPI và `run_in_threadpool` |
| Flask | REST API Python nhẹ | Dễ dùng nhưng cần nhiều extension cho validation/schema/tài liệu API |
| Django REST Framework | Web API đầy đủ | Mạnh cho CRUD lớn nhưng nặng hơn nhu cầu của hệ thống xử lý ảnh |
| Node.js/Express | REST API JavaScript | Không thuận lợi bằng Python vì pipeline ảnh dùng OpenCV/NumPy Python |

FastAPI phù hợp với endpoint upload ảnh như `/api/omr/grade`, `/api/omr/grade-batch`, `/api/omr/suggest-crop`. Các tác vụ blocking được gọi bằng `run_in_threadpool` để tránh chặn event loop ASGI.

### 3.4 Cơ sở dữ liệu - PostgreSQL + SQLAlchemy 2.0.44

Dữ liệu nghiệp vụ gồm tài khoản, bài thi, nhiều bộ đáp án theo mã đề và lịch sử chấm có cấu trúc lồng nhau. Vì vậy CSDL cần hỗ trợ dữ liệu quan hệ lẫn JSON.

| Công nghệ | Vai trò tương đương | Đánh giá |
|---|---|---|
| **PostgreSQL** | CSDL quan hệ có JSON | Được chọn vì hỗ trợ JSON tốt, phù hợp lưu `answer_sets`, metadata template và `result_json` của từng lượt chấm |
| MySQL/MariaDB | CSDL quan hệ | Có JSON nhưng PostgreSQL thuận lợi hơn với dữ liệu bán cấu trúc |
| SQLite | CSDL nhúng | Thuận tiện test local nhưng hạn chế khi nhiều người dùng đồng thời |
| MongoDB | CSDL document | Phù hợp document JSON nhưng hệ thống vẫn có quan hệ rõ giữa `users`, `omr_assignment`, `omr_test` |

SQLAlchemy được chọn thay vì SQL thuần vì codebase định nghĩa model bằng Python class, dùng session dependency trong FastAPI và có thể đổi kết nối bằng `DATABASE_URL`. Phiên bản hiện tại tạo bảng bằng `Base.metadata.create_all()` khi backend khởi động và có helper `ensure_runtime_schema()` để bổ sung các cột runtime mới cho bảng `omr_test` trong môi trường chưa dùng Alembic.

### 3.5 Frontend - React 19 + TypeScript 5.9.3 + Vite 7.2.2

Frontend là SPA mobile-first gồm nhiều trạng thái nghiệp vụ: danh sách bài thi, chấm bài, đáp án, thống kê, xuất file, camera scanner và route chi tiết bản ghi.

| Công nghệ | Vai trò tương đương | Đánh giá |
|---|---|---|
| **React + TypeScript** | SPA component-based | Được chọn vì phù hợp UI nhiều trạng thái, có type cho `FormProfile`, `OMRResult`, `GradeRecord` |
| Vue 3 + TypeScript | SPA component-based | Tương đương năng lực nhưng repo hiện đã dùng React, đổi framework không đem lại lợi ích đủ lớn |
| Angular | SPA framework đầy đủ | Mạnh nhưng boilerplate lớn hơn nhu cầu |
| Next.js | React SSR/full-stack | Không cần SSR vì ứng dụng là công cụ nội bộ chạy local/LAN |

Vite được chọn vì dev server nhanh, hỗ trợ HTTPS local qua `@vitejs/plugin-basic-ssl`, proxy `/api` và `/static` về backend. HTTPS local quan trọng vì camera browser yêu cầu secure context. Frontend cũng dùng `pdfjs-dist` trong chế độ dev để render trang đầu của PDF mẫu lên canvas khi cấu hình ROI họ tên.

### 3.6 WebRTC MediaDevices API

Smart Camera Scanner cần lấy khung hình camera trực tiếp trong trình duyệt và phân tích marker theo thời gian thực.

| Công nghệ | Vai trò tương đương | Đánh giá |
|---|---|---|
| **WebRTC MediaDevices API** | Truy cập camera từ browser | Được chọn vì không cần app native, hoạt động trên HTTPS/localhost và trả stream trực tiếp cho canvas |
| `<input type="file" capture>` | Chụp ảnh qua trình chọn file mobile | Dễ triển khai nhưng không phân tích frame realtime và không tự khóa khi đủ marker |
| React Native Camera | Camera native app | Trải nghiệm tốt hơn nhưng vượt phạm vi web app |
| Native Android/iOS | Camera native | Hiệu năng cao nhất nhưng tăng chi phí phát triển |

Lựa chọn WebRTC có cơ sở kỹ thuật vì scanner chỉ cần đọc pixel từ video frame qua canvas mỗi khoảng 130ms, tính độ sáng trung tâm và tỷ lệ pixel tối tại bốn marker. Thuật toán này đủ nhẹ để chạy client-side và giảm số ảnh gửi lỗi lên backend.

---

## CHƯƠNG 4. THIẾT KẾ, TRIỂN KHAI VÀ ĐÁNH GIÁ HỆ THỐNG

### 4.1 Thiết kế kiến trúc

#### 4.1.1 Lựa chọn kiến trúc phần mềm
Hệ thống được thiết kế theo kiến trúc ba tầng: frontend React, backend FastAPI và tầng dữ liệu gồm PostgreSQL cùng file system. Kiến trúc này phù hợp vì pipeline xử lý ảnh nặng được đặt ở backend, còn client chỉ xử lý giao diện, camera scanner và thao tác người dùng.

#### 4.1.2 Mô tả kiến trúc ba tầng cho hệ thống

**Hình 4.1: Kiến trúc Three-Tier Architecture**

```mermaid
graph TB
    subgraph Client["Presentation Tier"]
        Browser["React + TypeScript + Vite SPA"]
        Camera["WebRTC Camera Scanner"]
        Browser --> Camera
    end

    subgraph App["Application Tier"]
        FastAPI["FastAPI + Uvicorn"]
        Router["api/auth.py\napi/omr/*"]
        Service["services/omr/omr_service.py"]
        CV["OpenCV + NumPy"]
        Rescue["MCQ Map Search Rescue"]
        FastAPI --> Router --> Service
        Service --> CV
        Service --> Rescue
    end

    subgraph Data["Data Tier"]
        DB["PostgreSQL\nusers, omr_assignment, omr_test, omr_grade_result"]
        FS["storage/uploads\nảnh, crop, template, profile, zip"]
    end

    Browser -- "HTTP REST / multipart" --> FastAPI
    Router -- "SQLAlchemy" --> DB
    Service -- "cv2.imwrite/json" --> FS
```

Trong kiến trúc này, **Presentation Tier** là React SPA chạy trên trình duyệt. Tầng này quản lý đăng nhập, danh sách bài thi, nhập đáp án, upload ảnh, camera scanner, hiển thị ảnh overlay và xuất kết quả. Frontend không thực hiện chấm điểm chính; khi giáo viên chọn hoặc chụp ảnh, frontend chỉ kiểm tra trạng thái bài thi, tạo `FormData` và gửi request tới backend.

**Application Tier** là FastAPI backend. Router trong `be/app/api/` tiếp nhận HTTP request, kiểm tra dữ liệu đầu vào, đọc bài thi/profile/đáp án, sau đó gọi service xử lý ảnh. Pipeline OpenCV/NumPy, decode MSSV, decode mã đề, nhận diện đáp án và cơ chế MCQ Map Search Rescue đều nằm ở tầng backend để đảm bảo logic chấm điểm thống nhất và có thể kiểm thử tập trung.

**Data Tier** gồm PostgreSQL và file system. PostgreSQL lưu dữ liệu nghiệp vụ như user, bài thi, bộ đáp án, template và lịch sử chấm. Hệ thống không lưu ảnh lớn trực tiếp trong SQL; ảnh upload, ảnh overlay, crop MSSV/MCQ, JSON confidence và ZIP batch được ghi vào `storage/uploads`, còn database chỉ lưu metadata và đường dẫn tương ứng.

#### 4.1.3 Thiết kế tổng quan Frontend

**Hình 4.2: Thiết kế tổng quan Frontend**

```mermaid
flowchart TD
    subgraph Frontend["Frontend Package"]
        AppShell["AppShell"]
        Routing["Routing"]
        Pages["Pages"]
        Auth["Auth"]
        SharedUI["SharedUI"]
        OMRFeature["OMRFeature"]
        ApiClient["ApiClient"]
    end

    AppShell -.-> Routing
    Routing -.-> Pages
    Routing -.-> OMRFeature
    Pages -.-> Auth
    Pages -.-> SharedUI
    Auth -.-> ApiClient
    OMRFeature -.-> SharedUI
    OMRFeature -.-> ApiClient
    ApiClient -.-> Backend["Backend API"]
```

`AppShell` khởi tạo ứng dụng, `Routing` điều phối route, `Pages` chứa các màn hình chung, `Auth` xử lý xác thực, `SharedUI` chứa thành phần giao diện tái sử dụng, `OMRFeature` gom toàn bộ nghiệp vụ chấm phiếu và `ApiClient` là điểm giao tiếp duy nhất với backend.

Các phụ thuộc được biểu diễn bằng mũi tên nét đứt theo hướng từ gói sử dụng sang gói được sử dụng. `SharedUI` và `ApiClient` không phụ thuộc ngược lại vào `Pages` hoặc `OMRFeature`, nhờ đó frontend giữ được liên kết lỏng và tránh phụ thuộc vòng.

#### 4.1.4 Thiết kế tổng quan Backend

**Hình 4.3: Thiết kế tổng quan Backend**

```mermaid
flowchart TD
    subgraph Backend["Backend Package"]
        AppEntry["AppEntry"]
        ApiLayer["ApiLayer"]
        BusinessServices["BusinessServices"]
        OMRProcessing["OMRProcessing"]
        DataAccess["DataAccess"]
        Infrastructure["Infrastructure"]
    end

    AppEntry -.-> ApiLayer
    AppEntry -.-> DataAccess
    AppEntry -.-> Infrastructure
    ApiLayer -.-> BusinessServices
    BusinessServices -.-> OMRProcessing
    BusinessServices -.-> DataAccess
    BusinessServices -.-> Infrastructure
```

Hình 4.3 chỉ mô tả backend ở mức package tổng quan. `AppEntry` là điểm khởi động FastAPI, đăng ký middleware/router và static file. `ApiLayer` tiếp nhận HTTP request. `BusinessServices` chứa logic nghiệp vụ của xác thực, bài thi, mẫu phiếu, đáp án và kết quả. `OMRProcessing` đóng gói pipeline xử lý ảnh. `DataAccess` quản lý truy cập PostgreSQL, còn `Infrastructure` quản lý log, đường dẫn runtime và file upload.

Các phụ thuộc được thiết kế một chiều từ tầng ngoài vào tầng trong: entrypoint phụ thuộc API và hạ tầng khởi tạo, API phụ thuộc nghiệp vụ, nghiệp vụ phụ thuộc xử lý ảnh, dữ liệu và hạ tầng. `DataAccess`, `Infrastructure` và `OMRProcessing` không gọi ngược lên tầng API, giúp loại bỏ phụ thuộc vòng.

#### 4.1.5 Cấu trúc mã nguồn hệ thống

Cấu trúc mã nguồn được tổ chức theo hai khối chính: frontend React trong `fe/` và backend FastAPI trong `be/`. Các file runtime sinh ra khi chấm bài được tách khỏi mã nguồn và lưu trong `storage/`.

```txt
OCR_CRNN/
|-- fe/
|   |-- src/
|   |   |-- App.tsx
|   |   |-- pages/
|   |   |   |-- HomePage.tsx
|   |   |   |-- LoginPage.tsx
|   |   |   `-- RegisterPage.tsx
|   |   |-- components/
|   |   |-- config/
|   |   |   `-- api.ts
|   |   `-- features/omr/
|   |       |-- pages/
|   |       |   `-- MultichoicePage.tsx
|   |       |-- components/
|   |       |   |-- ExportPanel.tsx
|   |       |   |-- StatsPanel.tsx
|   |       |   `-- OmrProfileRoiEditor.tsx
|   |       |-- styles/
|   |       |   `-- OmrMobileApp.css
|   |       |-- types.ts
|   |       `-- utils.ts
|
|-- be/
|   |-- main.py
|   `-- app/
|       |-- api/
|       |   |-- auth.py
|       |   `-- omr/
|       |       |-- assignments.py
|       |       |-- grade_results.py
|       |       |-- grading.py
|       |       |-- profiles.py
|       |       |-- shared.py
|       |       `-- templates.py
|       |-- services/
|       |   `-- omr/
|       |       |-- omr_service.py
|       |       |-- omr_preprocess.py
|       |       |-- omr_layout.py
|       |       |-- omr_marker_utils.py
|       |       |-- omr_mcq.py
|       |       |-- omr_numeric.py
|       |       |-- omr_scoring.py
|       |       |-- omr_handwriting.py
|       |       `-- omr_visualize.py
|       |-- db/
|       |   |-- session.py
|       |   `-- models/
|       |       |-- user.py
|       |       `-- omr.py
|       `-- core/
|           |-- paths.py
|           `-- logging.py
|
|-- storage/
|   |-- uploads/
|   |   |-- omr/
|   |   |-- omr_templates/
|   |   |-- omr_data/profiles/
|   |   |-- answer_keys/omr/
|   |   `-- temp/
|   `-- logs/
|
|-- docs/
`-- README.md
```

**Bảng 4.1: Vai trò thư mục và module chính**

| Thành phần | Vai trò |
|---|---|
| `fe/src/features/omr/pages/MultichoicePage.tsx` | Màn hình nghiệp vụ OMR chính: tạo bài thi, nhập đáp án, chọn ảnh, gọi API chấm, camera scanner, xem kết quả |
| `fe/src/features/omr/components/StatsPanel.tsx` | Hiển thị thống kê kết quả chấm theo bài thi |
| `fe/src/features/omr/components/ExportPanel.tsx` | Xuất kết quả ra Excel/PDF trên frontend |
| `fe/src/features/omr/components/OmrProfileRoiEditor.tsx` | Công cụ cấu hình ROI cho profile trong môi trường phát triển |
| `fe/src/config/api.ts` | Khai báo endpoint frontend gọi tới backend |
| `be/app/api/auth.py` | Router đăng nhập/đăng ký người dùng |
| `be/app/api/omr/assignments.py` | Router tạo, đọc, cập nhật, xóa bài thi OMR |
| `be/app/api/omr/grading.py` | Router nhận request chấm một phiếu, chấm hàng loạt và gợi ý crop |
| `be/app/api/omr/grade_results.py` | Router đọc/xóa lịch sử từng lượt chấm trong `omr_grade_result` |
| `be/app/api/omr/profiles.py` | Router đọc/lưu Form Profile |
| `be/app/api/omr/shared.py` | Helper chung cho validate, serialize, lưu kết quả và xử lý URL tĩnh |
| `be/app/services/omr/omr_service.py` | Điều phối pipeline OMR end-to-end qua `process_omr_exam()` |
| `be/app/services/omr/omr_mcq.py` | Decode lưới câu hỏi, tính density/darkness, chọn đáp án và đánh dấu câu không chắc chắn |
| `be/app/services/omr/omr_numeric.py` | Decode lưới số MSSV và mã đề |
| `be/app/services/omr/omr_visualize.py` | Vẽ overlay/crop phục vụ kiểm tra trực quan |
| `be/app/db/models/omr.py` | Khai báo bảng `omr_assignment`, `omr_test`, `omr_grade_result` bằng SQLAlchemy |
| `be/app/core/paths.py` | Tập trung cấu hình đường dẫn runtime như `storage/uploads/omr` và `storage/uploads/omr_data/profiles` |
| `storage/uploads/` | Lưu ảnh upload, overlay, crop, JSON confidence, template, profile và file ZIP batch |
| PostgreSQL | Lưu user, bài thi, đáp án, template/profile metadata và lịch sử chấm |

**Bảng 4.2: Input/output các chức năng chính**

| Chức năng | Input | Xử lý chính | Output |
|---|---|---|---|
| Đăng nhập | Email, password | `auth.py` kiểm tra tài khoản | Thông tin user lưu ở frontend |
| Tạo bài thi | Title, profile, số câu, tổng điểm | `assignments.py` lưu `omr_assignment` | Bài thi mới trong danh sách |
| Nhập đáp án | Mã đề, danh sách đáp án | Frontend cập nhật `answer_sets` qua API assignment | Bộ đáp án theo mã đề |
| Chấm một ảnh | Ảnh, `uid`, `aid`, đáp án, profile | `grading.py` gọi `process_omr_exam()` | Điểm, MSSV, mã đề, overlay, chi tiết từng câu |
| Chấm hàng loạt | Danh sách ảnh, `uid`, `aid`, đáp án, profile | Backend lặp pipeline cho từng ảnh | `success_count`, `failed_count`, `results`, `zip_url` |
| Cấu hình ROI | Template, điểm chọn trên canvas | `OmrProfileRoiEditor.tsx` quy đổi ROI về hệ post-warp | Profile JSON trong `storage/uploads/omr_data/profiles` |
| Xem thống kê | Bài thi và lịch sử chấm | `StatsPanel.tsx` tổng hợp điểm | Bảng thống kê, trung bình, cao nhất, thấp nhất |
| Xuất kết quả | Danh sách bản ghi | `ExportPanel.tsx` tạo Excel/PDF | File tải về |

**Bảng 4.3: API chính của hệ thống**

| API | Method | Chức năng | Input chính | Output chính |
|---|---|---|---|---|
| `/api/login` | POST | Đăng nhập | email, password | Thông tin user |
| `/api/register` | POST | Đăng ký | user_name, email, password | Thông tin user mới |
| `/api/omr/assignments` | POST | Tạo bài thi | uid, title, profile, số câu | Assignment mới |
| `/api/omr/assignments/{uid}` | GET | Lấy danh sách bài thi | uid | Danh sách bài thi |
| `/api/omr/assignments/{uid}/{aid}` | PUT | Cập nhật bài thi/đáp án/lượt chấm gần nhất | FormData assignment | Assignment đã cập nhật |
| `/api/omr/assignments/{uid}/{aid}/grade-results` | GET | Lấy lịch sử chấm của một bài | uid, aid | `grade_results` |
| `/api/omr/grade` | POST | Chấm một phiếu | file, uid, aid, answers, profile | `OMRResult` + URL ảnh |
| `/api/omr/grade-batch` | POST | Chấm nhiều phiếu | files, uid, aid, answers, profile | Batch result |
| `/api/omr/form-profiles` | GET/POST | Đọc hoặc lưu Form Profile | profile JSON | Danh sách/profile đã lưu |
| `/api/omr/suggest-crop` | POST | Gợi ý tứ giác crop | image | Tọa độ 4 góc |

Nhờ cách tách này, request từ frontend luôn đi qua router rõ ràng trước khi vào service. Những module xử lý ảnh không phụ thuộc trực tiếp vào React hoặc HTTP, còn frontend không truy cập thẳng database/file system.

#### 4.1.6 Biểu đồ package chi tiết

Khác với hai hình tổng quan phía trên, phần này tách biểu đồ package chi tiết thành ba góc nhìn nhỏ: frontend, backend và phụ thuộc liên tầng. Cách tách này giúp mỗi hình chỉ tập trung vào một phạm vi, dễ đọc hơn so với việc gộp toàn bộ hệ thống vào một sơ đồ lớn.

**Hình 4.5a: Biểu đồ package chi tiết Frontend**

```mermaid
flowchart TD
    subgraph FrontendDetail["Frontend Detail"]
        AppShell["AppShell"]
        Routing["Routing"]
        Pages["Pages"]
        Auth["Auth"]
        SharedUI["SharedUI"]
        ApiClient["ApiClient"]

        subgraph OMRFeature["OMRFeature"]
            Grading["Grading"]
            Templates["Templates"]
            AnswerKeys["AnswerKeys"]
            Statistics["Statistics"]
            ExportPkg["Export"]
            DomainTypes["DomainTypes"]
            ClientUtils["ClientUtils"]
        end
    end

    AppShell -.-> Routing
    Routing -.-> Pages
    Routing -.-> OMRFeature
    Pages -.-> Auth
    Pages -.-> SharedUI
    Auth -.-> ApiClient
    OMRFeature -.-> SharedUI
    OMRFeature -.-> ApiClient
    Grading -.-> DomainTypes
    Templates -.-> DomainTypes
    AnswerKeys -.-> DomainTypes
    Statistics -.-> DomainTypes
    ExportPkg -.-> DomainTypes
    Grading -.-> ClientUtils
    Templates -.-> ClientUtils
```

Hình 4.5a phân rã package frontend chi tiết hơn so với hình tổng quan. `OMRFeature` được tách thành các gói nghiệp vụ con như `Grading`, `Templates`, `AnswerKeys`, `Statistics` và `Export`; các gói này dùng chung `DomainTypes` và `ClientUtils` để giữ độ kết dính cao trong miền OMR.

**Hình 4.5b: Biểu đồ package chi tiết Backend**

```mermaid
flowchart TD
    subgraph BackendDetail["Backend Detail"]
        AppEntry["AppEntry"]

        subgraph ApiLayer["ApiLayer"]
            AuthAPI["AuthAPI"]
            TemplateAPI["TemplateAPI"]
            ProfileAPI["ProfileAPI"]
            AssignmentAPI["AssignmentAPI"]
            GradingAPI["GradingAPI"]
            ResultAPI["ResultAPI"]
        end

        subgraph BusinessServices["BusinessServices"]
            AuthService["AuthService"]
            TemplateService["TemplateService"]
            ProfileService["ProfileService"]
            AssignmentService["AssignmentService"]
            GradingService["GradingService"]
            ResultService["ResultService"]
        end

        subgraph OMRProcessing["OMRProcessing"]
            Preprocessing["Preprocessing"]
            LayoutAnalysis["LayoutAnalysis"]
            Recognition["Recognition"]
            Scoring["Scoring"]
            Visualization["Visualization"]
            AnswerKeyParser["AnswerKeyParser"]
        end

        subgraph DataAccess["DataAccess"]
            DbSession["DbSession"]
            UserModel["UserModel"]
            OMRModels["OMRModels"]
            RuntimeSchema["RuntimeSchema"]
        end

        subgraph Infrastructure["Infrastructure"]
            RuntimePaths["RuntimePaths"]
            Logging["Logging"]
            FileStorage["FileStorage"]
        end
    end

    AppEntry -.-> ApiLayer
    AppEntry -.-> DataAccess
    AppEntry -.-> Infrastructure
    ApiLayer -.-> BusinessServices
    AuthService -.-> DataAccess
    TemplateService -.-> DataAccess
    ProfileService -.-> Infrastructure
    AssignmentService -.-> DataAccess
    GradingService -.-> OMRProcessing
    GradingService -.-> DataAccess
    GradingService -.-> Infrastructure
    ResultService -.-> DataAccess
    OMRProcessing -.-> Infrastructure
```

Hình 4.5b phân rã backend theo tầng. `ApiLayer` gom các router HTTP, `BusinessServices` gom nghiệp vụ, `OMRProcessing` gom các bước xử lý ảnh, `DataAccess` gom session/model/schema và `Infrastructure` gom log, đường dẫn, file runtime. Các mũi tên vẫn đi một chiều từ tầng sử dụng sang tầng được sử dụng để tránh phụ thuộc vòng.

**Hình 4.5c: Phụ thuộc giữa Frontend, Backend và Storage**

```mermaid
flowchart LR
    subgraph ClientSide["Frontend"]
        ApiClient["ApiClient"]
    end

    subgraph ServerSide["Backend"]
        AppEntry["AppEntry"]
        ApiLayer["ApiLayer"]
        BusinessServices["BusinessServices"]
        DataAccess["DataAccess"]
        Infrastructure["Infrastructure"]
    end

    subgraph Storage["Storage"]
        PostgreSQL["PostgreSQL"]
        Uploads["storage/uploads"]
    end

    ApiClient -.->|REST /api, /static| AppEntry
    AppEntry -.-> ApiLayer
    ApiLayer -.-> BusinessServices
    BusinessServices -.-> DataAccess
    BusinessServices -.-> Infrastructure
    DataAccess -.-> PostgreSQL
    Infrastructure -.-> Uploads
```

Hình 4.5c mô tả ranh giới giữa client, server và lưu trữ. Frontend chỉ giao tiếp qua `ApiClient`; backend xử lý request qua `AppEntry` và `ApiLayer`; dữ liệu nghiệp vụ được lưu trong PostgreSQL còn ảnh, overlay, crop và file runtime được lưu trong `storage/uploads`. Trong Mermaid, mỗi `subgraph` tương ứng một gói UML; khi dựng bản vẽ chính thức có thể biểu diễn bằng khung package có tab tên gói ở góc trên bên trái.

### 4.2 Thiết kế chi tiết

#### 4.2.1 Thiết kế giao diện
Giao diện được thiết kế mobile-first. Toàn bộ CSS chính của OMR nằm trong `fe/src/features/omr/styles/OmrMobileApp.css`.

**Bảng 4.4: Các màn hình chính và điều kiện hiển thị**

| Màn hình | Điều kiện/route | Mô tả |
|---|---|---|
| Trang chủ OMR | `navTab === "home"` | Danh sách bài thi |
| Kho mẫu OMR | `navTab === "templates"` | Danh sách Form Profile |
| Thông tin bài thi | `detailTestId !== null` | 4 tab: Chấm bài, Đáp án, Thống kê, Xuất |
| Chi tiết bản ghi | `/multichoice/record-detail/:testId/:recordId` | Ảnh kết quả và chi tiết bản ghi |

#### 4.2.2 Thiết kế lớp

**Bảng 4.5: State machine của MultichoicePage**

| State | Kiểu | Mô tả |
|---|---|---|
| `navTab` | `"home" \| "templates"` | Tab điều hướng chính |
| `detailTestId` | `number \| null` | ID bài thi đang xem |
| `detailTab` | `"grading" \| "answers" \| "stats" \| "export"` | Tab con |
| `scannerState` | `"idle" \| "searching" \| "locked"` | Trạng thái camera scanner |
| `pickedFiles` | `File[]` | Ảnh đã chọn, tối đa 50 |
| `submitting` | `boolean` | Đang gọi API |

```typescript
type GradeRecord = {
  id: string;
  graded_at: string;
  source: "single" | "batch";
  file_name: string;
  image_url?: string | null;
  data: OMRResult;
}

type OMRResult = {
  score?: number;
  student_id?: string;
  exam_code?: string;
  uncertain_count?: number;
  answer_compare?: AnswerCompareItem[];
}
```

#### 4.2.3 Luồng chấm một phiếu theo mã nguồn

**Hình 4.6: Luồng sequence chấm một phiếu**

```mermaid
sequenceDiagram
    actor Teacher as Giáo viên
    participant FE as MultichoicePage.tsx
    participant API as grading.py
    participant Shared as shared.py
    participant Service as omr_service.py
    participant DB as PostgreSQL
    participant FS as storage/uploads

    Teacher->>FE: Chọn/chụp 1 ảnh
    FE->>API: POST /api/omr/grade
    API->>Shared: Validate uid, resolve profile, answer key
    Shared->>DB: Đọc omr_assignment theo uid/aid
    API->>FS: Lưu ảnh upload
    API->>Service: process_omr_exam()
    Service->>Service: Warp, binarize, marker, ROI, decode
    Service->>Service: MCQ Map Search Rescue nếu cần
    Service->>FS: Ghi overlay/crop/telemetry JSON
    Service-->>API: OMRResult
    API->>DB: Thêm bản ghi omr_grade_result
    API-->>FE: JSON + static URLs
    FE->>API: PUT /api/omr/assignments/{uid}/{aid}
    API->>DB: Cập nhật last_result, graded_count
```

Luồng chấm một phiếu bắt đầu từ thao tác của giáo viên trên `MultichoicePage.tsx`. Khi giáo viên chọn ảnh từ thư viện hoặc để Smart Camera Scanner tự chụp, frontend kiểm tra bài thi đang mở, mã đề đang chọn và bộ đáp án tương ứng. Sau đó frontend tạo `FormData` gồm file ảnh, `uid`, `aid`, chuỗi đáp án, số câu, số lựa chọn, số chữ số MSSV, số dòng mỗi block và `form_profile_code` nếu bài thi có profile.

Request được gửi tới `POST /api/omr/grade`. Router `be/app/api/omr/grading.py` kiểm tra `uid`, đọc bài thi trong `omr_assignment`, xác định profile/template, lưu ảnh upload vào `storage/uploads/omr`, rồi gọi `process_omr_exam()` trong `be/app/services/omr/omr_service.py`. Service đọc ảnh, crop/warp phiếu về hệ tọa độ chuẩn, nhị phân hóa, tìm marker, dựng ROI, decode MSSV, mã đề và đáp án MCQ. Nếu số câu không chắc chắn vượt ngưỡng, cơ chế MCQ Map Search Rescue thử các biến thể lưới để tìm kết quả tốt hơn baseline.

Sau khi có `OMRResult`, backend so sánh với `answer_sets`, tính điểm, lưu ảnh overlay/crop/JSON confidence vào `storage/uploads/omr`, thêm bản ghi vào `omr_grade_result` thông qua helper trong `shared.py`, rồi trả JSON về frontend. Frontend hiển thị điểm, MSSV, mã đề, overlay và bảng đúng/sai; đồng thời cập nhật `last_result`/`graded_count` để danh sách bài thi phản ánh lượt chấm mới nhất.

**Bảng 4.6: Luồng chấm một phiếu theo mã nguồn**

| Bước | File/module liên quan | Dữ liệu vào | Dữ liệu ra |
|---|---|---|---|
| Chọn/chụp ảnh | `MultichoicePage.tsx` | File ảnh hoặc camera frame | `FormData` |
| Gọi API | `fe/src/config/api.ts` | `FormData`, endpoint `GRADE` | HTTP multipart request |
| Nhận request | `grading.py` | file, uid, aid, answers, profile | File upload + tham số đã validate |
| Đọc nghiệp vụ | `shared.py`, `models/omr.py` | uid, aid | `omr_assignment`, `answer_sets`, profile |
| Xử lý ảnh | `omr_service.py` và các helper OMR | Đường dẫn ảnh + runtime config | `OMRResult` |
| Decode MCQ | `omr_mcq.py` | ROI MCQ, ảnh xám, ảnh nhị phân | Đáp án, confidence, uncertain |
| Decode MSSV/mã đề | `omr_numeric.py` | ROI số | `student_id`, `exam_code` |
| Lưu file trực quan | `omr_visualize.py`, `storage/uploads/omr` | Ảnh chuẩn + kết quả decode | overlay, crop, confidence JSON |
| Lưu lịch sử | `shared.py`, `omr_grade_result` | `OMRResult`, metadata file | Bản ghi lịch sử chấm |
| Hiển thị | React component | JSON response | Điểm, overlay, bảng đáp án |

#### 4.2.4 Luồng chấm hàng loạt theo mã nguồn

**Hình 4.7: Luồng sequence chấm hàng loạt**

```mermaid
sequenceDiagram
    actor Teacher as Giáo viên
    participant FE as MultichoicePage.tsx
    participant API as grading.py
    participant Service as omr_service.py
    participant DB as PostgreSQL
    participant FS as storage/uploads

    Teacher->>FE: Chọn tối đa 50 ảnh
    FE->>API: POST /api/omr/grade-batch
    API->>API: Kiểm tra số file và bài thi
    loop Với từng ảnh
        API->>FS: Lưu file
        API->>Service: process_omr_exam()
        Service-->>API: success/error
        API->>DB: Lưu omr_grade_result nếu chấm thành công
    end
    API->>FS: Tạo zip overlay nếu có kết quả
    API-->>FE: success_count, failed_count, results, zip_url
```

Với chấm hàng loạt, người dùng chọn tối đa 50 ảnh trên frontend. Nếu `pickedFiles.length > 1`, `MultichoicePage.tsx` gọi `POST /api/omr/grade-batch` và gửi nhiều field `files` trong cùng một `FormData`. Backend kiểm tra số lượng file, bài thi, bộ đáp án và profile trước khi xử lý.

Trong `grade_exam_batch()`, backend lặp qua từng ảnh. Mỗi ảnh được lưu vào `storage/uploads/omr` và chạy cùng pipeline `process_omr_exam()` như luồng chấm một phiếu. Ảnh chấm thành công được thêm bản ghi `omr_grade_result` và đưa vào mảng `results`; ảnh lỗi không làm dừng toàn bộ batch mà được ghi thành phần tử `success: false` trong `results`, đồng thời tăng `failed_count`. Nếu có overlay thành công, backend tạo file ZIP trong `storage/uploads/omr` và trả `zip_url` để frontend cho phép tải về.

**Bảng 4.7: Trường kết quả chấm hàng loạt**

| Trường kết quả batch | Ý nghĩa |
|---|---|
| `success_count` | Số ảnh chấm thành công |
| `failed_count` | Số ảnh lỗi hoặc không decode được |
| `answer_source` | Nguồn đáp án backend dùng để so sánh |
| `results` | Danh sách kết quả từng ảnh; phần tử thành công có điểm, MSSV, mã đề, URL overlay/crop và `grade_result_id`, phần tử lỗi có `success: false` và `error` |
| `zip_url` | Đường dẫn file ZIP chứa overlay của các ảnh chấm thành công |

#### 4.2.5 Thiết kế cơ sở dữ liệu

**Hình 4.4: Sơ đồ ERD**

```mermaid
erDiagram
    users {
        int uuid PK
        varchar user_name
        varchar email
        varchar phone
        varchar password
    }

    omr_assignment {
        int aid PK
        int uuid FK
        varchar title
        varchar created_at_raw
        varchar created_at_label
        int question_count
        int total_points
        int graded_count
        json answer_sets
        varchar active_code
        json last_result
        datetime created_at
        datetime updated_at
    }

    omr_test {
        int omrid PK
        int uuid FK
        varchar omr_name
        varchar(3) omr_code
        int omr_quest
        json omr_answer
        varchar template_image
        json info_fields
        int options
        int rows_per_block
        int student_id_digits
        datetime created_at
        datetime updated_at
    }

    omr_grade_result {
        int grid PK
        int aid FK
        int uuid FK
        int omrid FK
        varchar source
        varchar file_name
        varchar student_id
        varchar exam_code
        varchar score
        varchar result_image
        varchar sid_crop_image
        varchar mcq_crop_image
        varchar bubble_confidence_json
        json result_json
        datetime created_at
    }

    users ||--o{ omr_assignment : "tạo"
    users ||--o{ omr_test : "tạo"
    users ||--o{ omr_grade_result : "có kết quả"
    omr_assignment ||--o{ omr_grade_result : "gồm"
    omr_test ||--o{ omr_grade_result : "dùng template"
```

**Bảng 4.8: Mô tả các cột chính**

| Bảng | Cột | Kiểu | Mô tả |
|---|---|---|---|
| `users` | `uuid` | INTEGER PK | ID người dùng |
| `users` | `password` | VARCHAR | Prototype hiện lưu plain text, cần hash ở production |
| `omr_assignment` | `answer_sets` | JSON | Danh sách bộ đáp án theo mã đề |
| `omr_assignment` | `last_result` | JSON | Bản ghi mới nhất/metadata tương thích giao diện cũ |
| `omr_test` | `omr_answer` | JSON | Đáp án zero-based của mẫu phiếu |
| `omr_test` | `template_image` | VARCHAR | Đường dẫn ảnh/PDF template trong `/static/omr_templates` |
| `omr_test` | `info_fields`, `options`, `rows_per_block`, `student_id_digits` | JSON/INTEGER | Metadata cấu hình template OMR |
| `omr_grade_result` | `result_json` | JSON | Kết quả chấm đầy đủ của một lượt chấm |
| `omr_grade_result` | `result_image`, `sid_crop_image`, `mcq_crop_image`, `bubble_confidence_json` | VARCHAR | Đường dẫn file runtime phục vụ xem lại lịch sử |

**Bảng 4.9: Vai trò bảng dữ liệu trong luồng hoạt động**

| Bảng | Vai trò trong luồng hoạt động |
|---|---|
| `users` | Xác định giáo viên đang đăng nhập và làm khóa ngoại cho bài thi, template, lịch sử chấm |
| `omr_assignment` | Lưu bài thi trên giao diện mobile, số câu, tổng điểm, `answer_sets` theo nhiều mã đề, `active_code`, `graded_count` và `last_result` |
| `omr_test` | Lưu template/profile phiếu OMR và metadata cấu hình như số lựa chọn, số dòng mỗi block, số chữ số MSSV |
| `omr_grade_result` | Lưu từng lượt chấm, gồm điểm, MSSV, mã đề, nguồn chấm single/batch, đường dẫn overlay/crop và JSON kết quả đầy đủ |

Trong luồng chấm, `answer_sets` của `omr_assignment` là nguồn đáp án chính khi giáo viên quản lý nhiều mã đề trong cùng một bài thi. Trường `last_result` chỉ giữ kết quả gần nhất để tương thích giao diện danh sách bài thi; lịch sử đầy đủ phải đọc từ `omr_grade_result`. Các file ảnh, crop, ZIP và JSON confidence không được nhúng vào SQL mà chỉ lưu đường dẫn để tránh làm phình database.

### 4.3 Xây dựng ứng dụng

#### 4.3.1 Thư viện và công cụ sử dụng

**Bảng 4.10: Thư viện và công cụ sử dụng**

| Mục đích | Công cụ | Phiên bản |
|---|---|---|
| Backend | FastAPI, Uvicorn, Pydantic | 0.121.3, 0.38.0, 2.12.5 |
| ORM/DB | SQLAlchemy, psycopg2-binary, PostgreSQL | 2.0.44, 2.9.11 |
| Xử lý ảnh | OpenCV-Python, NumPy, scikit-image, Pillow | 4.12.0.88, 2.2.6, 0.25.2, 10.2.0 |
| OCR/học sâu mở rộng | PyTorch, TorchVision, EasyOCR, VietOCR, Transformers | Theo `requirements.txt` |
| Frontend | React, TypeScript, Vite | 19.2.0, 5.9.3, 7.2.2 |
| Render PDF trên frontend | pdfjs-dist | Dùng trong ROI editor để hiển thị trang đầu PDF mẫu lên canvas |
| Xuất file | jsPDF, XLSX | 4.2.1, 0.18.5 |

#### 4.3.2 Pipeline xử lý ảnh OMR - 15 bước

**Hình 4.8: Pipeline xử lý ảnh OMR**

```mermaid
flowchart TD
    A[/"Ảnh đầu vào"/]
    B["1. cv2.imread"]
    C["2. Optional rect/manual crop"]
    D["3. Warp chuẩn 1000x1400"]
    E["4. Binarize Otsu/Adaptive"]
    F["5. Extract black markers"]
    G["6. Infer MCQ geometry"]
    H["7. Resolve anchors"]
    I["8. Build SID/Code/MCQ/handwriting ROI"]
    J["9. Refine MCQ ROI"]
    K["10. Decode MSSV/mã đề"]
    L["11. Decode MCQ with map"]
    M{"Uncertain vượt ngưỡng?"}
    N["12. MCQ Map Search Rescue"]
    O["13. Detect Q5 drift/expand"]
    P["14. Compare answer key, score"]
    Q["15. Draw overlay, save crop/telemetry"]
    R[/"JSON kết quả"/]
    A --> B --> C --> D --> E --> F --> G --> H --> I --> J --> K --> L --> M
    M -->|Có| N --> O
    M -->|Không| O
    O --> P --> Q --> R
```

Pipeline trên được triển khai trong `process_omr_exam()` và các helper trong `be/app/services/omr/`. Mỗi bước biến đổi dữ liệu theo hướng từ ảnh gốc sang ảnh chuẩn, sau đó từ ROI sang kết quả nhận diện có thể tính điểm.

**Bảng 4.11: Pipeline xử lý ảnh theo input/output**

| Bước | Mục đích | Input | Output |
|---|---|---|---|
| Đọc ảnh | Nạp ảnh vào OpenCV | File ảnh upload | Ma trận ảnh BGR |
| Crop/warp | Chuẩn hóa phối cảnh phiếu | Ảnh gốc, marker hoặc crop quad | Ảnh chuẩn theo kích thước profile |
| Binarize | Tách vùng tô đen khỏi nền giấy | Ảnh xám | Ảnh nhị phân đảo màu |
| Detect marker | Xác định vị trí phiếu và các anchor | Ảnh nhị phân | Tọa độ marker/anchor |
| Build ROI | Tạo vùng MSSV, mã đề, MCQ, họ tên | Profile + marker + ảnh đã warp | Danh sách ROI post-warp |
| Decode numeric | Đọc MSSV và mã đề | ROI số | `student_id`, `exam_code`, confidence |
| Decode MCQ | Đọc đáp án từng câu | ROI MCQ, ảnh xám, ảnh nhị phân | Danh sách đáp án, câu không chắc chắn |
| Rescue | Hiệu chỉnh lưới nếu nhiều câu không chắc chắn | Kết quả baseline + cấu hình `local_grid_search` | Kết quả MCQ tốt hơn nếu có |
| Scoring | So sánh với đáp án | Kết quả decode + `answer_sets` | Điểm và chi tiết đúng/sai |
| Save overlay | Tạo bằng chứng trực quan | Ảnh chuẩn + kết quả decode | Overlay, crop, confidence JSON |

#### 4.3.3 Cấu hình Form Profile và ROI

Form Profile là file JSON trong `storage/uploads/omr_data/profiles`. Backend đọc profile bằng `_resolve_profile()` và chuyển thành runtime config bằng `_build_runtime_config()`. Profile giúp hệ thống hỗ trợ nhiều mẫu phiếu khác nhau mà không phải sửa lại pipeline chính. Profile chứa vị trí marker, vùng MSSV, vùng mã đề, vùng câu hỏi, vùng chữ viết tay và các tham số decode. Các ROI được hiểu theo hệ tọa độ của ảnh đã warp, không phải ảnh gốc.

Ở tab **Kho mẫu OMR**, hệ thống có chế độ cấu hình dành cho môi trường phát triển. Nút **Cấu hình ROI** chỉ hiển thị khi chạy Vite dev (`import.meta.env.DEV`) hoặc bật biến `VITE_ENABLE_OMR_PROFILE_EDITOR=true`. Component `OmrProfileRoiEditor.tsx` render trang đầu của PDF bằng `pdfjs-dist` hoặc vẽ ảnh PNG/JPG/WebP lên canvas, sau đó cho người cấu hình click bốn góc vùng họ tên (`ho_ten`). Bốn điểm có thể chọn theo thứ tự bất kỳ; frontend lấy bounding rectangle từ `min/max x/y` và lưu vào:

```json
{
  "strategy": {
    "handwriting_fields": {
      "enabled": true,
      "save_crops": true,
      "field_rois": {
        "ho_ten": { "x": 0.2302, "y": 0.0410, "w": 0.2992, "h": 0.0677 }
      }
    }
  }
}
```

Điểm quan trọng là ROI không được lưu theo toàn trang PDF, mà được quy đổi sang hệ tọa độ sau warp của backend. Backend chấm ảnh bằng cách warp phiếu về kích thước chuẩn `1000x1400` theo bốn marker đen, rồi mới crop `ho_ten`. Vì vậy frontend dùng `strategy.corner_markers` để quy đổi:

```txt
left   = avg(tl.cx, bl.cx)
right  = avg(tr.cx, br.cx)
top    = avg(tl.cy, tr.cy)
bottom = avg(bl.cy, br.cy)
x' = (x - left) / (right - left)
y' = (y - top) / (bottom - top)
w' = w / (right - left)
h' = h / (bottom - top)
```

Khi mở lại editor, ROI đã lưu theo post-warp được quy đổi ngược về page-space để vẽ đúng vị trí trên PDF mẫu. Với profile 40 câu, ROI `ho_ten` sau quy đổi tương ứng khoảng `x=230, y=57, w=299, h=95` trên ảnh chuẩn `1000x1400`, đúng vùng họ tên khi kiểm thử với ảnh `20260418_121820_651438_camera_1776489482496.jpg`.

**Bảng 4.12: Thành phần Form Profile và ROI**

| Thành phần profile | Ý nghĩa |
|---|---|
| `corner_markers` | Vị trí 4 marker dùng để căn chỉnh và warp phiếu |
| `crop_quad` | Tứ giác crop thủ công nếu cần cắt phiếu theo vùng chọn |
| `sid_roi` | Vùng đọc MSSV bằng lưới số |
| `exam_code_roi` | Vùng đọc mã đề |
| `mcq_roi` | Vùng chứa câu hỏi trắc nghiệm |
| `mcq_decode` | Nhóm ngưỡng và tham số nhận diện bong bóng |
| `handwriting_fields` | Vùng crop chữ viết tay như `ho_ten`, phục vụ lưu crop và hướng mở rộng OCR |

Khi thêm mẫu phiếu mới, bước chính là tạo profile mới hoặc cập nhật ROI trong profile. Pipeline vẫn dùng cùng hàm `process_omr_exam()`, chỉ thay đổi runtime config được dựng từ profile tại thời điểm chấm.

**Bảng 4.13: Tham số `mcq_decode` và ý nghĩa**

| Tham số | Ý nghĩa |
|---|---|
| `min_mark_density` / `min_mark_score` | Ngưỡng tô tối thiểu |
| `min_margin` | Khoảng cách giữa đáp án cao nhất và nhì |
| `min_conf_ratio` | Tỷ lệ tin cậy của lựa chọn tốt nhất |
| `double_mark_gap` | Ngưỡng phát hiện tô nhiều đáp án |
| `soft_mark_floor`, `soft_margin` | Cứu đáp án tô nhẹ nhưng nổi bật |
| `noise_center_floor`, `noise_dark_floor` | Chặn nhiễu từ đường kẻ/vết bẩn |
| `local_grid_search` | Cho phép tìm kiếm cục bộ khi lưới lệch |

#### 4.3.4 Cơ chế MCQ Map Search Rescue

**Hình 4.9: Luồng MCQ Map Search Rescue**

```mermaid
flowchart TD
    A["Decode MCQ ban đầu"] --> B{"uncertain_count >= gate?"}
    B -->|Không| Z["Giữ kết quả ban đầu"]
    B -->|Có| C["Thử line_scales"]
    C --> D["Thử top_shift"]
    D --> E["Decode lại từng ứng viên"]
    E --> F["Rank theo uncertain, double-mark, quality"]
    F --> G{"Tốt hơn baseline?"}
    G -->|Có| H["Cập nhật mcq_result\nwarning MCQ_MAP_SEARCH_RESCUE"]
    G -->|Không| Z
```

#### 4.3.5 Kết quả đạt được

Hệ thống đã triển khai các chức năng chính: đăng nhập/đăng ký, tạo và quản lý bài thi, cấu hình nhiều mã đề, chấm một ảnh, chấm hàng loạt tối đa 50 ảnh, Smart Camera Scanner, xem thống kê, xem chi tiết bản ghi và xuất Excel/PDF. Backend có endpoint `/api/omr/suggest-crop` để gợi ý tứ giác crop; giao diện hiện tại ưu tiên căn chỉnh tự động bằng marker và Form Profile. Ngoài luồng người dùng chính, hệ thống có ROI editor dành cho dev để cấu hình nhanh vùng chữ viết tay `ho_ten` trên từng phiếu mẫu.

### 4.4 Kiểm thử

**Bảng 4.14: Kết quả kiểm thử chức năng**

| Mã TC | Chức năng | Module liên quan | Đầu vào | Kết quả kỳ vọng | Kết quả thực tế |
|---|---|---|---|---|---|
| TC-AUTH-01 | Đăng nhập hợp lệ | `auth.py`, `LoginPage.tsx` | Email/password đúng | Redirect `/home`, lưu user vào `localStorage` | Đạt |
| TC-AUTH-02 | Đăng nhập sai | `auth.py`, `LoginPage.tsx` | Password sai | Thông báo lỗi xác thực | Đạt |
| TC-ASSIGN-01 | Tạo bài thi | `assignments.py`, `MultichoicePage.tsx` | Title + profile | Bài thi xuất hiện trong danh sách | Đạt |
| TC-ASSIGN-02 | Thêm mã đề | `assignments.py`, `answer_sets` | code = `001` | Mã đề xuất hiện trong tab đáp án | Đạt |
| TC-GRADE-01 | Chấm 1 ảnh | `grading.py`, `omr_service.py`, `omr_mcq.py` | Ảnh phiếu chuẩn | Có điểm, MSSV, mã đề, overlay | Đạt |
| TC-GRADE-02 | Chấm ảnh có câu không rõ | `omr_service.py`, `omr_mcq.py` | Bong bóng tô mờ | Ghi `uncertain_count`, kích hoạt rescue nếu vượt ngưỡng | Đạt |
| TC-ROI-01 | Cấu hình ROI họ tên | `OmrProfileRoiEditor.tsx`, `profiles.py`, `omr_handwriting.py` | Profile 40 câu + ảnh `20260418_121820_651438_camera_1776489482496.jpg` | Crop `ho_ten` nằm đúng vùng họ tên, ROI post-warp khoảng `230,57,299,95` | Đạt |
| TC-BATCH-01 | Batch hợp lệ | `grading.py`, `omr_service.py` | 5 ảnh | `success_count = 5` | Đạt |
| TC-BATCH-02 | Batch có ảnh lỗi | `grading.py` | 4 ảnh đúng + 1 ảnh lỗi | `success_count = 4`, `failed_count = 1`, lỗi nằm trong `results` | Đạt |
| TC-CAM-01 | Bật camera | `MultichoicePage.tsx`, WebRTC MediaDevices API | HTTPS/localhost | Video hiển thị, scanner searching/locked | Đạt |
| TC-EXPORT-01 | Xuất Excel/PDF | `ExportPanel.tsx`, `grade_results.py` | Có bản ghi | Tải file `.xlsx` hoặc `.pdf` | Đạt |

### 4.5 Triển khai hệ thống

#### 4.5.1 Môi trường triển khai

**Bảng 4.15: Cấu hình triển khai hệ thống**

| Thành phần | Cấu hình thực tế | Ghi chú |
|---|---|---|
| Backend | Python 3.10, FastAPI, Uvicorn | Entry point `be/main.py` |
| Backend URL | `http://localhost:8000` | Host `0.0.0.0`, port `8000` |
| Frontend | React + TypeScript + Vite | Chạy từ thư mục `fe` |
| Frontend URL | `https://localhost:5173` | Dùng `@vitejs/plugin-basic-ssl` |
| Database | PostgreSQL | Mặc định `postgresql://postgres:1111@localhost:5432/postgres`, đổi bằng `DATABASE_URL` |
| Runtime storage | `storage/uploads`, `storage/logs` | Khai báo trong `be/app/core/paths.py` |
| Static files | `/static` | FastAPI mount `storage/uploads` |
| AWS production dự kiến | S3, CloudFront, ALB, ECS/Fargate hoặc EC2, RDS PostgreSQL | Tách frontend, backend, database và object storage |

#### 4.5.2 Quy trình chạy hệ thống

Backend:

```bash
cd be
python -m venv venv310
venv310\Scripts\activate
pip install -r requirements.txt
python main.py
```

Frontend:

```bash
cd fe
npm install
npm run dev
```

Khi truy cập từ điện thoại trong LAN, cần dùng HTTPS hoặc localhost-equivalent để trình duyệt cho phép `navigator.mediaDevices.getUserMedia`. Nếu dùng HTTP qua IP LAN, tính năng camera có thể bị chặn nhưng upload ảnh từ thư viện vẫn dùng được.

#### 4.5.3 Tổ chức file runtime

```txt
storage/
├── uploads/
│   ├── answer_keys/omr/      # file đáp án tạm, xóa sau khi parse
│   ├── omr/                  # ảnh upload, overlay, crop, confidence JSON, ZIP
│   ├── omr_data/profiles/
│   ├── omr_templates/        # ảnh/PDF template
│   └── temp/
└── logs/
```

Các thư mục runtime được tạo tự động bởi `ensure_runtime_dirs()` khi backend khởi động.

**Bảng 4.16: File runtime và storage**

| Loại file | Vị trí lưu | Mục đích |
|---|---|---|
| Ảnh upload | `storage/uploads/omr/` | Lưu ảnh gốc người dùng gửi để chấm |
| Ảnh overlay | `storage/uploads/omr/` | Hiển thị kết quả chấm trực quan trên frontend |
| Crop MSSV | `storage/uploads/omr/` | Kiểm tra vùng nhận diện MSSV |
| Crop MCQ | `storage/uploads/omr/` | Kiểm tra vùng câu hỏi trắc nghiệm |
| Confidence JSON | `storage/uploads/omr/` | Lưu điểm tin cậy, score từng bong bóng và thông tin câu không chắc chắn |
| ZIP batch | `storage/uploads/omr/` | Gom overlay của nhiều ảnh để tải về sau khi chấm hàng loạt |
| Template | `storage/uploads/omr_templates/` | Lưu ảnh/PDF mẫu phiếu |
| Profile JSON | `storage/uploads/omr_data/profiles/` | Lưu cấu hình ROI, marker và tham số xử lý ảnh |
| File đáp án tạm | `storage/uploads/answer_keys/omr/` | Lưu file đáp án upload trong quá trình parse |
| Log | `storage/logs/` | Lưu log runtime của backend |

#### 4.5.4 Định hướng triển khai trên AWS

Khi chuyển từ môi trường local/LAN sang AWS, kiến trúc triển khai nên tách rõ phần tĩnh, API, cơ sở dữ liệu và file runtime:

* **Frontend:** build React bằng `npm run build`, upload thư mục `dist` lên Amazon S3 và phân phối qua CloudFront. CloudFront gắn chứng chỉ TLS từ AWS Certificate Manager để đảm bảo HTTPS cho camera browser.
* **Backend:** đóng gói FastAPI thành Docker image và chạy trên ECS/Fargate hoặc EC2 sau Application Load Balancer. Uvicorn/Gunicorn phục vụ API, health check đặt qua ALB.
* **Database:** dùng Amazon RDS PostgreSQL thay cho PostgreSQL local. Biến `DATABASE_URL` được cấu hình qua Secrets Manager hoặc Parameter Store, không ghi trực tiếp trong repo.
* **Runtime file:** các file lớn trong `storage/uploads` nên chuyển sang S3 object storage. SQL chỉ lưu metadata và object key/URL; không lưu ảnh, PDF, ZIP hoặc JSON telemetry lớn trực tiếp trong database.
* **Static URL:** `/static/...` trong local có thể được thay bằng CloudFront/S3 URL hoặc một endpoint backend ký URL truy cập. Điều này giúp nhiều backend instance cùng đọc/ghi một kho file chung.
* **Quan sát và vận hành:** log backend gửi CloudWatch Logs; metric cơ bản gồm số request chấm bài, thời gian xử lý trung bình, số lỗi decode và dung lượng S3. Batch lớn có thể tách sang SQS + worker để tránh request HTTP chạy quá lâu.

---

## CHƯƠNG 5. CÁC GIẢI PHÁP VÀ ĐÓNG GÓP NỔI BẬT

Chương này tập trung vào các vấn đề kỹ thuật đã gặp trong quá trình triển khai và cách hệ thống được chỉnh sửa trong code. Mỗi giải pháp được trình bày theo chuỗi: vấn đề thực tế, nguyên nhân kỹ thuật, module đã sửa, cách sửa hoạt động và test case liên quan.

### 5.1 Cơ chế MCQ Map Search Rescue - Tự động hiệu chỉnh lưới câu hỏi

#### 5.1.1 Vấn đề gặp phải

Trong thực tế, giáo viên chụp ảnh phiếu trong nhiều điều kiện khác nhau: ánh đèn huỳnh quang chênh lệch, ảnh chụp hơi nghiêng, học sinh tô không đủ đậm hoặc có vết tẩy xóa. Sau khi warp phiếu về ảnh chuẩn, lưới MCQ đôi khi vẫn bị lệch vài pixel theo chiều dọc hoặc khoảng cách dòng, làm nhiều câu bị đọc là không chắc chắn (`uncertain_count` cao) hoặc tô nhiều đáp án giả.

#### 5.1.2 Nguyên nhân kỹ thuật

Profile chỉ cung cấp ROI và tham số hình học ban đầu. Với ảnh chụp thực tế, bốn marker và vùng giấy có thể không khớp hoàn toàn với mẫu lý tưởng, khiến `line_h`, `top_center_y` hoặc ranh giới block MCQ bị lệch. Nếu chỉ decode một lần theo baseline, `_decode_mcq_with_map()` có thể lấy mẫu bong bóng sai tâm, đặc biệt ở các câu giữa/cuối block.

#### 5.1.3 Cách sửa trong code

Phần sửa nằm trong `be/app/services/omr/omr_service.py`, tại luồng `process_omr_exam()`. Sau lần decode MCQ baseline, hệ thống tính:

```python
search_uncertain_gate = max(4, round(0.60 * questions))
```

Nếu `initial_uncertain >= search_uncertain_gate`, service kích hoạt MCQ Map Search Rescue. Cơ chế này không chạy lại toàn bộ pipeline, mà chỉ thử các biến thể hình học của lưới MCQ:

* Điều chỉnh `line_h` theo nhiều hệ số scale để khớp khoảng cách dòng thực tế.
* Dịch `top_center_y` bằng các `top_shift_px` nhỏ để sửa sai lệch dòng đầu tiên.
* Tận dụng `block_bands` nếu marker nội bộ cho phép suy luận ranh giới từng block câu hỏi.
* So sánh ứng viên bằng chất lượng decode, ưu tiên giảm `uncertain_questions` và `double_mark_questions`.

Ứng viên được xếp hạng theo bộ giá trị:

```
rank = (uncertain_count, double_mark_count, -quality_score)
```

Ứng viên có `rank` nhỏ hơn baseline được chọn. Nếu không có ứng viên cải thiện, kết quả ban đầu được giữ lại và metadata ghi lý do `no-better-candidate` hoặc `below-uncertain-gate`.

Khi tìm được ứng viên tốt hơn, `omr_service.py` cập nhật `mcq_result`, ghi `MCQ_MAP_SEARCH_RESCUE` vào `warning_codes` và lưu metadata như `line_h_after`, `top_shift_px`, `initial_uncertain`, `final_uncertain`. Nhờ đó kết quả trả về không chỉ có điểm mà còn có bằng chứng về việc cơ chế rescue đã được dùng.

**Hình 5.1: Luồng MCQ Map Search Rescue**

```mermaid
flowchart TD
    A["Decode MCQ ban đầu"] --> B{"uncertain_count >= search gate?"}
    B -->|Không| Z["Trả kết quả ban đầu"]
    B -->|Có| C["Sinh danh sách line_scale"]
    C --> D["Sinh danh sách top_shift"]
    D --> E["Decode lại từng ứng viên"]
    E --> F["Tính rank: uncertain, double-mark, quality"]
    F --> G{"Ứng viên tốt hơn baseline?"}
    G -->|Có| H["Cập nhật mcq_result và ghi warning"]
    G -->|Không| Z
```

#### 5.1.4 Kết quả sau khi sửa

Khi ảnh bị lệch nhẹ và số câu không chắc chắn vượt ngưỡng, hệ thống có thêm cơ hội sửa lưới MCQ thay vì trả ngay kết quả baseline. Nếu rescue cải thiện được rank, số câu không chắc chắn giảm và bảng đáp án trả về ổn định hơn; nếu không cải thiện, baseline vẫn được giữ lại để tránh thay đổi sai. Test liên quan là `TC-GRADE-02`, trong đó ảnh có bong bóng tô mờ phải ghi `uncertain_count` và kích hoạt rescue khi vượt ngưỡng.

### 5.2 Thuật toán tính điểm bong bóng tổng hợp

#### 5.2.1 Vấn đề gặp phải

Phương pháp đơn giản nhất để xác định bong bóng đã tô là đếm số pixel đen trong ô (`countNonZero`). Tuy nhiên, phương pháp này có độ chính xác thấp khi bong bóng tô mờ hoặc khi nền ảnh không hoàn toàn trắng sau khi nhị phân hóa.

#### 5.2.2 Nguyên nhân kỹ thuật

Ảnh camera có ánh sáng không đều, bóng giấy, nhiễu từ đường kẻ và nét tô không đồng nhất. Threshold nhị phân có thể làm mất nét tô mờ hoặc phóng đại vùng nhiễu. Nếu chỉ dựa vào số pixel đen, một ô bị đường kẻ chạy qua hoặc nền tối cục bộ có thể có score gần giống ô thật sự được tô.

#### 5.2.3 Cách sửa trong code

Phần sửa nằm trong `be/app/services/omr/omr_mcq.py`, chủ yếu ở các hàm `_cell_density()`, `_cell_score()`, `_row_candidate_quality()` và `_decode_mcq_with_map()`. Thay vì chỉ dùng pixel đen, hệ thống dùng công thức tổng hợp kết hợp mật độ pixel nhị phân và thông tin từ ảnh xám gốc:

```
cell_score = 0.90 × density + 0.10 × darkness

Trong đó:
  density  = countNonZero(cell_binary) / cell_area   (inner_ratio = 0.78)
  darkness = 0.55 × fill_ratio
           + 0.30 × dark_mean
           + 0.15 × dark_p25

  fill_ratio = countNonZero(cell_binary) / area
  dark_mean  = 1 − mean(gray_cell) / 255
  dark_p25   = 1 − percentile(gray_cell, 25%) / 255
```

Công thức `row_candidate_quality` để chọn đáp án tốt nhất trong một hàng:

```
quality = 1.10 × best_score + 1.45 × margin + 0.25 × best_center + 0.12 × best_dark − left_penalty
```

Trong đó `margin` là khoảng cách giữa điểm cao nhất và nhì, `left_penalty` phạt nếu ô đầu tiên (cột A) có điểm cao nhưng biên độ quá nhỏ (thường là nhiễu từ đường kẻ trái).

Các đại lượng có ý nghĩa như sau:

| Đại lượng | Ý nghĩa |
|---|---|
| `density` | Tỷ lệ pixel đen trong vùng bong bóng sau nhị phân hóa |
| `darkness` | Độ tối tổng hợp từ ảnh xám, giúp giữ tín hiệu tô mờ |
| `margin` | Khoảng cách giữa đáp án tốt nhất và đáp án đứng thứ hai |
| `best_score` | Điểm density của lựa chọn tốt nhất |
| `best_center` | Mật độ vùng trung tâm, giảm nhiễu từ viền ô |
| `best_dark` | Độ tối bổ sung từ ảnh xám |
| `left_penalty` | Hình phạt cho nhiễu ở cột A khi biên độ không đủ rõ |

#### 5.2.4 Kết quả sau khi sửa

Cách tính mới giúp hệ thống ổn định hơn với bong bóng tô nhẹ, vì ảnh xám gốc vẫn đóng góp một phần vào score khi ảnh nhị phân bị mất nét. Đồng thời `margin`, `best_center` và `left_penalty` giúp hạn chế chọn nhầm do đường kẻ hoặc vết bẩn. Test liên quan gồm `TC-GRADE-01` với ảnh phiếu chuẩn và `TC-GRADE-02` với ảnh có câu không rõ.

### 5.3 Smart Camera Scanner trên trình duyệt

#### 5.3.1 Vấn đề gặp phải

Để chụp và chấm phiếu nhanh chóng từng tờ một, giáo viên cần giao diện camera trực tiếp trên điện thoại không cần cài ứng dụng native. Việc phát hiện khi nào phiếu đã được đặt đúng vị trí (không nghiêng, đủ 4 góc) là thách thức kỹ thuật cần giải quyết hoàn toàn phía client.

#### 5.3.2 Nguyên nhân kỹ thuật

Trình duyệt chỉ cho phép dùng camera qua `navigator.mediaDevices.getUserMedia` trong secure context như HTTPS hoặc localhost. Ngoài ra, frontend không thể chạy toàn bộ pipeline OpenCV nặng như backend để chấm ngay trên video frame. Vì vậy cần một bộ kiểm tra nhẹ ở client để quyết định khi nào ảnh đủ điều kiện chụp: nhìn thấy giấy, đủ 4 marker tối, marker có tương phản với nền và trạng thái ổn định qua nhiều frame.

#### 5.3.3 Cách sửa trong code

Hệ thống triển khai thuật toán phát hiện marker tại hàm `evaluateAlignment()` trong `MultichoicePage.tsx`. Mỗi 130ms, thuật toán phân tích một frame video:

1. Vẽ frame lên canvas ẩn, đọc pixel data.
2. Tính `centerLuma` (độ sáng trung bình vùng giữa viewfinder) để xác nhận đang nhìn thấy tờ giấy trắng.
3. Tính `darkLumaThreshold = clamp(centerLuma − 26, 28, 110)` — ngưỡng thích nghi theo điều kiện ánh sáng.
4. Tại 4 vị trí góc được cấu hình trong Form Profile, tính `darkRatio`:
   ```javascript
   lum_pixel = 0.299×R + 0.587×G + 0.114×B
   darkRatio = count(lum < darkLumaThreshold) / totalPixels
   ```
5. Điều kiện LOCKED khi cả ba đều thỏa:
   ```javascript
   hasFourDarkMarkers = all(darkRatio[c] >= 0.14)  // 4 marker đủ đen
   paperInsideFrame   = centerLuma >= 52             // giấy trắng ở giữa
   markerContrastOk   = (centerLuma − mean(markerLuma)) >= 20  // đủ tương phản
   ```
6. Sau 4 frame liên tiếp thỏa điều kiện → trạng thái `"locked"`, khung chuyển xanh.
7. Sau 1300ms duy trì `"locked"` → tự động chụp JPEG (quality = 0.92) và gọi API.

Luồng camera dùng `navigator.mediaDevices.getUserMedia` với camera sau nếu có. Frame video được vẽ lên canvas ẩn, sau đó frontend chỉ tính toán các đại lượng đơn giản như `centerLuma`, `darkRatio` và `markerLuma`. Cách làm này đủ nhẹ để chạy trên điện thoại và vẫn tận dụng cùng Form Profile với backend, vì vị trí marker được lấy từ profile đang chọn.

#### 5.3.4 Kết quả sau khi sửa

Smart Camera Scanner cho phép giáo viên căn phiếu trong khung và để hệ thống tự chụp khi đủ điều kiện. Khi môi trường không phải HTTPS/localhost, frontend hiển thị lỗi rõ ràng và người dùng vẫn có thể dùng chế độ tải ảnh từ thư viện. Test liên quan là `TC-CAM-01`, kiểm tra video hiển thị, trạng thái `searching/locked` và điều kiện tự chụp.

---

## CHƯƠNG 6. KẾT LUẬN VÀ HƯỚNG PHÁT TRIỂN

### 6.1 Kết luận

Đồ án đã hoàn thành xây dựng hệ thống web chấm điểm phiếu trắc nghiệm tự động với đầy đủ các chức năng đề ra ban đầu:

**Về kết quả kỹ thuật:**
- Pipeline xử lý ảnh 15 bước hoạt động ổn định với ảnh chụp điện thoại trong nhiều điều kiện ánh sáng và góc chụp khác nhau.
- Cơ chế MCQ Map Search Rescue tự động hiệu chỉnh lưới câu hỏi khi phát hiện nhiều bong bóng không rõ ràng.
- Smart Camera Scanner phát hiện phiếu theo thời gian thực và tự động chụp mà không cần thao tác của người dùng.
- Module crop vùng chữ viết tay đã hỗ trợ ROI `ho_ten` theo profile và sẵn sàng để thu thập dữ liệu, tích hợp OCR/học sâu ở phiên bản sau.
- Dữ liệu lịch sử chấm đã được tách sang bảng `omr_grade_result`, giúp quản lý nhiều lượt chấm ổn định hơn so với chỉ lưu trong `last_result`.

**Về tính thực tiễn:**
- Hệ thống không yêu cầu phần cứng đặc biệt, chỉ cần điện thoại thông minh và kết nối mạng LAN.
- Giao diện mobile-first dễ sử dụng, hỗ trợ đầy đủ tiếng Việt.
- Hỗ trợ xuất dữ liệu Excel và PDF, dễ tích hợp với quy trình quản lý điểm hiện có.

Các test case chức năng chính được thiết kế đều đạt kết quả như kỳ vọng trong môi trường kiểm thử.

### 6.2 Hướng phát triển

#### 6.2.1 Cải thiện độ chính xác nhận diện

- Thu thập bộ dữ liệu phiếu thực tế và ảnh crop chữ viết tay để huấn luyện hoặc tích hợp mô hình OCR/học sâu ở các phiên bản sau.
- Nghiên cứu áp dụng mô hình Vision Transformer (ViT) cho bài toán phân loại bong bóng.
- Bổ sung xử lý với phiếu bị gấp hoặc nhàu.

#### 6.2.2 Cải thiện hiệu năng

- Xử lý batch song song thay vì tuần tự bằng `asyncio.gather` hoặc Celery task queue, giảm thời gian chờ khi tải lên nhiều ảnh.
- Cache Form Profile trong memory hoặc Redis, đồng thời invalidate cache khi dev/editor lưu profile mới.
- Tách xử lý batch dài sang hàng đợi SQS/Celery worker khi triển khai cloud để tránh timeout HTTP.

#### 6.2.3 Tăng cường bảo mật

- Tích hợp JWT (JSON Web Token) với refresh token để xác thực không lưu trong `localStorage`.
- Thêm rate limiting cho API endpoint chấm điểm để ngăn lạm dụng.
- Hash mật khẩu bằng bcrypt/argon2 và lưu secret trong AWS Secrets Manager hoặc Parameter Store khi triển khai production.

#### 6.2.4 Mở rộng tính năng

- Triển khai production trên AWS với S3 + CloudFront cho frontend/static file, RDS PostgreSQL cho database và ECS/Fargate hoặc EC2 cho backend FastAPI.
- Chuyển runtime file từ `storage/uploads` cục bộ sang S3, đồng thời giữ SQL làm nguồn sự thật cho metadata và kết quả nghiệp vụ.
- Thêm tính năng so sánh kết quả nhiều bài thi (thống kê lớp học) và biểu đồ phân phối điểm.
- Xây dựng mobile app native (React Native hoặc Flutter) để tối ưu hơn trải nghiệm camera.
- Hỗ trợ tích hợp API cho hệ thống quản lý học tập (LMS) như Moodle.

#### 6.2.5 Hỗ trợ offline

- Sử dụng Service Worker và IndexedDB để cho phép chụp ảnh và lưu cục bộ khi mất mạng, đồng bộ kết quả khi có kết nối trở lại.

---

## TÀI LIỆU THAM KHẢO

[1] **OpenCV Team** (2024). *OpenCV Documentation — Computer Vision Library*, version 4.12. https://docs.opencv.org

[2] **Bradski, G. & Kaehler, A.** (2008). *Learning OpenCV: Computer Vision with the OpenCV Library*. O'Reilly Media.

[3] **Otsu, N.** (1979). A threshold selection method from gray-level histograms. *IEEE Transactions on Systems, Man, and Cybernetics*, 9(1), 62–66.

[4] **Canny, J.** (1986). A computational approach to edge detection. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 8(6), 679–698.

[5] **LeCun, Y., Bengio, Y., & Hinton, G.** (2015). Deep learning. *Nature*, 521(7553), 436–444.

[6] **Ioffe, S. & Szegedy, C.** (2015). Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift. *ICML 2015*. arXiv:1502.03167.

[7] **Srivastava, N., Hinton, G., Krizhevsky, A., Sutskever, I., & Salakhutdinov, R.** (2014). Dropout: A Simple Way to Prevent Neural Networks from Overfitting. *Journal of Machine Learning Research*, 15, 1929–1958.

[8] **PyTorch Team** (2024). *PyTorch Documentation*, version 2.9. https://pytorch.org/docs

[9] **Tiangolo, S.** (2024). *FastAPI Documentation — Modern, Fast Web Framework for Python*. https://fastapi.tiangolo.com

[10] **SQLAlchemy Team** (2024). *SQLAlchemy 2.0 Documentation*. https://docs.sqlalchemy.org

[11] **React Team** (2024). *React — The Library for Web and Native User Interfaces*, version 19. https://react.dev

[12] **W3C** (2023). *Media Capture and Streams — WebRTC Specification*. https://www.w3.org/TR/mediacapture-streams/

[13] **PostgreSQL Global Development Group** (2024). *PostgreSQL Documentation*, version 16. https://www.postgresql.org/docs/

[14] **Vite Team** (2024). *Vite — Next Generation Frontend Tooling*, version 7. https://vitejs.dev
