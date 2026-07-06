## Biểu đồ Use Case Diagram — Hình 2.1 (overview.png)

Biểu đồ UML Use Case với 1 actor (Giáo viên) và 13 use case trong 4 nhóm bên trong system boundary. Quan hệ `«include»`: UC-08 và UC-09 đều bắt buộc gọi UC-10. Quan hệ `«extend»`: UC-09 là mở rộng tùy chọn của UC-08.

### PlantUML Use Case Diagram — Hình 2.1

```plantuml
@startuml Hinh-2.1-Use-Case
left to right direction
skinparam packageStyle rectangle
skinparam defaultTextAlignment center
skinparam usecase {
  BackgroundColor #ffffff
  BorderColor     #555555
}
skinparam actor {
  BackgroundColor #dae8fc
  BorderColor     #6c8ebf
}

actor "Giáo viên" as GV

rectangle "Hệ thống Quản lý Chấm thi OMR" {
  package "Nhóm Xác thực" {
    usecase "Đăng nhập\n(UC-01)" as UC01
    usecase "Đăng ký\n(UC-02)"  as UC02
  }
  package "Nhóm Mẫu phiếu & Form Profile" {
    usecase "Xem / chọn Form Profile\n& ảnh mẫu (UC-03)" as UC03
  }
  package "Nhóm Quản lý bài thi & đáp án" {
    usecase "Tạo / sửa / xóa bài thi\n(UC-04)"    as UC04
    usecase "Cấu hình đáp án\ntheo mã đề (UC-05)" as UC05
  }
  package "Nhóm Chấm điểm & kết quả" {
    usecase "Smart Camera Scanner\n(UC-06)"  as UC06
    usecase "Upload ảnh đơn / lô\n(UC-07)"  as UC07
    usecase "Chấm 1 phiếu\n(UC-08)"         as UC08
    usecase "Chấm hàng loạt\n(UC-09)"       as UC09
    usecase "Lưu lịch sử chấm\n(UC-10)"     as UC10
    usecase "Xem thống kê\n(UC-11)"         as UC11
    usecase "Xem chi tiết bản ghi\n(UC-12)" as UC12
    usecase "Xuất Excel / PDF\n(UC-13)"     as UC13
  }
}

GV --> UC01
GV .> UC02  : "chưa có tài khoản"
GV --> UC03
GV --> UC04
UC04 --> UC05
GV --> UC06
GV --> UC07
GV --> UC08
GV --> UC09
GV --> UC11
GV --> UC12
GV --> UC13
UC08 .> UC10 : <<include>>
UC09 .> UC10 : <<include>>
UC09 .> UC08 : <<extend>>

@enduml
```

---

## Biểu đồ Activity Diagram — Hình 2.2 (Quy trình nghiệp vụ chấm bài)

@startuml Hinh-2.2-Activity
skinparam defaultTextAlignment left

|Giáo viên|
start
:Đăng nhập / Đăng ký (UC-01, UC-02)\nauth.py — POST /login hoặc POST /auth/google;
:Xem Form Profile & ảnh mẫu (UC-03);
:Tạo / chỉnh sửa bài thi (UC-04)\nassignments.py;
:Cấu hình đáp án theo mã đề (UC-05)\nanswer_sets: {"001": [0,2,1,...], ...};

|Hệ thống|
if (Bài thi có đáp án?) then ([Không])
  :HTTP 400 — "Kho đáp án của bài thi đang trống";
  stop
else ([Có])
endif

|Giáo viên|
if (Cách lấy ảnh?) then ([Camera — UC-06])
  :Smart Camera Scanner\nWebRTC MediaDevices API\nPhân tích frame mỗi 130ms\nAuto-capture khi đủ 4 marker → pickedFiles;
else ([Upload — UC-07])
  :Chọn 1 hoặc nhiều file ảnh từ thư viện\n→ thêm vào pickedFiles;
endif
:Nhấn nút **Chấm bài**;

|Hệ thống|
if (pickedFiles.length > 1?) then ([Có])
  :POST /api/omr/grade-batch (UC-09)\ntối đa 50 ảnh · xử lý tuần tự;
else ([Không])
  :POST /api/omr/grade (UC-08);
endif

if (Lỗi đọc ảnh / warp?) then ([Có])
  :HTTP 400 · Không lưu lịch sử\nbatch: ghi success:false, không dừng lô;
  stop
else ([Không])
endif

:Pipeline OMR — 15 bước\nomr_service.py · process_omr_exam()\nBước 4 · Warp → 1000×1400 px\nBước 5–6 · Xám & nhị phân hóa\nBước 7 · Phát hiện 4 marker\nBước 8–10 · Dựng ROI\nBước 11 · Decode MSSV\nBước 12 · Decode mã đề (3 chữ số)\nBước 13 · Decode MCQ baseline;

if (uncertain ≥ max(4, ⌈0.60×Q⌉)?) then ([Có])
  :Bước 14 · MCQ Map Search Rescue\n54 ứng viên: 6 line_scale × 9 top_shift\nrank = (uncertain↑, double_mark↑, −quality)\nChỉ thay baseline nếu cải thiện đủ;
endif

:Bước 15 · Tính điểm & sinh ảnh\nTra answer_sets theo mã đề từ phiếu\nSo sánh từng câu · overlay màu\nCrop MSSV/MCQ/họ tên · JSON confidence;

:UC-10 · Lưu omr_grade_result\nuid · aid · MSSV · mã đề · điểm\nOverlay · crop · JSON confidence\nCập nhật last_result & graded_count;

if (Chấm lô?) then ([Có])
  :Đóng gói ZIP (ZIP_DEFLATED)\nzip_url · success_count · failed_count;
endif

:Trả JSON kết quả về frontend\nĐiểm · MSSV · Mã đề\nURL overlay · crop · grade_result_id;

|Giáo viên|
fork
  :UC-11 · Xem & lọc thống kê\nStatsPanel.tsx;
fork again
  :UC-12 · Xem chi tiết bản ghi\n+ ảnh overlay;
fork again
  :UC-13 · Xuất Excel / PDF\nExportPanel.tsx;
end fork
stop

@enduml

---

## PlantUML Activity Diagram — Hình 2.3 (Quy trình nghiệp vụ tổng thể)

Hình 2.3 được xuất ra 2 ảnh (`usecase_workflow1.png` và `usecase_workflow2.png`) — hai `@startuml` riêng tương ứng.

### Phần A — Luồng người dùng (UC-01 → UC-09) → `usecase_workflow1.png`

```plantuml
@startuml Hinh-2.3a-Workflow-User
skinparam defaultTextAlignment left

|Giáo viên|
start

group Giai đoạn 1 — Chuẩn bị bài thi
:Đăng nhập (UC-01)\nPOST /login hoặc POST /auth/google — auth.py;
note right: Đăng ký (UC-02)\nnếu chưa có tài khoản
:Xem / chọn Form Profile & ảnh mẫu (UC-03)\nprofiles.py · templates.py;
:Tạo / sửa bài thi (UC-04)\nassignments.py → omr_assignment;
:Cấu hình đáp án theo mã đề (UC-05)\nanswer_sets JSON: {"001":[0,2,1,...]};
end group

|Hệ thống|
if (Bài thi có đáp án?) then ([Không])
  :HTTP 400 — "Kho đáp án của bài thi đang trống";
  stop
else ([Có])
endif

|Giáo viên|
group Giai đoạn 2 — Thu nhận ảnh
if (Cách lấy ảnh?) then ([Camera — UC-06])
  :Smart Camera Scanner\nWebRTC MediaDevices API\nPhân tích frame mỗi 130ms\nTự động capture khi đủ 4 marker định vị\n→ thêm vào danh sách pickedFiles;
else ([Upload — UC-07])
  :Upload 1 hoặc nhiều file ảnh từ thiết bị\n→ thêm vào danh sách pickedFiles;
endif
:Nhấn nút **Chấm bài**;
end group

|Hệ thống|
group Giai đoạn 3 — Định tuyến
if (pickedFiles.length > 1?) then ([Có — Lô])
  :UC-09 · POST /api/omr/grade-batch\ntối đa 50 ảnh · xử lý tuần tự · grading.py;
else ([Không — Đơn])
  :UC-08 · POST /api/omr/grade\ngrading.py;
endif

if (Lỗi đọc ảnh / warp?) then ([Có])
  :HTTP 400 · Không lưu lịch sử\n(batch: ghi success:false, không dừng lô);
  stop
else ([Không])
endif
end group

:▶ Sang Phần B — Pipeline OMR;
stop
@enduml
```

### Phần B — Pipeline & kết quả (UC-10 → UC-13) → `usecase_workflow2.png`

```plantuml
@startuml Hinh-2.3b-Workflow-Backend
skinparam defaultTextAlignment left

|Hệ thống|
start
:process_omr_exam() — omr_service.py\n──────────────────────────────────────────\nBước 4 · Warp phối cảnh → 1000×1400 px\nBước 5–6 · Chuyển xám & nhị phân hóa thích nghi\nBước 7 · Phát hiện 4 marker định vị\nBước 8–10 · Dựng và tinh chỉnh các vùng ROI\nBước 11 · Decode MSSV\nBước 12 · Decode mã đề (3 chữ số)\nBước 13 · Decode MCQ baseline;

if (uncertain ≥ max(4, ⌈0.60×Q⌉)?) then ([Có])
  :Bước 14 · MCQ Map Search Rescue\n54 ứng viên: 6 line_scale × 9 top_shift\nrank = (uncertain↑, double_mark↑, −quality)\nChỉ thay baseline nếu cải thiện đủ;
endif

:Bước 15 · Tính điểm & sinh ảnh\nTra answer_sets theo mã đề đọc từ phiếu\nSo sánh từng câu · Tạo overlay màu\nCrop vùng MSSV/MCQ/họ tên · JSON confidence;

:UC-10 · Lưu omr_grade_result\nuid · aid · MSSV · mã đề · điểm\nảnh overlay · crop · JSON confidence\nCập nhật last_result & graded_count;

if (Chấm lô?) then ([Có])
  :Đóng gói ZIP (ZIP_DEFLATED)\nzip_url · success_count · failed_count;
endif

:Trả JSON kết quả về frontend\nĐiểm · MSSV · Mã đề · URL overlay\nURL crop · confidence · grade_result_id;

|Giáo viên|
fork
  :UC-11 · Xem & lọc thống kê\nStatsPanel.tsx;
fork again
  :UC-12 · Xem chi tiết bản ghi\n/multichoice/record-detail/:testId/:recordId;
fork again
  :UC-13 · Xuất Excel / PDF\nExportPanel.tsx;
end fork
stop
@enduml
```

---

## Biểu đồ trình tự — Luồng chấm một phiếu (Hình 4.x)

```plantuml
@startuml Seq-Grade-Single
skinparam sequenceMessageAlign center
skinparam defaultFontSize 12
skinparam responseMessageBelowArrow true

actor "Giáo viên" as GV
participant "MultichoicePage\n(:Frontend)" as FE
participant "grading.py\n(:Backend)" as BE
participant "omr_service.py\n(:OMRService)" as SVC
database "PostgreSQL\n(:DB)" as DB

GV -> FE : Chọn ảnh từ thư viện\nhoặc Camera tự chụp
FE -> FE : Kiểm tra bài thi đang mở\nvà bộ đáp án
FE -> BE : POST /api/omr/grade\n(FormData: file, uid, aid,\nanswers, profile_code)
activate BE
BE -> DB : SELECT omr_assignment\nVÀ answer_sets
DB --> BE : assignment, answer_sets, profile
BE -> SVC : process_omr_exam()\nqua run_in_threadpool
activate SVC
SVC -> SVC : Bước 1–6: Warp ảnh\nBinarize · Detect marker\nBuild ROI
SVC -> SVC : Bước 7–11: Decode MSSV\nDecode mã đề · Decode MCQ
alt uncertain ≥ max(4, ⌈0.6×Q⌉)
    SVC -> SVC : Bước 14: MCQ Map\nSearch Rescue\n(thử 54 biến thể lưới)
end
SVC -> SVC : Bước 15: Tính điểm\nvà sinh ảnh overlay/crop
SVC --> BE : OMRResult\n(score, student_id, exam_code,\nanswers, overlay_path)
deactivate SVC
BE -> DB : INSERT omr_grade_result\nUPDATE omr_assignment.last_result
BE --> FE : JSON\n(score, student_id, overlay_url, detail)
deactivate BE
FE --> GV : Hiển thị điểm, ảnh overlay\nvà bảng đúng/sai từng câu
@enduml
```

---

## Biểu đồ lớp — Mô hình dữ liệu backend (Hình 4.x)

```plantuml
@startuml Class-Backend-Models
skinparam classAttributeIconSize 0
skinparam classFontSize 12
left to right direction

class User {
  +uuid : Integer <<PK>>
  +user_name : String
  +email : String
  +phone : String
  +password : String
}

class OMRAssignment {
  +aid : Integer <<PK>>
  +uuid : Integer <<FK>>
  +title : String
  +question_count : Integer
  +total_points : Integer
  +graded_count : Integer
  +answer_sets : JSON
  +active_code : String
  +last_result : JSON
  +created_at : DateTime
}

class OMRTest {
  +omrid : Integer <<PK>>
  +uuid : Integer <<FK>>
  +omr_name : String
  +omr_code : String
  +omr_quest : Integer
  +omr_answer : JSON
  +template_image : String
  +options : Integer
  +rows_per_block : Integer
  +student_id_digits : Integer
}

class OMRGradeResult {
  +grid : Integer <<PK>>
  +aid : Integer <<FK>>
  +uuid : Integer <<FK>>
  +omrid : Integer <<FK>>
  +source : String
  +student_id : String
  +exam_code : String
  +score : String
  +result_image : String
  +result_json : JSON
  +created_at : DateTime
}

User "1" --o "0..*" OMRAssignment : sở hữu
User "1" --o "0..*" OMRTest : tạo
OMRAssignment "1" --o "0..*" OMRGradeResult : chứa
User "1" --o "0..*" OMRGradeResult : thuộc về
OMRTest "1" --o "0..*" OMRGradeResult : dùng profile
@enduml
```

---

## Biểu đồ máy trạng thái — Smart Camera Scanner (Hình 5.x)

```plantuml
@startuml State-CameraScanner
skinparam defaultFontSize 12

[*] --> IDLE

state "IDLE\n(Camera tắt)" as IDLE
state "ACTIVE\n(Stream video bắt đầu)" as ACTIVE
state "DETECTING\n(Phân tích frame 130ms)" as DETECTING
state "LOCKED\n(Đủ 4 marker liên tiếp ≥3 frame)" as LOCKED
state "AUTO_CAPTURED\n(canvas.toBlob → pickedFiles)" as CAPTURED

IDLE --> ACTIVE : Giáo viên nhấn "Bật Camera"
ACTIVE --> IDLE : Nhấn "Tắt Camera"
ACTIVE --> DETECTING : Stream video bắt đầu
DETECTING --> DETECTING : Chưa đủ điều kiện\n(centerLuma / darkRatio thấp)
DETECTING --> LOCKED : hasFourDarkMarkers\n& paperInsideFrame\n& markerContrastOk ≥ 3 frame
LOCKED --> DETECTING : Mất marker / điều kiện thay đổi
LOCKED --> CAPTURED : Giữ LOCKED ≥ 1300ms\ncanvas.toBlob(JPEG quality=0.92)
CAPTURED --> DETECTING : Ảnh thêm vào pickedFiles\ntiếp tục theo dõi phiếu tiếp
DETECTING --> IDLE : Nhấn "Tắt Camera"
LOCKED --> IDLE : Nhấn "Tắt Camera"
@enduml
```

---

## Biểu đồ gói kiến trúc Backend — figures/Backend.png

```plantuml
@startuml Backend-Package-Architecture
skinparam packageStyle rectangle
skinparam defaultFontSize 11

package "App Entry" as AppEntry {
  [App]
  [Middleware]
  [RuntimeInit]
}

package "API Layer" as ApiLayer {
  [AuthAPI]
  [AssignmentAPI]
  [GradingAPI]
  [GradeResultAPI]
  [ProfileAPI]
  [TemplateAPI]
}

package "Business Services" as BusinessSvc {
  [ValidateUID]
  [ResolveProfile]
  [BuildRuntimeConfig]
  [PersistGradeResult]
}

package "OMR Processing" as OMRProcessing {
  [OMRService]
  [Preprocess]
  [MarkerUtils]
  [Layout]
  [MCQ]
  [Numeric]
  [Visualize]
  [Scoring]
}

package "Data Access" as DataAccess {
  [OMRAssignment]
  [OMRTest]
  [OMRGradeResult]
  [User]
  [DatabaseSession]
}

package "Infrastructure" as Infrastructure {
  [RuntimePaths]
  [RuntimeSchema]
}

AppEntry ..> ApiLayer
ApiLayer ..> BusinessSvc
ApiLayer ..> DataAccess
BusinessSvc ..> OMRProcessing
BusinessSvc ..> DataAccess
OMRProcessing ..> Infrastructure

@enduml
```

---

## Biểu đồ gói kiến trúc Frontend — figures/Frontend.png

```plantuml
@startuml Frontend-Package-Architecture
skinparam packageStyle rectangle
skinparam defaultFontSize 11

package "App" as AppPkg {
  class App
  class Main
}

package "Pages" as MainPages {
  class LandingPage
  class LoginPage
  class RegisterPage
  class HomePage
}

package "Feature OMR" as FeatureOMR {

  package "OMR Pages" as OMRPages {
    class MultichoicePage
  }

  package "OMR Components" as OMRComponents {
    class StatsPanel
    class ExportPanel
    class OmrProfileRoiEditor
  }

  package "OMR Types" as OMRTypes {
    class Types
    class Utils
  }

}

package "Shared Components" as SharedComponents {
  class TopMenu
  class UserSidebar
  class ViewImageModal
  class GoogleAuthButton
}

package "Config and Utils" as ConfigUtils {
  class APIConfig
  class AuthStorage
}

package "Backend API" as BackendAPI {
  class FastAPI
}

AppPkg ..> MainPages
AppPkg ..> FeatureOMR
OMRPages ..> OMRComponents
OMRPages ..> SharedComponents
OMRPages ..> ConfigUtils
MainPages ..> SharedComponents
MainPages ..> ConfigUtils
ConfigUtils ..> BackendAPI

@enduml
```

---

## Biểu đồ ERD — Mô hình cơ sở dữ liệu — figures/ERD.png

```plantuml
@startuml ERD-Database
hide circle
hide empty methods
skinparam classAttributeIconSize 0
skinparam defaultFontSize 11
skinparam linetype ortho
skinparam entity {
  BackgroundColor #FEFEFE
  BorderColor #555555
}

entity "users" as users {
  * uuid : INTEGER <<PK>>
  --
  user_name : VARCHAR
  email : VARCHAR
  phone : VARCHAR
  password : VARCHAR
}

entity "omr_assignment" as assignment {
  * aid : INTEGER <<PK>>
  --
  # uuid : INTEGER <<FK>>
  title : VARCHAR
  question_count : INTEGER
  total_points : INTEGER
  graded_count : INTEGER
  active_code : VARCHAR
  answer_sets : JSON
  last_result : JSON
  created_at : TIMESTAMP
}

entity "omr_test" as test {
  * omrid : INTEGER <<PK>>
  --
  # uuid : INTEGER <<FK>>
  omr_name : VARCHAR
  omr_code : VARCHAR
  omr_quest : INTEGER
  options : INTEGER
  rows_per_block : INTEGER
  student_id_digits : INTEGER
  info_fields : JSON
  omr_answer : JSON
  template_image : VARCHAR
}

entity "omr_grade_result" as grade {
  * grid : INTEGER <<PK>>
  --
  # aid : INTEGER <<FK>>
  # uuid : INTEGER <<FK>>
  # omrid : INTEGER <<FK>>
  source : VARCHAR
  student_id : VARCHAR
  exam_code : VARCHAR
  score : VARCHAR
  result_image : VARCHAR
  sid_crop_image : VARCHAR
  mcq_crop_image : VARCHAR
  bubble_confidence_json : VARCHAR
  result_json : JSON
  created_at : TIMESTAMP
}

users ||--o{ assignment : "1 giáo viên\nsở hữu nhiều bài thi"
users ||--o{ test : "1 giáo viên\ntạo nhiều template"
users ||--o{ grade : "1 giáo viên\ncó nhiều lượt chấm"
assignment ||--o{ grade : "1 bài thi\nchứa nhiều lượt chấm"
test ||--o{ grade : "1 template\ndùng cho nhiều lượt chấm"

@enduml
```

---

## Biểu đồ hoạt động — Pipeline xử lý ảnh OMR 15 bước — figures/activity_diagram.png

```plantuml
@startuml OMR-Pipeline-15-Steps
skinparam defaultTextAlignment left
skinparam defaultFontSize 11
skinparam ActivityBorderColor #555555
skinparam ActivityBackgroundColor #f8f8f8
skinparam ActivityDiamondBorderColor #d6b656
skinparam ActivityDiamondBackgroundColor #fff2cc

start

:Bước 1 · Nhận request và đọc ảnh\nomr_service.py — cv2.imread()\nLưu file upload vào thư mục runtime\n→ Ma trận ảnh BGR;

if (Đọc ảnh thành công?) then ([Không])
  #ffcccc:HTTP 400 — không lưu lịch sử;
  stop
else ([Có])
endif

:Bước 2 · Chuẩn hóa tham số bài thi\nSố câu · số lựa chọn · số dòng/block\nSố chữ số MSSV · mã Form Profile;

if (Profile có crop quad?) then ([Có])
  :Bước 3 · Cắt ảnh ban đầu\nLoại bỏ vùng thừa ngoài tờ giấy;
else ([Không — bỏ qua])
endif

:Bước 4 · Chuẩn hóa phối cảnh toàn trang\nomr_preprocess.py\n① Corner-markers: dùng tâm 4 marker góc\n② Page-contour: Canny → approxPolyDP (4 điểm)\n③ Resize-only (fallback)\ncv2.getPerspectiveTransform → warpPerspective\n→ Ảnh chuẩn 1000×1400 px;

:Bước 5 · Chuyển xám & cân bằng nền\ncv2.cvtColor(BGR2GRAY)\nMorphology Close (ellipse k×k) → ước lượng nền cục bộ\ncv2.divide(gray, bg, scale=255) → chuẩn hóa ánh sáng\nGaussianBlur 3×3;

:Bước 6 · Nhị phân hóa ảnh\nomr_preprocess.py — profile_threshold_mode\n① Otsu: ngưỡng toàn cục tự động\n② Adaptive Gaussian: block 35×35, C=7\n③ Hybrid: bitwise_or(Otsu, Adaptive)\n→ Ảnh nhị phân đảo màu + MORPH_OPEN 2×2;

:Bước 7 · Phát hiện marker định vị\nomr_marker_utils.py\nTìm vùng đen đúng kích thước và hình dạng\n→ Tọa độ marker / anchor;

:Bước 8 · Suy luận hình học vùng MCQ\nomr_layout.py\nSố block · khoảng cách dòng (line_height)\nVị trí dòng đầu tiên trong mỗi block;

:Bước 9 · Dựng các ROI cần đọc\nomr_layout.py\nROI MSSV · ROI mã đề · ROI MCQ · ROI họ tên\nDựa trên profile + marker + ảnh đã warp;

:Bước 10 · Tinh chỉnh ROI vùng MCQ\nHiệu chỉnh sai lệch vài pixel theo chiều dọc\nDựa trên marker nội bộ hoặc cấu trúc lưới;

:Bước 11 · Decode MSSV\nomr_numeric.py — _decode_numeric_columns()\nLưới D cột × 10 hàng (chữ số 0–9)\nScore = 0.55×fill + 0.30×dark + 0.15×P₂₅\nAuto-switch nếu phát hiện lệch write-row;

:Bước 12 · Decode mã đề\nomr_numeric.py — 3 cột × 10 hàng\nZero-padding → tra answer_sets → chọn bộ đáp án;

:Bước 13 · Decode MCQ baseline\nomr_mcq.py — _decode_mcq_with_map()\nLưới (câu × lựa chọn) · dynamic_mark tự điều chỉnh\nPhát hiện: double-mark · noise gate · uncertain_questions;

if (uncertain ≥ max(4, ⌈0.60×Q⌉)?) then ([Có — kích hoạt Rescue])
  :Bước 14 · MCQ Map Search Rescue\nomr_mcq.py\n54 ứng viên: 6 line_scale × 9 top_shift\nrank = (uncertain↑, double_mark↑, −quality)\nChỉ thay baseline nếu đủ điều kiện bảo thủ\nghi MCQ_MAP_SEARCH_RESCUE → warning_codes;
else ([Không — giữ kết quả baseline])
endif

:Bước 15 · Tính điểm, lưu kết quả & sinh ảnh kiểm tra\nomr_scoring.py · omr_visualize.py · shared.py\nTra answer_sets theo mã đề đọc từ phiếu\nSo sánh từng câu → điểm + đúng/sai/bỏ/uncertain\nTạo overlay màu · crop MSSV/MCQ · JSON confidence\nINSERT omr_grade_result · UPDATE last_result & graded_count;

stop
@enduml
```

---

## Biểu đồ hoạt động — Cơ chế tự hiệu chỉnh lưới câu hỏi (MCQ Map Search Rescue) — figures/MCQ_Map_Search_Rescue.png

```plantuml
@startuml MCQ-Map-Search-Rescue-Flow
skinparam defaultTextAlignment left
skinparam defaultFontSize 11
skinparam ActivityBorderColor #555555
skinparam ActivityBackgroundColor #f8f8f8
skinparam ActivityDiamondBorderColor #d6b656
skinparam ActivityDiamondBackgroundColor #fff2cc

start

:Kết quả baseline MCQ (sau bước 13)\ninitial_uncertain = |uncertain_questions|\nbaseline_double = |double_mark_questions|\nbaseline_quality = _mcq_quality(mcq_result);

if (Rescue bị tắt qua profile\n(disable_mcq_rescue)\nhoặc long-form mode?) then ([Có])
  #f8cecc:Giữ kết quả baseline;
  stop
else ([Không])
endif

:gate = max(4, round(0.60 × Q));

note right
  Ví dụ: Q = 40 câu → gate = 24
  Q = 80 câu → gate = 48
  Rescue chỉ kích hoạt khi phiếu
  thực sự bị lệch lưới nghiêm trọng
end note

if (initial_uncertain ≥ gate?) then ([Không — dưới ngưỡng])
  #f8cecc:Giữ kết quả baseline\nMetadata: reason = "below-uncertain-gate";
  stop
else ([Có — vượt ngưỡng, kích hoạt Rescue])
endif

:Khởi tạo:\nbest_rank = (initial_uncertain, baseline_double, −baseline_quality)\nbest_result = mcq_result · best_line_h = line_h · best_shift_px = 0;

partition "Tìm kiếm lưới tối ưu — 54 ứng viên (6 scale × 9 shift)" {
  while (Còn ứng viên chưa thử?\n[6 line_scale × 9 shift_multiplier]) is ([Có])
    :Tính cand_line_h = clamp(line_h × scale, 6, 44)\nscale ∈ [1.00, 0.92, 0.88, 1.08, 0.84, 1.16]\ncand_shift_px = shift_mul × cand_line_h\nshift_mul ∈ [0.0, ±0.5, ±1.0, ±1.5, ±2.0];
    :Gọi _decode_mcq_with_map()\nvới line_h=cand_line_h, top_shift_px=cand_shift_px;
    :cand_rank = (uncertain_count↑, double_mark_count↑, −quality↓)\nquality = 10×|câu chắc| + 6×|4 câu đầu chắc| + 70×Σmargin;
    if (cand_rank < best_rank?) then ([Có — cải thiện])
      :best_rank ← cand_rank\nbest_result, best_line_h, best_shift_px ← ứng viên này;
    endif
  endwhile ([Không — đã thử hết 54 ứng viên])
}

:uncertain_gain = initial_uncertain − best_uncertain;

if (uncertain_gain ≥ 2?) then ([Có])
  :improved = True;
else ([Không])
  if (uncertain_gain ≥ 1\nvà double_mark không tăng?) then ([Có])
    :improved = True;
  else ([Không])
    if (uncertain_gain = 0\nvà double_mark giảm?) then ([Có])
      :improved = True;
    else ([Không])
      :improved = False;
    endif
  endif
endif

if (improved?) then ([Có — tìm được lưới tốt hơn])
  #d5e8d4:Áp dụng best_result làm kết quả MCQ\nCập nhật line_h ← best_line_h\ntop_center_y ← top_center_y + best_shift_px\nGhi MCQ_MAP_SEARCH_RESCUE → warning_codes;
else ([Không — không có ứng viên đủ tốt])
  #f8cecc:Giữ nguyên kết quả baseline;
endif

:Lưu metadata đầy đủ vào kết quả:\n  used (True/False) · reason · tested\n  initial_uncertain → final_uncertain\n  line_h_before / line_h_after · top_shift_px;

stop
@enduml
```

---

## Luồng chấm một phiếu — figures/seq_grade_single.png

```plantuml
@startuml seq_grade_single
title Luồng chấm một phiếu — POST /api/omr/grade

actor "Giáo viên" as User
participant "MultichoicePage\n(Frontend)" as FE
participant "grading.py\n(ApiLayer)" as API
participant "shared.py\n(BusinessServices)" as SVC
participant "omr_service.py\n(OMRProcessing)" as OMR
database "PostgreSQL" as DB
collections "FileSystem\n(storage/uploads/omr)" as FS

User -> FE : Chọn ảnh hoặc Smart Camera chụp
FE -> API : POST /api/omr/grade\n(file, uid, aid, answers,\nprofile_code, num_questions, ...)

API -> SVC : _resolve_profile(form_profile_code)
SVC -> FS : Đọc JSON profile
FS --> SVC : profile dict
SVC --> API : profile

API -> SVC : _build_runtime_config(profile, params)
SVC --> API : runtime config dict

alt uid != None
    API -> SVC : _validate_uid(uid)
    SVC -> DB : SELECT user WHERE uuid = uid
    DB --> SVC : user record
    SVC --> API : uid_checked
end

alt aid != None
    API -> DB : SELECT omr_assignment\nWHERE uuid=uid AND aid=aid
    DB --> API : record (answer_sets, active_code)
    API -> API : _build_assignment_answer_key_map()\nanswer_source = "assignment-code-map"
else omr_test_id != None
    API -> DB : SELECT omr_test WHERE omrid=omr_test_id
    DB --> API : answer_key
    API -> API : answer_source = "omr-test"
else answers trống — chế độ tự nhận diện
    API -> API : answer_key = None\nanswer_source = "auto"
else answers / file cung cấp trực tiếp
    API -> API : _resolve_shared_answer_key(answers, file)\nanswer_source = "manual"
end

API -> FS : Lưu file upload\n(storage/uploads/omr/<filename>)

alt auto mode (answer_key == None)
    API -> OMR : _resolve_answer_key_auto_from_sheet()\n(chạy pipeline để đọc mã đề từ ảnh)
    OMR --> API : answer_key, matched_test, detect_probe
end

API -> OMR : run_in_threadpool(\n  process_omr_exam,\n  image_path, answer_key,\n  answer_key_by_code, runtime config\n)

note over OMR
  Pipeline 15 bước:
  warp → nhị phân hóa → marker
  → layout ROI → decode MSSV
  → decode mã đề → decode MCQ
  → MCQ Map Search Rescue (nếu cần)
  → tính điểm → visualize
end note

OMR -> FS : Lưu overlay, sid_crop,\nmcq_crop, JSON confidence
FS --> OMR : file paths
OMR --> API : OMRResult dict

alt "error" in result
    API --> FE : HTTP 400 + {error}
else
    API -> SVC : _persist_omr_grade_result(\n  db, uid, aid, omrid,\n  source, file_name, result\n)
    SVC -> DB : INSERT omr_grade_result\nUPDATE omr_assignment.last_result
    DB --> SVC : grade_result_id
    SVC --> API : grade_result_id
    API --> FE : HTTP 200\n{score, student_id, exam_code,\nimage_url, sid_crop_url, mcq_crop_url,\nbubble_confidence_json_url,\ngrade_result_id, answer_source,\nmatched_omr_test, auto_detect}
end

FE --> User : Hiển thị điểm, overlay,\nbảng đúng/sai

@enduml
```

---

## Luồng chấm hàng loạt — figures/seq_grade_batch.png

```plantuml
@startuml seq_grade_batch
title Luồng chấm hàng loạt — POST /api/omr/grade-batch

actor "Giáo viên" as User
participant "MultichoicePage\n(Frontend)" as FE
participant "grading.py\n(ApiLayer)" as API
participant "shared.py\n(BusinessServices)" as SVC
participant "omr_service.py\n(OMRProcessing)" as OMR
database "PostgreSQL" as DB
collections "FileSystem\n(storage/uploads/omr)" as FS

User -> FE : Chọn nhiều ảnh (tối đa 50)
FE -> API : POST /api/omr/grade-batch\n(files[], uid, aid, answers,\nprofile_code, num_questions, ...)

note over API : Resolve profile, runtime config\nvà đáp án một lần cho cả lô

API -> SVC : _resolve_profile() + _build_runtime_config()
SVC --> API : runtime config

API -> SVC : _validate_uid(uid)
SVC -> DB : SELECT user WHERE uuid = uid
DB --> SVC : user record
SVC --> API : uid_checked

alt aid != None
    API -> DB : SELECT omr_assignment → assignment_answer_map
    DB --> API : answer_map, active_code
else omr_test_id != None
    API -> DB : SELECT omr_test → answer_key
    DB --> API : answer_key
else answers trống
    API -> API : auto_mode = True\nanswer_key = None
else answers trực tiếp
    API -> API : _resolve_shared_answer_key()
end

loop Mỗi file trong files[]
    API -> FS : Lưu file (timestamp_<tên gốc>)

    alt auto_mode
        API -> OMR : _resolve_answer_key_auto_from_sheet(\n  file_location, uid, ...\n)
        OMR --> API : one_answer_key, matched_one
        alt Không tìm thấy đề
            API -> API : results.append({success: false})\nfailed_count++
            note right : continue sang ảnh tiếp theo
        end
    end

    API -> OMR : run_in_threadpool(\n  process_omr_exam,\n  file, one_answer_key,\n  assignment_answer_map, runtime config\n)
    OMR -> FS : Lưu overlay, crop, JSON confidence
    OMR --> API : OMRResult dict

    alt "error" in result
        API -> API : results.append({success: false, error})\nfailed_count++
    else
        API -> SVC : _persist_omr_grade_result(...)
        SVC -> DB : INSERT omr_grade_result\nUPDATE omr_assignment.last_result
        DB --> SVC : grade_result_id
        API -> API : zip_image_paths.append(overlay_path)\nresults.append({success: true, ...})\nsuccess_count++
    end
end

alt success_count > 0
    API -> FS : zipfile.ZipFile(zip_image_paths, ZIP_DEFLATED)
    FS --> API : zip_url
end

API --> FE : HTTP 200\n{total_files, success_count, failed_count,\nresults[], zip_url, answer_source}

FE --> User : Bảng kết quả từng phiếu\n+ nút tải ZIP overlay

@enduml
```

