# OMR Method - Updated Notes

## 1. Scope
Tai lieu nay mo ta trang thai hien tai cua chuc nang OMR trong he thong, gom:
- Tao phieu OMR theo luong draft -> save.
- Cham 1 anh va cham batch nhieu anh.
- Tu dong nhan dien bai thi theo Ma de + Ten bai thi tren phieu.
- Quan ly kho phieu (list, download, delete).

Backend chinh:
- API: `be/app/api/omr_grading.py`
- OMR service: `be/app/services/omr/omr_service.py`
- DB model: `be/app/db/ocr_tables.py` (`OMRTest`)

Frontend chinh:
- OMR page: `fe/src/Page_Components/MultichoicePage.tsx`

## 2. Main workflows

### 2.1 Tao phieu OMR (draft -> save)
1. Frontend goi `POST /template` de tao anh phieu nhap.
2. API sinh file anh + metadata tam (draft), chua persist vao bang `omr_test`.
3. Nguoi dung xem truoc va bam Save.
4. Frontend goi `POST /template/save` de luu chinh thuc vao DB.
5. Backend kiem tra duplicate theo bo dinh danh:
   - `omr_name` + `omr_code` + `student_id_digits`
6. Neu trung, tra loi 409 de frontend thong bao trung mau phieu.

### 2.2 Cham OMR tu dong
1. Frontend goi `POST /grade` hoac `POST /grade-batch`.
2. Backend chay detect tren anh de lay:
   - Student ID
   - Ma de (3 digits)
   - Ten bai thi in tren phieu (OCR)
3. API auto match bai thi trong DB dua tren ma de/ten bai thi.
4. Lay dap an tu `omr_test` da match va cham diem.
5. Tra ket qua score + thong tin nhan dien + anh overlay.

### 2.3 Kho phieu (vault)
- `GET /tests/{uid}`: lay danh sach phieu da luu.
- `DELETE /tests/{uid}/{omrid}`: xoa phieu va file lien quan.
- Download anh template duoc frontend xu ly dang blob de dam bao tai file thay vi mo tab anh.

## 3. OMR CV pipeline

### 3.1 Chuan hoa anh
- Doc anh, resize ve kich thuoc chuan.
- Tien xu ly grayscale + denoise nhe.
- Tao cac kenh binary phuc vu detect box va bubble.

### 3.2 Perspective correction (multi-candidate)
He thong tao nhieu ung vien warp:
- Marker-based candidate.
- Page contour / outer-border candidate.
- Identity-style candidate trong truong hop phieu da gan dung template geometry.

Moi candidate duoc cham diem theo do hop ly ROI SID/MCQ + quality marker.
Candidate tot nhat duoc chon de decode tiep.

### 3.3 ROI detection
- Tim box SID, box MCQ, box Ma de theo marker/geometry.
- Neu detect box kem on dinh, fallback ve template ROI.
- Co tighten/expand ROI de giam vien den va lech canh.

### 3.4 Bubble decoding
- Tach luoi cell theo hang/cot cho SID, Ma de va MCQ.
- Tinh muc do to trong tam o (tranh bi nhieu do vien).
- Chon top1/top2 va confidence de phan loai:
  - `ok`
  - `blank`
  - `multiple`

## 4. Rule scoring and formulas

### 4.1 Chon dap an 1 cau
Voi vector gia tri to cua 1 hang:
- `top1`: lon nhat
- `top2`: lon thu 2
- `conf = top1 / max(1, top2)`

Ket hop nguong blank va `conf` de quyet dinh `ok/blank/multiple`.

### 4.2 Tinh diem
- `grading_questions = min(so cau detect, so cau dap an, so cau cau hinh)`
- Moi cau dung = 1 diem, sai/khong xac dinh = 0
- `score = (tong cau dung / grading_questions) * 100`

## 5. Auto-identification logic

### 5.1 Ma de
- Doc tu bang Ma de tren phieu (3 digits).
- Neu detect yeu, co fallback threshold/ROI de cuu.

### 5.2 Ten bai thi
- OCR text ten bai thi bang EasyOCR (vi/en).
- Chuan hoa text (normalize) de tang kha nang match.

### 5.3 Match bai thi trong DB
- Uu tien match bang Ma de.
- Neu can, ket hop ten bai thi de giam ambiguity.
- Ket qua match dung de lay bo dap an cham tu dong.

## 6. Template generation updates

Template moi ho tro:
- Student ID (so cot tuy bien)
- Ma de 3 so
- Vung thong tin in tren phieu
- Layout gian cach rong hon giua SID / Ma de / thong tin

Luu y giao dien in:
- Khong prefill so Ma de tren phieu trang.
- Bo tieu de "Thong tin" theo yeu cau UX.
- Ve text Unicode on dinh cho tieng Viet.

## 7. Outputs

Ket qua cham tra ve cac truong quan trong:
- `score`
- `student_id`
- `user_answers`
- `answer_compare`
- `exam_code_detected`
- `exam_title_detected`
- `matched_omr_test` (neu co)
- `result_image`

Anh ket qua co overlay:
- ROI SID/MCQ/Ma de
- Cac marker dap an dung/sai
- Vung Ma de detect de debug nhanh

## 8. Batch grading

He thong ho tro cham nhieu anh trong 1 request:
- Gioi han toi da 50 file / lan.
- Moi file co ket qua rieng (filename, score, sid, metadata nhan dien).
- Frontend hien thi danh sach ket qua sau khi cham.

## 9. Data model and persistence

Bang `omr_test` luu metadata bai thi:
- `omrid`
- `uuid` (owner)
- `omr_name`
- `omr_code`
- `omr_quest`
- `omr_answer`
- `created_at`, `updated_at`

Anh template + metadata phu tro duoc quan ly theo file storage va sidecar metadata.

## 10. Known limits and next improvements

Han che hien tai:
- Anh qua mo, rung, mat marker nang van co the giam do on dinh.
- OCR ten bai thi phu thuoc chat luong anh va font in.

Huong nang cap tiep:
1. Bo sung regression test set (anh mau + expected JSON).
2. Them debug mode chi tiet diem tung warp candidate.
3. Dua metadata sidecar vao DB migration de dong nhat luu tru.

## 11. Quick summary

He thong OMR hien tai la pipeline hybrid:
- Rule-based CV + multi-fallback geometry/threshold.
- Auto nhan dien Ma de + Ten bai thi de map dap an trong DB.
- Ho tro full lifecycle: Tao phieu (draft/save) -> Cham diem -> Quan ly kho phieu.
