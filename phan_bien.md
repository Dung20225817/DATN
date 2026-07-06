# Phản biện — Câu hỏi kỹ thuật hệ thống OMR chấm MCQ

---

## 1. Metric đánh giá, số ảnh, và kết quả

**Dataset:** 61 ảnh phiếu thực tế chụp bằng điện thoại, 4 nhóm điều kiện: Clear_answer (17), Shadow (30), Tilted_image (10), Vague_answer (4). Tổng số câu cần chấm: **1.220 câu** (20 câu/phiếu có đáp án đối chiếu).

**Metric chính — G-DetRate (Graded Detection Rate):**
Thesis gọi là "Acc. MCQ" và mô tả là "tỷ lệ câu được nhận diện đúng trên tổng câu có đáp án". Về mặt kỹ thuật, G-DetRate = (correct + wrong) / total_graded_q — tức là đếm cả câu đọc sai nhãn, chỉ loại câu uncertain. Tên "nhận diện đúng" trong thesis hơi sai lệch so với code thực tế.

**Metric mới bổ sung — True Detection Accuracy:**
True accuracy = correct / total_graded_q — chỉ đếm câu đọc đúng nhãn A/B/C/D. Đây là metric chưa có trong summary gốc, đã được thêm vào `eval/run_eval.py`.

**Kết quả phân tách 3 thành phần (Rescue + Weighted Scoring, 1.220 câu):**

| Nhóm         | Correct (True Acc) | Wrong (Misread) | Uncertain | Uncertain/phiếu |
|--------------|--------------------|-----------------|-----------|-----------------|
| Clear_answer | 99.1%              | 0.3%            | 0.6%      | 0.12            |
| Shadow       | 99.5%              | 0.0%            | 0.5%      | 0.10            |
| Tilted_image | 100.0%             | 0.0%            | 0.0%      | 0.00            |
| Vague_answer | 96.2%              | 0.0%            | 3.8%      | 0.75            |
| **Tổng**     | **99.3%**          | **0.1%**        | **0.7%**  | **0.13**        |

→ G-DetRate = Correct + Wrong = 99.3% + 0.1% ≈ 99.3% (khớp số liệu báo cáo)
→ True accuracy ≈ G-DetRate vì wrong gần như = 0 (chỉ 1/1.220 câu bị misread)

**Ý nghĩa của việc phân tách 3 thành phần — so sánh có/không Rescue:**

| Cấu hình     | Correct  | Wrong (Misread) | Uncertain |
|--------------|----------|-----------------|-----------|
| Có Rescue    | **99.3%** | **0.1%**       | **0.7%**  |
| Không Rescue | 61.9%    | 11.0%           | 27.1%     |

Không có Rescue: không chỉ uncertain tăng từ 0.7% lên 27.1%, mà **wrong tăng từ 0.1% lên 11.0%** — lệch lưới khiến hệ thống lấy mẫu nhầm bong bóng, đọc ra sai nhãn có hệ thống. G-DetRate (72.9%) đã che giấu phần này vì chỉ báo "không đọc được" mà không phân biệt "đọc sai".

Nguồn: `eval/results/eval_final_20260624.json`, `eval/results/eval_no_rescue.json`

---

## 2. MCQ Map Search Rescue — cải thiện và rủi ro

**Cải thiện so với baseline (không Rescue):**

| Cấu hình    | G-DetRate | Uncertain/phiếu | Score Accuracy |
|-------------|-----------|-----------------|----------------|
| Không Rescue| 72.9%     | 5.43            | 54.8%          |
| Có Rescue   | **99.3%** | **0.13**        | **100.0%**     |
| **Δ**       | **+26.4 pp** | **−5.30**    | **+45.2 pp**   |

Theo nhóm không có Rescue: Shadow chỉ đạt 70.7% (5.87 uncertain/phiếu), Tilted_image 66.0% — đây là hai nhóm được hưởng lợi nhiều nhất.

**Rescue có làm tệ hơn không?**
Không có trường hợp nào trong 62 ảnh bị tệ hơn (`regressed_count = 0`). Điều này được đảm bảo bởi **acceptance criteria** trong code (`omr_service.py` dòng 1706–1715):
- Chỉ chấp nhận candidate nếu: `ΔUncertain ≥ 2`, hoặc `ΔUncertain ≥ 1 && double_mark không tăng`, hoặc `ΔUncertain = 0 && double_mark giảm`
- Nếu không có candidate nào đạt ngưỡng → giữ nguyên kết quả baseline

---

## 3. Ngưỡng gate — discrepancy giữa thesis (60%) và code (40%)

**Thesis** (ch6, dòng 22–23 và ví dụ trace dòng 113) mô tả và xác nhận nhất quán: gate = max(4, ⌊0.60×Q⌋), Q=40 → gate=24. Đây là thiết kế gốc đã được document đầy đủ.

**Code thực tế** (`omr_service.py` dòng 1644) dùng 0.40:
```python
search_uncertain_gate = max(4, int(round(0.40 * float(max(1, questions)))))
```
Q=40 → gate = max(4, round(0.40×40)) = **16 câu** (40%).

**Đây là lỗi không đồng bộ giữa code và thesis.** 60% là ý định thiết kế đã được document rõ ràng với ví dụ trace cụ thể. Code hiện tại dùng 40% mà không có comment giải thích lý do thay đổi. Cần một trong hai: sửa code về 0.60 để khớp thesis, hoặc cập nhật thesis nếu 40% là quyết định có chủ ý sau thực nghiệm (kèm lý do).

**Ảnh hưởng thực tế:** Ngưỡng thấp hơn (40% vs 60%) khiến Rescue kích hoạt dễ hơn. Với Q=40: gate=16 thay vì 24 — phiếu chỉ cần 16 câu uncertain (40%) thay vì 24 câu (60%) để trigger Rescue. Điều này có thể làm tăng số lần Rescue chạy không cần thiết trên phiếu tô không rõ bình thường.

**Không có ablation study về threshold này:** Không có eval nào so sánh gate 40% vs 60%. Đây là điểm yếu — lựa chọn ngưỡng (dù là 40% hay 60%) chưa được validate thực nghiệm.

---

## 4. `_cell_score` vs `countNonZero` — so sánh số liệu

**Công thức `_cell_score`** (`omr_mcq.py` dòng 951–965):
```python
fill_ratio = cv2.countNonZero(binary_cell) / area        # từ ảnh nhị phân
dark_mean  = max(0, 1.0 - mean_grayscale / 255.0)        # từ ảnh grayscale
dark_p25   = max(0, 1.0 - percentile_25 / 255.0)         # tứ phân vị 25
score = 0.55 * fill_ratio + 0.30 * dark_mean + 0.15 * dark_p25
```

Hàm `_cell_density` (countNonZero đơn thuần) chỉ tính `fill_ratio`, không có thành phần grayscale.

**So sánh ablation** (`eval/results/eval_density_only.json` vs `eval_final_20260624.json`):

| Phương pháp            | G-DetRate | Uncertain/phiếu | Score Accuracy |
|------------------------|-----------|-----------------|----------------|
| density-only (countNonZero) | 97.7% | 0.45        | 100.0%         |
| weighted `_cell_score` | 97.7%     | 0.45            | 100.0%         |

→ **Độ chính xác hoàn toàn như nhau** trên 62 ảnh. Ưu điểm của `_cell_score` là tốc độ xử lý nhanh hơn (theo thesis ch6: 5.4s vs 9.7s/phiếu, tức nhanh hơn ~43%) vì tín hiệu grayscale giúp phân biệt ô tô mờ sớm hơn, giảm số lần kích hoạt Rescue không cần thiết.

**Trên ảnh tô mờ:** `fill_ratio` (countNonZero) có thể thấp (~0.11, dưới ngưỡng min_mark) và báo trống; `dark_mean` và `dark_p25` từ grayscale bổ sung tín hiệu phụ, giúp tổng score vượt ngưỡng và nhận diện đúng ô đã tô.

---

## 5. Không phát hiện 4 marker — tỷ lệ lỗi và cách xử lý

**Cơ chế fallback** (`omr_preprocess.py` dòng 164, `omr_service.py` dòng 1077–1079):
- Khi không tìm đủ 4 góc marker → strategy = `"resize-only"` (chỉ resize, không warp perspective)
- Ghi cờ `COORD_GLOBAL_WARP_FALLBACK` vào metadata kết quả
- Tương tự, khi anchor cục bộ thiếu → `COORD_ANCHOR_FALLBACK`

**Tỷ lệ lỗi:** Không được định lượng trong code hay eval. Hệ thống chỉ trả về boolean `fallback_used` và warning text, không có số liệu thống kê % ảnh bị fallback. Trong 62 ảnh eval, không có ảnh nào trong `error_images`, nhưng không rõ bao nhiêu ảnh đã kích hoạt fallback.

**Giáo viên xử lý như thế nào:** Cấu hình thủ công vùng crop và các ROI (Student ID, MCQ, Exam code) qua component `OmrProfileRoiEditor` (`fe/src/features/omr/components/OmrProfileRoiEditor.tsx`). Đây là bước cấu hình profile nâng cao, không bắt buộc với mọi giáo viên.

---

## 6. Bảo mật — mật khẩu plain-text và UID

**Mật khẩu plain-text: CÓ — vấn đề nghiêm trọng**

```python
# be/app/api/auth.py dòng 80–83
# Passwords are still stored as plain text to preserve current behavior.
# This should be replaced with proper password hashing in a later security pass.
if user.password != data.password:
    raise HTTPException(status_code=401, detail="Sai mật khẩu")
```
Mật khẩu lưu thẳng vào DB (`password=data.password`, dòng 98), không hash.

**UID lộ trong nhiều nơi:**
- Trả về trong response login/register: `"uid": user.uuid` (dòng 33)
- Gửi qua Form parameter: `uid: int = Form(...)` trong grading.py, templates.py
- Lộ trong URL path: `/api/omr/tests/{uid}`, `/api/omr/assignments/{uid}/{aid}`

**Token xác thực:** Hardcode chuỗi giả
```python
# be/app/api/auth.py dòng 37
"token": "fake_jwt_123",  # later: replace with a real JWT
```

**Kế hoạch production:** Được ghi là technical debt trong ARCHITECTURE.md ("Thay xác thực plain text bằng hash mật khẩu và token thật") nhưng **chưa được implement**. Hệ thống hiện tại không thể deploy production với cấu hình bảo mật này.

---

## 7. Smart Camera — số frame và calibrate darkRatio

**Số frame để chuyển LOCKED: 4 frame liên tiếp**

```typescript
// MultichoicePage.tsx dòng 383
if (detectHitRef.current >= 4 && scannerState !== "locked") {
  setScannerState("locked");
}
```
Ngược lại, 4 frame miss liên tiếp → quay về `"searching"` (dòng 389).

**Ngưỡng `darkRatio ≥ 0.14`** (`MultichoicePage.tsx` dòng 146):
```typescript
minDarkRatio: clampNumber(Number(hint?.min_dark_ratio ?? 0.14), 0.08, 0.35),
```
- Default: **0.14**, range hợp lệ: 0.08–0.35
- Configurable per profile qua `profile.strategy.scanner_hint.min_dark_ratio`
- **Quá trình calibrate:** Không có tài liệu, không có script calibration, không có comment giải thích lý do chọn 0.14. Đây là giá trị được chọn theo kinh nghiệm.

---

## 8. Form Profile — số bước cấu hình

**6 bước bắt buộc** cho giáo viên không chuyên IT (`fe/src/pages/AdminPage.tsx`):

| Bước | Trường         | Mặc định   | Ghi chú                        |
|------|----------------|------------|--------------------------------|
| 1    | Upload file mẫu | —         | PDF/PNG/JPG/WebP               |
| 2    | Tên profile    | tên file   | Tự điền                        |
| 3    | Số câu hỏi     | 40         | Range 1–300                    |
| 4    | Số lựa chọn/câu | 4 (A–D)   | 2, 4, hoặc 5 (A–B/A–D/A–E)    |
| 5    | Số câu/block + số chữ số MSSV | 20 + 6 | Range 1–60 và 1–20 |
| 6    | Có hàng ghi tay MSSV | true | Checkbox                      |

**Bước 7 (tùy chọn, nâng cao):** ROI Editor — kéo thủ công vùng MSSV, MCQ, mã đề, 4 góc marker trên ảnh mẫu. Bước này đòi hỏi hiểu khái niệm "vùng ảnh" và cần thiết khi marker không được detect tự động.

**Nhận xét UX:** 6 bước cơ bản tương đối thẳng thắn với giáo viên không chuyên IT vì phần lớn có giá trị mặc định hợp lý. Bước ROI Editor là rào cản kỹ thuật đáng kể nếu profile mới có layout khác chuẩn.
