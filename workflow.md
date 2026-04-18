# Workflow hien tai cua he thong OCR

Tai lieu nay duoc viet lai theo code backend dang chay trong:
- be/main.py
- be/app/api/handwritten_load_picture.py
- be/app/services/handwritten_services.py
- be/app/api/omr_grading.py
- be/app/services/omr/omr_service.py
- be/app/services/omr/ai/yolo_localizer.py

Muc tieu:
- Mo ta dung workflow hien tai (khong dua tren tai lieu cu).
- Xac dinh ro YOLO dang dong vai tro gi.

---

## 1) Tong quan he thong OCR

He thong co 2 pipeline OCR khac nhau:

1. OCR bai tu luan viet tay (handwritten essay)
- API prefix: /api/handwritten
- Dau vao: anh bai lam + dap an mau
- OCR chinh hien tai tren API grading: OpenAI Vision (gpt-4o/gpt-4o-mini)
- Dau ra: recognized_text + diem theo cau + phan tich TP/FP/FN

2. OCR OMR (phieu trac nghiem)
- API prefix: /api/omr
- Dau vao: anh phieu + answer key + profile tuy chon
- Xu ly chinh: OpenCV + heuristic + profile strategy + optional AI modules
- Dau ra: SID, exam code, user_answers, score, debug artifacts

---

## 2) Workflow OCR viet tay (API that su dang duoc goi)

### 2.1 Quan ly dap an

1. Upload draft dap an
- Endpoint: POST /api/handwritten/upload-answer-key
- Doc file .doc/.docx/.pdf/.txt
- Trich text thuan va parse cau hoi bang parse_answer_text_to_questions

2. Luu dap an
- Endpoint: POST /api/handwritten/save-answer-key
- Luu vao bang OCRTest (ocrid, ocr_name, ocr_answer)

3. Xem/xoa dap an
- GET /api/handwritten/answer-keys/{uid}
- GET /api/handwritten/answer-key/{ocrid}
- GET /api/handwritten/answer-key/{ocrid}/download
- DELETE /api/handwritten/answer-key/{ocrid}

### 2.2 Cham bai viet tay

1. Nhan request
- Endpoint: POST /api/handwritten/upload
- Input chinh: uid, essay_images[], ocrid, use_answer_correctness/use_ragas, ocr_model

2. Validate va luu anh
- Validate so anh (1..30)
- Validate model cho phep tren endpoint nay: openai_gpt4o, openai_gpt4o_mini
- Luu vao: uploads/handwritten/{uid}/{timestamp}/essay_XXX.ext

3. OCR va cham diem
- API goi process_handwritten_with_llm(...)
- Tao OCR engine OCRReaderOpenAI4oMini theo model map:
  - openai_gpt4o -> gpt-4o
  - openai_gpt4o_mini -> gpt-4o-mini
- OCR tung trang theo thu tu upload, ghep thanh recognized_text
- Parse dap an mau theo cau
- Align bai hoc sinh theo cau (_align_student_answers)
- Moi cau:
  - baseline semantic theo claim matching (_analyze_claims)
  - neu co OpenAI key: goi _llm_semantic_ragas_analysis de lay TP/FP/FN + ly do
  - tinh score (thang 10)
    - neu use_answer_correctness=True:
      correctness = 0.5 * answer_correctness + 0.5 * F1
    - nguoc lai: dung F1

4. Tra ket qua
- questions[]: id, content, score, ragas_analysis, strict_json_output
- total_score, total_max_score, recognized_text, processing_log

### 2.3 Luu y ky thuat

- Trong file handwritten_services.py co ham process_handwritten_essay (CRAFT/VietOCR/InternVL).
- Tuy nhien endpoint /api/handwritten/upload hien tai khong goi ham nay.
- Workflow runtime thuc te dang dung process_handwritten_with_llm (OpenAI Vision path).

---

## 3) Workflow OMR hien tai

### 3.1 Cac endpoint OMR

- POST /api/omr/grade
- POST /api/omr/grade-batch
- POST /api/omr/suggest-crop
- GET/POST profile: /api/omr/form-samples, /form-profiles, /form-profiles/{code}
- Assignment APIs: /api/omr/assignments...

### 3.2 Runtime config

Truoc khi cham, API _build_runtime_config se hop nhat:
- tham so request
- profile strategy (neu co)

Cac nhom config quan trong:
- ROI/profile: sid_roi, mcq_roi, exam_code_roi, sid_row_offsets
- MCQ decode: threshold_mode, ratios, row_offsets_px...
- AI optional: ai_yolo, ai_uncertainty, ai_sid_htr, agentic_rescue
- Marker/profile scanner: corner_markers, scanner_hint, page_size_pt

### 3.3 Luong process_omr_exam

1. Ingest + crop
- Doc anh
- Neu co manual crop quad hop le -> warp va lock
- Neu co crop rect -> cat truoc

2. Auto crop document + chuan hoa
- Neu khong bi lock: _auto_crop_document_region
- Resize canvas ve 1000x1400

3. Warp candidate selection
- Tao nhieu candidate: identity, marker-crop, marker-orig, page-contour, outer-border...
- Neu sid_has_write_row=True thi cham quality bang _score_warp_layout va chon candidate tot nhat
- Neu marker fail: co warning va co rescue bang outer border

4. Threshold
- _threshold_omr voi mode: otsu / weighted_adaptive / hybrid

5. Detect ROI SID/MCQ/Exam code
- Khoi tao tu heuristic _detect_form_boxes
- Neu profile co sid_roi + mcq_roi day du -> uu tien template geometry
- Neu khong: tighten/expand + guard hinh hoc
- Neu co corner_markers profile: marker-anchor refine cho SID/exam code

6. Decode SID
- Nhieu pass (global/local/crop-adaptive/gray/no-trim)
- Chon candidate tot bang _digit_sequence_score
- Co ho tro sid_row_offsets theo cot
- Optional ai_sid_htr de doi chieu hang viet tay SID

7. Decode exam code
- Nhieu pass: binary trim/full, gray trim/full, contour-column mode
- Chon pass tot nhat theo _digit_sequence_score

8. Decode MCQ
- Tach block theo rows_per_block/num_blocks
- Refine bubble grid, bo header khi can, split grid, pick mark
- Ho tro row-anchor re-align va row_offsets profile
- Optional ai_uncertainty (BubbleCellClassifier) de rescue/downgrade o uncertain

9. MCQ rescue branches
- Marker-original adaptive
- Secondary grayscale-adaptive decode
- Horizontal-shift rescue (khi mot option bi dominate bat thuong)

10. Grading
- detected_questions = len(user_answers)
- graded_questions = min(detected_questions, len(answer_key), questions)
- Cac cau vuot answer_key -> no_key (khong tinh diem)

11. Retry/orchestration optional
- Self-retry 1 lan neu warp hien tai yeu
- Optional agentic_rescue (run_agentic_rescue) so sanh 3 nhanh va chon payload tot nhat

12. Artifact + response
- Anh output: graded_*, sid_crop_*, mcq_crop_*
- Telemetry JSON: bubble_confidence_*.json
- Payload co score, answers, roi_detection, warp_strategy, warnings...

---

## 4) YOLO co vai tro gi?

### 4.1 Vai tro dung trong code

YOLO chi duoc dung cho OMR ROI localization, cu the:
- Module: be/app/services/omr/ai/yolo_localizer.py
- Ham: detect_omr_regions_yolo(...)
- Muc dich: tim box sid_region + mcq_region (+ document_page neu co)

Neu YOLO tra du 2 box sid_region va mcq_region:
- process_omr_exam override ROI heuristic bang box YOLO
- roi_detection.strategy gan ve yolo_localizer + guards

### 4.2 Dieu kien de YOLO thuc su chay

Can dong thoi:
1. profile_ai_yolo.enabled = true
2. profile_ai_yolo.model_path co file model hop le (.pt)
3. ultralytics import duoc trong runtime
4. detect_omr_regions_yolo tra ve du box can thiet

Neu thieu bat ky dieu kien nao:
- He thong fallback ve heuristic ROI
- Them warning runtime, khong hard fail

### 4.3 YOLO co duoc dung trong OCR viet tay khong?

Khong.
- Pipeline /api/handwritten/upload hien tai khong dung YOLO.
- OCR viet tay runtime hien tai dung OpenAI Vision OCR + semantic grading.

### 4.4 Trang thai hien tai trong profile workspace

Tu cac profile dang co trong be/uploads/omr_data/profiles:
- Chua thay cau hinh ai_yolo
- Nghia la mac dinh YOLO dang KHONG duoc bat trong van hanh hien tai

---

## 5) Ket luan nhanh

1. He thong OCR hien tai la 2 nhanh rieng: handwritten va OMR.
2. YOLO khong phai core bat buoc; YOLO la module optional cho OMR ROI localization.
3. Trong trang thai profile hien tai, pipeline van hanh chu yeu bang OpenCV heuristic + profile tune, khong phu thuoc YOLO.
