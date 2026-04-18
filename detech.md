# Cach he thong detect vung MCQ, SID, CODE (exam code)

Tai lieu nay mo ta dung theo code runtime hien tai, tap trung vao pipeline OMR:

- be/app/api/omr_grading.py
- be/app/services/omr/omr_service.py
- be/app/services/omr/ai/yolo_localizer.py

## 1. Dau vao va runtime config

Endpoint cham OMR la POST /api/omr/grade (va /grade-batch).

Truoc khi vao process_omr_exam, API hop nhat cau hinh trong _build_runtime_config:

- request params
- profile strategy (neu co)

Nhom config lien quan truc tiep den detect ROI:

- profile_sid_roi
- profile_mcq_roi
- profile_exam_code_roi
- profile_corner_markers
- profile_scanner_hint
- profile_ai_yolo

Luu y quan trong:

- use_template_geometry chi bat khi profile co dong thoi sid_roi va mcq_roi.
- profile_exam_code_roi co the ton tai doc lap, khong bat buoc phai co template geometry.

## 2. Tien xu ly truoc detect ROI

Trong process_omr_exam, anh duoc dua ve cung he toa do roi moi detect vung:

1. Doc anh, crop/warp (manual quad, auto crop, marker/page contour rescue).
2. Resize canvas ve 1000x1400.
3. Threshold qua _threshold_omr (otsu/weighted_adaptive/hybrid).

Toan bo detect SID/MCQ/CODE ben duoi chay tren anh threshold imgThresh (sau warp).

## 3. Detect SID + MCQ bang heuristic goc (_detect_form_boxes)

### 3.1 Tao contour candidates

_detect_form_boxes lam theo block morphology:

- MORPH_OPEN 3x3 de loai speckle.
- MORPH_CLOSE kernel 25x25 (2 lan) de "dinh" thanh tung khoi bang lon.
- Dilate nhe 3x3.

Sau do findContours va loc candidate theo hinh hoc:

- area_ratio >= 0.008
- area_ratio <= 0.82 (bo vien full page)
- khong duoc cham >= 3 canh anh (bo border component)
- extent >= 0.25

Moi candidate duoc tinh them:

- bubble_like_stats: so contour giong bubble, mat do, do dong deu
- contour_text_density: mat do blob nho + mat do line

Candidate bi loai neu giong text-table:

- bubble_count <= 4 va text_density > 28 va line_density > 0.045

### 3.2 Chon SID rect

Dieu kien sid_candidates:

- h > w (dang dung)
- y < 0.36 * H (nam nua tren)
- x >= 0.50 * W va (x + w) >= 0.74 * W (uu tien ben phai)
- 0.008 <= area_ratio <= 0.18
- bubble_count >= 6

Xep hang bang score tong hop:

- _score_sid_probe_candidate(...) (run h/vertical lines + area/aspect)
- cong bubble_density, bubble_uniformity
- tru text_density

Neu khong co sid_candidates, he thong probe 2 cua so tim kien o goc phai tren bang _find_rect_in_region, sau do loc lai bang _is_upper_right_rect.

Fallback cuoi cho SID la _fallback_sid_roi_right:

- x = 0.665W, y = 0.072H, w = 0.235W, h = 0.235H

### 3.3 Chon MCQ rect

Dieu kien mcq_candidates:

- w > h (ngang)
- y >= 0.34 * H (nua duoi)
- bubble_count >= 16
- bubble_uniformity >= 0.22

Xep hang uu tien:

- area
- bubble_count
- bubble_density
- bubble_uniformity
- phat text_density

Fallback MCQ la _fallback_mcq_roi:

- x = 0.075W, y = 0.430H, w = 0.855W, h = 0.470H

Guard cuoi trong _detect_form_boxes:

- neu area(MCQ) < area(SID) => ep ve fallback MCQ.

## 4. YOLO localizer (optional) cho SID + MCQ

Neu profile_ai_yolo bat va model hop le, process_omr_exam goi detect_omr_regions_yolo:

- Model labels duoc doc: document_page, sid_region, mcq_region.
- Moi class chi lay bbox co score cao nhat.
- Co tighten nhe bang projection tren anh binary.

Dieu kien de override ROI heuristic:

- YOLO tra du ca sid_region va mcq_region.

Khi do:

- sid_rect, mcq_rect duoc set tu YOLO box.
- roi_detection_strategy = yolo_localizer + fallback_geometry_guards.

Neu YOLO khong tra du box:

- he thong warning va fallback ve heuristic.

## 5. Uu tien profile ROI va post-process ROI

Sau khi co SID/MCQ ban dau (heuristic hoac YOLO), he thong ap thu tu uu tien:

1. Neu co template geometry (co dong thoi profile sid_roi + mcq_roi):
- sid_rect = profile_sid_rect
- mcq_rect = profile_mcq_rect

2. Neu khong dung template geometry:
- _tighten_rect_to_table cho SID/MCQ
- Contract/expand khac nhau tuy truong hop YOLO hay non-YOLO
- Guard SID bottom khong de cham header MCQ

3. Rieng form ben phai (identity-crop), neu SID qua rong:
- cat SID width con khoang 72% de tach khoi khu exam code.

## 6. Marker-anchor refine (neu co corner_markers)

Khi profile co corner_markers va khong dung template geometry:

- _refine_sid_rect_with_marker_anchor(...) tim lai SID bang cua so gan marker tren.
- _derive_exam_code_rect_with_anchor(...) suy ra CODE ROI dua theo SID + marker.

Scanner hint (min_dark_ratio...) duoc dung de loc candidate qua sang/qua yeu.

Neu refine thanh cong, roi_detection_strategy duoc them marker-anchor-refine.

## 7. Detect CODE ROI (exam code) theo thu tu uu tien

Thu tu lay CODE rect trong process_omr_exam:

1. Neu co profile_exam_code_roi => dung profile.
2. Else neu co anchor_exam_code_rect => dung marker-anchor.
3. Else => dung _fallback_exam_code_roi(w, h, sid_rect).

Cong thuc fallback CODE dua theo SID:

- width: min(w*0.16, sid_w*0.45), toi thieu 56 px
- gap voi SID: max(8, sid_w*0.07)
- dat cung hang y voi SID

Sau do co guard hinh hoc cho CODE (khi khong dung template geometry va khong co profile_exam_code_roi):

- neu khong nam dung upper-right (is_upper_right_rect, min_center_x_ratio=0.66, max_top_y_ratio=0.38)
- ep lai bang fallback_exam_code_roi dua tren sid_ref hop le.

## 8. Detect cot ben trong CODE ROI (de decode 3 chu so)

Sau khi co CODE ROI, he thong khong decode ngay theo chia deu 3 cot, ma uu tien tim cot that:

- _find_exam_code_columns(...) dung morphology close theo truc doc
- contour filter (area, height, width, vi tri)
- neu thieu cot thi fallback projection peaks
- neu van thieu thi fallback connected-components clustering
- neu van thieu nua thi thu lai tren gray-adaptive binarization

Khi tim du 3 cot, _decode_exam_code_by_columns moi split tung cot theo 10 hang (0..9) de pick mark.

Neu khong tim du cot, he thong van co cac decode candidate khac (bin/gray trim-full) va chon candidate tot nhat bang _digit_sequence_score.

## 9. Cac guard quan trong de tranh detect sai ROI

1. sid_upper_right_guard
- Neu SID ROI lech khoi 1/3 tren ben phai, ep ve _fallback_sid_roi_right.

2. code_upper_right_guard
- Neu CODE ROI lech upper-right, ep lai theo fallback dua tren SID.

3. MCQ-vs-SID area guard
- Neu MCQ nho hon SID, bo MCQ detect va quay ve fallback MCQ.

4. Marker/page rescue o buoc warp
- Giam truong hop ROI detect sai vi warp sai hinh.

## 10. Cach kiem tra detection da dung hay chua (payload debug)

Response cua process_omr_exam co cac truong dung de audit:

- roi_boxes.student_id/exam_code/mcq (toa do ROI cuoi)
- roi_detection.strategy
- roi_detection.used_yolo
- roi_detection.used_marker_anchor
- roi_detection.sid_upper_right_guard
- roi_detection.code_upper_right_guard
- roi_detection.code_contour_columns_found
- roi_detection.code_decode_source
- warnings
- warp_strategy

Thuc te, chi can nhin roi_boxes + roi_detection la biet vung SID/MCQ/CODE dang den tu:

- heuristic
- yolo
- profile template
- marker-anchor
- hay fallback guard.

## 11. Tom tat thu tu uu tien detect ROI

SID/MCQ:

1. Heuristic _detect_form_boxes
2. Neu YOLO hop le -> override SID/MCQ
3. Neu template geometry profile co du SID+MCQ -> override tiep theo profile
4. Neu khong template -> tighten/contract/expand + marker-anchor refine
5. Guard upper-right (SID)

CODE:

1. profile_exam_code_roi
2. marker-anchor derive (neu co)
3. fallback tu SID
4. upper-right guard cho CODE

Do do, profile dung se co uu tien cao nhat; YOLO la optional booster cho buoc localize, khong phai bat buoc de he thong cham duoc.