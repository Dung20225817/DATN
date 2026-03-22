# OMR Method - Implementation Notes

## 1. Muc tieu
Tai lieu nay mo ta:
- Luong hoat dong cua chuc nang OMR.
- Cac phuong phap da su dung trong pipeline hien tai.
- Cac phuong phap da thu de so sanh va ket qua.
- Ly do chon phuong phap cuoi cung.

Phuong an hien tai duoc toi uu cho form OMR template cua du an (SID 6 chu so + bang cau hoi MCQ) trong dieu kien anh chup thuc te co bong, nghieng va nhieu nen.

## 2. Luong hoat dong OMR (hien tai)
### B1. Nhan input va chuan bi
- Nhan anh bai lam, dap an (tu file hoac nhap tay), cau hinh layout (so cau, so lua chon, SID digits, ...).
- Co the nhan them crop thu cong tu frontend (crop_x, crop_y, crop_w, crop_h).

### B2. Tien xu ly anh
- Manual crop (neu co).
- Auto-crop vung giay (de bo vat the nen khong lien quan).
- Chuan hoa ve kich thuoc chuan 1000x1400.

### B3. Chon perspective warp tot nhat
Pipeline tao nhieu ung vien warp, sau do chon ung vien co diem chat luong cao nhat:
- marker-crop: marker tu anh da auto-crop.
- marker-orig: marker tu anh goc.
- page-contour: contour trang giay lon.
- outer-border-crop: border ngoai tu anh da auto-crop.
- outer-border-orig: border ngoai tu anh goc.

Diem chat luong warp duoc tinh dua tren:
- Co/khong detect duoc marker sau warp.
- Do khop ROI SID/MCQ voi ROI template fallback (IoU).
- Kich thuoc box SID/MCQ co nam trong mien hop ly hay khong.

### B4. Nhi phan hoa va trich ROI
- Nhi phan hoa chinh: Otsu + chong bong (illumination normalization) + morphology.
- Trich SID box va MCQ box bang detect form-box.
- Neu box bat thuong (sau marker warp), fallback ve ROI template.
- Tighten/expand ROI de bo vien du va can chinh bien.

### B5. Giai ma SID
- Tach luoi SID (10 hang x N cot), dem pixel trung tam o moi o.
- Chon hang duoc to cho moi cot bang _pick_marked_with_flags.
- Chay them local adaptive decoder tren SID crop de chong bong/nhieu.
- Hop nhat ket qua theo confidence de lay SID cuoi cung on dinh.

### B6. Giai ma MCQ
- Chia khoi MCQ theo block, cat bo cot so cau ben trai, trim header A/B/C/D.
- Tach luoi so hang cau x so lua chon.
- Dem pixel trung tam (inner-ratio) de giam anh huong vien bang va bong.
- Chon dap an tung cau bang co che blank/multiple/ok.

### B7. Fallback cho anh kho
Neu decode chinh con nhieu uncertain:
- Chay them nhanh fallback: marker-orig + adaptive threshold + template ROI cho MCQ.
- Chi thay ket qua neu uncertain giam ro rang.

### B8. Cham diem va xuat ket qua
- So sanh user_answers voi answer_key, tinh diem.
- Ve overlay (SID box, MCQ box, dap an dung/sai, score).
- Tra ve JSON ket qua + anh da ve.

## 3. Cac phuong phap da su dung
### 3.1 Perspective/geometry
- Marker-based corner detection (uu tien).
- Outer-border/page-contour fallback.
- Multi-candidate warp selection by quality score.

### 3.2 Thresholding va denoising
- Otsu + background normalization + CLAHE (nhanh, on dinh tong quat).
- Adaptive threshold (gaussian) cho fallback anh kho, dac biet voi SID crop.

### 3.3 ROI va grid reading
- Detect form boxes bang morphology block.
- Projection tighten/expand ROI.
- Cell center counting (inner_ratio) thay vi count toan bo o.
- Confidence-based mark selection (blank/multiple/ok).

## 4. Cac phuong phap da dung de so sanh
### So sanh A: marker-only vs multi-candidate warp
- marker-only nhanh nhung de fail khi marker bi che/crop.
- multi-candidate warp cho ket qua on dinh hon tren bo test co dieu kien khac nhau.

### So sanh B: count toan bo cell vs count vung trung tam
- count toan bo cell de bi anh huong boi vien o va bong.
- count trung tam cell giam nhieu false positive/false blank.

### So sanh C: chi Otsu vs Otsu + adaptive fallback
- Otsu la luong chinh vi on dinh va it overfit.
- Adaptive fallback huu ich cho anh kho (bong manh, do tuong phan cuc bo kem).

### So sanh D: SID decode 1 pass vs SID decode nhieu pass
- 1 pass de sai lech 1 chu so trong anh kho.
- Nhieu pass (global + local adaptive) ket hop confidence cho SID on dinh hon.

## 5. Ly do chon phuong phap hien tai
Phuong phap hien tai duoc chon vi can bang duoc 4 yeu to:
- Do chinh xac: dung tot tren bo anh test noi bo, ke ca anh kho.
- Do ben: co fallback theo nhieu tang (warp, threshold, ROI, decode).
- Kha nang mo rong: van giu duoc cau truc ro rang de tinh chinh theo tung mau form.
- Hieu nang: van du nhanh cho API backend (khong dung model hoc sau trong runtime).

## 6. Han che va huong cai tien
Han che con lai:
- Rat nhay voi truong hop anh qua mo/nhoe, mat chi tiet marker hoan toan.
- Co the can profile rieng cho form khac (layout khac template hien tai).

Huong cai tien de xuat:
- Them bo regression test tu dong (batch folder test + expected JSON).
- Luu them diagnostic score cua tung warp candidate trong response debug mode.
- Bo sung quality gate de canh bao nguoi dung chup lai khi anh qua xau.

## 7. Tom tat
Pipeline OMR hien tai la mot he thong hybrid:
- Rule-based CV + multi-fallback + confidence voting.
- Chon warp tot nhat, doc SID/MCQ theo nhieu lop bao ve, va chi thay fallback khi ket qua tot hon ro rang.

Dieu nay giup he thong dung duoc trong dieu kien chup thuc te, khong chi tren anh scan dep.
