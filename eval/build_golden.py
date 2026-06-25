"""
build_golden.py — Tạo golden data cho eval dataset OMR.

Chạy 1 lần (hoặc chạy lại để cập nhật sau khi có thêm DB records).
Không cần backend đang chạy. Cần kết nối DB PostgreSQL.

Usage:
    python eval/build_golden.py
    python eval/build_golden.py --dataset-dir D:/GR2/OCR_CRNN/ocr_dataset
    python eval/build_golden.py --db-url postgresql://postgres:1111@localhost:5432/postgres
"""

import argparse
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path

DB_URL_DEFAULT = "postgresql://postgres:1111@localhost:5432/postgres"
DATASET_DIR_DEFAULT = Path(__file__).parent.parent / "ocr_dataset"
STORAGE_DIR_DEFAULT = Path(__file__).parent.parent / "storage" / "uploads" / "omr"

LABEL_MAP = ["A", "B", "C", "D", "E"]
INT_TO_LABEL = {i: LABEL_MAP[i] for i in range(5)}
LABEL_TO_INT = {v: k for k, v in INT_TO_LABEL.items()}

PROFILE_BY_QCOUNT = {
    20: "20-cau-pdf",
    40: "40-cau-pdf",
    50: "50-cau",
    120: "120pdf",
}

# Giới hạn số câu có đáp án tối đa theo form profile.
# 120pdf: assignment DB có đáp án lẻ tẻ ở Q21-Q44 nhưng thực tế chỉ chấm Q1-Q20.
FORM_PROFILE_MAX_GRADED = {
    "120pdf": 20,
}


def extract_camera_ts(filename: str) -> str | None:
    """Trích xuất camera timestamp từ tên file."""
    m = re.search(r"camera_(\d+)", filename)
    return m.group(1) if m else None


def int_list_to_labels(int_list: list) -> list:
    """Chuyển [0,1,2,3] → ['A','B','C','D']. -1 → '-'."""
    result = []
    for v in int_list:
        if v is None or v == -1:
            result.append("-")
        else:
            result.append(INT_TO_LABEL.get(int(v), "-"))
    return result


def build_bubble_confidence_map(storage_dir: Path) -> dict:
    """
    Quét tất cả bubble_confidence_*.json, trả về map:
      camera_ts → {"path": filepath, "data": parsed_json}
    """
    bc_map = {}
    for filepath in storage_dir.glob("bubble_confidence_*.json"):
        try:
            with open(filepath, encoding="utf-8") as f:
                data = json.load(f)
            image_field = data.get("image", "")
            ts = extract_camera_ts(image_field)
            if ts:
                # Ưu tiên file mới nhất nếu có nhiều file cùng camera_ts
                if ts not in bc_map or filepath.stat().st_mtime > bc_map[ts]["mtime"]:
                    bc_map[ts] = {
                        "path": str(filepath),
                        "filename": filepath.name,
                        "data": data,
                        "mtime": filepath.stat().st_mtime,
                    }
        except Exception as e:
            print(f"  [WARN] bubble_confidence: {filepath.name}: {e}")
    return bc_map


def query_db(db_url: str) -> tuple[dict, dict]:
    """
    Query DB, trả về:
      db_map: camera_ts → DB record (record mới nhất)
      answer_sets_map: exam_code → answer_list (int)
    """
    try:
        import psycopg2
        import psycopg2.extras
    except ImportError:
        print("[ERROR] psycopg2 chưa cài. Chạy: pip install psycopg2-binary")
        sys.exit(1)

    db_map = {}
    answer_sets_map = {}

    try:
        conn = psycopg2.connect(db_url)
        cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)

        # Lấy tất cả grading records
        cur.execute("""
            SELECT
                grid, file_name, exam_code, score, aid, source,
                result_json->>'correct_answers' AS correct_answers_raw,
                result_json->>'user_answers' AS user_answers_raw,
                result_json->>'answer_key_source' AS answer_key_source,
                bubble_confidence_json,
                created_at
            FROM omr_grade_result
            ORDER BY created_at ASC
        """)
        rows = cur.fetchall()
        for row in rows:
            file_name = row["file_name"] or ""
            ts = extract_camera_ts(file_name)
            if not ts:
                continue
            try:
                correct_raw = row["correct_answers_raw"]
                user_raw = row["user_answers_raw"]
                correct_list = json.loads(correct_raw) if correct_raw else []
                user_list = json.loads(user_raw) if user_raw else []
                score_val = float(row["score"]) if row["score"] else None
            except Exception:
                correct_list, user_list, score_val = [], [], None

            # Lấy record mới nhất (overwrite nếu trùng ts)
            db_map[ts] = {
                "grid": row["grid"],
                "file_name": file_name,
                "exam_code": row["exam_code"],
                "score": score_val,
                "aid": row["aid"],
                "source": row["source"],
                "correct_answers": correct_list,
                "user_answers": user_list,
                "answer_key_source": row["answer_key_source"],
                "bubble_confidence_json": row["bubble_confidence_json"],
                "created_at": str(row["created_at"]),
            }

        # Lấy answer_sets từ tất cả assignments
        cur.execute("SELECT aid, answer_sets FROM omr_assignment")
        for row in cur.fetchall():
            answer_sets_raw = row["answer_sets"]
            if not answer_sets_raw:
                continue
            try:
                sets = answer_sets_raw if isinstance(answer_sets_raw, list) else json.loads(answer_sets_raw)
                for entry in sets:
                    code = str(entry.get("code", "")).strip()
                    answers = entry.get("answers", [])
                    if code and answers:
                        # Ưu tiên giữ entry đã có (hoặc overwrite — tuỳ)
                        if code not in answer_sets_map:
                            answer_sets_map[code] = {"answers": answers, "aid": row["aid"]}
            except Exception as e:
                print(f"  [WARN] answer_sets parse: {e}")

        cur.close()
        conn.close()
        print(f"  DB: {len(db_map)} grading records, {len(answer_sets_map)} answer-set codes")
    except Exception as e:
        print(f"  [WARN] Không kết nối được DB: {e}")
        print("         Golden sẽ chỉ dùng bubble_confidence (không có answer_key từ DB).")

    return db_map, answer_sets_map


def get_exam_code_from_bc(bc_data: dict) -> str | None:
    """Trích exam_code từ coordinate_mapping trong bubble_confidence."""
    cm = bc_data.get("coordinate_mapping", {})
    # Có thể ở nhiều nơi
    for field in ["exam_code", "exam_code_detected"]:
        val = cm.get(field)
        if val and str(val).strip() and str(val).strip() not in ("", "None", "null"):
            return str(val).strip()
    # Từ roi_boxes
    roi = bc_data.get("roi_boxes", {})
    ec = roi.get("exam_code_value") or roi.get("exam_code")
    if ec and isinstance(ec, str):
        return ec.strip() or None
    return None


def infer_profile(num_questions: int) -> str:
    return PROFILE_BY_QCOUNT.get(num_questions, f"{num_questions}-cau")


def build_golden_entry(
    image_path: Path,
    category: str,
    bc_entry: dict | None,
    db_record: dict | None,
    answer_sets_map: dict,
) -> dict:
    """Xây dựng 1 golden entry cho 1 ảnh."""
    image_name = image_path.name

    golden: dict = {
        "image": image_name,
        "category": category,
        "form_profile": None,
        "questions": None,
        "exam_code": None,
        "answer_key": None,
        "answer_key_int": None,
        "expected_score": None,
        "expected_user_answers": None,
        "expected_uncertain": [],
        "answer_key_source": None,
        "aid": None,
        "source_bubble_confidence": None,
        "has_db_record": db_record is not None,
        "needs_annotation": False,
        "created_at": datetime.now().isoformat(),
    }

    # Thông tin từ bubble_confidence
    if bc_entry:
        bc_data = bc_entry["data"]
        golden["source_bubble_confidence"] = bc_entry["filename"]
        rows = bc_data.get("rows", [])
        golden["questions"] = len(rows)
        golden["form_profile"] = infer_profile(len(rows))

        # expected_user_answers và uncertain từ bubble_confidence
        user_ans_bc = []
        uncertain_bc = []
        for r in rows:
            label = r.get("selected_label", "-")
            if r.get("uncertain") or label == "-" or label is None:
                user_ans_bc.append("-")
                uncertain_bc.append(r.get("question", len(user_ans_bc)))
            else:
                user_ans_bc.append(str(label))
        golden["expected_user_answers"] = user_ans_bc
        golden["expected_uncertain"] = uncertain_bc

        # Thử lấy exam_code từ bubble_confidence (dùng làm fallback)
        ec_bc = get_exam_code_from_bc(bc_data)
        if ec_bc:
            golden["exam_code"] = ec_bc

    # Thông tin từ DB record (ưu tiên hơn bubble_confidence)
    if db_record:
        golden["exam_code"] = db_record.get("exam_code") or golden.get("exam_code")
        golden["aid"] = db_record.get("aid")
        golden["expected_score"] = db_record.get("score")
        golden["answer_key_source"] = db_record.get("answer_key_source")

        correct_int = db_record.get("correct_answers", [])
        user_int = db_record.get("user_answers", [])

        if correct_int:
            # Lọc -1 (no-key)
            valid = [v for v in correct_int if v is not None and v != -1]
            if valid:
                golden["answer_key_int"] = correct_int
                golden["answer_key"] = int_list_to_labels(correct_int)

        if user_int:
            # Xây dựng expected_user_answers:
            # - Câu không có đáp án (answer_key_int=-1) → None (bỏ qua khi eval)
            # - Câu có đáp án nhưng uncertain → "-"
            # - Câu có đáp án và detect được → "A"/"B"/"C"/"D"
            answer_key_int_local = golden.get("answer_key_int") or []
            expected_user = []
            for i, user_val in enumerate(user_int):
                key_val = answer_key_int_local[i] if i < len(answer_key_int_local) else -1
                if key_val is None or key_val == -1:
                    expected_user.append(None)  # no-key → null (bỏ qua trong eval)
                elif user_val is None or user_val == -1:
                    expected_user.append("-")   # có key nhưng uncertain
                else:
                    expected_user.append(LABEL_MAP[int(user_val)])
            golden["expected_user_answers"] = expected_user
            # expected_uncertain chỉ là câu CÓ đáp án nhưng detect uncertain (lỗi thật)
            golden["expected_uncertain"] = [
                i + 1 for i, (u, k) in enumerate(
                    zip(user_int, answer_key_int_local + [-1] * 200)
                )
                if (u is None or u == -1) and (k is not None and k != -1)
            ]

        if bc_entry:
            # Lấy questions count từ DB user_answers nếu bubble_confidence không có
            if not golden["questions"] and user_int:
                golden["questions"] = len(user_int)
            if not golden["form_profile"] and golden["questions"]:
                golden["form_profile"] = infer_profile(golden["questions"])

    # Fallback: lookup answer_key từ answer_sets_map nếu chưa có
    if not golden["answer_key"] and golden["exam_code"]:
        code = str(golden["exam_code"]).strip()
        if code in answer_sets_map:
            entry = answer_sets_map[code]
            raw = entry["answers"]
            if isinstance(raw, list):
                valid = [v for v in raw if v is not None and v != -1]
                if valid:
                    golden["answer_key_int"] = raw
                    golden["answer_key"] = int_list_to_labels(raw)
                    golden["aid"] = golden["aid"] or entry["aid"]
                    golden["answer_key_source"] = golden["answer_key_source"] or "assignment-lookup"

    # Cap answer_key theo FORM_PROFILE_MAX_GRADED (một số profile có đáp án DB
    # ngoài phạm vi thực tế cần chấm, ví dụ 120pdf chỉ chấm Q1-Q20)
    fp_key = golden.get("form_profile") or ""
    max_graded = FORM_PROFILE_MAX_GRADED.get(fp_key)
    if max_graded and golden.get("answer_key_int"):
        cap = int(max_graded)
        ak_int = list(golden["answer_key_int"])
        for i in range(cap, len(ak_int)):
            ak_int[i] = -1
        golden["answer_key_int"] = ak_int
        golden["answer_key"] = int_list_to_labels(ak_int)

    # Đánh dấu cần annotation thủ công nếu không có answer_key
    if not golden["answer_key"]:
        golden["needs_annotation"] = True

    # Thêm graded_questions: số câu thực sự có đáp án (answer_key_int != -1)
    golden["graded_questions"] = sum(
        1 for v in (golden.get("answer_key_int") or [])
        if v is not None and v != -1
    )

    return golden


def update_golden_from_eval(eval_json_path: str, dataset_dir: Path):
    """
    Đọc eval report JSON, cập nhật golden files cho ảnh có score thay đổi.
    Chỉ cập nhật: expected_score, expected_user_answers, expected_uncertain.
    Giữ nguyên: answer_key, answer_key_int (đáp án đúng không thay đổi).
    """
    with open(eval_json_path, encoding="utf-8") as f:
        report = json.load(f)

    images_report = report.get("images", [])
    updated, skipped, not_found = 0, 0, 0

    for img_result in images_report:
        direction = img_result.get("score_direction")
        if direction not in ("improved", "regressed"):
            skipped += 1
            continue

        image_name = img_result.get("image", "")
        category = img_result.get("category", "normal")

        # Tìm golden file
        if category and category != "normal":
            golden_path = dataset_dir / category / Path(image_name).with_suffix(".golden.json").name
        else:
            golden_path = dataset_dir / Path(image_name).with_suffix(".golden.json").name

        if not golden_path.exists():
            not_found += 1
            print(f"  [!] Không tìm thấy golden: {golden_path}")
            continue

        with open(golden_path, encoding="utf-8") as f:
            golden = json.load(f)

        # Lấy kết quả mới từ eval report
        new_score = img_result.get("api_score")
        api_user_answers_int = img_result.get("api_user_answers_raw")  # nếu có
        new_uncertain_qs = img_result.get("api_uncertain_questions") or []

        # Rebuild expected_user_answers từ answer_compare trong report
        answer_compare = img_result.get("question_results") or []
        answer_key_int = golden.get("answer_key_int") or []

        if answer_compare:
            new_expected_user = []
            for i, ac in enumerate(answer_compare):
                key_val = answer_key_int[i] if i < len(answer_key_int) else -1
                if key_val is None or key_val == -1:
                    new_expected_user.append(None)
                else:
                    sel = ac.get("selected_label") or "-"
                    new_expected_user.append(sel)
            golden["expected_user_answers"] = new_expected_user
        else:
            # Không có answer_compare, dùng uncertain list để build
            old_user = golden.get("expected_user_answers") or []
            new_expected_user = []
            graded_q = golden.get("graded_questions") or len(answer_key_int)
            for i, old_val in enumerate(old_user):
                key_val = answer_key_int[i] if i < len(answer_key_int) else -1
                if key_val is None or key_val == -1:
                    new_expected_user.append(None)
                elif (i + 1) in new_uncertain_qs:
                    new_expected_user.append("-")
                else:
                    new_expected_user.append(old_val)
            golden["expected_user_answers"] = new_expected_user

        # Cập nhật expected_uncertain: chỉ câu có key nhưng uncertain
        golden["expected_uncertain"] = [
            q for q in new_uncertain_qs
            if q - 1 < len(answer_key_int) and
            answer_key_int[q - 1] is not None and answer_key_int[q - 1] != -1
        ]

        if new_score is not None:
            golden["expected_score"] = new_score

        golden["golden_source"] = "eval_update"
        golden["updated_at"] = datetime.now().isoformat()
        golden["score_direction_at_update"] = direction

        with open(golden_path, "w", encoding="utf-8") as f:
            json.dump(golden, f, ensure_ascii=False, indent=2)
        updated += 1
        print(f"  [UPD] {category}/{image_name}: {direction}, score -> {new_score}")

    print(f"\n  Cập nhật: {updated}, bỏ qua (matched/no-baseline): {skipped}, không tìm thấy: {not_found}")


def collect_images(dataset_dir: Path) -> list[tuple[Path, str]]:
    """Trả về [(image_path, category)] cho tất cả ảnh trong dataset_dir."""
    images = []
    # Root folder
    for p in sorted(dataset_dir.iterdir()):
        if p.is_file() and p.suffix.lower() in (".jpg", ".jpeg", ".png"):
            images.append((p, "normal"))
    # Subfolders
    for sub in sorted(dataset_dir.iterdir()):
        if sub.is_dir():
            cat = sub.name
            for p in sorted(sub.iterdir()):
                if p.is_file() and p.suffix.lower() in (".jpg", ".jpeg", ".png"):
                    images.append((p, cat))
    return images


def main():
    parser = argparse.ArgumentParser(description="Tạo golden data cho eval dataset OMR")
    parser.add_argument("--dataset-dir", default=str(DATASET_DIR_DEFAULT))
    parser.add_argument("--storage-dir", default=str(STORAGE_DIR_DEFAULT))
    parser.add_argument("--db-url", default=DB_URL_DEFAULT)
    parser.add_argument("--force", action="store_true", help="Ghi đè file golden đã tồn tại")
    parser.add_argument(
        "--update-from-eval",
        default="",
        metavar="EVAL_JSON",
        help="Cập nhật golden từ kết quả eval (chỉ ảnh improved/regressed)",
    )
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)

    # Chế độ update-from-eval: chỉ cập nhật, không rebuild toàn bộ
    if args.update_from_eval:
        print(f"=== Update Golden From Eval ===")
        print(f"Dataset: {dataset_dir}")
        print(f"Eval:    {args.update_from_eval}")
        print()
        update_golden_from_eval(args.update_from_eval, dataset_dir)
        return

    storage_dir = Path(args.storage_dir)

    print(f"=== Build Golden Data ===")
    print(f"Dataset: {dataset_dir}")
    print(f"Storage: {storage_dir}")
    print()

    # Bước 1: Build map bubble_confidence
    print("[1/4] Quét bubble_confidence JSONs...")
    bc_map = build_bubble_confidence_map(storage_dir)
    print(f"  Tìm thấy {len(bc_map)} bubble_confidence files")

    # Bước 2 & 3: Query DB
    print("[2/4] Query DB...")
    db_map, answer_sets_map = query_db(args.db_url)

    # Bước 4: Tạo golden JSON cho từng ảnh
    print("[3/4] Tạo golden JSON...")
    images = collect_images(dataset_dir)
    print(f"  Tìm thấy {len(images)} ảnh")

    stats = {
        "total": 0,
        "with_bc": 0,
        "with_db": 0,
        "with_answer_key": 0,
        "needs_annotation": 0,
        "skipped_existing": 0,
        "categories": {},
        "form_profiles": {},
    }

    all_entries = []

    for image_path, category in images:
        stats["total"] += 1
        stats["categories"][category] = stats["categories"].get(category, 0) + 1

        # Tìm bubble_confidence
        ts = extract_camera_ts(image_path.name)
        bc_entry = bc_map.get(ts) if ts else None
        db_record = db_map.get(ts) if ts else None

        if bc_entry:
            stats["with_bc"] += 1
        if db_record:
            stats["with_db"] += 1

        # Tạo golden entry
        golden = build_golden_entry(image_path, category, bc_entry, db_record, answer_sets_map)

        if golden["answer_key"]:
            stats["with_answer_key"] += 1
        if golden["needs_annotation"]:
            stats["needs_annotation"] += 1
        if golden["form_profile"]:
            fp = golden["form_profile"]
            stats["form_profiles"][fp] = stats["form_profiles"].get(fp, 0) + 1

        # Ghi file
        golden_path = image_path.with_suffix(".golden.json")
        if golden_path.exists() and not args.force:
            stats["skipped_existing"] += 1
        else:
            with open(golden_path, "w", encoding="utf-8") as f:
                json.dump(golden, f, ensure_ascii=False, indent=2)

        all_entries.append({
            "image": image_path.name,
            "path": str(image_path),
            "golden_path": str(golden_path),
            "category": category,
            "ts": ts,
            "has_bc": bc_entry is not None,
            "has_db": db_record is not None,
            "has_answer_key": golden["answer_key"] is not None,
            "needs_annotation": golden["needs_annotation"],
            "form_profile": golden["form_profile"],
            "questions": golden["questions"],
            "graded_questions": golden.get("graded_questions"),
            "exam_code": golden["exam_code"],
        })

    # Ghi eval_manifest.json
    print("[4/4] Ghi eval_manifest.json...")
    manifest = {
        "created_at": datetime.now().isoformat(),
        "dataset_dir": str(dataset_dir),
        "stats": stats,
        "images": all_entries,
    }
    manifest_path = dataset_dir / "eval_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    # Báo cáo
    print()
    print("=== Kết quả ===")
    print(f"  Tổng ảnh:           {stats['total']}")
    print(f"  Có bubble_conf:     {stats['with_bc']}")
    print(f"  Có DB record:       {stats['with_db']}")
    print(f"  Có answer_key:      {stats['with_answer_key']}")
    print(f"  Cần annotation:     {stats['needs_annotation']}")
    if stats["skipped_existing"]:
        print(f"  Bỏ qua (đã có):    {stats['skipped_existing']}  (dùng --force để ghi đè)")
    print(f"  Manifest:           {manifest_path}")
    print()
    print("  Phân loại:")
    for cat, cnt in sorted(stats["categories"].items()):
        print(f"    {cat or 'normal'}: {cnt}")
    print("  Form profiles:")
    for fp, cnt in sorted(stats["form_profiles"].items()):
        print(f"    {fp}: {cnt}")

    # Liệt kê ảnh cần annotation
    needs = [e for e in all_entries if e["needs_annotation"]]
    if needs:
        print()
        print(f"  [!] {len(needs)} ảnh chưa có answer_key:")
        for e in needs:
            ts_info = f"ts={e['ts']}" if e["ts"] else "ts=?"
            bc_info = "bc=OK" if e["has_bc"] else "bc=MISS"
            print(f"      {e['category']}/{e['image']}  ({ts_info}, {bc_info})")


if __name__ == "__main__":
    main()
