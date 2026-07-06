"""
run_eval.py — Chạy eval pipeline OMR trên toàn bộ ocr_dataset.

Gửi từng ảnh lên API /api/omr/grade, so sánh với golden JSON, báo cáo metrics.
Cần backend đang chạy (localhost:8000) và các golden JSON đã được build_golden.py tạo trước.

Usage:
    python eval/run_eval.py
    python eval/run_eval.py --category "không bóng che"
    python eval/run_eval.py --category nghiêng --verbose
    python eval/run_eval.py --api-url http://localhost:8000 --output eval/results/eval.json
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

try:
    import requests
except ImportError:
    print("[ERROR] requests chưa cài. Chạy: pip install requests")
    sys.exit(1)

DATASET_DIR_DEFAULT = Path(__file__).parent.parent / "ocr_dataset"
API_URL_DEFAULT = "http://localhost:8000"
SCORE_TOLERANCE = 0.5
LABEL_MAP = ["A", "B", "C", "D", "E"]


def int_list_to_labels(int_list: list) -> list:
    result = []
    for v in int_list:
        if v is None or v == -1:
            result.append("-")
        else:
            result.append(LABEL_MAP[int(v)] if 0 <= int(v) < len(LABEL_MAP) else "-")
    return result


def load_manifest(dataset_dir: Path) -> dict:
    path = dataset_dir / "eval_manifest.json"
    if not path.exists():
        print(f"[ERROR] Không tìm thấy eval_manifest.json tại {path}")
        print("        Chạy build_golden.py trước.")
        sys.exit(1)
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_golden(image_path: Path) -> dict | None:
    golden_path = image_path.with_suffix(".golden.json")
    if not golden_path.exists():
        return None
    with open(golden_path, encoding="utf-8") as f:
        return json.load(f)


def grade_image(api_url: str, image_path: Path, golden: dict, no_rescue: bool = False, density_only: bool = False) -> dict | None:
    """Gửi ảnh lên API và lấy kết quả. Trả về None nếu lỗi."""
    import tempfile

    form_profile = golden.get("form_profile") or ""
    answer_key = golden.get("answer_key") or []
    num_questions = golden.get("questions") or len(answer_key)

    # Bỏ qua "-" (no-key questions)
    valid_answers = [lbl for lbl in answer_key if lbl and lbl != "-"]
    answers_str = ",".join(valid_answers)

    try:
        with open(image_path, "rb") as img_file:
            img_bytes = img_file.read()

        # Nếu số answers < num_questions (có "-" no-key entries), dùng file upload
        # để bypass validation "len(answer_key) != num_questions".
        # Khi dùng file source: API dùng len(answer_key) làm parsed_questions
        # → questions ngoài phạm vi answer_key sẽ là "no-key".
        use_file_upload = len(valid_answers) < int(num_questions)

        if use_file_upload:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".txt", delete=False, encoding="utf-8"
            ) as tmp:
                tmp.write(answers_str)
                tmp_path = tmp.name
            try:
                with open(tmp_path, "rb") as key_file:
                    files = {
                        "file": (image_path.name, img_bytes, "image/jpeg"),
                        "answer_key_file": ("answer_key.txt", key_file, "text/plain"),
                    }
                    data = {
                        "form_profile_code": form_profile,
                        "answers": "",
                        "num_questions": str(num_questions),
                    }
                    if no_rescue:
                        data["disable_mcq_rescue"] = "true"
                    if density_only:
                        data["density_only_scoring"] = "true"
                    resp = requests.post(
                        f"{api_url}/api/omr/grade",
                        files=files,
                        data=data,
                        timeout=60,
                    )
            finally:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
        else:
            files = {"file": (image_path.name, img_bytes, "image/jpeg")}
            data = {
                "form_profile_code": form_profile,
                "answers": answers_str,
                "num_questions": str(num_questions),
            }
            if no_rescue:
                data["disable_mcq_rescue"] = "true"
            if density_only:
                data["density_only_scoring"] = "true"
            resp = requests.post(
                f"{api_url}/api/omr/grade",
                files=files,
                data=data,
                timeout=60,
            )

        if resp.status_code != 200:
            try:
                err_detail = resp.json().get("detail") or resp.text[:200]
            except Exception:
                err_detail = resp.text[:200]
            return {"api_error": resp.status_code, "detail": err_detail}
        return resp.json()
    except requests.exceptions.ConnectionError:
        return {"api_error": "connection_refused", "detail": "Backend không chạy"}
    except Exception as e:
        return {"api_error": "exception", "detail": str(e)}


def compare_result(api_resp: dict, golden: dict) -> dict:
    """So sánh kết quả API với golden data. Trả về metrics cho 1 ảnh."""
    graded_q = golden.get("graded_questions") or golden.get("questions") or 0

    result = {
        "image": golden["image"],
        "category": golden["category"],
        "form_profile": golden.get("form_profile"),
        "exam_code_golden": golden.get("exam_code"),
        "questions": golden.get("questions"),
        "graded_questions": graded_q,
        "has_db_record": golden.get("has_db_record", False),

        # API response fields
        "api_exam_code": None,
        "api_score": None,
        "api_uncertain_count": 0,
        "api_uncertain_questions": [],

        # Per-question (graded range only)
        "total_questions": 0,
        "graded_detected_count": 0,
        "meaningful_uncertain_count": 0,
        "meaningful_uncertain": [],
        "detected_count": 0,
        "uncertain_count": 0,
        "correct_count": 0,
        "wrong_count": 0,

        # Score comparison
        "expected_score": golden.get("expected_score"),
        "score_match": None,
        "score_diff": None,
        "score_direction": "no-baseline",

        # Regression (new detection vs old golden detection, bỏ qua None)
        "regression_agree": 0,
        "regression_total": 0,
        "regression_rate": None,
        "perfect_regression": False,

        # question_results: cho --update-from-eval
        "question_results": [],

        # Errors
        "api_error": None,
    }

    data = api_resp.get("data", {})
    if not data or "api_error" in api_resp:
        result["api_error"] = api_resp.get("api_error") or api_resp.get("detail") or "unknown"
        return result

    result["api_exam_code"] = data.get("exam_code")
    result["api_score"] = data.get("score")
    api_user_answers = data.get("user_answers") or []
    uncertain_qs = data.get("uncertain_questions") or []
    result["api_uncertain_questions"] = uncertain_qs
    result["api_uncertain_count"] = len(uncertain_qs)

    # Thống kê từ answer_compare
    answer_compare = data.get("answer_compare") or []
    result["total_questions"] = len(answer_compare)
    result["question_results"] = [
        {"question": ac.get("question"), "selected_label": ac.get("selected_label"), "status": ac.get("status")}
        for ac in answer_compare
    ]
    correct_c, wrong_c, uncertain_c = 0, 0, 0
    graded_detected = 0
    for ac in answer_compare:
        status = ac.get("status", "")
        q = ac.get("question", 0)
        if status == "correct":
            correct_c += 1
            if q <= graded_q:
                graded_detected += 1
        elif status == "wrong":
            wrong_c += 1
            if q <= graded_q:
                graded_detected += 1
        elif status == "uncertain":
            uncertain_c += 1
    result["detected_count"] = correct_c + wrong_c
    result["uncertain_count"] = uncertain_c
    result["correct_count"] = correct_c
    result["wrong_count"] = wrong_c
    result["graded_detected_count"] = graded_detected

    # meaningful_uncertain: chỉ câu trong graded range
    meaningful_uncertain = [q for q in uncertain_qs if q <= graded_q]
    result["meaningful_uncertain_count"] = len(meaningful_uncertain)
    result["meaningful_uncertain"] = meaningful_uncertain

    # Score comparison + direction
    expected_score = golden.get("expected_score")
    new_score = result["api_score"]
    if expected_score is not None and new_score is not None:
        diff = abs(float(new_score) - float(expected_score))
        result["score_diff"] = round(diff, 3)
        result["score_match"] = diff <= SCORE_TOLERANCE
        if float(new_score) > float(expected_score) + SCORE_TOLERANCE:
            result["score_direction"] = "improved"
        elif float(new_score) < float(expected_score) - SCORE_TOLERANCE:
            result["score_direction"] = "regressed"
        else:
            result["score_direction"] = "matched"

    # Regression: so sánh new detection vs golden expected_user_answers
    # Bỏ qua vị trí golden_user[i] == None (no-key questions)
    golden_user = golden.get("expected_user_answers") or []
    if golden_user and api_user_answers:
        new_labels = int_list_to_labels(api_user_answers)
        agree, total_reg = 0, 0
        for i, g in enumerate(golden_user):
            if g is None:
                continue  # no-key, bỏ qua
            if i >= len(new_labels):
                break
            total_reg += 1
            if g == new_labels[i]:
                agree += 1
        result["regression_agree"] = agree
        result["regression_total"] = total_reg
        result["regression_rate"] = round(agree / total_reg, 4) if total_reg > 0 else None
        result["perfect_regression"] = agree == total_reg

    return result


def print_verbose_result(img_result: dict, golden: dict, api_resp: dict):
    """In chi tiết per-ảnh khi --verbose."""
    img = img_result["image"]
    cat = img_result["category"]
    score = img_result.get("api_score")
    exp = img_result.get("expected_score")
    score_str = f"{score:.1f}" if score is not None else "?"
    exp_str = f"{exp:.1f}" if exp is not None else "?"
    direction = img_result.get("score_direction", "no-baseline")
    m_unc = img_result.get("meaningful_uncertain_count", 0)
    reg = img_result.get("regression_rate")
    reg_str = f"{reg*100:.1f}%" if reg is not None else "N/A"

    error = img_result.get("api_error")
    if error:
        print(f"  [ERR] {cat}/{img}: {error}")
        return

    print(f"  [{direction.upper():<8}] {cat}/{img}")
    print(f"         score={score_str}/{exp_str}  g-unc={m_unc}  regression={reg_str}")

    # In per-question nếu có lỗi
    if img_result.get("score_match") is False or uncertain > 0:
        data = api_resp.get("data", {})
        answer_compare = data.get("answer_compare") or []
        golden_user = golden.get("expected_user_answers") or []
        answer_key = golden.get("answer_key") or []
        new_labels = int_list_to_labels(data.get("user_answers") or [])
        for ac in answer_compare:
            q = ac.get("question", 0)
            status = ac.get("status", "")
            sel = ac.get("selected_label", "-")
            key = answer_key[q-1] if q-1 < len(answer_key) else "?"
            old_det = golden_user[q-1] if q-1 < len(golden_user) else "?"
            if status in ("wrong", "uncertain"):
                reg_flag = " REG_DIFF" if old_det != sel else ""
                print(f"         Q{q:02d}: key={key} got={sel} old={old_det} [{status.upper()}]{reg_flag}")


def aggregate(results: list) -> dict:
    """Tính metrics tổng hợp từ list kết quả."""
    total = len(results)
    errors = sum(1 for r in results if r.get("api_error"))
    valid = [r for r in results if not r.get("api_error")]

    score_matches = [r for r in valid if r.get("score_match") is True]
    score_evaluated = [r for r in valid if r.get("score_match") is not None]
    perfect_regressions = [r for r in valid if r.get("perfect_regression")]
    improved = [r for r in valid if r.get("score_direction") == "improved"]
    regressed = [r for r in valid if r.get("score_direction") == "regressed"]

    total_graded_q = sum(r.get("graded_questions") or 0 for r in valid)
    total_graded_detected = sum(r.get("graded_detected_count") or 0 for r in valid)
    total_correct = sum(r.get("correct_count") or 0 for r in valid)
    total_wrong = sum(r.get("wrong_count") or 0 for r in valid)
    total_meaningful_unc = sum(r.get("meaningful_uncertain_count") or 0 for r in valid)
    total_reg_agree = sum(r["regression_agree"] for r in valid if r.get("regression_total", 0) > 0)
    total_reg_total = sum(r["regression_total"] for r in valid if r.get("regression_total", 0) > 0)

    # Câu thường xuyên uncertain (chỉ trong graded range)
    uncertain_q_freq: dict = {}
    for r in valid:
        graded_q = r.get("graded_questions") or 0
        for q in r.get("api_uncertain_questions") or []:
            if q <= graded_q:
                uncertain_q_freq[q] = uncertain_q_freq.get(q, 0) + 1
    top_uncertain = sorted(uncertain_q_freq.items(), key=lambda x: -x[1])[:10]

    graded_detection_rate = round(total_graded_detected / total_graded_q, 4) if total_graded_q else None

    return {
        "total_images": total,
        "valid_images": len(valid),
        "error_images": errors,
        "score_accuracy": round(len(score_matches) / len(score_evaluated), 4) if score_evaluated else None,
        "score_accuracy_pct": f"{len(score_matches)/len(score_evaluated)*100:.1f}%" if score_evaluated else "N/A",
        "graded_detection_rate": graded_detection_rate,
        "graded_detection_rate_pct": f"{graded_detection_rate*100:.1f}%" if graded_detection_rate is not None else "N/A",
        "true_detection_accuracy": round(total_correct / total_graded_q, 4) if total_graded_q else None,
        "true_detection_accuracy_pct": f"{total_correct/total_graded_q*100:.1f}%" if total_graded_q else "N/A",
        "wrong_detection_rate": round(total_wrong / total_graded_q, 4) if total_graded_q else None,
        "wrong_detection_rate_pct": f"{total_wrong/total_graded_q*100:.1f}%" if total_graded_q else "N/A",
        "uncertain_rate": round(total_meaningful_unc / total_graded_q, 4) if total_graded_q else None,
        "uncertain_rate_pct": f"{total_meaningful_unc/total_graded_q*100:.1f}%" if total_graded_q else "N/A",
        "mean_meaningful_uncertain": round(total_meaningful_unc / len(valid), 2) if valid else None,
        "regression_rate": round(total_reg_agree / total_reg_total, 4) if total_reg_total else None,
        "regression_rate_pct": f"{total_reg_agree/total_reg_total*100:.1f}%" if total_reg_total else "N/A",
        "perfect_images": round(len(perfect_regressions) / len(valid), 4) if valid else None,
        "perfect_images_pct": f"{len(perfect_regressions)/len(valid)*100:.1f}%" if valid else "N/A",
        "improved_count": len(improved),
        "regressed_count": len(regressed),
        "top_uncertain_questions": [
            {"question": q, "count": c, "pct": f"{c/len(valid)*100:.0f}%"}
            for q, c in top_uncertain
        ],
    }


def print_summary_table(all_results: list, category: str):
    cats = sorted(set(r["category"] for r in all_results))
    print()
    print("=== OMR Eval Report ===")
    header = f"{'Category':<20} | {'Images':>6} | {'True-Acc':>8} | {'G-DetRate':>9} | {'Wrong':>6} | {'Uncertain':>9} | {'G-Unc':>5} | {'BETTER/WORSE':>12}"
    print(header)
    print("-" * len(header))
    for cat in cats:
        cat_results = [r for r in all_results if r["category"] == cat]
        m = aggregate(cat_results)
        g_unc = str(m["mean_meaningful_uncertain"] or "?")
        bw = f"{m['improved_count']}/{m['regressed_count']}"
        print(f"{cat:<20} | {m['total_images']:>6} | {m['true_detection_accuracy_pct']:>8} | {m['graded_detection_rate_pct']:>9} | {m['wrong_detection_rate_pct']:>6} | {m.get('uncertain_rate_pct', '?'):>9} | {g_unc:>5} | {bw:>12}")
    print("-" * len(header))
    total_m = aggregate(all_results)
    g_unc = str(total_m["mean_meaningful_uncertain"] or "?")
    bw = f"{total_m['improved_count']}/{total_m['regressed_count']}"
    print(f"{'TOTAL':<20} | {total_m['total_images']:>6} | {total_m['true_detection_accuracy_pct']:>8} | {total_m['graded_detection_rate_pct']:>9} | {total_m['wrong_detection_rate_pct']:>6} | {total_m.get('uncertain_rate_pct', '?'):>9} | {g_unc:>5} | {bw:>12}")
    if total_m["top_uncertain_questions"]:
        print()
        top = ", ".join(f"Q{e['question']}({e['pct']})" for e in total_m["top_uncertain_questions"][:5])
        print(f"Câu thường uncertain (graded): {top}")
    errors = [r for r in all_results if r.get("api_error")]
    if errors:
        print(f"\n[!] {len(errors)} ảnh bị lỗi API:")
        for r in errors:
            print(f"    {r['category']}/{r['image']}: {r['api_error']}")


def main():
    parser = argparse.ArgumentParser(description="Eval pipeline OMR")
    parser.add_argument("--dataset-dir", default=str(DATASET_DIR_DEFAULT))
    parser.add_argument("--api-url", default=API_URL_DEFAULT)
    parser.add_argument("--category", default="all",
                        help="all | normal | Clear_answer | Shadow | Tilted_image | Vague_answer")
    parser.add_argument("--output", default="",
                        help="Đường dẫn file JSON output (mặc định: eval/results/eval_YYYYMMDD_HHMMSS.json)")
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--skip-no-db", action="store_true",
                        help="Bỏ qua ảnh không có DB record (không có expected_score)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Chỉ liệt kê ảnh, không gọi API")
    parser.add_argument("--no-rescue", action="store_true",
                        help="Tắt MCQ Map Search Rescue (ablation)")
    parser.add_argument("--density-only", action="store_true",
                        help="Chỉ dùng density score, bỏ darkness (ablation)")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)

    # Load manifest
    print(f"=== OMR Eval ===")
    print(f"Dataset: {dataset_dir}")
    print(f"API:     {args.api_url}")
    print(f"Filter:  category={args.category}")
    if args.no_rescue:
        print(f"Mode:    --no-rescue (MCQ Map Search Rescue DISABLED)")
    if args.density_only:
        print(f"Mode:    --density-only (darkness bỏ qua, chỉ dùng density score)")
    print()

    manifest = load_manifest(dataset_dir)
    all_images = manifest.get("images", [])

    # Filter theo category
    if args.category != "all":
        all_images = [e for e in all_images if e["category"] == args.category]
        print(f"Sau filter: {len(all_images)} ảnh")

    # Chỉ ảnh có answer_key
    images_with_key = [e for e in all_images if e.get("has_answer_key")]
    images_no_key = [e for e in all_images if not e.get("has_answer_key")]

    if images_no_key:
        print(f"[!] {len(images_no_key)} ảnh không có answer_key (needs_annotation=true), bỏ qua.")

    if args.skip_no_db:
        images_with_key = [e for e in images_with_key if e.get("has_db")]
        print(f"    (--skip-no-db): chỉ giữ {len(images_with_key)} ảnh có DB record")

    print(f"Sẽ eval: {len(images_with_key)} ảnh")
    print()

    if args.dry_run:
        for e in images_with_key:
            print(f"  {e['category']}/{e['image']}  (fp={e['form_profile']}, ec={e['exam_code']})")
        return

    # Kiểm tra API health
    try:
        r = requests.get(f"{args.api_url}/health", timeout=5)
        if r.status_code != 200:
            print(f"[WARN] API health check trả về {r.status_code}")
        else:
            print("API health check: OK")
    except Exception as e:
        print(f"[ERROR] Không kết nối được API tại {args.api_url}: {e}")
        sys.exit(1)

    # Chạy eval
    all_results = []
    t_start = time.time()

    for i, entry in enumerate(images_with_key, 1):
        image_path = Path(entry["path"])
        if not image_path.exists():
            print(f"  [{i}/{len(images_with_key)}] SKIP (file không tồn tại): {image_path.name}")
            continue

        golden = load_golden(image_path)
        if not golden:
            print(f"  [{i}/{len(images_with_key)}] SKIP (không có golden JSON): {image_path.name}")
            continue

        print(f"  [{i}/{len(images_with_key)}] {entry['category']}/{image_path.name}", end=" ", flush=True)
        t0 = time.time()

        api_resp = grade_image(args.api_url, image_path, golden, no_rescue=args.no_rescue, density_only=args.density_only)
        elapsed = time.time() - t0

        img_result = compare_result(api_resp, golden)
        all_results.append(img_result)

        # In kết quả ngắn
        if img_result.get("api_error"):
            print(f"[ERR {elapsed:.1f}s] {img_result['api_error']}")
        else:
            score = img_result.get("api_score")
            exp = img_result.get("expected_score")
            direction = img_result.get("score_direction", "no-baseline")
            m_unc = img_result.get("meaningful_uncertain_count", 0)
            reg = img_result.get("regression_rate")
            prefix_map = {
                "matched": "OK    ",
                "improved": "BETTER",
                "regressed": "WORSE ",
                "no-baseline": "?     ",
            }
            pfx = prefix_map.get(direction, "?     ")
            score_str = f"{score:.1f}" if score is not None else "?"
            exp_str = f"{exp:.1f}" if exp is not None else "?"
            reg_str = f"{reg*100:.0f}%" if reg is not None else "?"
            print(f"[{pfx}] score={score_str}/{exp_str} g-unc={m_unc} reg={reg_str} ({elapsed:.1f}s)")

        if args.verbose:
            print_verbose_result(img_result, golden, api_resp)

    t_total = time.time() - t_start

    if not all_results:
        print("\n[!] Không có ảnh nào được eval.")
        return

    # Báo cáo tổng hợp
    print_summary_table(all_results, args.category)

    total_m = aggregate(all_results)
    print(f"\nThời gian: {t_total:.1f}s / {len(all_results)} ảnh = {t_total/len(all_results):.1f}s/ảnh")

    # Lưu JSON
    if args.output:
        out_path = Path(args.output)
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = Path(__file__).parent / "results"
        out_dir.mkdir(exist_ok=True)
        out_path = out_dir / f"eval_{ts}.json"

    report = {
        "created_at": datetime.now().isoformat(),
        "config": {
            "api_url": args.api_url,
            "dataset_dir": str(dataset_dir),
            "category_filter": args.category,
        },
        "summary": total_m,
        "by_category": {
            cat: aggregate([r for r in all_results if r["category"] == cat])
            for cat in sorted(set(r["category"] for r in all_results))
        },
        "images": all_results,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\nReport: {out_path}")


if __name__ == "__main__":
    main()
