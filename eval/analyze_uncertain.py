"""
analyze_uncertain.py — Phân tích nguyên nhân câu uncertain từ bubble_confidence JSON.

Đọc bubble_confidence JSON sẵn có, không cần chạy lại pipeline.

Usage:
    python eval/analyze_uncertain.py
    python eval/analyze_uncertain.py --targets Q7,Q9,Q11 --baseline Q1,Q3,Q5
    python eval/analyze_uncertain.py --question Q7        # chi tiết từng ảnh cho Q7
    python eval/analyze_uncertain.py --profile 20-cau-pdf # chỉ form profile này
"""

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

DATASET_DIR = Path(__file__).parent.parent / "ocr_dataset"
STORAGE_DIR = Path(__file__).parent.parent / "storage" / "uploads" / "omr"
RESULTS_DIR = Path(__file__).parent / "results"

SUBFOLDER_ROOTS = [
    DATASET_DIR,
    DATASET_DIR / "Clear_answer",
    DATASET_DIR / "Shadow",
    DATASET_DIR / "Tilted_image",
    DATASET_DIR / "Vague_answer",
]


def parse_q(s: str) -> int:
    return int(s.strip().lstrip("Qq"))


def load_all_golden() -> list[dict]:
    entries = []
    for subdir in SUBFOLDER_ROOTS:
        if not subdir.exists():
            continue
        for f in sorted(subdir.glob("*.golden.json")):
            with open(f, encoding="utf-8") as fp:
                d = json.load(fp)
            d["_golden_path"] = str(f)
            entries.append(d)
    return entries


def load_bubble_confidence(bc_filename: str) -> list[dict] | None:
    path = STORAGE_DIR / bc_filename
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as fp:
        data = json.load(fp)
    return data.get("rows") or []


def collect_rows(golden_entries: list[dict], profile_filter: str | None) -> dict[str, list[dict]]:
    """
    Returns: {question_key: [row_dict, ...]} across all valid images.
    row_dict includes extra fields: _form_profile, _category, _image
    """
    question_rows: dict[str, list[dict]] = defaultdict(list)

    for g in golden_entries:
        if g.get("needs_annotation"):
            continue
        profile = g.get("form_profile") or ""
        if profile_filter and profile != profile_filter:
            continue
        bc_file = g.get("source_bubble_confidence")
        if not bc_file:
            continue
        rows = load_bubble_confidence(bc_file)
        if rows is None:
            continue
        for row in rows:
            q = row.get("question")
            if q is None:
                continue
            row = dict(row)
            row["_form_profile"] = profile
            row["_category"] = g.get("category")
            row["_image"] = g.get("image")
            question_rows[f"Q{q}"].append(row)

    return question_rows


def median_or_none(vals: list) -> float | None:
    clean = [v for v in vals if v is not None]
    if not clean:
        return None
    return round(statistics.median(clean), 4)


def pct(count: int, total: int) -> str:
    if total == 0:
        return "  n/a"
    return f"{100 * count / total:5.1f}%"


def analyze_question(rows: list[dict]) -> dict:
    if not rows:
        return {}
    n = len(rows)
    uncertain_rows = [r for r in rows if r.get("uncertain")]
    n_unc = len(uncertain_rows)

    noise_rej = sum(1 for r in rows if r.get("noise_rejected"))
    double_mk = sum(1 for r in rows if r.get("double_mark"))
    below_thr = sum(1 for r in rows if not r.get("uncertain") is False and not r.get("noise_rejected") and not r.get("double_mark"))
    grid_used = sum(1 for r in rows if r.get("grid_search_used"))

    best_scores = [r.get("best_score") for r in rows]
    margins = [r.get("margin") for r in rows]
    center_dens = [r.get("best_center_density") for r in rows]
    dark_p20 = [r.get("best_dark_p20") for r in rows]

    # D-column (idx=3) scores specifically
    d_scores = [r["scores"][3] for r in rows if r.get("scores") and len(r["scores"]) > 3]

    # Uncertain-only rows
    unc_best = [r.get("best_score") for r in uncertain_rows]
    unc_margins = [r.get("margin") for r in uncertain_rows]
    unc_center = [r.get("best_center_density") for r in uncertain_rows]
    unc_d_scores = [r["scores"][3] for r in uncertain_rows if r.get("scores") and len(r["scores"]) > 3]

    # Selected label distribution
    label_counts: dict[str, int] = defaultdict(int)
    for r in rows:
        lbl = r.get("selected_label") or "-"
        label_counts[lbl] += 1

    return {
        "n": n,
        "uncertain_count": n_unc,
        "uncertain_pct": round(100 * n_unc / n, 1) if n else 0,
        "noise_rejected_count": noise_rej,
        "noise_rejected_pct": round(100 * noise_rej / n, 1) if n else 0,
        "double_mark_count": double_mk,
        "double_mark_pct": round(100 * double_mk / n, 1) if n else 0,
        "grid_used_pct": round(100 * grid_used / n, 1) if n else 0,
        "median_best_score": median_or_none(best_scores),
        "median_margin": median_or_none(margins),
        "median_center_density": median_or_none(center_dens),
        "median_d_score": median_or_none(d_scores),
        "uncertain_median_best": median_or_none(unc_best),
        "uncertain_median_margin": median_or_none(unc_margins),
        "uncertain_median_center": median_or_none(unc_center),
        "uncertain_median_d_score": median_or_none(unc_d_scores),
        "label_distribution": dict(label_counts),
    }


def print_comparison_table(stats: dict[str, dict], targets: list[str], baseline: list[str]):
    all_qs = baseline + targets
    header = (
        f"{'Q':>4} | {'N':>4} | {'Unc%':>6} | {'D-score':>7} | "
        f"{'margin':>7} | {'center':>7} | {'noise_rej%':>10} | {'dbl_mk%':>8} | {'grid%':>6}"
    )
    sep = "-" * len(header)
    print()
    print("=== Per-Question Analysis ===")
    print(header)
    print(sep)
    for q in all_qs:
        s = stats.get(q, {})
        if not s:
            print(f"{q:>4} | (no data)")
            continue
        tag = "  ← baseline" if q in baseline else ""
        print(
            f"{q:>4} | {s['n']:>4} | {s['uncertain_pct']:>5.1f}% | "
            f"{s.get('median_d_score', 0) or 0:>7.4f} | "
            f"{s.get('median_margin', 0) or 0:>7.4f} | "
            f"{s.get('median_center_density', 0) or 0:>7.4f} | "
            f"{s['noise_rejected_pct']:>9.1f}% | "
            f"{s['double_mark_pct']:>7.1f}% | "
            f"{s['grid_used_pct']:>5.1f}%"
            f"{tag}"
        )
    print(sep)
    print()
    print("Ghi chú cột:")
    print("  D-score = median density score của D-column trên TẤT CẢ ảnh (kể cả khi selected != D)")
    print("  margin  = best_score - second_score (< 0.05 → nguy hiểm)")
    print("  center  = center density của bubble được chọn")
    print("  noise_rej = pipeline detect được nhưng noise gate reject")
    print("  dbl_mk  = phát hiện nhiều bubble cùng lúc")
    print("  grid%   = % ảnh có dùng local grid search adaptive")


def print_uncertain_detail(rows: list[dict], q_label: str):
    uncertain_rows = [r for r in rows if r.get("uncertain")]
    print()
    print(f"=== Chi tiết từng ảnh có {q_label} uncertain ({len(uncertain_rows)}/{len(rows)}) ===")
    print(
        f"{'Image':>45} | {'Profile':>12} | {'best':>6} | {'margin':>7} | "
        f"{'center':>7} | {'D-sc':>6} | noise | dbl | reason"
    )
    print("-" * 120)
    for r in sorted(uncertain_rows, key=lambda x: x.get("_image", "")):
        d_sc = r["scores"][3] if r.get("scores") and len(r["scores"]) > 3 else -1
        noise = "Y" if r.get("noise_rejected") else "."
        dbl = "Y" if r.get("double_mark") else "."
        reason = []
        if r.get("noise_rejected"):
            reason.append("noise_gate")
        if r.get("double_mark"):
            reason.append("double_mark")
        if not reason:
            reason.append("low_score/margin")
        print(
            f"{str(r.get('_image', ''))[:45]:>45} | {r.get('_form_profile', ''):>12} | "
            f"{r.get('best_score') or 0:>6.4f} | {r.get('margin') or 0:>7.4f} | "
            f"{r.get('best_center_density') or 0:>7.4f} | {d_sc:>6.4f} | "
            f"{noise:>5} | {dbl:>3} | {', '.join(reason)}"
        )


def diagnose(stats: dict[str, dict], targets: list[str], baseline: list[str]):
    print()
    print("=== Diagnosis ===")

    baseline_margin = statistics.median([
        stats[q]["median_margin"] for q in baseline
        if q in stats and stats[q].get("median_margin") is not None
    ] or [0])
    baseline_d = statistics.median([
        stats[q]["median_d_score"] for q in baseline
        if q in stats and stats[q].get("median_d_score") is not None
    ] or [0])

    for q in targets:
        s = stats.get(q)
        if not s or s.get("n", 0) == 0:
            continue
        issues = []
        if s.get("uncertain_pct", 0) < 5:
            print(f"{q}: Uncertain rate thấp ({s['uncertain_pct']}%) — không cần điều tra")
            continue

        d_sc = s.get("median_d_score") or 0
        margin = s.get("median_margin") or 0
        noise_rej = s.get("noise_rejected_pct", 0)
        dbl_mk = s.get("double_mark_pct", 0)
        grid_pct = s.get("grid_used_pct", 0)

        if d_sc < baseline_d * 0.75:
            issues.append(f"D-column score thấp ({d_sc:.3f} vs baseline {baseline_d:.3f}) → column drift hoặc band_right quá aggressive")
        if margin < baseline_margin * 0.5:
            issues.append(f"Margin thấp ({margin:.3f} vs baseline {baseline_margin:.3f}) → bubble yếu hoặc nhiều bóng đổ")
        if noise_rej > 5:
            issues.append(f"noise_rejected cao ({noise_rej:.1f}%) → noise gate quá chặt hoặc ink density thấp")
        if dbl_mk > 10:
            issues.append(f"double_mark cao ({dbl_mk:.1f}%) → block boundary blur hoặc shadow từ row trước")
        if grid_pct < 20 and s.get("uncertain_pct", 0) > 10:
            issues.append(f"grid_search_used={grid_pct:.0f}% → block này không có adaptive positioning")

        if not issues:
            issues.append("Uncertain nhẹ, chưa xác định rõ nguyên nhân — cần xem chi tiết từng ảnh")

        print(f"\n{q} (unc={s['uncertain_pct']}%):")
        for iss in issues:
            print(f"  • {iss}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--targets", default="Q7,Q9,Q11", help="Câu cần phân tích (VD: Q7,Q9,Q11)")
    parser.add_argument("--baseline", default="Q1,Q3,Q5", help="Câu baseline ổn định để so sánh")
    parser.add_argument("--question", default=None, help="In chi tiết từng ảnh cho câu này (VD: Q7)")
    parser.add_argument("--profile", default=None, help="Lọc theo form profile (VD: 20-cau-pdf)")
    parser.add_argument("--output", default=None, help="Lưu kết quả JSON (mặc định: không lưu)")
    args = parser.parse_args()

    targets = [f"Q{parse_q(q)}" for q in args.targets.split(",")]
    baseline = [f"Q{parse_q(q)}" for q in args.baseline.split(",")]

    print(f"Đọc golden data từ {DATASET_DIR}...")
    golden_entries = load_all_golden()
    total = len(golden_entries)
    valid = sum(1 for g in golden_entries if not g.get("needs_annotation") and g.get("source_bubble_confidence"))

    if args.profile:
        valid_profile = sum(1 for g in golden_entries if g.get("form_profile") == args.profile and not g.get("needs_annotation"))
        print(f"  {total} golden files, {valid_profile} có BC JSON và profile={args.profile}")
    else:
        print(f"  {total} golden files, {valid} có bubble_confidence JSON")

    print(f"  Profile filter: {args.profile or 'tất cả'}")
    print(f"  Targets: {', '.join(targets)}")
    print(f"  Baseline: {', '.join(baseline)}")

    print("\nĐọc bubble_confidence JSON...")
    all_qs_needed = list(set(targets + baseline))
    question_rows = collect_rows(golden_entries, args.profile)

    # Thống kê cơ bản
    unique_images = set()
    for rows in question_rows.values():
        for r in rows:
            unique_images.add(r.get("_image"))
    print(f"  {len(unique_images)} ảnh có dữ liệu")

    # Phân tích
    stats: dict[str, dict] = {}
    for q in all_qs_needed:
        rows = question_rows.get(q, [])
        stats[q] = analyze_question(rows)

    # In bảng
    print_comparison_table(stats, targets, baseline)

    # Diagnosis
    diagnose(stats, targets, baseline)

    # Chi tiết nếu --question
    if args.question:
        q_label = f"Q{parse_q(args.question)}"
        rows = question_rows.get(q_label, [])
        if rows:
            print_uncertain_detail(rows, q_label)
        else:
            print(f"\nKhông có dữ liệu cho {q_label}")

    # Label distribution
    print()
    print("=== Label distribution cho target questions ===")
    for q in targets:
        s = stats.get(q, {})
        dist = s.get("label_distribution", {})
        if not dist:
            continue
        total_q = sum(dist.values())
        dist_str = "  ".join(f"{k}:{v}({100*v//total_q}%)" for k, v in sorted(dist.items()))
        print(f"  {q}: {dist_str}")

    # Save JSON
    if args.output:
        out = {
            "config": {"targets": targets, "baseline": baseline, "profile_filter": args.profile},
            "stats": stats,
        }
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as fp:
            json.dump(out, fp, ensure_ascii=False, indent=2)
        print(f"\nSaved: {args.output}")
    else:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        from datetime import datetime
        out_path = RESULTS_DIR / f"analysis_uncertain_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        out = {
            "config": {"targets": targets, "baseline": baseline, "profile_filter": args.profile},
            "stats": stats,
        }
        with open(out_path, "w", encoding="utf-8") as fp:
            json.dump(out, fp, ensure_ascii=False, indent=2)
        print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
