from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pypdfium2 as pdfium


ROOT_DIR = Path(__file__).resolve().parents[1]
OMR_DATA_DIR = ROOT_DIR / "be" / "uploads" / "omr_data"
PROFILE_DIR = OMR_DATA_DIR / "profiles"
DEBUG_DIR = PROFILE_DIR / "marker_debug"


def _safe_profile_code(raw: str) -> str:
    text = "".join(ch.lower() if ch.isalnum() or ch in "-_" else "-" for ch in str(raw or "").strip())
    while "--" in text:
        text = text.replace("--", "-")
    return text.strip("-")[:80]


def _load_pdf_page_rgb(pdf_path: Path, scale: float = 2.2) -> Tuple[np.ndarray, float, float]:
    doc = pdfium.PdfDocument(str(pdf_path))
    if len(doc) <= 0:
        raise ValueError(f"PDF has no page: {pdf_path.name}")

    page = doc[0]
    page_w_pt, page_h_pt = page.get_size()
    bitmap = page.render(scale=scale, rotation=0)
    image = bitmap.to_numpy()

    if image.ndim != 3:
        raise ValueError(f"Unexpected PDF raster shape: {image.shape}")

    if image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    elif image.shape[2] != 3:
        raise ValueError(f"Unsupported channel count: {image.shape[2]}")

    return image, float(page_w_pt), float(page_h_pt)


def _find_corner_markers(image_rgb: np.ndarray) -> Tuple[Dict[str, Dict[str, float]], List[Dict[str, float]]]:
    h, w = image_rgb.shape[:2]
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    _, binary_inv = cv2.threshold(blur, 95, 255, cv2.THRESH_BINARY_INV)
    binary_inv = cv2.morphologyEx(binary_inv, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)

    contours, _ = cv2.findContours(binary_inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    min_area = (w * h) * 0.00002
    max_area = (w * h) * 0.0100
    candidates: List[Dict[str, float]] = []

    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < min_area or area > max_area:
            continue

        x, y, bw, bh = cv2.boundingRect(contour)
        rect_area = float(max(1, bw * bh))
        aspect = bw / float(max(1, bh))
        fill_ratio = area / rect_area
        if aspect < 0.55 or aspect > 1.45:
            continue
        if fill_ratio < 0.55:
            continue

        cx = x + (bw / 2.0)
        cy = y + (bh / 2.0)
        candidates.append(
            {
                "x": float(x),
                "y": float(y),
                "w": float(bw),
                "h": float(bh),
                "cx": float(cx),
                "cy": float(cy),
                "area": area,
                "fill_ratio": fill_ratio,
            }
        )

    if len(candidates) < 4:
        raise ValueError("Could not detect enough black-square candidates from PDF")

    corners = {
        "tl": (0.0, 0.0),
        "tr": (float(w), 0.0),
        "bl": (0.0, float(h)),
        "br": (float(w), float(h)),
    }

    diag = math.hypot(w, h)
    used: set[int] = set()
    selected: Dict[str, Dict[str, float]] = {}

    for key, target in corners.items():
        scored = []
        for idx, item in enumerate(candidates):
            if idx in used:
                continue
            dist = math.hypot(item["cx"] - target[0], item["cy"] - target[1])
            score = dist - (item["area"] * 0.003)
            scored.append((score, idx, dist))

        if not scored:
            raise ValueError(f"No marker candidate for corner {key}")

        scored.sort(key=lambda x: x[0])
        _, best_idx, dist = scored[0]
        if dist / max(1.0, diag) > 0.38:
            raise ValueError(f"Best marker for {key} is too far from page corner")

        used.add(best_idx)
        selected[key] = candidates[best_idx]

    marker_boxes = {}
    for key in ["tl", "tr", "bl", "br"]:
        marker = selected[key]
        marker_boxes[key] = {
            "x": round(marker["x"] / w, 6),
            "y": round(marker["y"] / h, 6),
            "w": round(marker["w"] / w, 6),
            "h": round(marker["h"] / h, 6),
            "cx": round(marker["cx"] / w, 6),
            "cy": round(marker["cy"] / h, 6),
        }

    return marker_boxes, candidates


def _draw_debug(image_rgb: np.ndarray, marker_boxes: Dict[str, Dict[str, float]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    h, w = image_rgb.shape[:2]
    debug = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    for key, color in [("tl", (0, 255, 0)), ("tr", (255, 180, 0)), ("bl", (0, 180, 255)), ("br", (255, 0, 255))]:
        box = marker_boxes[key]
        x = int(round(box["x"] * w))
        y = int(round(box["y"] * h))
        bw = int(round(box["w"] * w))
        bh = int(round(box["h"] * h))
        cv2.rectangle(debug, (x, y), (x + bw, y + bh), color, 3)
        cv2.putText(debug, key, (x, max(18, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)

    cv2.imwrite(str(out_path), debug)


def _update_profile(pdf_path: Path) -> Dict[str, object]:
    code = _safe_profile_code(pdf_path.stem)
    profile_path = PROFILE_DIR / f"{code}.json"
    if not profile_path.exists():
        raise FileNotFoundError(f"Profile not found for PDF: {pdf_path.name}")

    image_rgb, page_w_pt, page_h_pt = _load_pdf_page_rgb(pdf_path)
    marker_boxes, _ = _find_corner_markers(image_rgb)
    _draw_debug(image_rgb, marker_boxes, DEBUG_DIR / f"{code}_markers.png")

    with open(profile_path, "r", encoding="utf-8") as f:
        profile = json.load(f)

    strategy = profile.get("strategy") if isinstance(profile.get("strategy"), dict) else {}
    marker_sizes = [min(v["w"], v["h"]) for v in marker_boxes.values()]

    strategy["sheet_aspect_ratio"] = round(page_h_pt / max(1.0, page_w_pt), 6)
    strategy["page_size_pt"] = {
        "width": round(page_w_pt, 3),
        "height": round(page_h_pt, 3),
    }
    strategy["corner_markers"] = marker_boxes
    strategy["scanner_hint"] = {
        "min_dark_ratio": 0.14,
        "min_center_luma": 52,
        "min_marker_contrast": 20,
        "sample_size_norm": round(float(np.median(marker_sizes) * 1.8), 6),
    }

    profile["strategy"] = strategy

    with open(profile_path, "w", encoding="utf-8") as f:
        json.dump(profile, f, ensure_ascii=False, indent=2)

    return {
        "code": code,
        "pdf": pdf_path.name,
        "profile": profile_path.name,
        "page_size_pt": strategy["page_size_pt"],
        "sheet_aspect_ratio": strategy["sheet_aspect_ratio"],
        "corner_markers": strategy["corner_markers"],
    }


def main() -> None:
    pdf_files = sorted(OMR_DATA_DIR.glob("*.pdf"))
    if not pdf_files:
        raise SystemExit("No PDF samples found under be/uploads/omr_data")

    results = []
    errors = []
    for pdf_path in pdf_files:
        try:
            results.append(_update_profile(pdf_path))
        except Exception as ex:  # noqa: BLE001
            errors.append({"pdf": pdf_path.name, "error": str(ex)})

    print(json.dumps({"updated": results, "errors": errors}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
