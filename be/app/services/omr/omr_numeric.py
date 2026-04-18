from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np

from .omr_mcq import _cell_score, _extract_cell


def _safe_int(raw, default=0) -> int:
    try:
        return int(raw)
    except Exception:
        return int(default)


def _parse_sid_row_offsets(profile_sid_row_offsets, digits: int) -> List[int]:
    if isinstance(profile_sid_row_offsets, dict):
        raw = profile_sid_row_offsets.get("offsets")
    else:
        raw = profile_sid_row_offsets

    out = [0 for _ in range(max(1, int(digits)))]
    if not isinstance(raw, (list, tuple)):
        return out

    for idx in range(min(len(raw), len(out))):
        out[idx] = _safe_int(raw[idx], 0)
    return out


def _decode_numeric_columns(
    gray_img,
    binary_inv,
    roi,
    digits: int,
    has_write_row: bool = False,
    row_offsets: Optional[Sequence[int]] = None,
):
    x = int(roi["x"])
    y = int(roi["y"])
    w = int(roi["w"])
    h = int(roi["h"])

    total_rows = 11 if has_write_row else 10
    row_edges = np.linspace(y, y + h, total_rows + 1, dtype=np.float32)
    col_edges = np.linspace(x, x + w, int(digits) + 1, dtype=np.float32)

    score_matrix = np.zeros((total_rows, int(digits)), dtype=np.float32)

    offsets = [0 for _ in range(int(digits))]
    if isinstance(row_offsets, (list, tuple)):
        for i in range(min(len(offsets), len(row_offsets))):
            offsets[i] = _safe_int(row_offsets[i], 0)

    value_chars: List[str] = []
    confs: List[float] = []

    for c in range(int(digits)):
        col_scores = []
        for r in range(total_rows):
            src_r = min(max(r + int(offsets[c]), 0), total_rows - 1)
            cell_gray, cell_bin = _extract_cell(
                gray_img,
                binary_inv,
                col_edges[c],
                row_edges[src_r],
                col_edges[c + 1],
                row_edges[src_r + 1],
                inner_ratio=0.74,
            )
            score = _cell_score(cell_gray, cell_bin)
            col_scores.append(score)
            score_matrix[r, c] = float(score)

        valid_start = 1 if has_write_row else 0
        valid_scores = np.asarray(col_scores[valid_start : valid_start + 10], dtype=np.float32)
        if valid_scores.size <= 0:
            value_chars.append("?")
            confs.append(0.0)
            continue

        best_rel = int(np.argmax(valid_scores))
        best = float(valid_scores[best_rel])

        if valid_scores.size >= 2:
            two_best = np.partition(valid_scores, -2)[-2:]
            second = float(np.min(two_best))
        else:
            second = 0.0

        conf = min(9.99, best / max(1e-4, second))
        confs.append(float(conf))

        if best < 0.07:
            value_chars.append("?")
        else:
            value_chars.append(str(best_rel))

    value = "".join(value_chars)
    mean_conf = float(np.mean(confs)) if confs else 0.0
    status = "ok" if ("?" not in value and mean_conf >= 1.03) else "uncertain"

    return {
        "value": value,
        "status": status,
        "confidence": round(mean_conf, 4),
        "scores": score_matrix.tolist(),
    }
