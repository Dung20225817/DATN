from __future__ import annotations

from typing import Dict


MAP_LABEL: Dict[int, str] = {
    0: "A",
    1: "B",
    2: "C",
    3: "D",
}


def choice_label(index: int) -> str:
    idx = int(index)
    if idx < 0:
        return "-"
    if idx in MAP_LABEL:
        return MAP_LABEL[idx]
    if idx < 26:
        return chr(ord("A") + idx)
    return f"C{idx + 1}"