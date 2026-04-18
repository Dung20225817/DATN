from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


@dataclass
class RescueAttempt:
    branch: str
    payload: Dict[str, Any]
    warp_layout_score: float
    uncertain_count: int
    sid_min_confidence: float


@dataclass
class AgentGradingState:
    initial: Dict[str, Any]
    attempts: List[RescueAttempt] = field(default_factory=list)
    selected: Optional[RescueAttempt] = None


def should_retry(result: Dict[str, Any], sid_conf_threshold: float = 1.05) -> bool:
    uncertain = int(result.get("uncertain_count", 0))
    sid_conf = result.get("student_id_confidence") or []
    sid_min = min(sid_conf) if sid_conf else 0.0
    return (uncertain > 0) or (sid_min < float(sid_conf_threshold))


def score_attempt(payload: Dict[str, Any]) -> float:
    uncertain = int(payload.get("uncertain_count", 9999))
    sid_conf = payload.get("student_id_confidence") or []
    sid_min = min(sid_conf) if sid_conf else 0.0
    warp_score = float(payload.get("warp_layout_score", 0.0))

    # Lower penalty is better; higher warp score is better.
    penalty = uncertain * 2.2 + max(0.0, 1.05 - float(sid_min)) * 3.6
    return float(warp_score) - float(penalty)


def pick_best_attempt(candidates: List[RescueAttempt]) -> Optional[RescueAttempt]:
    if not candidates:
        return None
    ranked = sorted(candidates, key=lambda a: score_attempt(a.payload), reverse=True)
    return ranked[0]


def run_agentic_rescue(
    initial_result: Dict[str, Any],
    branch_runners: Dict[str, Callable[[], Dict[str, Any]]],
    sid_conf_threshold: float = 1.05,
) -> Dict[str, Any]:
    """Simple agentic orchestrator mirroring a LangGraph-like decision flow.

    branch_runners expects callables for e.g.:
    - horizontal_shift_rescue
    - gray_darkness_pass
    - marker_original_adaptive
    """
    state = AgentGradingState(initial=initial_result)

    if not should_retry(initial_result, sid_conf_threshold=sid_conf_threshold):
        return initial_result

    for branch_name in ["marker_original_adaptive", "gray_darkness_pass", "horizontal_shift_rescue"]:
        fn = branch_runners.get(branch_name)
        if fn is None:
            continue
        try:
            payload = fn()
        except Exception:
            continue
        if not isinstance(payload, dict) or bool(payload.get("error")):
            continue

        attempt = RescueAttempt(
            branch=branch_name,
            payload=payload,
            warp_layout_score=float(payload.get("warp_layout_score", 0.0)),
            uncertain_count=int(payload.get("uncertain_count", 9999)),
            sid_min_confidence=min(payload.get("student_id_confidence") or [0.0]),
        )
        state.attempts.append(attempt)

    best = pick_best_attempt(state.attempts)
    if best is None:
        return initial_result
    state.selected = best
    return best.payload


def build_agentic_stategraph_spec() -> Dict[str, Any]:
    """StateGraph-like spec for documentation/implementation alignment."""
    return {
        "nodes": [
            "grade_first_pass",
            "check_uncertainty",
            "run_marker_original_adaptive",
            "run_gray_darkness_pass",
            "run_horizontal_shift_rescue",
            "compare_attempts",
            "finalize_result",
        ],
        "edges": [
            ("grade_first_pass", "check_uncertainty"),
            ("check_uncertainty", "finalize_result", "if no_retry"),
            ("check_uncertainty", "run_marker_original_adaptive", "if retry"),
            ("run_marker_original_adaptive", "run_gray_darkness_pass"),
            ("run_gray_darkness_pass", "run_horizontal_shift_rescue"),
            ("run_horizontal_shift_rescue", "compare_attempts"),
            ("compare_attempts", "finalize_result"),
        ],
        "state": {
            "initial_result": "dict",
            "attempt_results": "list[dict]",
            "selected_result": "dict",
            "decision_metrics": {
                "uncertain_count": "int",
                "student_id_min_confidence": "float",
                "warp_layout_score": "float",
            },
        },
    }
