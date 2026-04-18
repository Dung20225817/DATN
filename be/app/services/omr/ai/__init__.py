"""Optional AI helpers for OMR pipeline upgrades.

Modules in this package are designed to be optional at runtime.
When a model or dependency is unavailable, caller code should fallback
back to heuristic CV logic.
"""

from .yolo_localizer import detect_omr_regions_yolo  # noqa: F401
from .thresholding import threshold_weighted_adaptive, threshold_hybrid_shadow_robust  # noqa: F401
from .uncertainty_classifier import BubbleCellClassifier, SimpleBubbleCNN  # noqa: F401
from .htr_sid import SidHandwritingRecognizer  # noqa: F401
from .agent_workflow import build_agentic_stategraph_spec, run_agentic_rescue  # noqa: F401
