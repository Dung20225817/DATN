from fastapi import APIRouter

from .assignments import router as assignments_router
from .grading import router as grading_router
from .profiles import router as profiles_router
from .templates import router as templates_router

router = APIRouter()
router.include_router(templates_router)
router.include_router(profiles_router)
router.include_router(assignments_router)
router.include_router(grading_router)
