# main.py - FastAPI Application Entry Point

import json
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
import logging

from app.core.paths import UPLOADS_DIR, ensure_runtime_dirs
from app.core.logging import setup_logging, get_logger
from app.api import auth
from app.api.omr import router as omr_router
from app.db.session import engine, Base
from app.db.models import user, omr  # noqa: F401 - import to register models

# Setup logging
setup_logging(level=logging.INFO)
logger = get_logger(__name__)

logger.info("Starting OCR Grading API...")


class UnicodeJSONResponse(JSONResponse):
    """JSONResponse that keeps Vietnamese (and all Unicode) characters readable."""
    def render(self, content) -> bytes:
        return json.dumps(content, ensure_ascii=False, allow_nan=False).encode("utf-8")


# Create database tables if they don't exist
Base.metadata.create_all(bind=engine)
logger.info("Database tables initialized")

# Initialize FastAPI app
app = FastAPI(
    title="OMR Grading API",
    description="Vietnamese OMR grading system",
    version="1.0",
    default_response_class=UnicodeJSONResponse
)

# Create runtime directories
ensure_runtime_dirs()

logger.info("Upload directories initialized")

# Mount static files for image serving
app.mount("/static", StaticFiles(directory=str(UPLOADS_DIR)), name="static")

# Configure CORS middleware - Allow localhost on all ports for development
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"^https?://(localhost|127\.0\.0\.1|192\.168\.\d+\.\d+):\d+$",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register API routers
logger.info("Registering API routers...")
app.include_router(auth.router, prefix="/api", tags=["auth"])
app.include_router(omr_router, prefix="/api/omr", tags=["omr"])

logger.info("All routers registered successfully")

# Health check endpoint
@app.get("/")
def root():
    return {
        "message": "OMR Grading API is running",
        "version": "1.0",
        "endpoints": {
            "auth": "/api/login",
            "omr": "/api/omr",
        }
    }

@app.get("/health")
def health_check():
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )
