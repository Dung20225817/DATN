# main.py - FastAPI Application Entry Point

import json
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
import os
import logging

from app.logging_config import setup_logging, get_logger
from app.api import auth
from app.api.handwritten_load_picture import router as handwritten_router
from app.api.omr_grading import router as omr_router
from app.db_connect import engine, Base
from app.db import table, ocr_tables  # noqa: F401 - import to register models

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
    title="OCR Grading API",
    description="Vietnamese handwritten essay and OMR grading system",
    version="1.0",
    default_response_class=UnicodeJSONResponse
)

# Create upload directories
os.makedirs("uploads/omr", exist_ok=True)
os.makedirs("uploads/answer_keys", exist_ok=True)
os.makedirs("uploads/handwritten", exist_ok=True)
os.makedirs("uploads/temp", exist_ok=True)

logger.info("Upload directories initialized")

# Mount static files for image serving
app.mount("/static", StaticFiles(directory="uploads"), name="static")

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
app.include_router(handwritten_router, prefix="/api/handwritten", tags=["handwritten"])
app.include_router(omr_router, prefix="/api/omr", tags=["omr"])

logger.info("All routers registered successfully")

# Health check endpoint
@app.get("/")
def root():
    return {
        "message": "OCR Grading API is running",
        "version": "1.0",
        "endpoints": {
            "auth": "/api/login",
            "handwritten": "/api/handwritten",
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
