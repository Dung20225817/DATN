# OCR CRNN - Comprehensive Technical Analysis

## 📊 Executive Summary

**Project**: Vietnamese OMR Grading System (Hệ thống chấm điểm trắc nghiệm)
**Main Technologies**: FastAPI, React, PostgreSQL, OpenCV, PyTorch
**Core Modules**: 5913 lines of code (main components)
**Status**: Production-ready (v1.0+)

---

## 1. API ENDPOINTS SUMMARY

### 16 Main Endpoints

**Authentication** (1):
- POST /api/login

**Template Management** (4):
- POST /api/omr/template → Generate template from config
- POST /api/omr/template/save → Save to database
- GET /api/omr/tests/{uid} → List saved tests
- DELETE /api/omr/tests/{uid}/{omrid} → Delete test

**Form Profiles** (4):
- GET /api/omr/form-samples
- GET /api/omr/form-profiles
- GET /api/omr/form-profiles/{code}
- POST /api/omr/form-profiles

**Grading** (3):
- POST /api/omr/grade (single sheet)
- POST /api/omr/grade-batch (up to 50 sheets)
- POST /api/omr/suggest-crop (auto-detect crop quad)

**Assignments** (4):
- POST /api/omr/assignments
- GET /api/omr/assignments/{uid}
- PUT /api/omr/assignments/{uid}/{aid}
- DELETE /api/omr/assignments/{uid}/{aid}

---

## 2. OMR PROCESSING PIPELINE (12 STAGES)

### Overview

Input Image → [12 Processing Stages] → Grading Result

**Stages**:
1. Preprocessing (load, resize, binarize)
2. Page Quad Detection (find 4 corners)
3. Perspective Warp (correct angle)
4. Layout Detection (find SID/MCQ regions)
5. Marker Detection (calibration markers)
6. SID Decoding (student ID recognition)
7. MCQ Decoding (bubble mark detection)
8. Uncertainty Detection (AI classification)
9. Handwriting Extraction (OCR of fields)
10. Agentic Rescue (alternative methods if needed)
11. Scoring (compare with answer key)
12. Visualization (draw result on image)

---

## 3. DATABASE SCHEMA

### 2 Main Tables

**omr_test**: Saved OMR templates
- omrid, uuid, omr_name, omr_code (3 digits), omr_quest, omr_answer (JSON)

**omr_assignment**: Mobile assignments
- aid, uuid, title, created_at_raw, question_count, answer_sets (JSON), last_result (JSON)

**Profile Storage**: JSON files in uploads/omr_data/profiles/

---

## 4. KEY CONFIGURATION PARAMETERS

### MCQ Decode (13 params)
- threshold_mode: otsu|adaptive|weighted_adaptive|hybrid
- bubble_left_ratio, bubble_right_ratio: Define bubble horizontal bounds
- inner_ratio: Inner bubble region for fill calculation
- min_mark_density: Threshold to mark bubble as filled
- adaptive_block_size, adaptive_c: Adaptive thresholding params

### AI Uncertainty (5 params)
- enabled: bool
- model_path: Path to bubble classifier model
- marked_conf_threshold, empty_conf_threshold: Confidence thresholds

### AI SID HTR (4 params)
- enabled: bool
- model_path: Path to digit recognition model
- min_confidence: Confidence threshold

### Agentic Rescue (2 params)
- enabled: bool
- sid_conf_threshold: Trigger threshold

---

## 5. AI/ML COMPONENTS

### 1. Student ID Handwriting Recognition (ai/htr_sid.py)
- Model: Tiny CNN (28×28 input)
- Output: Digit 0-9 + confidence
- Used when sid_has_write_row=true

### 2. Bubble Uncertainty Classifier (ai/uncertainty_classifier.py)
- Model: SimpleBubbleCNN (32×32 input)
- Classes: marked | empty | erased
- Optional, improves detection accuracy

### 3. Thresholding Methods (ai/thresholding.py)
- Otsu: Automatic (fast)
- Adaptive: Local threshold (robust)
- Hybrid: Multi-pass (accurate)

### 4. Agentic Rescue (ai/agent_workflow.py)
- If uncertain_count > 0 or sid_confidence < threshold
- Try 3 alternative processing branches
- Score & pick best result

---

## 6. FRONTEND STATE MANAGEMENT

### React Hooks in MultichoicePage.tsx (2300 lines)

**Key State**:
- assignments: TestCardItem[]
- selectedAssignment: TestCardItem | null
- formProfiles: FormProfile[]
- selectedProfile: FormProfile | null
- gradeRecords: GradeRecord[]
- currentGradeResult: OMRResult | null
- scannerState: "idle" | "searching" | "locked"
- navTab: "home" | "templates"
- detailTab: "grading" | "answers" | "stats" | "export"

---

## 7. TECHNOLOGY VERSIONS

**Backend**:
- FastAPI 0.121.3
- SQLAlchemy 2.0.44
- PyTorch 2.9.1
- OpenCV 4.12.0.88
- NumPy 2.2.6

**Frontend**:
- React 19.2.0
- TypeScript 5.9.3
- Vite 7.2.2
- jsPDF 4.2.1
- XLSX 0.18.5

---

## 8. FILE STRUCTURE

**Backend (~5900 lines)**:
- be/main.py (94) - Entry point
- be/app/api/omr_grading.py (2141) - API endpoints
- be/app/services/omr/omr_service.py (1472) - Orchestration
- be/app/services/omr/*.py - Processing modules
- be/app/db/*.py - Database models

**Frontend (~2300 lines)**:
- fe/src/App.tsx (22) - Router
- fe/src/config/api.ts (102) - API wrapper
- fe/src/Page_Components/MultichoicePage.tsx (2300) - Main UI

---

## 9. DATA FLOW EXAMPLE

User → Upload Image → resolve answer key → process_omr_exam() →
[12 processing stages] → Result JSON → Display UI → Save to DB

---

## 10. PERFORMANCE

**Single Sheet**: ~3.4 seconds (core processing)
**Batch (50 sheets)**: ~3 minutes
**Supports**: GPU acceleration via CUDA

---

## 11. SECURITY VALIDATION

**Input Validation**: All parameters sanitized
**File Upload**: Basename extraction, extension checking
**Database**: ORM prevents SQL injection
**User Isolation**: All queries filtered by uuid

---

## 12. EXTENSIBILITY

**Plugin Points**:
- Custom OCR engines
- Custom AI models
- Custom export formats
- Custom scoring logic

**Future**: Multi-page OMR, WebSocket updates, distributed processing

---

**Analysis Date**: April 24, 2026
**Version**: 1.0.0
