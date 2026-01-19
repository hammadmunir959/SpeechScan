# System & Architecture Analysis: SpeechScan

## 1. System Design Analysis

### 🎯 Problem Clarification
- **Core Problem:** The system aims to detect dysarthria (speech disorders) from audio samples.
- **Status:** well-defined.
- **Issues:**
  - **Medical Disclaimer:** While present in README, the "Mock" fallback mode in `api/main.py` (returning random results when models fail) is extremely dangerous for a health-related application. It degrades trust and could be misleading.

### 2. Requirements & Scope
- **Functional:** 
  - API and Web Interface are functional.
  - Live recording is implemented.
- **Non-Functional:**
  - **Latency:** Model inference time (Wav2Vec2) is significant. `api/main.py` attempts to handle this with async loading, but the `predict` endpoint creates a temporary file on disk for every request (`temp_audio_...`), adding I/O latency.
  - **Availability:** The system relies on a "loading" state. If several workers are spawned (e.g., via Gunicorn), each will independently load the heavy model into RAM, potentially crashing small instances.

### 5. API Design
- **Endpoints:** Simple `/predict` endpoint.
- **Issues:**
  - **No Versioning:** API path is `/predict`, not `/api/v1/predict`.
  - **Security:** No rate limiting or authentication on the API itself. The frontend does some Firebase auth, but the backend implementation in `api/main.py` does not verify any tokens. It's an open API.
  - **Input Validation:** Relies on file content processing. Large files (>50MB) are checked, but writing them to disk *before* processing allows a disk-filling attack.

### 7. Database & Data
- **Status:** No backend database integration found in `api/main.py`. simple ephemeral processing.
- **Issues:**
  - **Persistence:** Analysis results are returned but not saved. There is no historical record of predictions for the user or the system (observability).
  - **Frontend Mismatch:** Frontend initializes Firebase Firestore (`window.db`), but the backend does not interact with it. Data flow is disjointed.

## 2. Architecture Design Analysis

### 2. Architecture Style
- **Type:** Single-Service Monolith.
- **Verdict:** Appropriate for this scale, but the heavy ML model integration makes it "heavyweight".

### 4. Inter-Service Communication
- **Async vs Sync:** The `/predict` endpoint is synchronous (waiting for inference).
- **Issue:** For long audio files, this might timeout. A better architecture would be Async (Submit Job -> Return ID -> Poll Status/Webhook), especially for deep learning inference.

### 6. Infrastructure & Deployment
- **Docker:** **CRITICAL FAILURE**.
  - The `Dockerfile` attempts to `COPY api/main_docker.py api/main.py`.
  - **Fact:** `api/main_docker.py` does not exist in the codebase. This build will fail.
  - **Security:** `Dockerfile` runs as root initially but correctly switches to `appuser`.
- **Compute:** CPU-based inference (implied by `tensorflow-cpu` and `torch...cpu` in `requirements.txt`). This will be slow for concurrent users.

### 8. Reliability & Resilience
- **Mock Fallback:**
  - `api/main.py` contains logic to return *randomized mock data* if the model fails to load.
  - **Severity: HIGH**. In a medical context, failing silently and returning fake "Normal/Abnormal" results is unacceptable. It should return a 500 Error.

### 9. Code Quality & Maintainability
- **Hardcoded Paths:** `sys.path.append` and relative paths (`../ProcessedData/models`) make the code brittle and hard to test in isolation.
- **Global State:** The `global predictor` variable in `api/main.py` is a classic singleton pattern that makes unit testing difficult.
- **Configuration:** Secrets (if any) and configuration (model paths) are partly hardcoded.

## Summary of Critical Issues (The "Fix Immediately" List)

1.  **Broken Docker Build:** `main_docker.py` is missing.
2.  **Dangerous Mock Fallback:** Remove the specific logic that returns random medical diagnoses when the model is down.
3.  **Missing Security:** Add API rate limiting and basic token authentication (even if just validating the Firebase token from client).
4.  **Sync Inference:** For a heavy ML model, move to an asynchronous task queue (e.g., Celery + Redis) architecture to avoid holding HTTP connections open.
5.  **Disk I/O:** Avoid writing temp files to disk (`temp_audio_...`) if possible; process in memory stream or use a proper temp volume cleanup strategy to prevent disk exhaustion.
