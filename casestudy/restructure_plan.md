# 🏗️ SpeechScan Restructure Plan: Path to Production

> **Goal:** Transform the experimental SpeechScan prototype into a secure, scalable, and production-ready medical-grade application.

---

## 🛑 Phase 1: Critical Fixes (Immediate Action)

### 1.1 Docker & Build Stabilization
- [x] **Fix Dockerfile:** Remove reference to non-existent `api/main_docker.py`. Use `api/main.py`.
- [x] **Optimize Build:** Implement multi-stage build to separate build dependencies (compilers) from runtime (slim python).
- [x] **Pin Versions:** specific versions in `requirements.txt` with hashes for reproducibility.

### 1.2 Safety & Medical Integrity
- [x] **Remove Mock Fallback:** **CRITICAL.** Remove logic in `api/main.py` that returns random results on model failure. Return `503 Service Unavailable` instead.
- [x] **Add Disclaimers:** Ensure API responses include the medical disclaimer field.

### 1.3 Security Hardening (Level 1)
- [x] **Input Validation:** Restrict file uploads by magic bytes (not just extension) to prevent malware uploads. Limit size strictly (e.g., 10MB).
- [x] **Sanitize Filenames:** Stop using original filenames in disk operations. Use UUIDs.
- [x] **CORS Policy:** Restrict `allow_origins=["*"]` to specific frontend domains.

---

## 🔄 Phase 2: Architectural Restructuring

### 2.1 Asynchronous Inference Engine
The current synchronous blocking model is unscalable for deep learning.
- [x] **Introduce Redis:** Use as a message broker.
- [x] **Celery/Dramatiq Worker:** Move `DysarthriaGenderPredictor` into a dedicated worker process.
- [x] **Job Queue Flow:**
  - `POST /predict` -> Uploads file -> Pushes job to Redis -> Returns `job_id`.
  - `GET /jobs/{job_id}` -> Polling endpoint for status.
  - *Refactor frontend to support polling.*

### 2.2 Storage & Persistence
- [x] **Object Storage:** Move from local `temp_audio_` files to S3-compatible storage (MinIO for local dev, AWS S3 for prod).
- [x] **Database Integration:**
  - Connect Backend to **Firestore** (since Frontend uses it) or **PostgreSQL**.
  - Store: `job_id`, `timestamp`, `prediction_result`, `user_hash`.

### 2.3 API Modernization
- [x] **Structure:** Move to `app/v1/routers/...`.
- [x] **Schemas:** Strict Pydantic models for all Request/Response objects.
- [x] **Error Handling:** Centralized exception handler returning structured JSON errors.

---

## 🛠️ Phase 3: Code Quality & DevOps

### 3.1 Refactoring
- [x] **Dependency Injection:** Remove global `predictor` variable. Pass model references via FastAPI `lifespan` or dependency override.
- [x] **Configuration:** Use `pydantic-settings` to load config from `.env` files (Model paths, timeout settings, thresholds).

### 3.2 Testing
- [x] **Unit Tests:** Test valid/invalid inputs without loading the heavy model (mock the predictor).
- [x] **Integration Tests:** Test the full flow (API -> Worker -> Result).
- [ ] **Regression:** Ensure accuracy doesn't drop with updates.

### 3.3 Observability
- [x] **Structured Logging:** JSON logs for production parsing.
- [ ] **Metrics:** Expose Prometheus metrics (`/metrics`) for:
  - Inference latency.
  - Error rates.
  - Queue depth.

---

## 📋 Execution Checklist

### Week 1: Stability
- [x] Fix Dockerfile
- [x] Remove Random Mock Data
- [x] Implement UUID filenames

### Week 2: Architecture
- [x] Set up Redis + Worker
- [x] Refactor API to Async pattern (Job ID)

### Week 3: Persistence & Polish
- [x] Integrate Database (Firestore/Postgres)
- [x] Add Authentication middleware
- [x] Final Security Review
