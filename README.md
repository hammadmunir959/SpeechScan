# SpeechScan: AI-Powered Dysarthria Detection

SpeechScan is a production-grade machine learning platform for detecting dysarthria (speech disorders) and classifying gender from raw audio samples. It leverages state-of-the-art Wav2Vec2 features combined with a optimized neural network architecture, redesigned for security, scalability, and medical integrity.

## 🎯 Key Features

-   **High Accuracy:** 94.2% test accuracy and 98.7% AUC for voice health analysis.
-   **Asynchronous Processing:** Decoupled ML inference using Celery and Redis to handle concurrent requests without blocking.
-   **Production-Ready Architecture:** Multi-container orchestration via Docker Compose.
-   **Secure & Safe:** Explicit medical disclaimers, sanitized UUID-based file handling, and restricted CORS policies.
-   **Multi-Interface:** Interactive Web UI (with live recording), REST API, and CLI.
-   **Persistence:** Built-in support for Firestore result history and abstract storage for audio samples.

## 📊 Model Performance & Evaluation

The engine is trained on 10,115 audio samples from the Mozilla Common Voice and TORGO datasets.

| Metric | Score |
| :--- | :--- |
| **Test Accuracy** | 94.2% |
| **Precision** | 93.7% |
| **Recall** | 94.9% |
| **F1 Score** | 94.3% |
| **ROC-AUC** | 98.7% |

### Confusion Matrix Insights
- **Normal Class:** High specificity ensures low false alarm rates for healthy speakers.
- **Abnormal Class:** High sensitivity (94.9% recall) ensures critical speech patterns are not missed.

## 🛠️ Tech Stack

-   **Backend:** FastAPI (Python), Celery, Redis.
-   **ML Engine:** TensorFlow, PyTorch, Transformers (Wav2Vec2), Librosa.
-   **Persistence:** Google Cloud Firestore (Admin SDK).
-   **Infrastructure:** Docker, Docker Compose, Nginx.
-   **Frontend:** Vanilla JS, Tailwind CSS, Lucide Icons.

## 🧠 System Architecture & Approach

SpeechScan was transformed from an experimental prototype into a production-ready system through a systematic three-phase approach:

### Phase 1: Security & Safety Hardening
- **Mock Internal Removal:** Completely removed experimental mock fallback logic that returned randomized predictions on system failure. The API now returns standard HTTP 5xx errors to maintain medical integrity.
- **Filename Sanitization:** Switched from preserving original filenames to UUID-based naming to prevent path traversal and disk collisions.
- **CORS Restriction:** Hardened the API by restricting allowed origins to specific production domains.

### Phase 2: Decoupled Async Inference
- Moved heavy ML inference (Wav2Vec2 feature extraction) out of the FastAPI request loop.
- Implemented a **Task Queue pattern** using **Celery** as the worker and **Redis** as the message broker.
- Introduced a polling mechanism in the Frontend and API (`/results/{job_id}`) to handle long-running analysis gracefully.

### Phase 3: Observability & Persistence
- **Standardized Config:** Implemented `pydantic-settings` for centralized, environment-driven configuration.
- **Structured Logging:** Switched to JSON logging for easier integration with ELK/Cloudwatch monitoring.
- **Data Persistence:** Automated saving of analysis history to **Firestore** and established an abstract **Storage Service** for managing audio files.

## 🚀 Usage Instructions

### Docker Deployment (Recommended)

Ensure you have Docker and Docker Compose (v2+) installed.

```bash
# Clone the repository
git clone https://github.com/hammadmunir959/SpeechScan
cd SpeechScan

# Build and start the services (Redis, API, Worker)
docker compose up --build -d

# Visit the dashboard
open http://localhost:8000
```

### Local API Development

```bash
# Install dependencies (CPU-optimized versions recommended)
pip install -r api/requirements.txt

# Run Redis locally or via Docker
docker run -p 6379:6379 -d redis

# Start the Celery worker
python -m celery -A api.celery_app worker --loglevel=info

# Start the FastAPI server
python api/main.py
```

## 🎤 API Endpoints

-   `POST /predict`: Submit an audio file (`.wav`, `.mp3`, `.webm`). Returns a `job_id`.
-   `GET /results/{job_id}`: Poll for analysis status and final report.
-   `GET /health`: System health and model status check.

---

**⚠️ Medical Disclaimer:** This tool is for research and educational purposes only. It is NOT a diagnostic tool. Always consult a qualified healthcare professional for medical concerns.
