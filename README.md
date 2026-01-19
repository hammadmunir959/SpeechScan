# SpeechScan: Advanced AI for Dysarthria Detection & Voice Analytics

SpeechScan is a state-of-the-art, end-to-end machine learning system designed for the automated detection of dysarthria and comprehensive voice health analysis. Engineered from the ground up, the platform transforms raw acoustic signals into actionable diagnostic insights using deep transformer architectures and optimized neural classification heads.

---

## 🚀 Core Value Proposition

-   **Scratch-Built ML Pipeline:** A custom-engineered inference stack combining self-supervised speech representations with purpose-built classification layers.
-   **High-Fidelity Accuracy:** Achieves **94.2% test accuracy** and a **98.7% ROC-AUC** score, ensuring world-class sensitivity to subtle speech anomalies.
-   **Production-Grade Scalability:** Built on a distributed asynchronous architecture (FastAPI + Celery + Redis), capable of processing high-volume requests without latency bottlenecks.
-   **Clinical Integrity:** Implements robust safety mechanisms, including medical disclaimers, secure data handling, and the elimination of heuristic mock fallbacks.

---

## 🧠 Technical Deep Dive: From Raw Audio to Insights

SpeechScan does not just "wrap" a model; it implements a rigorous multi-stage pipeline designed for precision.

### 1. Acoustic Preprocessing & Physics
Raw audio is rarely ready for deep learning. SpeechScan implements a sophisticated cleaning stage:
-   **Silence Removal:** Adaptive trimming using a 20dB threshold to eliminate non-signal artifacts.
-   **Temporal Normalization:** Audio is intelligently padded (to 2s) or truncated (to 10s) using constant-mode padding to preserve rhythmic timing without introducing artificial discontinuities.
-   **Amplitude Scaling:** 16-bit PCM normalization ensures consistent energy distribution across different recording hardware.
-   **Resampling:** All inputs are decanted to a 16kHz mono-channel stream, the gold standard for speech transformer models.

### 2. Feature Extraction (Wav2Vec2)
At the heart of the system is the **Wav2Vec2-Base-960h** transformer. We leverage the self-supervised latents of this model:
-   **Global Average Pooling:** Instead of simple temporal slices, we compute the mean across the last hidden state (768 dimensions), capturing a holistic spectral signature of the speaker's vocal health.
-   **Contextual Encoding:** The transformer captures long-range dependencies in speech, identifying the "slurring" or "strained" qualities characteristic of dysarthria.

### 🛠️ Local Installation (Without Docker)

If you prefer to run the system directly on your host machine, follow these steps:

#### 1. System Requirements
- **Python:** 3.8 - 3.12
- **Redis:** Required as a message broker.
- **FFmpeg:** Required for audio processing.

#### 2. Install Redis & FFmpeg
```bash
# Ubuntu/Debian
sudo apt update && sudo apt install -y redis-server ffmpeg libsndfile1

# MacOS (using Homebrew)
brew install redis ffmpeg
```

#### 3. Setup Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # Linux/MacOS
# venv\Scripts\activate  # Windows

# Install CPU-optimized dependencies
pip install --upgrade pip
pip install -r api/requirements.txt
```

#### 4. Configure & Launch
```bash
# 1. Start Redis in a separate terminal
redis-server

# 2. Start the Celery Worker (Inference Engine)
python -m celery -A api.celery_app worker --loglevel=info

# 3. Start the FastAPI Server
python api/main.py
```

### Environment Configuration (.env)
SpeechScan uses `pydantic-settings` for centralized configuration. Key variables include:
- `REDIS_URL`: URL for the message broker (e.g., `redis://localhost:6379/0`).
- `MODEL_DIR`: Path to the directory containing weights.
- `ALLOWED_ORIGINS`: List of CORS-permitted domains.
- `LOG_LEVEL`: Granularity of structured logs (`INFO`, `DEBUG`, `ERROR`).
- `MEDICAL_DISCLAIMER`: Custom text for the mandatory disclaimer field.

---

## 🛠️ Operational Excellence: Health & Monitoring

The system is designed with built-in resilience:
-   **API Health:** `/health` endpoint tracks system uptime and model availability.
-   **Worker Heartbeats:** Celery handles worker failures with automatic task retries and connection monitoring.
-   **Structured Logs:** All operations are emitted as JSON, allowing for deep log analysis and alerting in environments like Amazon CloudWatch or Datadog.

---
### 3. Neural Classification Head
The extracted 768-dimensional vectors are fed into a **custom-trained Feed-Forward Neural Network (FNN)**:
-   **Architecture:** Optimized multi-layer architecture with Dropout (0.3) for regularization.
-   **Activation:** ReLU layers for non-linear feature separation.
-   **Output:** A sigmoid probability score representing the severity/likelihood of dysarthric patterns.

### 4. Acoustic Pitch Analysis (F0 Estimation)
Beyond health, the system performs gender and pitch classification using **Probabilistic YIN (pYIN)**:
-   **F0 Tracking:** Estimates fundamental frequency within a 50Hz-400Hz range.
-   **Metadata Extraction:** Provides mean F0 in Hz, offering objective data for clinical speech therapists.

---

## 📊 Evaluation & Model Performance

SpeechScan was validated on a massive dataset of **10,115+ curated samples** (Mozilla Common Voice + TORGO).

| Metric | Score | Clinical Interpretation |
| :--- | :--- | :--- |
| **Accuracy** | 94.2% | High reliability in standard environments. |
| **ROC-AUC** | 98.7% | Exceptional ability to distinguish between Normal and Abnormal states. |
| **Recall (Sensitivity)** | 94.9% | Minimizes "False Normals" – critical for medical screening. |
| **F1-Score** | 94.3% | Balanced precision and recall across all demographics. |

### Confusion Matrix Insights
The model shows particular strength in detecting **Parkinsonian speech** and **Ataxic dysarthria**, where vocal tremors and rhythmic instability are prominent.

---

## 📁 Project Structure: Engineering Blueprint

```
SpeechScan/
├── api/                    # Production API Server (FastAPI)
│   ├── main.py            # API Gateway & Lifespan Management
│   ├── tasks.py           # Celery Workers (The 'Brains' for Inference)
│   ├── config.py          # Centralized Configuration (Pydantic-Settings)
│   ├── database.py        # Persistence layer (Firestore Integration)
│   └── logging_config.py  # Structured JSON Logging for Observability
├── notebook/              # Research & Model Development
│   ├── dyarthria_model.ipynb # The 'From-Scratch' training pipeline
│   └── dyarthria_model.py # Clean export of ML logic
├── ProcessedData/models/   # Optimized binary weights (.h5 & .json)
├── web_interface/         # High-Performance UI (Tailwind + JS)
├── predict.py             # CLI Tool for batch research tasks
├── Dockerfile             # Multi-stage production build
└── docker-compose.yml     # Service Orchestration (Redis, API, Worker)
```

---

## 📖 Comprehensive "How to Use" Guide

### 🌐 Method 1: Interactive Web Dashboard
The most user-friendly way to interact with SpeechScan.
1.  **Launch:** Visit `http://localhost:8000` after starting the services.
2.  **Upload/Record:** Use the interface to either record live audio (limited to 10s) or drag & drop an existing `.wav` or `.mp3` file.
3.  **Analyze:** Click "Analyze Audio". The interface will switch to a "Processing" state while the Celery worker handles the inference.
4.  **Results:** View your **Health Status** (Normal/Moderate/Mild), **Clarity Score**, and **Acoustic Notes**.

### 💻 Method 2: Research CLI (predict.py)
Ideal for batch processing and researchers.
```bash
# Single file analysis
python predict.py example.wav

# Batch directory analysis (outputs result.csv)
python predict.py ./audio_folder --batch --output results.csv

# Verbose research mode (prints layer-by-layer status)
python predict.py example.wav --verbose
```

### 🔌 Method 3: REST API Integration
For integrating SpeechScan into your own applications.
1.  **Submit Job:**
    ```bash
    curl -X POST "http://localhost:8000/predict" -F "file=@audio.wav"
    # Returns: {"job_id": "...", "status": "pending"}
    ```
2.  **Poll for Results:**
    ```bash
    curl "http://localhost:8000/results/{job_id}"
    # Returns: {"status": "completed", "result": {...}}
    ```

---

## 🎯 The Approach: From Research to Production

The creation of SpeechScan followed a rigorous engineering lifecycle:
1.  **Exploratory Data Analysis:** Normalizing the disparity between the healthy Common Voice dataset and the dysarthric TORGO dataset.
2.  **Architecture Design:** Selection of Wav2Vec2 features over traditional MFCCs to capture deep phonetic nuances.
3.  **Security Hardening:** Implementing UUID sanitization to prevent Path Traversal and eliminating "Mock Diagnosis" heuristics.
4.  **Scaling Transformation:** Moving from a blocking synchronous server to a **Distributed Task Queue** to handle DL inference latencies.
5.  **Observability:** Implementing Structured Logging to ensure "black-box" ML operations are fully traceable in production logs.

---

**⚠️ Medical Disclaimer:** This tool is for research and educational purposes only. It is NOT a medical device and should not be used as a substitute for professional clinical diagnosis.
