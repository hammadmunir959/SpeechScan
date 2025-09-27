# Dysarthria Detection Model

A machine learning model for detecting dysarthria (abnormal speech patterns) in audio files using Wav2Vec2 features and neural network classification.

## 🎯 Overview

This project provides a complete solution for dysarthria detection with:
- **High Accuracy**: 94.2% test accuracy, 98.7% AUC score
- **Multiple Interfaces**: API, CLI, and Web UI
- **Live Recording**: Record audio directly in the browser
- **Easy Deployment**: Docker containerization
- **Production Ready**: FastAPI backend with proper error handling

## 📊 Model Performance

| Metric | Score |
|--------|-------|
| Test Accuracy | 94.2% |
| Precision | 93.7% |
| Recall | 94.9% |
| F1 Score | 94.3% |
| AUC | 98.7% |

## 🚀 Quick Start

### Option 1: Docker (Recommended)

```bash
# Clone and navigate to the project
cd dysarthria_model

# Build and run with Docker Compose
docker-compose up --build

# Access the web interface
open http://localhost
```

### Option 2: Local Installation

```bash
# Install dependencies
pip install -r api/requirements.txt

# Run the API server
python api/main.py

# Access the web interface
open web_interface/index.html
```

### Option 3: Command Line

```bash
# Make prediction on audio file
python predict.py path/to/audio.wav

# With custom model directory
python predict.py path/to/audio.wav --model-dir model/

# Save results to file
python predict.py path/to/audio.wav --output results.json
```

## 📁 Project Structure

```
dysarthria_model/
├── api/                    # FastAPI backend
│   ├── main.py            # API server
│   └── requirements.txt   # Python dependencies
├── model/                 # Trained model files
│   ├── config.json        # Model configuration
│   ├── metadata.json      # Model metadata
│   └── model.weights.h5   # Model weights
├── web_interface/         # Web UI
│   └── index.html         # Frontend interface
├── predict.py             # CLI prediction script
├── Dockerfile             # Docker configuration
├── docker-compose.yml     # Docker Compose setup
├── nginx.conf             # Nginx configuration
└── README.md              # This file
```

## 🔧 API Usage

### Health Check
```bash
curl http://localhost:8000/health
```

### Predict Dysarthria
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@audio.wav"
```

### Response Format
```json
{
  "prediction": "Normal",
  "probability": 0.123,
  "confidence": "High",
  "processing_time": 1.456
}
```

## 🎤 Supported Audio Formats

- **WAV** (.wav)
- **MP3** (.mp3)
- **FLAC** (.flac)
- **OPUS** (.opus)
- **WebM** (.webm) - For browser recordings
- **OGG** (.ogg)

### Audio Requirements
- **Duration**: 2-10 seconds (automatically padded/truncated)
- **Sample Rate**: Automatically resampled to 16kHz
- **Channels**: Mono or stereo (converted to mono)
- **File Size**: Up to 50MB
- **Recording**: Real-time recording with automatic noise suppression

## 🏗️ Model Architecture

### Feature Extraction
- **Wav2Vec2**: Pre-trained transformer model for speech representation
- **Input**: Raw audio waveform (16kHz)
- **Output**: 768-dimensional feature vector

### Classification
- **Architecture**: 3-layer neural network
- **Input**: 768 Wav2Vec2 features
- **Hidden Layers**: 128 → 64 neurons (ReLU activation)
- **Output**: Binary classification (Normal/Abnormal)
- **Regularization**: Dropout (0.3) and early stopping

## 📈 Training Data

- **Total Samples**: 10,115 audio files
- **Normal Voices**: 5,017 samples
- **Abnormal Voices**: 5,100 samples
- **Data Sources**: Mozilla Common Voice + TORGO dataset
- **Class Balance**: 0.98:1 (Normal:Abnormal)

## 🔍 Usage Examples

### Web Interface
1. Open `http://localhost` in your browser
2. **Option A - Upload File**: Upload an audio file using drag & drop or click to browse
3. **Option B - Record Live**: Click "Start Recording" to record audio directly in your browser
4. Click "Analyze Audio" to get results
5. View prediction, probability, and confidence level

### Live Recording Feature
The web interface now includes a **live recording feature** that allows users to:
- **Record directly in the browser** using their microphone
- **Real-time feedback** with recording timer and visual indicators
- **Automatic noise suppression** and echo cancellation
- **Preview recorded audio** before analysis
- **One-click analysis** of recorded audio
- **Auto-stop after 10 seconds** to prevent long recordings

#### Recording Requirements:
- **Browser Support**: Modern browsers with MediaRecorder API
- **Microphone Access**: User must grant microphone permissions
- **HTTPS**: Recording requires secure context (HTTPS or localhost)
- **Audio Quality**: Optimized for speech recognition (16kHz sample rate)

### Command Line
```bash
# Basic prediction
python predict.py sample.wav

# Verbose output
python predict.py sample.wav --verbose

# Custom model directory
python predict.py sample.wav --model-dir /path/to/model

# Save results
python predict.py sample.wav --output results.json
```

### Python API
```python
import requests

# Upload and analyze audio
with open('audio.wav', 'rb') as f:
    response = requests.post(
        'http://localhost:8000/predict',
        files={'file': f}
    )

result = response.json()
print(f"Prediction: {result['prediction']}")
print(f"Probability: {result['probability']:.3f}")
```

## 🐳 Docker Deployment

### Development
```bash
# Build and run
docker-compose up --build

# Run in background
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

### Production
```bash
# Build production image
docker build -t dysarthria-detection .

# Run with custom configuration
docker run -p 8000:8000 -v ./model:/app/model:ro dysarthria-detection
```

## 🔧 Configuration

### Environment Variables
- `PYTHONUNBUFFERED=1`: Enable Python output buffering
- `MODEL_DIR`: Path to model directory (default: `model/`)

### API Configuration
- **Host**: `0.0.0.0` (all interfaces)
- **Port**: `8000`
- **Max File Size**: 50MB
- **Timeout**: 30 seconds

### Model Parameters
- **Target Sample Rate**: 16,000 Hz
- **Max Duration**: 10 seconds
- **Min Duration**: 2 seconds
- **Confidence Thresholds**:
  - High: > 0.8
  - Medium: 0.6 - 0.8
  - Low: < 0.6

## 🚨 Troubleshooting

### Common Issues

**1. Model Loading Error**
```
Error: Model files not found
```
- Ensure `model/` directory contains all required files
- Check file permissions

**2. Audio Processing Error**
```
Error: Audio preprocessing failed
```
- Verify audio file format is supported
- Check file is not corrupted
- Ensure file size is under 50MB

**3. API Connection Error**
```
Error: API Disconnected
```
- Verify API server is running on port 8000
- Check firewall settings
- Ensure no port conflicts

**4. Docker Build Error**
```
Error: Failed to build image
```
- Check Docker is running
- Verify all files are present
- Try `docker-compose build --no-cache`

**5. Recording Not Working**
```
Error: Unable to access microphone
```
- Check browser permissions for microphone access
- Ensure you're using HTTPS or localhost
- Try refreshing the page and granting permissions again
- Check if another application is using the microphone

**6. Recording Quality Issues**
- Ensure good microphone quality
- Minimize background noise
- Speak clearly and at normal volume
- Check microphone settings in browser/system

### Performance Optimization

**For Large Files:**
- Pre-process audio to 2-10 seconds
- Use WAV format for fastest processing
- Ensure good audio quality

**For High Throughput:**
- Use GPU acceleration (modify Dockerfile)
- Increase worker processes
- Implement request queuing

## 📚 Technical Details

### Dependencies
- **TensorFlow**: 2.15.0 (Model inference)
- **Transformers**: 4.35.0 (Wav2Vec2)
- **Librosa**: 0.10.1 (Audio processing)
- **FastAPI**: 0.104.1 (API framework)
- **PyTorch**: 2.1.0 (Wav2Vec2 backend)

### System Requirements
- **RAM**: Minimum 4GB (8GB recommended)
- **CPU**: Multi-core processor
- **Storage**: 2GB free space
- **OS**: Linux, macOS, Windows

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Mozilla Common Voice** for normal speech data
- **TORGO** dataset for dysarthric speech data
- **Hugging Face** for Wav2Vec2 model
- **FastAPI** team for the excellent framework

## 📞 Support

For issues and questions:
1. Check the troubleshooting section
2. Review the API documentation at `/docs`
3. Open an issue on GitHub
4. Contact the development team

---

**⚠️ Medical Disclaimer**: This tool is for research and educational purposes only. It should not be used as a substitute for professional medical diagnosis or treatment. Always consult with qualified healthcare professionals for medical concerns.
