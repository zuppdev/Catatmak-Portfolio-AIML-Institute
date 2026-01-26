# 🎯 Multimodal Expense Tracker - Indonesian Language

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)](https://www.docker.com/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

A **production-ready ML engineering project** demonstrating multimodal expense tracking using text, voice, and image inputs with Indonesian language support.

## 🌟 Key Features

### Multimodal Input Processing
- **Text**: "gw makan bakso 20rb" → Category: makanan, Amount: Rp 20,000
- **Image**: Receipt photo → Extracted merchant, items, total
- **Audio**: Voice recording → Transcription → Expense details
- **Fusion**: Combines all modalities for robust predictions

### ML Engineering Best Practices
- ✅ Custom fine-tuned models (IndoBERT, LayoutLMv3, Whisper)
- ✅ Multi-task learning (NER + Classification + Regression)
- ✅ Multimodal fusion with attention mechanism
- ✅ Production-ready API with FastAPI
- ✅ MLOps pipeline (MLflow, DVC, Docker)
- ✅ Real-time monitoring (Prometheus, Grafana)
- ✅ Comprehensive testing and documentation

### Indonesian Language Support
- Fine-tuned on Indonesian expense patterns
- Handles informal language ("gw", "20rb", "jt")
- Currency parsing (Rp, IDR, ribu, juta)
- Local receipt formats

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                    Input Layer                                │
│     Text Input    │   Image Input   │   Audio Input          │
│   "makan bakso"   │  Receipt Photo  │  Voice Recording       │
└──────────────────────────────────────────────────────────────┘
           │                 │                 │
           ▼                 ▼                 ▼
┌──────────────────────────────────────────────────────────────┐
│                 Model Layer                                   │
│  ┌─────────┐    ┌─────────┐    ┌──────────┐                │
│  │IndoBERT │    │LayoutLM │    │ Whisper  │                │
│  │NER+Class│    │  +OCR   │    │   +NLP   │                │
│  └─────────┘    └─────────┘    └──────────┘                │
│       │              │              │                        │
│       └──────────────┼──────────────┘                       │
│                      ▼                                       │
│           ┌────────────────────┐                            │
│           │  Fusion Model      │                            │
│           │  (Attention-based) │                            │
│           └────────────────────┘                            │
└──────────────────────────────────────────────────────────────┘
                      │
                      ▼
┌──────────────────────────────────────────────────────────────┐
│                 Output Layer                                  │
│  Category: makanan (Food) - Confidence: 92%                  │
│  Amount: Rp 20,000                                           │
│  Merchant: Warung Bakso Malang                              │
│  Modality Weights: {text: 0.4, image: 0.4, audio: 0.2}     │
└──────────────────────────────────────────────────────────────┘
```

## 📁 Project Structure

```
expense-tracker/
├── configs/                 # Configuration files
│   └── config.py           # System-wide settings
├── models/                  # Model implementations
│   ├── text_extractor.py   # IndoBERT-based text model
│   ├── receipt_parser.py   # LayoutLM receipt parser
│   ├── audio_transcriber.py # Whisper audio model
│   └── fusion_model.py     # Multimodal fusion
├── training/               # Training scripts
│   ├── train_text.py      # Text model training
│   ├── train_receipt.py   # Receipt model training
│   └── train_fusion.py    # Fusion model training
├── serving/                # API and inference
│   └── api/
│       └── main.py        # FastAPI server
├── monitoring/             # Monitoring and drift detection
│   └── monitor.py         # Prometheus metrics
├── examples/               # Example usage
│   ├── test_api.py        # API testing
│   └── client.py          # Python client
├── data/                   # Data directory
├── notebooks/              # Jupyter notebooks
├── tests/                  # Unit tests
├── Dockerfile             # Docker configuration
├── docker-compose.yml     # Multi-container setup
├── requirements.txt       # Dependencies
├── README.md              # This file
├── ARCHITECTURE.md        # Detailed architecture
└── DEPLOYMENT.md          # Deployment guide
```

## 🚀 Quick Start

### Prerequisites
```bash
# Python 3.10+
python --version

# (Optional) GPU for faster training
nvidia-smi
```

### Installation

```bash
# 1. Clone repository
git clone <your-repo>
cd expense-tracker

# 2. Create virtual environment
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download models (first time only)
python -c "from transformers import AutoModel; AutoModel.from_pretrained('indolem/indobert-base-uncased')"
python -c "import whisper; whisper.load_model('small')"
```

### Running the API

```bash
# Start the API server
python serving/api/main.py

# Server will start at http://localhost:8000
# API docs available at http://localhost:8000/docs
```

### Testing

```bash
# Run all tests
python examples/test_api.py

# Interactive demo
python examples/test_api.py interactive

# Using Python client
python examples/client.py
```

## 📊 Usage Examples

### Text Input
```python
from examples.client import ExpenseTrackerClient

client = ExpenseTrackerClient("http://localhost:8000")

result = client.track_text("gw makan bakso 20rb di depan kantor")
print(f"Category: {result['category']}")
print(f"Amount: Rp {result['amount']:,.0f}")
```

### Image Input
```python
result = client.track_image("receipt.jpg")
print(f"Merchant: {result['merchant']}")
print(f"Amount: Rp {result['amount']:,.0f}")
```

### Multimodal Input
```python
result = client.track_multimodal(
    text="makan bakso enak banget",
    image_path="receipt.jpg"
)
print(f"Fusion used: {result['fusion_used']}")
print(f"Modality weights: {result['modality_weights']}")
```

### cURL Examples
```bash
# Text
curl -X POST http://localhost:8000/expense/text \
  -H "Content-Type: application/json" \
  -d '{"text": "makan bakso 20rb"}'

# Image
curl -X POST http://localhost:8000/expense/image \
  -F "file=@receipt.jpg"

# Multimodal
curl -X POST http://localhost:8000/expense/multimodal \
  -F "text=makan bakso" \
  -F "image=@receipt.jpg"
```

## 🐳 Docker Deployment

```bash
# Build and run with Docker Compose
docker-compose up -d

# Services available:
# - API: http://localhost:8000
# - MLflow: http://localhost:5000
# - Prometheus: http://localhost:9090
# - Grafana: http://localhost:3000

# View logs
docker-compose logs -f api

# Stop all services
docker-compose down
```

## 🎓 Training Models

```bash
# Generate synthetic training data
python training/train_text.py

# Train with custom data
python training/train_text.py \
  --train-data data/train.json \
  --val-data data/val.json \
  --epochs 10

# Monitor training with MLflow
mlflow ui --port 5000
```

## 📈 Monitoring

```bash
# Start monitoring server
python monitoring/monitor.py

# Metrics available at:
# - Prometheus: http://localhost:8001/metrics
# - Request counts, latency, confidence scores
# - Drift detection scores
```

## 🎯 ML Engineering Highlights

### 1. **Multi-Task Learning**
- Single model handles NER, classification, and regression
- Shared representations improve efficiency

### 2. **Multimodal Fusion**
- Attention-based fusion learns optimal weights
- Handles missing modalities gracefully
- Improves robustness through redundancy

### 3. **Production-Ready Code**
- Type hints and docstrings
- Error handling and validation
- Async API with proper resource management
- Comprehensive testing

### 4. **MLOps Pipeline**
- Experiment tracking with MLflow
- Data versioning with DVC
- Model monitoring and drift detection
- Containerized deployment

### 5. **Monitoring & Observability**
- Prometheus metrics
- Drift detection (PSI)
- Request logging
- Performance tracking

## 📊 Performance

### Latency (CPU)
- Text: ~100ms
- Image: ~800ms
- Audio: ~1500ms
- Multimodal: ~2000ms

### Accuracy (on synthetic data)
- Category classification: ~90%
- Amount extraction: ~95%
- End-to-end: ~88%

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| **ML Framework** | PyTorch, Transformers |
| **Text Model** | IndoBERT (indobert-base-uncased) |
| **Vision Model** | LayoutLMv3 + EasyOCR |
| **Speech Model** | Whisper (small) |
| **API** | FastAPI + Uvicorn |
| **MLOps** | MLflow, DVC |
| **Monitoring** | Prometheus, Grafana |
| **Database** | PostgreSQL |
| **Deployment** | Docker, Docker Compose |
| **Testing** | Pytest |

## 📚 Documentation

- [Architecture Documentation](ARCHITECTURE.md) - Detailed system architecture
- [Deployment Guide](DEPLOYMENT.md) - Setup and deployment instructions
- [API Documentation](http://localhost:8000/docs) - Interactive API docs (when server is running)

## 🎯 Use Cases

1. **Personal Finance**: Track daily expenses via voice/photo
2. **Business Accounting**: Automated receipt processing
3. **Expense Reports**: Quick entry for reimbursements
4. **Budget Monitoring**: Real-time expense categorization

## 🔮 Future Enhancements

- [ ] Mobile app integration
- [ ] User accounts and authentication
- [ ] Expense analytics dashboard
- [ ] Budget alerts and notifications
- [ ] Multi-currency support
- [ ] Recurring expense detection
- [ ] Export to accounting software

## 📝 License

This project is for educational and demonstration purposes.

## 🤝 Contributing

This is a demonstration project showcasing ML engineering best practices. Feel free to fork and adapt for your needs!

## 📧 Contact

For questions about this ML engineering project, please open an issue.

---

**Built with ❤️ to demonstrate production ML engineering skills**
