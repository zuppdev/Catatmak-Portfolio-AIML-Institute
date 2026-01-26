# Setup and Deployment Guide

## Table of Contents
1. [Local Development Setup](#local-development-setup)
2. [Data Preparation](#data-preparation)
3. [Model Training](#model-training)
4. [Running the API](#running-the-api)
5. [Docker Deployment](#docker-deployment)
6. [Monitoring](#monitoring)
7. [Testing](#testing)

## Local Development Setup

### Prerequisites
- Python 3.10+
- pip
- Git
- (Optional) CUDA-enabled GPU for faster training

### Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd expense-tracker

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download required models (first time only)
python -c "from transformers import AutoModel, AutoTokenizer; \
           AutoModel.from_pretrained('indolem/indobert-base-uncased'); \
           AutoTokenizer.from_pretrained('indolem/indobert-base-uncased')"

python -c "import whisper; whisper.load_model('small')"
```

## Data Preparation

### Creating Training Data

You can use the synthetic data generator or prepare your own dataset:

```python
# Generate synthetic data
from training.train_text import create_synthetic_data
import json

train_data = create_synthetic_data(num_samples=1000)
val_data = create_synthetic_data(num_samples=200)

# Save
with open('data/train.json', 'w') as f:
    json.dump(train_data, f, ensure_ascii=False)

with open('data/val.json', 'w') as f:
    json.dump(val_data, f, ensure_ascii=False)
```

### Real Data Format

For your own data, use this JSON format:

```json
[
  {
    "text": "makan bakso 20 ribu",
    "category": "makanan",
    "amount": 20000,
    "ner_labels": [0, 0, 0, ...]  // Optional: BIO tags for NER
  }
]
```

## Model Training

### Training Text Extractor

```bash
# With synthetic data
python training/train_text.py

# With your own data
python training/train_text.py \
    --train-data data/train.json \
    --val-data data/val.json \
    --epochs 10 \
    --batch-size 32
```

### Training Other Models

```bash
# Receipt parser
python training/train_receipt.py

# Audio transcriber (fine-tuning Whisper)
python training/train_audio.py

# Fusion model
python training/train_fusion.py
```

### Monitoring Training with MLflow

```bash
# Start MLflow UI
mlflow ui --port 5000

# View at http://localhost:5000
```

## Running the API

### Development Mode

```bash
# Start API server
python serving/api/main.py

# Or with uvicorn directly
uvicorn serving.api.main:app --reload --host 0.0.0.0 --port 8000
```

### Production Mode

```bash
# With multiple workers
uvicorn serving.api.main:app \
    --host 0.0.0.0 \
    --port 8000 \
    --workers 4
```

## Docker Deployment

### Building the Image

```bash
# Build
docker build -t expense-tracker:latest .

# Run
docker run -p 8000:8000 expense-tracker:latest
```

### Using Docker Compose

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f api

# Stop all services
docker-compose down

# Rebuild and restart
docker-compose up -d --build
```

### Services Available

After `docker-compose up`:

- API: http://localhost:8000
- API Docs: http://localhost:8000/docs
- MLflow: http://localhost:5000
- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000 (admin/admin)

## Monitoring

### Starting Monitoring

```bash
# Start monitoring server
python monitoring/monitor.py
```

### Prometheus Metrics

Available at: http://localhost:8001/metrics

Key metrics:
- `expense_requests_total`: Total requests by modality and category
- `expense_request_duration_seconds`: Request latency
- `model_confidence`: Model confidence distribution
- `model_drift_score`: Drift detection score

### Grafana Dashboard

1. Access: http://localhost:3000
2. Login: admin/admin
3. Add Prometheus datasource: http://prometheus:9090
4. Import dashboard from `monitoring/grafana/dashboards/`

## Testing

### Unit Tests

```bash
# Run all tests
pytest tests/

# With coverage
pytest --cov=. --cov-report=html tests/
```

### API Testing

```bash
# Test all endpoints
python examples/test_api.py

# Interactive demo
python examples/test_api.py interactive
```

### Example Usage

#### Text Input
```bash
curl -X POST http://localhost:8000/expense/text \
  -H "Content-Type: application/json" \
  -d '{"text": "makan bakso 20rb"}'
```

#### Image Input
```bash
curl -X POST http://localhost:8000/expense/image \
  -F "file=@receipt.jpg"
```

#### Multimodal Input
```bash
curl -X POST http://localhost:8000/expense/multimodal \
  -F "text=makan bakso enak" \
  -F "image=@receipt.jpg" \
  -F "audio=@recording.wav"
```

## Production Deployment

### Kubernetes Deployment

```bash
# Apply configurations
kubectl apply -f k8s/

# Check status
kubectl get pods -n expense-tracker

# View logs
kubectl logs -f deployment/expense-tracker-api -n expense-tracker
```

### Environment Variables

Required environment variables:

```bash
API_HOST=0.0.0.0
API_PORT=8000
DATABASE_URL=postgresql://user:pass@host:5432/db
MLFLOW_TRACKING_URI=http://mlflow:5000
PROMETHEUS_PORT=8001
ENABLE_MONITORING=true
```

### Performance Tuning

1. **GPU Acceleration**: Set `CUDA_VISIBLE_DEVICES` for GPU usage
2. **Worker Processes**: Adjust `API_WORKERS` based on CPU cores
3. **Model Caching**: Models are loaded once at startup
4. **Batch Processing**: For bulk processing, use batch endpoints

### Security Considerations

1. Add authentication middleware
2. Use HTTPS in production
3. Implement rate limiting
4. Validate and sanitize inputs
5. Keep dependencies updated

## Troubleshooting

### Common Issues

**Models not loading:**
```bash
# Clear cache and re-download
rm -rf ~/.cache/huggingface
python scripts/download_models.py
```

**Out of memory:**
```bash
# Use smaller batch sizes
# Use CPU instead of GPU
# Reduce model sizes (use 'tiny' or 'base' instead of 'large')
```

**Slow inference:**
```bash
# Enable GPU if available
# Use quantized models
# Implement model caching
# Use batching for multiple requests
```

## Next Steps

1. **Collect Real Data**: Replace synthetic data with actual expense data
2. **Fine-tune Models**: Train on domain-specific Indonesian expenses
3. **Add Features**: Implement user accounts, expense analytics
4. **Scale**: Deploy to cloud (AWS, GCP, Azure)
5. **Mobile App**: Create mobile client for easy data entry

## Support

For issues and questions:
- Check the documentation
- Review example scripts
- Check logs: `tail -f logs/api.log`
- Review monitoring metrics

## License

This project is for educational and demonstration purposes.
