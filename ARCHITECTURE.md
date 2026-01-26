# System Architecture Documentation

## Overview

This document describes the architecture of the Multimodal Indonesian Expense Tracker, a production-ready ML system that processes text, images, and audio to extract expense information.

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         Input Layer                              │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐                  │
│  │   Text   │    │  Image   │    │  Audio   │                  │
│  │ "makan   │    │ Receipt  │    │  Voice   │                  │
│  │ bakso"   │    │   Photo  │    │Recording │                  │
│  └──────────┘    └──────────┘    └──────────┘                  │
└─────────────────────────────────────────────────────────────────┘
            │              │              │
            ▼              ▼              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Preprocessing Layer                           │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐                  │
│  │   Text   │    │  Image   │    │  Audio   │                  │
│  │  Clean   │    │ Enhance  │    │  Clean   │                  │
│  │Normalize │    │   OCR    │    │ Denoise  │                  │
│  └──────────┘    └──────────┘    └──────────┘                  │
└─────────────────────────────────────────────────────────────────┘
            │              │              │
            ▼              ▼              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Model Layer                                  │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐                  │
│  │ IndoBERT │    │LayoutLM  │    │ Whisper  │                  │
│  │   NER +  │    │  Receipt │    │   +      │                  │
│  │  Class   │    │  Parser  │    │IndoBERT  │                  │
│  └──────────┘    └──────────┘    └──────────┘                  │
│       │                │                │                        │
│       └────────────────┼────────────────┘                       │
│                        ▼                                         │
│              ┌──────────────────┐                               │
│              │  Fusion Model    │                               │
│              │  Multi-head      │                               │
│              │  Attention       │                               │
│              └──────────────────┘                               │
└─────────────────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Output Layer                                 │
│  ┌─────────────────────────────────────────────────────┐        │
│  │  Category: makanan                                  │        │
│  │  Amount: Rp 20,000                                  │        │
│  │  Merchant: Warung Bakso                             │        │
│  │  Confidence: 0.92                                   │        │
│  │  Modality Weights: {text: 0.4, image: 0.4, ...}   │        │
│  └─────────────────────────────────────────────────────┘        │
└─────────────────────────────────────────────────────────────────┘
```

## Component Architecture

### 1. Text Extraction Pipeline

**Model**: IndoBERT (indobert-base-uncased)
**Tasks**:
- Named Entity Recognition (NER)
- Category Classification
- Amount Extraction

**Architecture**:
```python
Input Text → Tokenizer → BERT Encoder → [NER Head, Category Head, Amount Head]
```

**Features**:
- Multi-task learning
- Indonesian language support
- Handles informal language ("gw", "20rb")
- Regex fallback for amount extraction

### 2. Image Processing Pipeline

**Model**: LayoutLMv3
**OCR**: EasyOCR (Indonesian + English)

**Architecture**:
```python
Receipt Image → Preprocessing → OCR → LayoutLM → [Category, Key Fields]
```

**Preprocessing Steps**:
1. Grayscale conversion
2. Adaptive thresholding
3. Denoising
4. Contrast enhancement

**Features**:
- Document understanding with layout information
- Handles various receipt formats
- Extracts merchant, date, amount, items

### 3. Audio Processing Pipeline

**Model**: Whisper (small)
**Language**: Indonesian (id)

**Architecture**:
```python
Audio → Preprocessing → Whisper → Transcription → Text Extractor
```

**Audio Enhancement**:
1. Noise reduction (spectral gating)
2. Volume normalization
3. Pre-emphasis filter

**Features**:
- Supports multiple audio formats
- Timestamp extraction
- Confidence scoring per segment

### 4. Multimodal Fusion

**Architecture**: Attention-based fusion

**Components**:
1. **Modality Projection**: Projects each modality to common dimension
2. **Cross-Modal Attention**: Multi-head attention across modalities
3. **Learned Weighting**: Dynamically weights modalities based on input
4. **Fusion Layer**: Combines weighted representations

**Formula**:
```
Fused = W_text * E_text + W_image * E_image + W_audio * E_audio

where W are learned weights that sum to 1
```

**Benefits**:
- Handles missing modalities gracefully
- Learns to trust different sources
- Improves robustness through redundancy

## Data Flow

### Single Modality Request

```
1. Client sends text/image/audio
2. API validates input
3. Preprocessor cleans data
4. Model extracts features
5. Post-processor formats output
6. Monitoring logs prediction
7. Response returned to client
```

### Multimodal Request

```
1. Client sends multiple inputs
2. API validates all inputs
3. Each modality processed in parallel:
   - Text → Text Extractor
   - Image → Receipt Parser
   - Audio → Whisper → Text Extractor
4. Embeddings extracted from each
5. Fusion model combines embeddings
6. Weighted prediction generated
7. Individual results + fused result returned
```

## Training Pipeline

### Phase 1: Individual Model Training

```
Data Collection → Data Cleaning → Annotation → Training → Validation → Model Registry
```

**Tools**:
- MLflow for experiment tracking
- DVC for data versioning
- Weights & Biases for metrics visualization

### Phase 2: Fusion Model Training

```
Collect Individual Embeddings → Create Fusion Dataset → Train Fusion Model → Validate
```

**Strategy**:
- Freeze individual models initially
- Train fusion weights
- Fine-tune end-to-end if needed

### Training Configuration

```yaml
Text Extractor:
  epochs: 10
  batch_size: 32
  learning_rate: 2e-5
  optimizer: AdamW
  scheduler: Linear warmup

Receipt Parser:
  epochs: 15
  batch_size: 8
  learning_rate: 5e-5
  optimizer: AdamW

Audio Transcriber:
  epochs: 8
  batch_size: 16
  learning_rate: 1e-5
  fine_tune: True

Fusion Model:
  epochs: 20
  batch_size: 32
  learning_rate: 1e-4
  hidden_dim: 256
```

## API Architecture

### FastAPI Server

**Endpoints**:
- `GET /health` - Health check
- `GET /categories` - List categories
- `POST /expense/text` - Text extraction
- `POST /expense/image` - Image extraction
- `POST /expense/audio` - Audio extraction
- `POST /expense/multimodal` - Multimodal extraction

**Features**:
- Async request handling
- File upload validation
- CORS support
- Automatic API documentation (Swagger)
- Error handling and logging

**Performance**:
- Request timeout: 30s
- Max file size: 10MB
- Concurrent requests: Handled by workers
- Model caching: Models loaded once at startup

## Monitoring Architecture

### Metrics Collection

**Prometheus Metrics**:
1. `expense_requests_total` - Total requests by modality
2. `expense_request_duration_seconds` - Latency histogram
3. `model_confidence` - Confidence distribution
4. `expense_amount_idr` - Amount distribution
5. `prediction_errors_total` - Error counts
6. `model_drift_score` - Drift detection

### Drift Detection

**Method**: Population Stability Index (PSI)

**Formula**:
```
PSI = Σ (P_current - P_baseline) * ln(P_current / P_baseline)
```

**Thresholds**:
- PSI < 0.1: No significant drift
- 0.1 ≤ PSI < 0.25: Moderate drift
- PSI ≥ 0.25: Significant drift (retrain recommended)

**Monitoring Window**: 1000 predictions
**Check Interval**: 5 minutes

## Database Schema

### Expenses Table

```sql
CREATE TABLE expenses (
    id SERIAL PRIMARY KEY,
    user_id INTEGER,
    category VARCHAR(50),
    amount DECIMAL(12, 2),
    currency VARCHAR(3) DEFAULT 'IDR',
    description TEXT,
    merchant VARCHAR(255),
    date DATE,
    confidence FLOAT,
    modality VARCHAR(20),
    modality_weights JSONB,
    created_at TIMESTAMP DEFAULT NOW()
);
```

### Model Predictions Table

```sql
CREATE TABLE model_predictions (
    id SERIAL PRIMARY KEY,
    expense_id INTEGER REFERENCES expenses(id),
    model_name VARCHAR(50),
    raw_output JSONB,
    processing_time FLOAT,
    created_at TIMESTAMP DEFAULT NOW()
);
```

## Deployment Architecture

### Container Structure

```
expense-tracker/
├── api (FastAPI + Models)
├── postgres (Database)
├── mlflow (Experiment tracking)
├── prometheus (Metrics)
└── grafana (Visualization)
```

### Scaling Strategy

**Horizontal Scaling**:
- Multiple API containers behind load balancer
- Database replication for read queries
- Model serving separated if needed

**Vertical Scaling**:
- GPU instances for faster inference
- More memory for larger batch sizes
- SSD for faster model loading

### Production Deployment

```
User → Load Balancer → API Pods (K8s)
                         ├─ Model Cache
                         ├─ Database Pool
                         └─ Monitoring
```

## Security Considerations

### API Security
1. API key authentication
2. Rate limiting per user
3. Input validation and sanitization
4. HTTPS only in production
5. CORS configuration

### Data Security
1. Encrypted database connections
2. No storage of sensitive info
3. Anonymized logging
4. GDPR compliance ready

### Model Security
1. Model versioning and signing
2. Isolated model serving
3. Input size limits
4. Timeout protection

## Performance Benchmarks

### Inference Latency (CPU)

| Modality | Average | P95 | P99 |
|----------|---------|-----|-----|
| Text     | 100ms   | 150ms | 200ms |
| Image    | 800ms   | 1200ms | 1500ms |
| Audio    | 1500ms  | 2000ms | 2500ms |
| Multimodal | 2000ms | 2800ms | 3500ms |

### Inference Latency (GPU)

| Modality | Average | P95 | P99 |
|----------|---------|-----|-----|
| Text     | 50ms    | 70ms | 90ms |
| Image    | 200ms   | 300ms | 400ms |
| Audio    | 500ms   | 700ms | 900ms |
| Multimodal | 600ms | 900ms | 1200ms |

### Throughput

- Text: 100 req/s (CPU), 500 req/s (GPU)
- Image: 10 req/s (CPU), 50 req/s (GPU)
- Audio: 5 req/s (CPU), 25 req/s (GPU)

## Future Enhancements

### Model Improvements
1. Fine-tune on larger Indonesian dataset
2. Implement active learning pipeline
3. Add online learning for user patterns
4. Develop domain-specific models

### Feature Additions
1. Expense categorization rules
2. Budget tracking and alerts
3. Receipt verification
4. Multi-user support
5. Mobile app integration

### Infrastructure
1. Model quantization for edge deployment
2. ONNX export for cross-platform
3. Kubernetes autoscaling
4. Multi-region deployment

## References

### Models
- IndoBERT: https://huggingface.co/indolem/indobert-base-uncased
- LayoutLMv3: https://huggingface.co/microsoft/layoutlmv3-base
- Whisper: https://github.com/openai/whisper

### Papers
- BERT: Devlin et al. (2018)
- LayoutLM: Xu et al. (2020)
- Whisper: Radford et al. (2022)

### Tools
- FastAPI: https://fastapi.tiangolo.com
- MLflow: https://mlflow.org
- Prometheus: https://prometheus.io
