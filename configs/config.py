import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
LOGS_DIR = BASE_DIR / "logs"

# Create directories
for dir_path in [DATA_DIR, MODEL_DIR, LOGS_DIR]:
    dir_path.mkdir(exist_ok=True, parents=True)

# Model configurations
MODELS = {
    "text": {
        "name": "indolem/indobert-base-uncased",
        "max_length": 128,
        "batch_size": 32,
        "learning_rate": 2e-5,
        "epochs": 10
    },
    "image": {
        "name": "microsoft/layoutlmv3-base",
        "max_length": 512,
        "batch_size": 8,
        "learning_rate": 5e-5,
        "epochs": 15
    },
    "audio": {
        "name": "openai/whisper-small",
        "language": "id",
        "batch_size": 16,
        "learning_rate": 1e-5,
        "epochs": 8
    },
    "fusion": {
        "hidden_dim": 256,
        "dropout": 0.3,
        "batch_size": 32,
        "learning_rate": 1e-4,
        "epochs": 20
    }
}

# Categories for expense classification
CATEGORIES = [
    "makanan",          # food
    "transportasi",     # transportation
    "belanja",          # shopping
    "hiburan",          # entertainment
    "kesehatan",        # health
    "pendidikan",       # education
    "tagihan",          # bills/utilities
    "lainnya"           # others
]

# Entity types for NER
ENTITY_TYPES = [
    "O",           # Outside
    "B-AMOUNT",    # Beginning of amount
    "I-AMOUNT",    # Inside amount
    "B-ITEM",      # Beginning of item
    "I-ITEM",      # Inside item
    "B-MERCHANT",  # Beginning of merchant
    "I-MERCHANT",  # Inside merchant
    "B-DATE",      # Beginning of date
    "I-DATE"       # Inside date
]

# Data augmentation settings
AUGMENTATION = {
    "text": {
        "synonym_replacement": True,
        "random_insertion": True,
        "random_swap": True,
        "random_deletion": True,
        "alpha": 0.1
    },
    "image": {
        "rotation": 5,
        "brightness": 0.2,
        "contrast": 0.2,
        "noise": 0.01
    },
    "audio": {
        "noise_injection": True,
        "time_stretch": True,
        "pitch_shift": True
    }
}

# MLflow settings
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MLFLOW_EXPERIMENT_NAME = "expense-tracker"

# API settings
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", 8000))
API_WORKERS = int(os.getenv("API_WORKERS", 4))

# Database settings
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://user:password@localhost:5432/expense_tracker"
)

# Monitoring settings
PROMETHEUS_PORT = int(os.getenv("PROMETHEUS_PORT", 8001))
ENABLE_MONITORING = os.getenv("ENABLE_MONITORING", "true").lower() == "true"

# Inference settings
CONFIDENCE_THRESHOLD = 0.7
FUSION_WEIGHTS = {
    "text": 0.4,
    "image": 0.4,
    "audio": 0.2
}

# Currency settings
CURRENCY = "IDR"
AMOUNT_PATTERNS = [
    r"(?:Rp\.?\s*)?(\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{2})?)",
    r"(\d+)\s*(?:ribu|rb)",
    r"(\d+)\s*(?:juta|jt)",
    r"IDR\s*(\d+)"
]

# Minimum sensible amount in IDR - amounts below this are likely
# shorthand (e.g., "20" meaning "20k" = 20,000 IDR)
MIN_SENSIBLE_AMOUNT_IDR = 1000

# Threshold below which we assume the user meant thousands
# e.g., if amount < 1000, multiply by 1000 to get the real amount
AMOUNT_SHORTHAND_THRESHOLD = 1000
