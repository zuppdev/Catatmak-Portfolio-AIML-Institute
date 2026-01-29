from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Depends, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from pydantic import BaseModel
from typing import Optional, List
import numpy as np
from PIL import Image
import io
import base64
import tempfile
import os
import secrets
from datetime import datetime

from models.text_extractor import TextExtractor
from models.receipt_parser import ReceiptParser
from models.audio_transcriber import AudioTranscriber
from models.fusion_model import MultimodalExpenseExtractor
from configs.config import API_HOST, API_PORT, CATEGORIES

# =============================================================================
# SECURITY CONFIGURATION
# =============================================================================

# API Key - load from environment variable (set this on your VPS)
API_KEY = os.environ.get("EXPENSE_API_KEY", "")
API_KEY_ENABLED = bool(API_KEY)  # Disable auth if no key set

# Allowed origins for CORS (add your frontend domains)
ALLOWED_ORIGINS = os.environ.get("ALLOWED_ORIGINS", "").split(",")
if not ALLOWED_ORIGINS or ALLOWED_ORIGINS == [""]:
    # Default: allow all in development, restrict in production
    ALLOWED_ORIGINS = ["*"] if not API_KEY_ENABLED else []

# API Key header
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(api_key: str = Security(api_key_header)):
    """Verify API key if authentication is enabled"""
    if not API_KEY_ENABLED:
        return True  # Skip auth if no key configured

    if not api_key:
        raise HTTPException(
            status_code=401,
            detail="Missing API key. Include 'X-API-Key' header."
        )

    if not secrets.compare_digest(api_key, API_KEY):
        raise HTTPException(
            status_code=403,
            detail="Invalid API key"
        )

    return True


# =============================================================================
# APP INITIALIZATION
# =============================================================================

# Initialize FastAPI app
app = FastAPI(
    title="Multimodal Expense Tracker API",
    description="Indonesian language expense tracking with text, voice, and image support",
    version="1.0.0",
    docs_url="/docs" if not API_KEY_ENABLED else None,  # Disable docs in production
    redoc_url="/redoc" if not API_KEY_ENABLED else None,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["X-API-Key", "Content-Type"],
)


# Pydantic models
class ExpenseResponse(BaseModel):
    category: str
    category_confidence: float
    amount: int  # Always full amount in IDR (e.g., 20000 not 20)
    amount_formatted: str  # Human readable (e.g., "Rp 20.000")
    currency: str = "IDR"
    date: str
    description: Optional[str] = None
    merchant: Optional[str] = None
    modality_weights: Optional[dict] = None
    fusion_used: bool = False


class TextExpenseRequest(BaseModel):
    text: str


class HealthResponse(BaseModel):
    status: str
    models_loaded: dict
    timestamp: str


class ModelInfoResponse(BaseModel):
    name: str
    version: str
    description: str
    supported_categories: List[dict]
    supported_inputs: List[str]
    amount_formats: List[str]
    example_queries: List[dict]
    model_details: dict


def format_idr(amount: float) -> str:
    """Format amount as Indonesian Rupiah with thousand separators"""
    return f"Rp {int(amount):,}".replace(",", ".")


# Global model instances
text_extractor = None
receipt_parser = None
audio_transcriber = None
multimodal_extractor = None


@app.on_event("startup")
async def startup_event():
    """Initialize models on startup"""
    global text_extractor, receipt_parser, audio_transcriber, multimodal_extractor
    
    print("Loading models...")
    
    try:
        # Load text extractor
        text_extractor = TextExtractor()
        print("✓ Text extractor loaded")
        
        # Load receipt parser
        receipt_parser = ReceiptParser()
        print("✓ Receipt parser loaded")
        
        # Load audio transcriber
        audio_transcriber = AudioTranscriber(model_size="small")
        print("✓ Audio transcriber loaded")
        
        # Initialize multimodal extractor
        multimodal_extractor = MultimodalExpenseExtractor(
            text_extractor=text_extractor,
            receipt_parser=receipt_parser,
            audio_transcriber=audio_transcriber
        )
        print("✓ Multimodal extractor initialized")
        
        print("All models loaded successfully!")
        
    except Exception as e:
        print(f"Error loading models: {e}")
        raise


@app.get("/", response_model=dict)
async def root():
    """Root endpoint"""
    return {
        "message": "Multimodal Expense Tracker API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "text": "/expense/text",
            "image": "/expense/image",
            "audio": "/expense/audio",
            "multimodal": "/expense/multimodal"
        }
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        models_loaded={
            "text_extractor": text_extractor is not None,
            "receipt_parser": receipt_parser is not None,
            "audio_transcriber": audio_transcriber is not None,
            "multimodal_extractor": multimodal_extractor is not None
        },
        timestamp=datetime.now().isoformat()
    )


@app.get("/model/info", response_model=ModelInfoResponse)
async def get_model_info():
    """Get detailed model information and capabilities"""
    return ModelInfoResponse(
        name="Catatmak - Indonesian Expense Tracker",
        version="1.0.0",
        description="Multimodal AI model for extracting expense information from Indonesian text, images, and audio",
        supported_categories=[
            {"id": "makanan", "name": "Makanan & Minuman", "description": "Food, drinks, restaurants, cafes"},
            {"id": "transportasi", "name": "Transportasi", "description": "Fuel, parking, ride-hailing, public transport"},
            {"id": "belanja", "name": "Belanja", "description": "Shopping, clothing, electronics, household"},
            {"id": "tagihan", "name": "Tagihan", "description": "Bills, utilities, subscriptions, rent"},
            {"id": "hiburan", "name": "Hiburan", "description": "Entertainment, movies, games, streaming"},
            {"id": "kesehatan", "name": "Kesehatan", "description": "Health, medicine, doctor, hospital"},
            {"id": "pendidikan", "name": "Pendidikan", "description": "Education, courses, books, training"},
        ],
        supported_inputs=[
            "Text (Indonesian language)",
            "Image (receipt photos - JPG, PNG)",
            "Audio (voice recordings - WAV, MP3, M4A)"
        ],
        amount_formats=[
            "20000 - Plain number",
            "20k - Thousands shorthand",
            "20rb / 20 ribu - Indonesian thousands",
            "2jt / 2 juta - Indonesian millions",
            "Rp 20.000 - Formatted with prefix",
            "Rp20000 - Prefix without separator"
        ],
        example_queries=[
            {"input": "makan bakso 20k", "category": "makanan", "amount": 20000},
            {"input": "isi bensin 50rb", "category": "transportasi", "amount": 50000},
            {"input": "beli baju 150 ribu", "category": "belanja", "amount": 150000},
            {"input": "bayar listrik 500000", "category": "tagihan", "amount": 500000},
            {"input": "nonton bioskop 75k", "category": "hiburan", "amount": 75000},
        ],
        model_details={
            "text_model": {
                "base": "IndoBERT (indobenchmark/indobert-base-p1)",
                "type": "Multi-task classification + NER",
                "features": ["Category classification", "Amount extraction", "Entity recognition"]
            },
            "image_model": {
                "base": "LayoutLMv3",
                "type": "Document understanding",
                "features": ["OCR", "Receipt parsing", "Amount detection"]
            },
            "audio_model": {
                "base": "OpenAI Whisper (small)",
                "type": "Speech-to-text",
                "features": ["Indonesian transcription", "Voice expense input"]
            },
            "fusion": {
                "type": "Confidence-weighted multimodal fusion",
                "features": ["Cross-modal validation", "Weighted averaging"]
            }
        }
    )


@app.get("/model/test")
async def test_model():
    """
    Run sample queries to demonstrate model capabilities.
    Returns evaluation results with expected vs actual outputs.
    """
    test_cases = [
        {"text": "makan bakso 20k", "expected_category": "makanan", "expected_amount": 20000},
        {"text": "isi bensin pertamax 100rb", "expected_category": "transportasi", "expected_amount": 100000},
        {"text": "beli baju di mall 250 ribu", "expected_category": "belanja", "expected_amount": 250000},
        {"text": "bayar listrik bulan ini 450000", "expected_category": "tagihan", "expected_amount": 450000},
        {"text": "nonton film di bioskop 50k", "expected_category": "hiburan", "expected_amount": 50000},
        {"text": "beli obat di apotek 35rb", "expected_category": "kesehatan", "expected_amount": 35000},
        {"text": "bayar les bahasa inggris 500k", "expected_category": "pendidikan", "expected_amount": 500000},
        {"text": "ngopi di starbucks 75 ribu", "expected_category": "makanan", "expected_amount": 75000},
        {"text": "gojek ke kantor 25000", "expected_category": "transportasi", "expected_amount": 25000},
        {"text": "langganan netflix 150rb", "expected_category": "hiburan", "expected_amount": 150000},
    ]

    results = []
    correct_category = 0
    correct_amount = 0
    total = len(test_cases)

    for test in test_cases:
        try:
            prediction = text_extractor.predict(test["text"])
            amount = int(prediction["amount"])

            cat_match = prediction["category"] == test["expected_category"]
            amt_match = amount == test["expected_amount"]

            if cat_match:
                correct_category += 1
            if amt_match:
                correct_amount += 1

            results.append({
                "input": test["text"],
                "expected": {
                    "category": test["expected_category"],
                    "amount": test["expected_amount"]
                },
                "actual": {
                    "category": prediction["category"],
                    "amount": amount,
                    "amount_formatted": format_idr(amount),
                    "confidence": round(prediction["category_confidence"], 2)
                },
                "category_correct": cat_match,
                "amount_correct": amt_match
            })
        except Exception as e:
            results.append({
                "input": test["text"],
                "error": str(e)
            })

    return {
        "summary": {
            "total_tests": total,
            "category_accuracy": f"{(correct_category / total) * 100:.1f}%",
            "amount_accuracy": f"{(correct_amount / total) * 100:.1f}%",
            "category_correct": correct_category,
            "amount_correct": correct_amount
        },
        "results": results,
        "timestamp": datetime.now().isoformat()
    }


@app.post("/expense/text", response_model=ExpenseResponse)
async def extract_from_text(request: TextExpenseRequest, _: bool = Depends(verify_api_key)):
    """
    Extract expense information from text
    
    Example:
    {
        "text": "gw makan bakso 20rb di depan kantor"
    }
    """
    try:
        result = text_extractor.predict(request.text)
        amount = int(result["amount"])

        return ExpenseResponse(
            category=result["category"],
            category_confidence=result["category_confidence"],
            amount=amount,
            amount_formatted=format_idr(amount),
            date=datetime.now().strftime("%Y-%m-%d"),
            description=result["processed_text"],
            merchant=result["entities"].get("merchant", "")
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/expense/image", response_model=ExpenseResponse)
async def extract_from_image(file: UploadFile = File(...), _: bool = Depends(verify_api_key)):
    """
    Extract expense information from receipt image
    
    Accepts: JPG, PNG, JPEG
    """
    try:
        # Validate file type
        if not file.content_type.startswith("image/"):
            raise HTTPException(
                status_code=400, 
                detail="File must be an image"
            )
        
        # Read image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # Parse receipt
        result = receipt_parser.predict(image)
        amount = int(result["amount"])

        return ExpenseResponse(
            category=result["category"],
            category_confidence=result["category_confidence"],
            amount=amount,
            amount_formatted=format_idr(amount),
            date=result.get("date", datetime.now().strftime("%Y-%m-%d")),
            merchant=result.get("merchant", ""),
            description=result.get("ocr_text", "")[:200]
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/expense/audio", response_model=ExpenseResponse)
async def extract_from_audio(file: UploadFile = File(...), _: bool = Depends(verify_api_key)):
    """
    Extract expense information from audio recording
    
    Accepts: WAV, MP3, M4A, OGG
    """
    try:
        # Validate file type
        valid_types = ["audio/wav", "audio/mpeg", "audio/mp4", "audio/ogg"]
        if file.content_type not in valid_types:
            raise HTTPException(
                status_code=400,
                detail=f"File must be audio. Accepted: {valid_types}"
            )
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            contents = await file.read()
            tmp.write(contents)
            tmp_path = tmp.name
        
        try:
            # Transcribe and extract
            transcription = audio_transcriber.transcribe(tmp_path)
            text_result = text_extractor.predict(transcription["text"])
            amount = int(text_result["amount"])

            return ExpenseResponse(
                category=text_result["category"],
                category_confidence=text_result["category_confidence"],
                amount=amount,
                amount_formatted=format_idr(amount),
                date=datetime.now().strftime("%Y-%m-%d"),
                description=transcription["text"],
                merchant=text_result["entities"].get("merchant", "")
            )

        finally:
            # Clean up temp file
            os.unlink(tmp_path)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/expense/multimodal", response_model=ExpenseResponse)
async def extract_multimodal(
    text: Optional[str] = Form(None),
    image: Optional[UploadFile] = File(None),
    audio: Optional[UploadFile] = File(None),
    _: bool = Depends(verify_api_key)
):
    """
    Extract expense information from multiple modalities
    
    Accepts any combination of text, image, and audio
    """
    try:
        # Validate at least one input
        if not any([text, image, audio]):
            raise HTTPException(
                status_code=400,
                detail="At least one input (text, image, or audio) is required"
            )
        
        # Process image
        image_data = None
        if image:
            contents = await image.read()
            image_data = Image.open(io.BytesIO(contents))
        
        # Process audio
        audio_data = None
        if audio:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                contents = await audio.read()
                tmp.write(contents)
                audio_data = tmp.name
        
        try:
            # Extract with fusion
            result = multimodal_extractor.extract(
                text=text,
                image=image_data,
                audio=audio_data
            )
            
            # Get description from individual results
            description_parts = []
            if result.get("individual_results"):
                if result["individual_results"].get("text_result"):
                    description_parts.append(
                        result["individual_results"]["text_result"].get("processed_text", "")
                    )
                if result["individual_results"].get("audio_result"):
                    description_parts.append(
                        result["individual_results"]["audio_result"].get("transcription", "")
                    )
            
            description = " | ".join(filter(None, description_parts)) or text
            amount = int(result["amount"])

            return ExpenseResponse(
                category=result["category"],
                category_confidence=result["category_confidence"],
                amount=amount,
                amount_formatted=format_idr(amount),
                date=datetime.now().strftime("%Y-%m-%d"),
                description=description[:200] if description else None,
                modality_weights=result.get("modality_weights"),
                fusion_used=result.get("fusion_used", False)
            )

        finally:
            # Clean up audio temp file
            if audio_data and os.path.exists(audio_data):
                os.unlink(audio_data)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/categories", response_model=List[str])
async def get_categories():
    """Get list of supported expense categories"""
    return CATEGORIES


if __name__ == "__main__":
    import uvicorn
    
    print(f"Starting server on {API_HOST}:{API_PORT}")
    uvicorn.run(
        "main:app",
        host=API_HOST,
        port=API_PORT,
        reload=True
    )
