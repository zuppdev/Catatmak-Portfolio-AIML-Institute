from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import numpy as np
from PIL import Image
import io
import base64
import tempfile
import os
from datetime import datetime

from models.text_extractor import TextExtractor
from models.receipt_parser import ReceiptParser
from models.audio_transcriber import AudioTranscriber
from models.fusion_model import MultimodalExpenseExtractor
from configs.config import API_HOST, API_PORT, CATEGORIES


# Initialize FastAPI app
app = FastAPI(
    title="Multimodal Expense Tracker API",
    description="Indonesian language expense tracking with text, voice, and image support",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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


@app.post("/expense/text", response_model=ExpenseResponse)
async def extract_from_text(request: TextExpenseRequest):
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
async def extract_from_image(file: UploadFile = File(...)):
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
async def extract_from_audio(file: UploadFile = File(...)):
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
    audio: Optional[UploadFile] = File(None)
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
