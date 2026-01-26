import torch
import torch.nn as nn
from transformers import LayoutLMv3Processor, LayoutLMv3Model
from PIL import Image
import easyocr
import cv2
import numpy as np
from typing import Dict, List, Tuple
import re
from configs.config import MODELS, CATEGORIES, AMOUNT_PATTERNS


class ReceiptParserModel(nn.Module):
    """
    LayoutLMv3-based model for Indonesian receipt parsing
    """
    
    def __init__(self, num_categories: int):
        super().__init__()
        
        model_name = MODELS["image"]["name"]
        self.layoutlm = LayoutLMv3Model.from_pretrained(model_name)
        self.processor = LayoutLMv3Processor.from_pretrained(
            model_name,
            apply_ocr=False  # We'll use our own OCR
        )
        
        hidden_size = self.layoutlm.config.hidden_size
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_categories)
        )
        
        # Key information extraction head
        self.key_info_head = nn.Linear(hidden_size, 4)  # [amount, date, merchant, item]
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        bbox: torch.Tensor,
        pixel_values: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Forward pass"""
        outputs = self.layoutlm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            bbox=bbox,
            pixel_values=pixel_values
        )
        
        sequence_output = outputs.last_hidden_state
        pooled_output = sequence_output[:, 0, :]  # CLS token
        
        # Category classification
        category_logits = self.classifier(pooled_output)
        
        # Key information extraction
        key_info_logits = self.key_info_head(sequence_output)
        
        return {
            "category_logits": category_logits,
            "key_info_logits": key_info_logits,
            "pooled_output": pooled_output
        }


class ReceiptParser:
    """
    High-level interface for receipt parsing
    """
    
    def __init__(self, model_path: str = None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize OCR
        self.ocr_reader = easyocr.Reader(['id', 'en'], gpu=torch.cuda.is_available())
        
        # Initialize model
        self.model = ReceiptParserModel(num_categories=len(CATEGORIES)).to(self.device)
        
        if model_path:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        
        self.model.eval()
        self.processor = self.model.processor
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocess receipt image for better OCR"""
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        
        # Denoise
        denoised = cv2.fastNlMeansDenoising(thresh)
        
        # Enhance contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(denoised)
        
        return enhanced
    
    def perform_ocr(self, image: np.ndarray) -> Tuple[List[str], List[List[int]]]:
        """
        Perform OCR on receipt image
        
        Returns:
            words: List of detected words
            boxes: List of bounding boxes [x1, y1, x2, y2]
        """
        # Preprocess
        processed = self.preprocess_image(image)
        
        # Run OCR
        results = self.ocr_reader.readtext(processed)
        
        words = []
        boxes = []
        
        for bbox, text, confidence in results:
            if confidence > 0.3:  # Filter low confidence
                words.append(text)
                
                # Convert bbox format
                x_coords = [point[0] for point in bbox]
                y_coords = [point[1] for point in bbox]
                box = [
                    int(min(x_coords)),
                    int(min(y_coords)),
                    int(max(x_coords)),
                    int(max(y_coords))
                ]
                boxes.append(box)
        
        return words, boxes
    
    def normalize_bbox(
        self, 
        boxes: List[List[int]], 
        image_width: int, 
        image_height: int
    ) -> List[List[int]]:
        """Normalize bounding boxes to 0-1000 range for LayoutLM"""
        normalized = []
        for box in boxes:
            x1, y1, x2, y2 = box
            normalized_box = [
                int(1000 * x1 / image_width),
                int(1000 * y1 / image_height),
                int(1000 * x2 / image_width),
                int(1000 * y2 / image_height)
            ]
            normalized.append(normalized_box)
        return normalized
    
    def extract_amount_from_text(self, words: List[str]) -> float:
        """Extract amount from OCR text using patterns"""
        full_text = " ".join(words)
        
        for pattern in AMOUNT_PATTERNS:
            matches = re.findall(pattern, full_text, re.IGNORECASE)
            if matches:
                # Get the largest amount (likely the total)
                amounts = []
                for match in matches:
                    try:
                        amount_str = match.replace('.', '').replace(',', '')
                        amounts.append(float(amount_str))
                    except ValueError:
                        continue
                
                if amounts:
                    return max(amounts)
        
        return 0.0
    
    def extract_date(self, words: List[str]) -> str:
        """Extract date from OCR text"""
        full_text = " ".join(words)
        
        # Common Indonesian date patterns
        date_patterns = [
            r'\d{1,2}[-/]\d{1,2}[-/]\d{2,4}',
            r'\d{1,2}\s+(?:Jan|Feb|Mar|Apr|Mei|Jun|Jul|Agt|Sep|Okt|Nov|Des)\s+\d{2,4}',
            r'\d{2,4}[-/]\d{1,2}[-/]\d{1,2}'
        ]
        
        for pattern in date_patterns:
            match = re.search(pattern, full_text, re.IGNORECASE)
            if match:
                return match.group(0)
        
        return ""
    
    def extract_merchant(self, words: List[str]) -> str:
        """Extract merchant name (usually at top of receipt)"""
        if len(words) > 0:
            # Take first few words as merchant name
            merchant_words = words[:min(3, len(words))]
            return " ".join(merchant_words)
        return ""
    
    def predict(self, image_input) -> Dict:
        """
        Parse receipt image and extract information
        
        Args:
            image_input: PIL Image, numpy array, or path to image
            
        Returns:
            Dictionary with extracted information
        """
        # Load image
        if isinstance(image_input, str):
            image = Image.open(image_input).convert("RGB")
            image_np = np.array(image)
        elif isinstance(image_input, Image.Image):
            image = image_input.convert("RGB")
            image_np = np.array(image)
        else:
            image_np = image_input
            image = Image.fromarray(cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB))
        
        # Perform OCR
        words, boxes = self.perform_ocr(image_np)
        
        if not words:
            return {
                "category": "lainnya",
                "category_confidence": 0.0,
                "amount": 0.0,
                "merchant": "",
                "date": "",
                "ocr_text": "",
                "embedding": None
            }
        
        # Extract information using regex
        amount = self.extract_amount_from_text(words)
        date = self.extract_date(words)
        merchant = self.extract_merchant(words)
        
        # Prepare model input
        height, width = image_np.shape[:2]
        normalized_boxes = self.normalize_bbox(boxes, width, height)
        
        # Encode with LayoutLM
        encoding = self.processor(
            image,
            words,
            boxes=normalized_boxes,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MODELS["image"]["max_length"]
        )
        
        # Move to device
        for k, v in encoding.items():
            if isinstance(v, torch.Tensor):
                encoding[k] = v.to(self.device)
        
        # Get predictions
        with torch.no_grad():
            outputs = self.model(**encoding)
        
        # Extract category
        category_probs = torch.softmax(outputs["category_logits"], dim=-1)
        category_idx = category_probs.argmax(dim=-1).item()
        category_conf = category_probs[0, category_idx].item()
        
        return {
            "category": CATEGORIES[category_idx],
            "category_confidence": category_conf,
            "amount": amount,
            "merchant": merchant,
            "date": date,
            "ocr_text": " ".join(words),
            "num_words": len(words),
            "embedding": outputs["pooled_output"].cpu().numpy()[0]
        }


if __name__ == "__main__":
    # Test the parser
    parser = ReceiptParser()
    
    # Create a dummy receipt image for testing
    img = np.ones((600, 400, 3), dtype=np.uint8) * 255
    
    # Add some text (this would be a real receipt in practice)
    cv2.putText(img, "Warung Bakso Malang", (50, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    cv2.putText(img, "Bakso Special", (50, 100), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    cv2.putText(img, "Total: Rp 20.000", (50, 150), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    cv2.putText(img, "25/01/2026", (50, 200), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    
    result = parser.predict(img)
    print("\nReceipt Parsing Result:")
    print(f"Category: {result['category']} ({result['category_confidence']:.2f})")
    print(f"Amount: Rp {result['amount']:,.0f}")
    print(f"Merchant: {result['merchant']}")
    print(f"Date: {result['date']}")
    print(f"OCR Text: {result['ocr_text'][:100]}...")
