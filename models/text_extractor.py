import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, AutoConfig
from typing import Dict, List, Tuple
import re
from configs.config import MODELS, CATEGORIES, ENTITY_TYPES, AMOUNT_PATTERNS, AMOUNT_SHORTHAND_THRESHOLD


class TextExtractorModel(nn.Module):
    """
    Multi-task model for Indonesian text:
    1. Named Entity Recognition (NER) for amount, item, merchant
    2. Category classification
    """
    
    def __init__(self, num_categories: int, num_entities: int):
        super().__init__()
        
        model_name = MODELS["text"]["name"]
        self.config = AutoConfig.from_pretrained(model_name)
        self.bert = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        hidden_size = self.config.hidden_size
        
        # NER head
        self.ner_dropout = nn.Dropout(0.3)
        self.ner_classifier = nn.Linear(hidden_size, num_entities)
        
        # Category classification head
        self.category_dropout = nn.Dropout(0.3)
        self.category_classifier = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_categories)
        )
        
        # Amount extraction head (regression)
        self.amount_head = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass

        Args:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            token_type_ids: Token type IDs [batch_size, seq_len] (optional, ignored)

        Returns:
            Dictionary with NER logits, category logits, and amount predictions
        """
        # Get BERT outputs (token_type_ids not passed to avoid compatibility issues)
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        sequence_output = outputs.last_hidden_state  # [batch, seq_len, hidden]
        pooled_output = outputs.pooler_output  # [batch, hidden]
        
        # NER predictions (token-level)
        ner_logits = self.ner_classifier(
            self.ner_dropout(sequence_output)
        )
        
        # Category predictions (sequence-level)
        category_logits = self.category_classifier(
            self.category_dropout(pooled_output)
        )
        
        # Amount prediction (sequence-level)
        amount_pred = self.amount_head(pooled_output)
        
        return {
            "ner_logits": ner_logits,
            "category_logits": category_logits,
            "amount_pred": amount_pred,
            "pooled_output": pooled_output
        }


class TextExtractor:
    """
    High-level interface for text extraction
    """
    
    def __init__(self, model_path: str = None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.model = TextExtractorModel(
            num_categories=len(CATEGORIES),
            num_entities=len(ENTITY_TYPES)
        ).to(self.device)
        
        if model_path:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        
        self.model.eval()
        self.tokenizer = self.model.tokenizer
    
    def preprocess_text(self, text: str) -> str:
        """Clean and normalize Indonesian text"""
        # Lowercase
        text = text.lower()
        
        # Normalize common abbreviations
        replacements = {
            "gw": "saya",
            "gue": "saya",
            "lu": "kamu",
            "org": "orang",
            "yg": "yang",
            "dgn": "dengan",
            "utk": "untuk",
            "jd": "jadi",
            "tdk": "tidak",
            "sdh": "sudah",
            "blm": "belum"
        }
        
        for abbr, full in replacements.items():
            text = re.sub(r'\b' + abbr + r'\b', full, text)
        
        # Normalize currency
        text = re.sub(r'\b(\d+)k\b', r'\1000', text)
        text = re.sub(r'\b(\d+)rb\b', r'\1000', text)
        text = re.sub(r'\b(\d+)jt\b', r'\1000000', text)
        
        return text.strip()
    
    def extract_amount_regex(self, text: str) -> float:
        """Extract amount using regex patterns"""
        for pattern in AMOUNT_PATTERNS:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                amount_str = matches[0]
                # Remove formatting
                amount_str = amount_str.replace('.', '').replace(',', '')
                try:
                    return float(amount_str)
                except ValueError:
                    continue
        return 0.0
    
    def predict(self, text: str) -> Dict:
        """
        Extract expense information from text
        
        Args:
            text: Indonesian text describing expense
            
        Returns:
            Dictionary with extracted information
        """
        # Preprocess
        processed_text = self.preprocess_text(text)
        
        # Tokenize
        inputs = self.tokenizer(
            processed_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MODELS["text"]["max_length"]
        ).to(self.device)
        
        # Get predictions
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Extract category
        category_probs = torch.softmax(outputs["category_logits"], dim=-1)
        category_idx = category_probs.argmax(dim=-1).item()
        category_conf = category_probs[0, category_idx].item()
        
        # Extract NER entities
        ner_probs = torch.softmax(outputs["ner_logits"], dim=-1)
        ner_preds = ner_probs.argmax(dim=-1)[0]  # [seq_len]
        
        # Decode entities
        tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        entities = self._decode_entities(tokens, ner_preds.cpu().numpy())
        
        # Extract amount (combine model and regex)
        model_amount = outputs["amount_pred"].item()
        regex_amount = self.extract_amount_regex(text)
        amount = regex_amount if regex_amount > 0 else model_amount

        # Apply shorthand multiplier for suspiciously low amounts
        # In Indonesian context, "20" for food typically means "20k" (20,000 IDR)
        # Check if text already had explicit multiplier (k, rb, ribu, jt, juta, Rp)
        has_explicit_multiplier = bool(re.search(
            r'\d+\s*(?:k|rb|ribu|jt|juta)\b|Rp\.?\s*\d|IDR\s*\d|\d{1,3}[.,]\d{3}',
            text.lower()
        ))

        if amount > 0 and amount < AMOUNT_SHORTHAND_THRESHOLD and not has_explicit_multiplier:
            amount = amount * 1000
        
        return {
            "text": text,
            "processed_text": processed_text,
            "category": CATEGORIES[category_idx],
            "category_confidence": category_conf,
            "amount": amount,
            "entities": entities,
            "embedding": outputs["pooled_output"].cpu().numpy()[0]
        }
    
    def _decode_entities(
        self, 
        tokens: List[str], 
        predictions: List[int]
    ) -> Dict[str, str]:
        """Decode BIO tags to entities"""
        entities = {
            "item": [],
            "merchant": [],
            "date": []
        }
        
        current_entity = None
        current_type = None
        
        for token, pred_idx in zip(tokens, predictions):
            if token in ['[CLS]', '[SEP]', '[PAD]']:
                continue
            
            tag = ENTITY_TYPES[pred_idx]
            
            if tag.startswith("B-"):
                # Save previous entity
                if current_entity:
                    entity_type = current_type.lower()
                    if entity_type in entities:
                        entities[entity_type].append(" ".join(current_entity))
                
                # Start new entity
                current_type = tag.split("-")[1]
                current_entity = [token.replace("##", "")]
                
            elif tag.startswith("I-") and current_entity:
                # Continue entity
                current_entity.append(token.replace("##", ""))
                
            else:
                # Save and reset
                if current_entity:
                    entity_type = current_type.lower()
                    if entity_type in entities:
                        entities[entity_type].append(" ".join(current_entity))
                current_entity = None
                current_type = None
        
        # Save last entity
        if current_entity:
            entity_type = current_type.lower()
            if entity_type in entities:
                entities[entity_type].append(" ".join(current_entity))
        
        # Join lists to strings
        return {k: ", ".join(v) if v else "" for k, v in entities.items()}


if __name__ == "__main__":
    # Test the model
    extractor = TextExtractor()
    
    test_cases = [
        "gw makan bakso 20rb di depan kantor",
        "beli bensin shell 50000",
        "nonton bioskop 45 ribu sama temen",
        "bayar listrik 200000 bulan ini"
    ]
    
    for text in test_cases:
        result = extractor.predict(text)
        print(f"\nInput: {text}")
        print(f"Category: {result['category']} ({result['category_confidence']:.2f})")
        print(f"Amount: Rp {result['amount']:,.0f}")
        print(f"Entities: {result['entities']}")
