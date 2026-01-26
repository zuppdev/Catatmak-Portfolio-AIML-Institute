import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional
import numpy as np
from configs.config import MODELS, CATEGORIES, FUSION_WEIGHTS, CONFIDENCE_THRESHOLD


class MultimodalFusionModel(nn.Module):
    """
    Fusion model to combine text, image, and audio embeddings
    Uses attention mechanism to weight different modalities
    """
    
    def __init__(
        self,
        text_dim: int = 768,
        image_dim: int = 768,
        audio_dim: int = 768,
        hidden_dim: int = 256,
        num_categories: int = len(CATEGORIES)
    ):
        super().__init__()
        
        # Projection layers to common dimension
        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        self.image_proj = nn.Sequential(
            nn.Linear(image_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        self.audio_proj = nn.Sequential(
            nn.Linear(audio_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Cross-modal attention
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        # Fusion layers
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Category classifier
        self.category_head = nn.Linear(hidden_dim, num_categories)
        
        # Amount regressor
        self.amount_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1)
        )
        
        # Modality weight predictor (learns to weight modalities)
        self.modality_weights = nn.Sequential(
            nn.Linear(hidden_dim * 3, 64),
            nn.ReLU(),
            nn.Linear(64, 3),
            nn.Softmax(dim=-1)
        )
    
    def forward(
        self,
        text_emb: Optional[torch.Tensor] = None,
        image_emb: Optional[torch.Tensor] = None,
        audio_emb: Optional[torch.Tensor] = None,
        text_mask: Optional[torch.Tensor] = None,
        image_mask: Optional[torch.Tensor] = None,
        audio_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with optional modalities
        
        Args:
            text_emb: Text embeddings [batch, text_dim]
            image_emb: Image embeddings [batch, image_dim]
            audio_emb: Audio embeddings [batch, audio_dim]
            *_mask: Boolean masks indicating which modalities are present
            
        Returns:
            Dictionary with predictions and learned weights
        """
        batch_size = (
            text_emb.size(0) if text_emb is not None else
            image_emb.size(0) if image_emb is not None else
            audio_emb.size(0)
        )
        
        device = (
            text_emb.device if text_emb is not None else
            image_emb.device if image_emb is not None else
            audio_emb.device
        )
        
        # Project to common dimension
        projected = []
        masks = []
        
        if text_emb is not None and (text_mask is None or text_mask.any()):
            text_proj = self.text_proj(text_emb)
            projected.append(text_proj)
            masks.append(text_mask if text_mask is not None else torch.ones(batch_size, dtype=torch.bool, device=device))
        else:
            projected.append(torch.zeros(batch_size, self.text_proj[0].out_features, device=device))
            masks.append(torch.zeros(batch_size, dtype=torch.bool, device=device))
        
        if image_emb is not None and (image_mask is None or image_mask.any()):
            image_proj = self.image_proj(image_emb)
            projected.append(image_proj)
            masks.append(image_mask if image_mask is not None else torch.ones(batch_size, dtype=torch.bool, device=device))
        else:
            projected.append(torch.zeros(batch_size, self.image_proj[0].out_features, device=device))
            masks.append(torch.zeros(batch_size, dtype=torch.bool, device=device))
        
        if audio_emb is not None and (audio_mask is None or audio_mask.any()):
            audio_proj = self.audio_proj(audio_emb)
            projected.append(audio_proj)
            masks.append(audio_mask if audio_mask is not None else torch.ones(batch_size, dtype=torch.bool, device=device))
        else:
            projected.append(torch.zeros(batch_size, self.audio_proj[0].out_features, device=device))
            masks.append(torch.zeros(batch_size, dtype=torch.bool, device=device))
        
        # Stack modalities [batch, 3, hidden_dim]
        stacked = torch.stack(projected, dim=1)
        
        # Apply cross-modal attention
        attended, attention_weights = self.attention(
            stacked, stacked, stacked,
            need_weights=True
        )
        
        # Concatenate all modalities
        concatenated = torch.cat([
            attended[:, 0, :],  # text
            attended[:, 1, :],  # image
            attended[:, 2, :]   # audio
        ], dim=-1)
        
        # Predict modality weights
        learned_weights = self.modality_weights(concatenated)
        
        # Weighted combination
        weighted = (
            learned_weights[:, 0:1] * attended[:, 0, :] +
            learned_weights[:, 1:2] * attended[:, 1, :] +
            learned_weights[:, 2:3] * attended[:, 2, :]
        )
        
        # Fusion
        fused = self.fusion(concatenated)
        
        # Combine weighted and fused
        final = fused + weighted
        
        # Predictions
        category_logits = self.category_head(final)
        amount_pred = self.amount_head(final)
        
        return {
            "category_logits": category_logits,
            "amount_pred": amount_pred,
            "modality_weights": learned_weights,
            "attention_weights": attention_weights,
            "fused_embedding": final
        }


class MultimodalExpenseExtractor:
    """
    High-level interface for multimodal expense extraction
    Combines text, image, and audio models with fusion
    """
    
    def __init__(
        self,
        text_extractor=None,
        receipt_parser=None,
        audio_transcriber=None,
        fusion_model_path: Optional[str] = None
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize individual models
        self.text_extractor = text_extractor
        self.receipt_parser = receipt_parser
        self.audio_transcriber = audio_transcriber
        
        # Initialize fusion model
        self.fusion_model = MultimodalFusionModel(
            hidden_dim=MODELS["fusion"]["hidden_dim"]
        ).to(self.device)
        
        if fusion_model_path:
            self.fusion_model.load_state_dict(
                torch.load(fusion_model_path, map_location=self.device)
            )
        
        self.fusion_model.eval()
    
    def extract(
        self,
        text: Optional[str] = None,
        image = None,
        audio = None
    ) -> Dict:
        """
        Extract expense information from multiple modalities
        
        Args:
            text: Text description of expense
            image: Receipt image (PIL Image, numpy array, or path)
            audio: Audio recording (numpy array or path)
            
        Returns:
            Dictionary with fused expense information
        """
        results = {
            "text_result": None,
            "image_result": None,
            "audio_result": None
        }
        
        embeddings = {
            "text": None,
            "image": None,
            "audio": None
        }
        
        # Process text
        if text and self.text_extractor:
            text_result = self.text_extractor.predict(text)
            results["text_result"] = text_result
            embeddings["text"] = torch.tensor(
                text_result["embedding"], 
                device=self.device
            ).unsqueeze(0)
        
        # Process image
        if image is not None and self.receipt_parser:
            image_result = self.receipt_parser.predict(image)
            results["image_result"] = image_result
            if image_result["embedding"] is not None:
                embeddings["image"] = torch.tensor(
                    image_result["embedding"],
                    device=self.device
                ).unsqueeze(0)
        
        # Process audio
        if audio is not None and self.audio_transcriber:
            # Transcribe first
            transcription = self.audio_transcriber.transcribe(audio)
            
            # Then extract from transcription
            if self.text_extractor:
                audio_result = self.text_extractor.predict(transcription["text"])
                audio_result["transcription"] = transcription["text"]
                results["audio_result"] = audio_result
                embeddings["audio"] = torch.tensor(
                    audio_result["embedding"],
                    device=self.device
                ).unsqueeze(0)
        
        # Fusion if multiple modalities present
        if sum(e is not None for e in embeddings.values()) > 1:
            with torch.no_grad():
                fusion_output = self.fusion_model(
                    text_emb=embeddings["text"],
                    image_emb=embeddings["image"],
                    audio_emb=embeddings["audio"]
                )
            
            # Get fused predictions
            category_probs = torch.softmax(fusion_output["category_logits"], dim=-1)
            category_idx = category_probs.argmax(dim=-1).item()
            category_conf = category_probs[0, category_idx].item()
            fused_amount = fusion_output["amount_pred"].item()
            
            modality_weights = fusion_output["modality_weights"][0].cpu().numpy()
            
            # Aggregate amounts with confidence weighting
            amounts = []
            confidences = []
            
            if results["text_result"]:
                amounts.append(results["text_result"]["amount"])
                confidences.append(results["text_result"]["category_confidence"])
            
            if results["image_result"]:
                amounts.append(results["image_result"]["amount"])
                confidences.append(results["image_result"]["category_confidence"])
            
            if results["audio_result"]:
                amounts.append(results["audio_result"]["amount"])
                confidences.append(results["audio_result"]["category_confidence"])
            
            # Weighted average of amounts
            if amounts and sum(confidences) > 0:
                weighted_amount = sum(
                    a * c for a, c in zip(amounts, confidences)
                ) / sum(confidences)
            else:
                weighted_amount = fused_amount
            
            return {
                "category": CATEGORIES[category_idx],
                "category_confidence": category_conf,
                "amount": weighted_amount,
                "modality_weights": {
                    "text": float(modality_weights[0]),
                    "image": float(modality_weights[1]),
                    "audio": float(modality_weights[2])
                },
                "individual_results": results,
                "fusion_used": True
            }
        
        # Single modality - return that result
        else:
            for modality, result in results.items():
                if result:
                    return {
                        **result,
                        "fusion_used": False,
                        "primary_modality": modality.replace("_result", "")
                    }
        
        return {
            "category": "lainnya",
            "category_confidence": 0.0,
            "amount": 0.0,
            "error": "No valid input provided"
        }


if __name__ == "__main__":
    # Test fusion model
    print("Testing Multimodal Fusion Model...")
    
    # Create dummy embeddings
    batch_size = 2
    text_emb = torch.randn(batch_size, 768)
    image_emb = torch.randn(batch_size, 768)
    audio_emb = torch.randn(batch_size, 768)
    
    model = MultimodalFusionModel()
    
    output = model(
        text_emb=text_emb,
        image_emb=image_emb,
        audio_emb=audio_emb
    )
    
    print(f"Category logits shape: {output['category_logits'].shape}")
    print(f"Amount predictions shape: {output['amount_pred'].shape}")
    print(f"Modality weights: {output['modality_weights']}")
    print(f"Learned weights sum to 1: {output['modality_weights'].sum(dim=1)}")
