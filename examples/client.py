"""
Simple Python client for the Expense Tracker API
"""

import requests
from typing import Optional, Dict, Union
from pathlib import Path
import json


class ExpenseTrackerClient:
    """
    Client for interacting with the Expense Tracker API
    
    Example:
        client = ExpenseTrackerClient("http://localhost:8000")
        
        # Text input
        result = client.track_text("makan bakso 20rb")
        
        # Image input
        result = client.track_image("receipt.jpg")
        
        # Multimodal
        result = client.track_multimodal(
            text="makan bakso",
            image_path="receipt.jpg"
        )
    """
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
    
    def health_check(self) -> Dict:
        """Check API health status"""
        response = self.session.get(f"{self.base_url}/health")
        response.raise_for_status()
        return response.json()
    
    def get_categories(self) -> list:
        """Get list of supported categories"""
        response = self.session.get(f"{self.base_url}/categories")
        response.raise_for_status()
        return response.json()
    
    def track_text(self, text: str) -> Dict:
        """
        Track expense from text description
        
        Args:
            text: Indonesian text describing the expense
            
        Returns:
            Dictionary with expense information
            
        Example:
            result = client.track_text("beli bakso 20 ribu")
            print(f"Category: {result['category']}")
            print(f"Amount: Rp {result['amount']:,.0f}")
        """
        response = self.session.post(
            f"{self.base_url}/expense/text",
            json={"text": text}
        )
        response.raise_for_status()
        return response.json()
    
    def track_image(self, image_path: Union[str, Path]) -> Dict:
        """
        Track expense from receipt image
        
        Args:
            image_path: Path to receipt image
            
        Returns:
            Dictionary with expense information
            
        Example:
            result = client.track_image("receipt.jpg")
            print(f"Merchant: {result['merchant']}")
            print(f"Amount: Rp {result['amount']:,.0f}")
        """
        with open(image_path, 'rb') as f:
            files = {'file': f}
            response = self.session.post(
                f"{self.base_url}/expense/image",
                files=files
            )
        response.raise_for_status()
        return response.json()
    
    def track_audio(self, audio_path: Union[str, Path]) -> Dict:
        """
        Track expense from audio recording
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Dictionary with expense information
            
        Example:
            result = client.track_audio("recording.wav")
            print(f"Transcription: {result['description']}")
            print(f"Amount: Rp {result['amount']:,.0f}")
        """
        with open(audio_path, 'rb') as f:
            files = {'file': f}
            response = self.session.post(
                f"{self.base_url}/expense/audio",
                files=files
            )
        response.raise_for_status()
        return response.json()
    
    def track_multimodal(
        self,
        text: Optional[str] = None,
        image_path: Optional[Union[str, Path]] = None,
        audio_path: Optional[Union[str, Path]] = None
    ) -> Dict:
        """
        Track expense using multiple modalities
        
        Args:
            text: Optional text description
            image_path: Optional path to receipt image
            audio_path: Optional path to audio file
            
        Returns:
            Dictionary with fused expense information
            
        Example:
            result = client.track_multimodal(
                text="makan bakso enak",
                image_path="receipt.jpg"
            )
            
            print(f"Modality weights: {result['modality_weights']}")
            print(f"Amount: Rp {result['amount']:,.0f}")
        """
        data = {}
        files = {}
        
        if text:
            data['text'] = text
        
        if image_path:
            files['image'] = open(image_path, 'rb')
        
        if audio_path:
            files['audio'] = open(audio_path, 'rb')
        
        try:
            response = self.session.post(
                f"{self.base_url}/expense/multimodal",
                data=data,
                files=files
            )
            response.raise_for_status()
            return response.json()
        finally:
            # Close file handles
            for f in files.values():
                f.close()
    
    def batch_track_text(self, texts: list) -> list:
        """
        Track multiple expenses from text (batch processing)
        
        Args:
            texts: List of text descriptions
            
        Returns:
            List of expense dictionaries
        """
        results = []
        for text in texts:
            try:
                result = self.track_text(text)
                results.append(result)
            except Exception as e:
                results.append({"error": str(e), "text": text})
        return results
    
    def format_expense(self, expense: Dict) -> str:
        """
        Format expense dictionary as readable string
        
        Args:
            expense: Expense dictionary from API
            
        Returns:
            Formatted string
        """
        lines = [
            f"Category: {expense['category']}",
            f"Amount: Rp {expense['amount']:,.0f}",
            f"Confidence: {expense['category_confidence']:.1%}",
        ]
        
        if expense.get('merchant'):
            lines.append(f"Merchant: {expense['merchant']}")
        
        if expense.get('date'):
            lines.append(f"Date: {expense['date']}")
        
        if expense.get('description'):
            lines.append(f"Description: {expense['description'][:50]}...")
        
        if expense.get('modality_weights'):
            weights = expense['modality_weights']
            lines.append(f"Modality Weights: text={weights['text']:.2f}, "
                        f"image={weights['image']:.2f}, audio={weights['audio']:.2f}")
        
        return "\n".join(lines)


# Example usage
def main():
    """Example usage of the client"""
    
    # Initialize client
    client = ExpenseTrackerClient("http://localhost:8000")
    
    # Check health
    print("Checking API health...")
    health = client.health_check()
    print(f"Status: {health['status']}")
    print()
    
    # Get categories
    print("Supported categories:")
    categories = client.get_categories()
    for cat in categories:
        print(f"  - {cat}")
    print()
    
    # Test text tracking
    print("="*50)
    print("TEXT TRACKING")
    print("="*50)
    
    text_examples = [
        "gw makan bakso 20rb di depan kantor",
        "beli bensin 50000 di shell",
        "nonton film avengers 45 ribu",
    ]
    
    for text in text_examples:
        print(f"\nInput: {text}")
        result = client.track_text(text)
        print(client.format_expense(result))
    
    # Batch processing
    print("\n" + "="*50)
    print("BATCH PROCESSING")
    print("="*50)
    
    batch_texts = [
        "parkir motor 5000",
        "makan siang 35rb",
        "bayar listrik 200000"
    ]
    
    results = client.batch_track_text(batch_texts)
    
    total = sum(r['amount'] for r in results if 'amount' in r)
    print(f"\nProcessed {len(results)} expenses")
    print(f"Total amount: Rp {total:,.0f}")
    
    # Export to JSON
    with open('expense_report.json', 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print("\nExported to expense_report.json")


if __name__ == "__main__":
    main()
