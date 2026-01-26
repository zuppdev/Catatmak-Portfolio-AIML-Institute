"""
Example script demonstrating multimodal expense tracking
"""

import requests
import json
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import io
import base64


# API base URL
API_URL = "http://localhost:8000"


def test_health():
    """Test health endpoint"""
    print("\n" + "="*50)
    print("Testing Health Endpoint")
    print("="*50)
    
    response = requests.get(f"{API_URL}/health")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    
    return response.status_code == 200


def test_text_extraction():
    """Test text-based expense extraction"""
    print("\n" + "="*50)
    print("Testing Text Extraction")
    print("="*50)
    
    test_cases = [
        "gw makan bakso 20rb di depan kantor",
        "beli bensin shell 50000 rupiah",
        "nonton bioskop sama temen 45 ribu",
        "bayar listrik bulan ini 250000",
        "parkir motor 5000",
        "beli kopi susu 15rb di starbucks"
    ]
    
    for text in test_cases:
        print(f"\nInput: {text}")
        
        response = requests.post(
            f"{API_URL}/expense/text",
            json={"text": text}
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"Category: {result['category']}")
            print(f"Confidence: {result['category_confidence']:.2%}")
            print(f"Amount: Rp {result['amount']:,.0f}")
            print(f"Description: {result['description']}")
        else:
            print(f"Error: {response.status_code} - {response.text}")


def create_sample_receipt():
    """Create a sample receipt image for testing"""
    # Create white background
    img = Image.new('RGB', (400, 600), color='white')
    draw = ImageDraw.Draw(img)
    
    # Try to use default font
    try:
        font_large = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
        font_medium = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
        font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    except:
        font_large = ImageFont.load_default()
        font_medium = ImageFont.load_default()
        font_small = ImageFont.load_default()
    
    # Draw receipt content
    y = 30
    
    # Header
    draw.text((50, y), "WARUNG BAKSO MALANG", fill='black', font=font_large)
    y += 40
    
    draw.text((50, y), "Jl. Sudirman No. 123", fill='black', font=font_small)
    y += 25
    
    draw.text((50, y), "Jakarta Pusat", fill='black', font=font_small)
    y += 40
    
    # Separator
    draw.line([(40, y), (360, y)], fill='black', width=2)
    y += 30
    
    # Items
    items = [
        ("Bakso Special", "20.000"),
        ("Es Teh Manis", "5.000"),
        ("Kerupuk", "2.000"),
    ]
    
    for item, price in items:
        draw.text((50, y), item, fill='black', font=font_medium)
        draw.text((280, y), f"Rp {price}", fill='black', font=font_medium)
        y += 30
    
    # Separator
    y += 10
    draw.line([(40, y), (360, y)], fill='black', width=2)
    y += 30
    
    # Total
    draw.text((50, y), "TOTAL", fill='black', font=font_large)
    draw.text((250, y), "Rp 27.000", fill='black', font=font_large)
    y += 50
    
    # Date
    draw.text((50, y), "25/01/2026  14:30", fill='black', font=font_small)
    y += 30
    
    # Footer
    draw.text((100, y), "Terima Kasih", fill='black', font=font_medium)
    
    return img


def test_image_extraction():
    """Test image-based expense extraction"""
    print("\n" + "="*50)
    print("Testing Image Extraction")
    print("="*50)
    
    # Create sample receipt
    receipt_img = create_sample_receipt()
    
    # Save to bytes
    img_bytes = io.BytesIO()
    receipt_img.save(img_bytes, format='PNG')
    img_bytes.seek(0)
    
    print("\nUploading sample receipt image...")
    
    response = requests.post(
        f"{API_URL}/expense/image",
        files={"file": ("receipt.png", img_bytes, "image/png")}
    )
    
    if response.status_code == 200:
        result = response.json()
        print(f"Category: {result['category']}")
        print(f"Confidence: {result['category_confidence']:.2%}")
        print(f"Amount: Rp {result['amount']:,.0f}")
        print(f"Merchant: {result.get('merchant', 'N/A')}")
        print(f"Date: {result.get('date', 'N/A')}")
        print(f"OCR Text Preview: {result.get('description', '')[:100]}...")
    else:
        print(f"Error: {response.status_code} - {response.text}")


def test_multimodal():
    """Test multimodal extraction"""
    print("\n" + "="*50)
    print("Testing Multimodal Extraction")
    print("="*50)
    
    # Create sample receipt
    receipt_img = create_sample_receipt()
    img_bytes = io.BytesIO()
    receipt_img.save(img_bytes, format='PNG')
    img_bytes.seek(0)
    
    # Combine text and image
    text = "tadi siang makan bakso enak banget 27 ribu"
    
    print(f"\nText: {text}")
    print("Image: Receipt uploaded")
    
    response = requests.post(
        f"{API_URL}/expense/multimodal",
        data={"text": text},
        files={"image": ("receipt.png", img_bytes, "image/png")}
    )
    
    if response.status_code == 200:
        result = response.json()
        print(f"\nCategory: {result['category']}")
        print(f"Confidence: {result['category_confidence']:.2%}")
        print(f"Amount: Rp {result['amount']:,.0f}")
        print(f"Fusion Used: {result['fusion_used']}")
        
        if result.get('modality_weights'):
            print("\nModality Weights:")
            for modality, weight in result['modality_weights'].items():
                print(f"  {modality}: {weight:.2%}")
    else:
        print(f"Error: {response.status_code} - {response.text}")


def test_categories():
    """Test categories endpoint"""
    print("\n" + "="*50)
    print("Testing Categories Endpoint")
    print("="*50)
    
    response = requests.get(f"{API_URL}/categories")
    
    if response.status_code == 200:
        categories = response.json()
        print(f"\nSupported Categories ({len(categories)}):")
        for i, cat in enumerate(categories, 1):
            print(f"  {i}. {cat}")
    else:
        print(f"Error: {response.status_code}")


def run_all_tests():
    """Run all test cases"""
    print("\n" + "="*60)
    print("MULTIMODAL EXPENSE TRACKER - API TESTING")
    print("="*60)
    
    try:
        # Test health first
        if not test_health():
            print("\n❌ API is not healthy. Please start the server first:")
            print("   python serving/api/main.py")
            return
        
        print("\n✓ API is healthy!")
        
        # Run tests
        test_categories()
        test_text_extraction()
        test_image_extraction()
        test_multimodal()
        
        print("\n" + "="*60)
        print("ALL TESTS COMPLETED")
        print("="*60)
        
    except requests.exceptions.ConnectionError:
        print("\n❌ Cannot connect to API. Please start the server first:")
        print("   python serving/api/main.py")
    except Exception as e:
        print(f"\n❌ Error running tests: {e}")


def interactive_demo():
    """Interactive demo for testing"""
    print("\n" + "="*60)
    print("INTERACTIVE EXPENSE TRACKER DEMO")
    print("="*60)
    
    while True:
        print("\nOptions:")
        print("1. Extract from text")
        print("2. Extract from image (sample receipt)")
        print("3. Extract multimodal (text + image)")
        print("4. View categories")
        print("5. Run all tests")
        print("0. Exit")
        
        choice = input("\nEnter choice: ").strip()
        
        if choice == "0":
            print("Goodbye!")
            break
        elif choice == "1":
            text = input("\nEnter expense text (Indonesian): ").strip()
            if text:
                response = requests.post(
                    f"{API_URL}/expense/text",
                    json={"text": text}
                )
                print(json.dumps(response.json(), indent=2))
        elif choice == "2":
            test_image_extraction()
        elif choice == "3":
            test_multimodal()
        elif choice == "4":
            test_categories()
        elif choice == "5":
            run_all_tests()
        else:
            print("Invalid choice")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "interactive":
        interactive_demo()
    else:
        run_all_tests()
