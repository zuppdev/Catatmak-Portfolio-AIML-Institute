import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
import numpy as np
from typing import Dict, List
import mlflow
from tqdm import tqdm
import json
from pathlib import Path

from models.text_extractor import TextExtractorModel
from configs.config import MODELS, CATEGORIES, ENTITY_TYPES, MODEL_DIR


class ExpenseTextDataset(Dataset):
    """Dataset for expense text"""
    
    def __init__(self, data_path: str, tokenizer, max_length: int = 128):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Load data
        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Tokenize text
        encoding = self.tokenizer(
            item['text'],
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # Prepare NER labels
        ner_labels = item.get('ner_labels', [0] * self.max_length)
        ner_labels = ner_labels[:self.max_length] + [0] * (self.max_length - len(ner_labels))
        
        # Category label
        category_label = CATEGORIES.index(item['category'])
        
        # Amount
        amount = float(item.get('amount', 0))
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'ner_labels': torch.tensor(ner_labels, dtype=torch.long),
            'category_label': torch.tensor(category_label, dtype=torch.long),
            'amount': torch.tensor([amount], dtype=torch.float)
        }


def create_synthetic_data(num_samples: int = 5000) -> List[Dict]:
    """Create synthetic training data with diverse Indonesian patterns"""

    # Expanded templates with informal Indonesian
    templates = {
        "makanan": [
            "makan {item} {amount}",
            "beli {item} {amount}",
            "jajan {item} {amount}",
            "{item} {amount}",
            "sarapan {item} {amount}",
            "makan siang {item} {amount}",
            "makan malam {item} {amount}",
            "lunch {item} {amount}",
            "dinner {item} {amount}",
            "breakfast {item} {amount}",
            "beli makan {item} {amount}",
            "pesan {item} {amount}",
            "order {item} {amount}",
            "gofood {item} {amount}",
            "grabfood {item} {amount}",
            "makan di {place} {amount}",
            "makan {item} di {place} {amount}",
            "{item} di {place} {amount}",
            "ngopi {amount}",
            "ngopi di {place} {amount}",
            "beli kopi {amount}",
            "kopi {amount}",
            "minum {item} {amount}",
        ],
        "transportasi": [
            "bensin {amount}",
            "isi bensin {amount}",
            "beli bensin {amount}",
            "pertamax {amount}",
            "pertalite {amount}",
            "solar {amount}",
            "parkir {amount}",
            "bayar parkir {amount}",
            "parkir motor {amount}",
            "parkir mobil {amount}",
            "grab {amount}",
            "gojek {amount}",
            "ojol {amount}",
            "ojek online {amount}",
            "taxi {amount}",
            "taksi {amount}",
            "uber {amount}",
            "maxim {amount}",
            "naik {vehicle} {amount}",
            "ongkos {vehicle} {amount}",
            "tiket {vehicle} {amount}",
            "tol {amount}",
            "bayar tol {amount}",
            "e-toll {amount}",
        ],
        "belanja": [
            "beli {item} {amount}",
            "belanja {item} {amount}",
            "shopping {item} {amount}",
            "beli {item} di {place} {amount}",
            "{item} {amount}",
            "beli oleh-oleh {amount}",
            "beli souvenir {amount}",
            "beli kado {amount}",
            "beli hadiah {amount}",
        ],
        "tagihan": [
            "bayar {bill} {amount}",
            "{bill} {amount}",
            "bayar tagihan {bill} {amount}",
            "lunasi {bill} {amount}",
            "transfer {bill} {amount}",
            "bayar cicilan {amount}",
            "bayar kredit {amount}",
            "bayar hutang {amount}",
        ],
        "hiburan": [
            "nonton {item} {amount}",
            "nonton film {amount}",
            "bioskop {amount}",
            "tiket bioskop {amount}",
            "karaoke {amount}",
            "main {item} {amount}",
            "beli game {amount}",
            "langganan {item} {amount}",
            "netflix {amount}",
            "spotify {amount}",
            "youtube premium {amount}",
            "disney+ {amount}",
            "konser {amount}",
            "tiket konser {amount}",
            "wisata {amount}",
            "piknik {amount}",
            "jalan-jalan {amount}",
            "liburan {amount}",
        ],
        "kesehatan": [
            "beli obat {amount}",
            "obat {amount}",
            "apotek {amount}",
            "dokter {amount}",
            "periksa {amount}",
            "medical checkup {amount}",
            "vitamin {amount}",
            "beli vitamin {amount}",
            "rumah sakit {amount}",
            "klinik {amount}",
            "tes lab {amount}",
            "gigi {amount}",
            "dokter gigi {amount}",
        ],
        "pendidikan": [
            "bayar kursus {amount}",
            "kursus {item} {amount}",
            "les {item} {amount}",
            "beli buku {amount}",
            "buku {amount}",
            "bayar SPP {amount}",
            "SPP {amount}",
            "uang kuliah {amount}",
            "bimbel {amount}",
            "pelatihan {amount}",
            "seminar {amount}",
            "workshop {amount}",
            "kelas online {amount}",
            "udemy {amount}",
            "coursera {amount}",
            "beli alat tulis {amount}",
        ],
    }

    # Expanded items per category
    items = {
        "makanan": [
            "bakso", "mie ayam", "nasi goreng", "sate", "gado-gado",
            "soto", "rawon", "rendang", "nasi padang", "ayam geprek",
            "ayam goreng", "bebek goreng", "ikan bakar", "seafood",
            "pizza", "burger", "kentang goreng", "ayam kfc", "mcd",
            "ramen", "sushi", "dimsum", "martabak", "pempek",
            "bubur ayam", "lontong sayur", "ketoprak", "siomay",
            "batagor", "cilok", "cireng", "gorengan", "es teh",
            "es jeruk", "jus", "kopi", "teh", "susu", "boba",
            "chatime", "kopi kenangan", "starbucks", "janji jiwa"
        ],
        "belanja": [
            "baju", "celana", "sepatu", "tas", "jaket", "kaos",
            "kemeja", "dress", "rok", "sandal", "topi", "jam tangan",
            "kacamata", "dompet", "ikat pinggang", "aksesoris",
            "kosmetik", "skincare", "parfum", "sabun", "shampoo",
            "hp", "laptop", "charger", "earphone", "case hp",
            "peralatan dapur", "furniture", "dekorasi"
        ],
        "hiburan": [
            "film", "konser", "game", "netflix", "spotify", "disney+",
            "playstation", "xbox", "nintendo", "steam"
        ],
        "pendidikan": [
            "bahasa inggris", "matematika", "programming", "coding",
            "musik", "gitar", "piano", "menggambar", "desain"
        ],
    }

    # Places
    places = [
        "warung", "resto", "restoran", "cafe", "kafe", "mall",
        "kantor", "kampus", "sekolah", "rumah", "kosan", "apartemen",
        "pinggir jalan", "depan kantor", "dekat rumah", "samping kampus",
        "indomaret", "alfamart", "supermarket", "pasar", "tokopedia",
        "shopee", "lazada", "bukalapak", "mcd", "kfc", "starbucks"
    ]

    # Vehicles
    vehicles = ["bus", "kereta", "pesawat", "kapal", "angkot", "mrt", "lrt", "krl", "transjakarta"]

    # Bills
    bills = [
        "listrik", "air", "internet", "wifi", "pulsa", "paket data",
        "gas", "pln", "pdam", "telkom", "indihome", "tv kabel",
        "asuransi", "pajak", "iuran", "sewa", "kontrakan", "kos"
    ]

    # Amount formats - more variety
    def format_amount(amount):
        formats = [
            f"Rp {amount:,}".replace(',', '.'),
            f"Rp{amount:,}".replace(',', '.'),
            f"Rp. {amount:,}".replace(',', '.'),
            f"{amount:,}".replace(',', '.'),
            f"{amount}",
            f"{amount // 1000}k" if amount >= 1000 else f"{amount}",
            f"{amount // 1000}rb" if amount >= 1000 else f"{amount}",
            f"{amount // 1000} ribu" if amount >= 1000 else f"{amount}",
            f"{amount // 1000000}jt" if amount >= 1000000 else (f"{amount // 1000}rb" if amount >= 1000 else f"{amount}"),
        ]
        return np.random.choice(formats)

    # Informal prefixes (slang)
    prefixes = [
        "", "", "",  # Empty prefix most common
        "gw ", "gue ", "gua ",  # I (informal Jakarta)
        "aku ", "saya ",  # I (formal)
        "abis ", "habis ", "baru ",  # just did
        "td ", "tadi ",  # earlier
        "lg ", "lagi ",  # currently
        "mau ", "pengen ", "pingin ",  # want to
        "udah ", "sudah ",  # already
    ]

    # Informal suffixes
    suffixes = [
        "", "", "",  # Empty suffix most common
        " nih", " dong", " deh", " sih", " lho", " ya",
        " tadi", " barusan", " kemarin", " td pagi", " td siang", " td malam",
        " sama temen", " sama pacar", " sama keluarga", " sendirian",
        " buat makan", " buat jajan", " buat bensin",
    ]

    amounts = [
        5000, 8000, 10000, 12000, 15000, 18000, 20000, 22000, 25000,
        28000, 30000, 35000, 40000, 45000, 50000, 55000, 60000, 65000,
        70000, 75000, 80000, 85000, 90000, 95000, 100000, 120000,
        150000, 175000, 200000, 250000, 300000, 350000, 400000, 500000,
        750000, 1000000, 1500000, 2000000, 2500000, 3000000, 5000000
    ]

    data = []

    for _ in range(num_samples):
        # Pick category and template
        category = np.random.choice(list(templates.keys()))
        template = np.random.choice(templates[category])

        # Get item if needed
        if "{item}" in template:
            category_items = items.get(category, ["sesuatu"])
            item = np.random.choice(category_items)
        else:
            item = ""

        # Get place if needed
        place = np.random.choice(places) if "{place}" in template else ""

        # Get vehicle if needed
        vehicle = np.random.choice(vehicles) if "{vehicle}" in template else ""

        # Get bill if needed
        bill = np.random.choice(bills) if "{bill}" in template else ""

        # Get amount
        amount = int(np.random.choice(amounts))
        amount_text = format_amount(amount)

        # Build text
        text = template.format(
            item=item,
            amount=amount_text,
            place=place,
            vehicle=vehicle,
            bill=bill
        )

        # Add prefix and suffix randomly
        if np.random.random() < 0.4:
            text = np.random.choice(prefixes) + text
        if np.random.random() < 0.3:
            text = text + np.random.choice(suffixes)

        # Clean up extra spaces
        text = ' '.join(text.split())

        data.append({
            "text": text,
            "category": category,
            "amount": amount,
            "ner_labels": [0] * 128  # Simplified for demo
        })

    return data


def train_text_extractor(
    train_data_path: str,
    val_data_path: str = None,
    epochs: int = 10,
    batch_size: int = 32,
    learning_rate: float = 2e-5,
    output_dir: str = None
):
    """
    Train the text extractor model
    
    Args:
        train_data_path: Path to training data JSON
        val_data_path: Path to validation data JSON
        epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        output_dir: Directory to save model
    """
    
    if output_dir is None:
        output_dir = MODEL_DIR / "text_extractor"
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Initialize MLflow
    mlflow.set_experiment("text-extractor-training")
    
    with mlflow.start_run():
        # Log parameters
        mlflow.log_params({
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "model": MODELS["text"]["name"]
        })
        
        # Device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        
        # Initialize model
        model = TextExtractorModel(
            num_categories=len(CATEGORIES),
            num_entities=len(ENTITY_TYPES)
        ).to(device)
        
        tokenizer = model.tokenizer
        
        # Create datasets
        train_dataset = ExpenseTextDataset(train_data_path, tokenizer)
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True
        )
        
        if val_data_path:
            val_dataset = ExpenseTextDataset(val_data_path, tokenizer)
            val_loader = DataLoader(val_dataset, batch_size=batch_size)
        else:
            val_loader = None
        
        # Optimizer and scheduler
        optimizer = AdamW(model.parameters(), lr=learning_rate)
        
        total_steps = len(train_loader) * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=total_steps // 10,
            num_training_steps=total_steps
        )
        
        # Compute class weights for balanced training
        with open(train_data_path, 'r', encoding='utf-8') as f:
            train_data_raw = json.load(f)
        category_counts = {}
        for item in train_data_raw:
            cat = item['category']
            category_counts[cat] = category_counts.get(cat, 0) + 1
        total_samples = len(train_data_raw)
        num_classes = len(CATEGORIES)
        class_weights = []
        for cat in CATEGORIES:
            count = category_counts.get(cat, 1)
            weight = total_samples / (num_classes * count)
            class_weights.append(weight)
        class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
        print(f"Class weights: {dict(zip(CATEGORIES, class_weights.tolist()))}")

        # Loss functions with class weights
        ner_criterion = nn.CrossEntropyLoss(ignore_index=0)
        category_criterion = nn.CrossEntropyLoss(weight=class_weights)
        amount_criterion = nn.MSELoss()
        
        # Training loop
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            model.train()
            total_loss = 0
            total_ner_loss = 0
            total_category_loss = 0
            total_amount_loss = 0
            
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
            
            for batch in progress_bar:
                # Move to device
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                ner_labels = batch['ner_labels'].to(device)
                category_labels = batch['category_label'].to(device)
                amounts = batch['amount'].to(device)
                
                # Forward pass
                outputs = model(input_ids, attention_mask)
                
                # Calculate losses
                ner_loss = ner_criterion(
                    outputs['ner_logits'].view(-1, len(ENTITY_TYPES)),
                    ner_labels.view(-1)
                )
                
                category_loss = category_criterion(
                    outputs['category_logits'],
                    category_labels
                )
                
                amount_loss = amount_criterion(
                    outputs['amount_pred'],
                    amounts
                )
                
                # Combined loss
                loss = ner_loss + category_loss + 0.1 * amount_loss
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                
                # Track losses
                total_loss += loss.item()
                total_ner_loss += ner_loss.item()
                total_category_loss += category_loss.item()
                total_amount_loss += amount_loss.item()
                
                progress_bar.set_postfix({'loss': loss.item()})
            
            # Average losses
            avg_loss = total_loss / len(train_loader)
            avg_ner_loss = total_ner_loss / len(train_loader)
            avg_category_loss = total_category_loss / len(train_loader)
            avg_amount_loss = total_amount_loss / len(train_loader)
            
            print(f"\nEpoch {epoch+1} Training:")
            print(f"  Loss: {avg_loss:.4f}")
            print(f"  NER Loss: {avg_ner_loss:.4f}")
            print(f"  Category Loss: {avg_category_loss:.4f}")
            print(f"  Amount Loss: {avg_amount_loss:.4f}")
            
            # Log to MLflow
            mlflow.log_metrics({
                "train_loss": avg_loss,
                "train_ner_loss": avg_ner_loss,
                "train_category_loss": avg_category_loss,
                "train_amount_loss": avg_amount_loss
            }, step=epoch)
            
            # Validation
            if val_loader:
                model.eval()
                val_loss = 0
                
                with torch.no_grad():
                    for batch in val_loader:
                        input_ids = batch['input_ids'].to(device)
                        attention_mask = batch['attention_mask'].to(device)
                        ner_labels = batch['ner_labels'].to(device)
                        category_labels = batch['category_label'].to(device)
                        amounts = batch['amount'].to(device)
                        
                        outputs = model(input_ids, attention_mask)
                        
                        ner_loss = ner_criterion(
                            outputs['ner_logits'].view(-1, len(ENTITY_TYPES)),
                            ner_labels.view(-1)
                        )
                        category_loss = category_criterion(
                            outputs['category_logits'],
                            category_labels
                        )
                        amount_loss = amount_criterion(
                            outputs['amount_pred'],
                            amounts
                        )
                        
                        loss = ner_loss + category_loss + 0.1 * amount_loss
                        val_loss += loss.item()
                
                avg_val_loss = val_loss / len(val_loader)
                print(f"  Validation Loss: {avg_val_loss:.4f}")
                
                mlflow.log_metric("val_loss", avg_val_loss, step=epoch)
                
                # Save best model
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    torch.save(
                        model.state_dict(),
                        output_dir / "best_model.pt"
                    )
                    print("  Saved best model!")
        
        # Save final model
        final_path = output_dir / "final_model.pt"
        torch.save(model.state_dict(), final_path)
        
        # Log model to MLflow
        mlflow.pytorch.log_model(model, "model")
        
        print(f"\nTraining complete! Model saved to {output_dir}")
        
        return model


if __name__ == "__main__":
    # Create synthetic data with diverse patterns
    print("Creating synthetic training data...")
    train_data = create_synthetic_data(num_samples=10000)
    val_data = create_synthetic_data(num_samples=2000)

    # Save data
    train_path = MODEL_DIR / "train_data.json"
    val_path = MODEL_DIR / "val_data.json"

    with open(train_path, 'w', encoding='utf-8') as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)

    with open(val_path, 'w', encoding='utf-8') as f:
        json.dump(val_data, f, ensure_ascii=False, indent=2)

    print(f"Created {len(train_data)} training samples")
    print(f"Created {len(val_data)} validation samples")

    # Train model with better parameters
    print("\nStarting training...")
    model = train_text_extractor(
        train_data_path=str(train_path),
        val_data_path=str(val_path),
        epochs=10,
        batch_size=32,
        learning_rate=2e-5
    )

    print("\nTraining script completed!")
