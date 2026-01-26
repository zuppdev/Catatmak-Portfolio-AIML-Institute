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


def create_synthetic_data(num_samples: int = 1000) -> List[Dict]:
    """Create synthetic training data for demonstration"""
    
    templates = [
        ("beli {item} {amount}", "belanja"),
        ("makan {item} {amount}", "makanan"),
        ("bayar {item} {amount}", "tagihan"),
        ("bensin {amount}", "transportasi"),
        ("parkir {amount}", "transportasi"),
        ("nonton {item} {amount}", "hiburan"),
        ("beli obat {amount}", "kesehatan"),
        ("bayar kursus {amount}", "pendidikan"),
    ]
    
    items = {
        "belanja": ["baju", "sepatu", "tas", "celana", "jaket"],
        "makanan": ["bakso", "nasi goreng", "mie ayam", "sate", "gado-gado"],
        "hiburan": ["film", "konser", "game"],
    }
    
    amounts = [10000, 15000, 20000, 25000, 30000, 50000, 75000, 100000]
    
    data = []
    
    for _ in range(num_samples):
        template, category = templates[np.random.randint(len(templates))]
        
        if "{item}" in template:
            category_items = items.get(category, ["sesuatu"])
            item = np.random.choice(category_items)
        else:
            item = ""
        
        amount = np.random.choice(amounts)
        amount_text = f"Rp {amount:,}".replace(',', '.')
        
        text = template.format(item=item, amount=amount_text)
        
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
        
        # Loss functions
        ner_criterion = nn.CrossEntropyLoss(ignore_index=0)
        category_criterion = nn.CrossEntropyLoss()
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
    # Create synthetic data for demonstration
    print("Creating synthetic training data...")
    train_data = create_synthetic_data(num_samples=1000)
    val_data = create_synthetic_data(num_samples=200)
    
    # Save data
    train_path = MODEL_DIR / "train_data.json"
    val_path = MODEL_DIR / "val_data.json"
    
    with open(train_path, 'w', encoding='utf-8') as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)
    
    with open(val_path, 'w', encoding='utf-8') as f:
        json.dump(val_data, f, ensure_ascii=False, indent=2)
    
    print(f"Created {len(train_data)} training samples")
    print(f"Created {len(val_data)} validation samples")
    
    # Train model
    print("\nStarting training...")
    model = train_text_extractor(
        train_data_path=str(train_path),
        val_data_path=str(val_path),
        epochs=5,  # Reduced for demo
        batch_size=16
    )
    
    print("\nTraining script completed!")
