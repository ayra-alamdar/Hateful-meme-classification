import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
import argparse
import json
from tqdm import tqdm
import numpy as np
from sklearn.metrics import roc_auc_score
import logging

from config import *
from utils.dataset import get_dataloader
from models.fusion_models import LateFusionModel, EarlyFusionModel
from utils.visualization import (
    plot_confusion_matrix,
    plot_roc_curve,
    plot_training_history
)

def parse_args():
    parser = argparse.ArgumentParser(description='Train multimodal fusion model')
    parser.add_argument('--model', type=str, choices=['early', 'late'],
                      required=True, help='Fusion model type')
    parser.add_argument('--text_model', type=str, choices=['lstm', 'bert'],
                      default='bert', help='Text processing model')
    parser.add_argument('--image_model', type=str, choices=['cnn', 'resnet'],
                      default='resnet', help='Image processing model')
    parser.add_argument('--batch_size', type=int, default=32,
                      help='Batch size')
    parser.add_argument('--epochs', type=int, default=10,
                      help='Number of epochs')
    parser.add_argument('--lr', type=float, default=2e-5,
                      help='Learning rate')
    parser.add_argument('--device', type=str, default='cuda',
                      help='Device to use')
    parser.add_argument('--vocab_size', type=int,
                      help='Vocabulary size for LSTM model')
    return parser.parse_args()

def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    predictions = []
    labels = []
    
    for batch in tqdm(dataloader, desc='Training'):
        # Move data to device
        image = batch['image'].to(device)
        label = batch['label'].to(device)
        text_data = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
            if k not in ['image', 'label']
        }
        
        # Forward pass
        optimizer.zero_grad()
        output = model(image, text_data)
        loss = criterion(output, label)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        predictions.extend(torch.softmax(output, dim=1)[:, 1].cpu().detach())
        labels.extend(label.cpu().detach())
    
    # Calculate metrics
    predictions = torch.stack(predictions)
    labels = torch.stack(labels)
    auroc = roc_auc_score(labels, predictions)
    
    return total_loss / len(dataloader), auroc

def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    predictions = []
    labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Validation'):
            # Move data to device
            image = batch['image'].to(device)
            label = batch['label'].to(device)
            text_data = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
                if k not in ['image', 'label']
            }
            
            # Forward pass
            output = model(image, text_data)
            loss = criterion(output, label)
            
            total_loss += loss.item()
            predictions.extend(torch.softmax(output, dim=1)[:, 1].cpu().detach())
            labels.extend(label.cpu().detach())
    
    # Calculate metrics
    predictions = torch.stack(predictions)
    labels = torch.stack(labels)
    auroc = roc_auc_score(labels, predictions)
    
    return total_loss / len(dataloader), auroc, predictions, labels

def main():
    args = parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Create directories
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device: {device}')
    
    # Create dataloaders
    train_loader = get_dataloader(
        DATA_DIR,
        'train',
        args.batch_size,
        args.text_model,
        NUM_WORKERS,
        augment=True
    )
    val_loader = get_dataloader(
        DATA_DIR,
        'val',
        args.batch_size,
        args.text_model,
        NUM_WORKERS
    )
    
    # Create model
    model_class = EarlyFusionModel if args.model == 'early' else LateFusionModel
    model = model_class(
        text_model=args.text_model,
        image_model=args.image_model,
        vocab_size=args.vocab_size if args.text_model == 'lstm' else None
    ).to(device)
    
    # Setup training
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.1,
        patience=2,
        verbose=True
    )
    
    # Setup TensorBoard
    writer = SummaryWriter(TENSORBOARD_DIR / f'{args.model}_{args.text_model}_{args.image_model}')
    
    # Training loop
    best_auroc = 0
    history = {
        'train_loss': [], 'train_auroc': [],
        'val_loss': [], 'val_auroc': []
    }
    
    for epoch in range(args.epochs):
        logging.info(f'Epoch {epoch+1}/{args.epochs}')
        
        # Train
        train_loss, train_auroc = train_epoch(
            model, train_loader,
            criterion, optimizer,
            device
        )
        
        # Validate
        val_loss, val_auroc, predictions, labels = validate(
            model, val_loader,
            criterion, device
        )
        
        # Update learning rate
        scheduler.step(val_auroc)
        
        # Log metrics
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('AUROC/train', train_auroc, epoch)
        writer.add_scalar('AUROC/val', val_auroc, epoch)
        
        # Update history
        history['train_loss'].append(train_loss)
        history['train_auroc'].append(train_auroc)
        history['val_loss'].append(val_loss)
        history['val_auroc'].append(val_auroc)
        
        # Save best model
        if val_auroc > best_auroc:
            best_auroc = val_auroc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_auroc': val_auroc,
                'args': vars(args)
            }, CHECKPOINT_DIR / f'best_{args.model}_{args.text_model}_{args.image_model}.pt')
            
            # Plot and save validation metrics
            plot_confusion_matrix(
                labels,
                predictions > 0.5,
                save_path=CHECKPOINT_DIR / f'confusion_matrix_{args.model}.png'
            )
            plot_roc_curve(
                labels,
                predictions,
                save_path=CHECKPOINT_DIR / f'roc_curve_{args.model}.png'
            )
        
        logging.info(
            f'Train Loss: {train_loss:.4f}, Train AUROC: {train_auroc:.4f}, '
            f'Val Loss: {val_loss:.4f}, Val AUROC: {val_auroc:.4f}'
        )
    
    # Plot training history
    plot_training_history(
        history,
        ['loss', 'auroc'],
        save_path=CHECKPOINT_DIR / f'training_history_{args.model}.png'
    )
    
    writer.close()
    logging.info('Training completed!')

if __name__ == '__main__':
    main() 