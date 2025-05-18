import torch
import argparse
import logging
from pathlib import Path
from tqdm import tqdm
import json

from config import *
from utils.dataset import get_dataloader
from models.fusion_models import LateFusionModel, EarlyFusionModel
from utils.visualization import (
    plot_confusion_matrix,
    plot_roc_curve,
    plot_sample_memes
)

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate multimodal fusion model')
    parser.add_argument('--checkpoint', type=str, required=True,
                      help='Path to model checkpoint')
    parser.add_argument('--batch_size', type=int, default=32,
                      help='Batch size')
    parser.add_argument('--device', type=str, default='cuda',
                      help='Device to use')
    parser.add_argument('--output_dir', type=str, default='evaluation_results',
                      help='Directory to save evaluation results')
    return parser.parse_args()

def evaluate(model, dataloader, device):
    model.eval()
    predictions = []
    labels = []
    all_outputs = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Evaluating'):
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
            probs = torch.softmax(output, dim=1)
            
            predictions.extend(probs[:, 1].cpu())
            labels.extend(label.cpu())
            all_outputs.append({
                'probs': probs.cpu().numpy().tolist(),
                'label': label.cpu().numpy().tolist()
            })
    
    return (
        torch.stack(predictions),
        torch.stack(labels),
        all_outputs
    )

def main():
    args = parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    checkpoint_args = checkpoint['args']
    
    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device: {device}')
    
    # Create model
    model_class = EarlyFusionModel if checkpoint_args['model'] == 'early' else LateFusionModel
    model = model_class(
        text_model=checkpoint_args['text_model'],
        image_model=checkpoint_args['image_model']
    ).to(device)
    
    # Load model weights
    model.load_state_dict(checkpoint['model_state_dict'])
    logging.info(f'Loaded checkpoint from epoch {checkpoint["epoch"]}')
    
    # Create test dataloader
    test_loader = get_dataloader(
        DATA_DIR,
        'test',
        args.batch_size,
        checkpoint_args['text_model'],
        NUM_WORKERS
    )
    
    # Evaluate
    predictions, labels, outputs = evaluate(model, test_loader, device)
    
    # Save predictions
    with open(output_dir / 'predictions.json', 'w') as f:
        json.dump(outputs, f)
    
    # Plot metrics
    plot_confusion_matrix(
        labels,
        predictions > 0.5,
        save_path=output_dir / 'test_confusion_matrix.png'
    )
    plot_roc_curve(
        labels,
        predictions,
        save_path=output_dir / 'test_roc_curve.png'
    )
    
    # Plot sample predictions
    test_data = next(iter(test_loader))
    plot_sample_memes(
        test_data['image'][:5],
        test_data['text'][:5] if 'text' in test_data else [''] * 5,
        test_data['label'][:5],
        save_path=output_dir / 'sample_predictions.png'
    )
    
    logging.info('Evaluation completed!')

if __name__ == '__main__':
    main() 