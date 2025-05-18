import torch
import subprocess
import itertools
from pathlib import Path
import json
import pandas as pd
from datetime import datetime
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('model_combinations.log'),
        logging.StreamHandler()
    ]
)

# Define all possible combinations
COMBINATIONS = {
    'model': ['early', 'late'],
    'text_model': ['lstm', 'bert'],
    'image_model': ['cnn', 'resnet']
}

# Create results directory
RESULTS_DIR = Path('results')
RESULTS_DIR.mkdir(exist_ok=True)

def run_model_combination(model_type, text_model, image_model):
    """Run a single model combination and return its results."""
    try:
        cmd = [
            'python', 'train.py',
            '--model', model_type,
            '--text_model', text_model,
            '--image_model', image_model,
            '--device', 'cuda'  # Change to 'cpu' if no GPU available
        ]
        
        logging.info(f"Running combination: {' '.join(cmd)}")
        
        # Run the training process
        process = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )
        
        # Check if the process was successful
        if process.returncode == 0:
            # Load the best model results
            checkpoint_path = Path('checkpoints') / f'best_{model_type}_{text_model}_{image_model}.pt'
            if checkpoint_path.exists():
                checkpoint = torch.load(checkpoint_path, map_location='cpu')
                return {
                    'model_type': model_type,
                    'text_model': text_model,
                    'image_model': image_model,
                    'best_val_auroc': checkpoint['val_auroc'],
                    'epoch': checkpoint['epoch'],
                    'status': 'success'
                }
        
        return {
            'model_type': model_type,
            'text_model': text_model,
            'image_model': image_model,
            'best_val_auroc': None,
            'epoch': None,
            'status': 'failed'
        }
        
    except Exception as e:
        logging.error(f"Error running combination: {str(e)}")
        return {
            'model_type': model_type,
            'text_model': text_model,
            'image_model': image_model,
            'best_val_auroc': None,
            'epoch': None,
            'status': f'error: {str(e)}'
        }

def main():
    # Generate all possible combinations
    combinations = list(itertools.product(
        COMBINATIONS['model'],
        COMBINATIONS['text_model'],
        COMBINATIONS['image_model']
    ))
    
    # Store results
    results = []
    
    # Run all combinations
    for model_type, text_model, image_model in combinations:
        logging.info(f"\n{'='*50}")
        logging.info(f"Starting new combination:")
        logging.info(f"Model Type: {model_type}")
        logging.info(f"Text Model: {text_model}")
        logging.info(f"Image Model: {image_model}")
        logging.info(f"{'='*50}\n")
        
        result = run_model_combination(model_type, text_model, image_model)
        results.append(result)
        
        # Save intermediate results
        df = pd.DataFrame(results)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        df.to_csv(RESULTS_DIR / f'model_results_{timestamp}.csv', index=False)
        
        # Also save as JSON for better readability
        with open(RESULTS_DIR / f'model_results_{timestamp}.json', 'w') as f:
            json.dump(results, f, indent=4)
    
    # Create final summary
    df = pd.DataFrame(results)
    df = df.sort_values('best_val_auroc', ascending=False)
    
    # Save final results
    final_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    df.to_csv(RESULTS_DIR / f'final_results_{final_timestamp}.csv', index=False)
    
    # Print best combination
    best_model = df.iloc[0]
    logging.info("\nBest Model Combination:")
    logging.info(f"Model Type: {best_model['model_type']}")
    logging.info(f"Text Model: {best_model['text_model']}")
    logging.info(f"Image Model: {best_model['image_model']}")
    logging.info(f"Best Validation AUROC: {best_model['best_val_auroc']}")
    logging.info(f"Achieved at epoch: {best_model['epoch']}")

if __name__ == '__main__':
    main() 