import torch
import subprocess
import itertools
from pathlib import Path
import json
import pandas as pd
from datetime import datetime
import logging
import sys
import pkg_resources
import os
from importlib import import_module
import shutil
import kagglehub

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('model_combinations.log'),
        logging.StreamHandler()
    ]
)

def check_and_install_dependencies():
    """Check and install required dependencies."""
    logging.info("Checking dependencies...")
    
    # Read requirements
    with open('requirements.txt', 'r') as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]
    
    # Add kagglehub to requirements if not present
    if not any('kagglehub' in req for req in requirements):
        requirements.append('kagglehub')
    
    # Check each requirement
    for requirement in requirements:
        package_name = requirement.split('>=')[0]
        try:
            pkg_resources.require(requirement)
            logging.info(f"✓ {requirement} already satisfied")
        except (pkg_resources.DistributionNotFound, pkg_resources.VersionConflict):
            logging.info(f"Installing {requirement}...")
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", requirement])
                logging.info(f"✓ Successfully installed {requirement}")
            except subprocess.CalledProcessError as e:
                logging.error(f"Failed to install {requirement}: {str(e)}")
                raise

def prepare_dataset():
    """Prepare the dataset using kagglehub"""
    try:
        # First check if data already exists
        data_dir = Path("data")
        if data_dir.exists() and (data_dir / "train.jsonl").exists():
            logging.info("Dataset already exists, skipping download...")
            return
        
        logging.info("Downloading dataset using kagglehub...")
        
        # Download dataset
        dataset_path = kagglehub.dataset_download("marafey/hateful-memes-dataset")
        logging.info(f"Dataset downloaded to: {dataset_path}")
        
        # Create data directory
        data_dir.mkdir(exist_ok=True)
        
        # Move all files to data directory
        for file_path in Path(dataset_path).glob("*"):
            if file_path.is_file():
                shutil.copy2(file_path, data_dir / file_path.name)
        
        # Create images directory and move images
        images_dir = data_dir / "images"
        images_dir.mkdir(exist_ok=True)
        
        # Move all image files to images directory
        for file in data_dir.glob("*.png"):
            shutil.move(str(file), str(images_dir / file.name))
        
        # Process annotations
        process_annotations(data_dir)
        
        logging.info("Dataset preparation completed!")
    except Exception as e:
        logging.error(f"Error preparing dataset: {str(e)}")
        raise

def process_annotations(data_dir):
    """Process and split annotations."""
    # Read annotations
    with open(data_dir / "labels.json", "r") as f:
        annotations = json.load(f)
    
    # Convert to list format if it's a dict
    if isinstance(annotations, dict):
        annotations = [
            {"id": k, **v} for k, v in annotations.items()
        ]
    
    # Sort by id for reproducibility
    annotations.sort(key=lambda x: x["id"])
    
    # Calculate split sizes
    total = len(annotations)
    train_size = int(0.7 * total)
    val_size = int(0.15 * total)
    
    # Split data
    train_data = annotations[:train_size]
    val_data = annotations[train_size:train_size + val_size]
    test_data = annotations[train_size + val_size:]
    
    # Save splits
    splits = {
        "train": train_data,
        "val": val_data,
        "test": test_data
    }
    
    for split_name, split_data in splits.items():
        output_file = data_dir / f"{split_name}.jsonl"
        with open(output_file, "w") as f:
            for item in split_data:
                # Add image path
                item["img"] = f"images/{item['id']}.png"
                f.write(json.dumps(item) + "\n")
    
    logging.info(f"Data split into: train({len(train_data)}), val({len(val_data)}), test({len(test_data)})")

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
            sys.executable,  # Use the same Python interpreter
            'train.py',
            '--model', model_type,
            '--text_model', text_model,
            '--image_model', image_model,
            '--device', 'cuda' if torch.cuda.is_available() else 'cpu'
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
    try:
        # Setup phase
        logging.info("Starting setup phase...")
        check_and_install_dependencies()
        prepare_dataset()
        logging.info("Setup completed successfully!")
        
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
        
    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
        raise

if __name__ == '__main__':
    main() 