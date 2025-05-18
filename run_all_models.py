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
import shutil
from tqdm import tqdm
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

def check_environment():
    """Check if the environment is properly set up."""
    logging.info("Checking environment...")
    
    # Check CUDA availability
    if torch.cuda.is_available():
        logging.info(f"CUDA is available. Found device: {torch.cuda.get_device_name(0)}")
    else:
        logging.warning("CUDA is not available. Training will be done on CPU.")
    
    # Check Python version
    logging.info(f"Python version: {sys.version}")
    
    # Check if we're in Google Colab
    try:
        import google.colab
        logging.info("Running in Google Colab environment")
        
        # Check if we're connected to a GPU runtime
        if not torch.cuda.is_available():
            logging.warning("Running in Colab but no GPU detected. Please make sure to select GPU runtime.")
    except ImportError:
        logging.info("Not running in Google Colab environment")
    
    # Check current working directory and its contents
    cwd = Path.cwd()
    logging.info(f"Current working directory: {cwd}")
    logging.info("Directory contents:")
    for item in cwd.iterdir():
        logging.info(f"  {item.name}")
    
    # Create necessary directories
    for directory in ['data', 'checkpoints', 'results']:
        Path(directory).mkdir(exist_ok=True)
        logging.info(f"Created directory: {directory}")

def check_and_install_dependencies():
    """Check and install required dependencies."""
    logging.info("Checking dependencies...")
    
    # Read requirements
    with open('requirements.txt', 'r') as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]
    
    # Add required packages for download
    additional_requirements = ['kagglehub']
    for req in additional_requirements:
        if not any(req in r for r in requirements):
            requirements.append(req)
    
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
    
    # Download required NLTK data
    logging.info("Checking NLTK data...")
    try:
        import nltk
        nltk_resources = ['punkt', 'wordnet']
        for resource in nltk_resources:
            try:
                nltk.data.find(f'tokenizers/{resource}')
                logging.info(f"✓ NLTK {resource} already downloaded")
            except LookupError:
                logging.info(f"Downloading NLTK {resource}...")
                nltk.download(resource)
                logging.info(f"✓ Successfully downloaded NLTK {resource}")
    except Exception as e:
        logging.error(f"Error downloading NLTK data: {str(e)}")
        raise

def prepare_dataset():
    """Prepare the dataset using kagglehub"""
    try:
        # First check if data already exists
        data_dir = Path("data")
        data_dir.mkdir(exist_ok=True)
        
        # Check if dataset is already prepared
        if (data_dir / "train.jsonl").exists() and (data_dir / "img").exists():
            # Verify images exist
            img_dir = data_dir / "img"
            if list(img_dir.glob("*.png")):
                logging.info("Dataset already exists and appears valid, skipping download...")
                return
            else:
                logging.warning("Image directory exists but no images found. Re-downloading dataset...")
        
        logging.info("Downloading dataset using kagglehub...")
        
        # Download dataset using kagglehub (no authentication needed for public datasets)
        dataset_path = Path(kagglehub.dataset_download("marafey/hateful-memes-dataset"))
        logging.info(f"Dataset downloaded to: {dataset_path}")
        
        # Create img directory (not images - to match the JSONL files)
        img_dir = data_dir / "img"
        img_dir.mkdir(exist_ok=True)
        
        # Look for the img directory in possible locations
        possible_img_dirs = [
            dataset_path / "img",
            dataset_path / "data" / "img",
            dataset_path / "images",
            dataset_path / "data" / "images"
        ]
        
        source_img_dir = None
        for dir_path in possible_img_dirs:
            if dir_path.exists() and dir_path.is_dir():
                source_img_dir = dir_path
                break
                
        if source_img_dir is None:
            # Try to find any directory containing PNG files
            for root, _, _ in os.walk(dataset_path):
                root_path = Path(root)
                if list(root_path.glob("*.png")):
                    source_img_dir = root_path
                    logging.info(f"Found images in: {root_path}")
                    break
                    
        if source_img_dir is None:
            raise FileNotFoundError(f"Image directory not found in any of these locations: {[str(p) for p in possible_img_dirs]}")
            
        # Copy all images from img directory
        logging.info(f"Copying images from {source_img_dir}")
        image_count = 0
        for file_path in source_img_dir.glob("*.png"):
            shutil.copy2(file_path, img_dir / file_path.name)
            image_count += 1
            
        if image_count == 0:
            raise FileNotFoundError("No images found to copy")
            
        logging.info(f"Copied {image_count} images")
            
        # Look for the JSONL files
        jsonl_files = ['train.jsonl', 'dev.jsonl', 'test.jsonl']
        for jsonl_file in jsonl_files:
            possible_locations = [
                dataset_path / jsonl_file,
                dataset_path / "data" / jsonl_file
            ]
            
            file_found = False
            for loc in possible_locations:
                if loc.exists():
                    shutil.copy2(loc, data_dir / jsonl_file)
                    logging.info(f"Copied {jsonl_file} from {loc}")
                    file_found = True
                    break
                    
            if not file_found:
                logging.warning(f"Could not find {jsonl_file} in any location")
        
        # Rename dev.jsonl to val.jsonl for consistency with our code
        if (data_dir / "dev.jsonl").exists():
            shutil.move(data_dir / "dev.jsonl", data_dir / "val.jsonl")
            logging.info("Renamed dev.jsonl to val.jsonl")
        
        logging.info("Dataset preparation completed!")
        
        # Verify the dataset structure
        expected_files = ["train.jsonl", "val.jsonl", "test.jsonl"]
        missing_files = [f for f in expected_files if not (data_dir / f).exists()]
        if missing_files:
            raise FileNotFoundError(f"Missing required files: {missing_files}")
            
        # Verify images
        image_files = list(img_dir.glob("*.png"))
        if not image_files:
            raise FileNotFoundError("No images were copied to the img directory")
        else:
            logging.info(f"Found {len(image_files)} images in the img directory")
            
        # Verify image paths in JSONL files
        for jsonl_file in expected_files:
            with open(data_dir / jsonl_file, 'r') as f:
                for line in f:
                    data = json.loads(line)
                    img_path = img_dir / Path(data['img']).name
                    if not img_path.exists():
                        logging.warning(f"Missing image file: {img_path}")
            
    except Exception as e:
        logging.error(f"Error preparing dataset: {str(e)}")
        raise

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
        # Build vocabulary size for LSTM if needed
        vocab_size = None
        if text_model == 'lstm':
            # Read training data to build vocabulary
            with open('data/train.jsonl', 'r') as f:
                texts = [json.loads(line)['text'] for line in f]
            # Simple tokenization and vocabulary building
            words = set()
            for text in texts:
                words.update(text.lower().split())
            vocab_size = len(words) + 2  # +2 for <pad> and <unk>
            logging.info(f"Built vocabulary with size: {vocab_size}")
        
        cmd = [
            sys.executable,  # Use the same Python interpreter
            'train.py',
            '--model', model_type,
            '--text_model', text_model,
            '--image_model', image_model,
            '--device', 'cuda' if torch.cuda.is_available() else 'cpu',
            '--epochs', '2'  # Set epochs to 2
        ]
        
        # Add vocab_size if using LSTM
        if vocab_size is not None:
            cmd.extend(['--vocab_size', str(vocab_size)])
        
        logging.info(f"Running combination: {' '.join(cmd)}")
        
        # Run the training process
        process = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False  # Don't raise exception immediately
        )
        
        # Log the output regardless of success/failure
        if process.stdout:
            logging.info("Process output:")
            logging.info(process.stdout)
        
        if process.stderr:
            logging.error("Process errors:")
            logging.error(process.stderr)
            
        # Now check if the process was successful
        process.check_returncode()  # This will raise CalledProcessError if return code != 0
        
        # If we get here, the process was successful
        # Check if the process was successful
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
        
    except subprocess.CalledProcessError as e:
        logging.error(f"Command failed with return code {e.returncode}")
        if e.stdout:
            logging.error("Process output:")
            logging.error(e.stdout)
        if e.stderr:
            logging.error("Process errors:")
            logging.error(e.stderr)
        return {
            'model_type': model_type,
            'text_model': text_model,
            'image_model': image_model,
            'best_val_auroc': None,
            'epoch': None,
            'status': f'error: Command failed with return code {e.returncode}'
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
        # Environment and setup phase
        logging.info("Starting environment check...")
        check_environment()
        
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