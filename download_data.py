import os
import json
import shutil
from pathlib import Path
import kaggle
import logging
from tqdm import tqdm

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def download_dataset():
    """Download dataset from Kaggle."""
    logging.info("Downloading dataset from Kaggle...")
    
    # Create data directory
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    # Download dataset
    kaggle.api.authenticate()
    kaggle.api.dataset_download_files(
        'marafey/hateful-memes-dataset',
        path=data_dir,
        unzip=True
    )
    
    # Create images directory
    images_dir = data_dir / "images"
    images_dir.mkdir(exist_ok=True)
    
    # Move all image files to images directory
    for file in data_dir.glob("*.png"):
        shutil.move(str(file), str(images_dir / file.name))
    
    logging.info("Dataset downloaded and organized successfully!")

def process_annotations():
    """Process and split annotations."""
    data_dir = Path("data")
    
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

def main():
    setup_logging()
    
    try:
        download_dataset()
        process_annotations()
        logging.info("Dataset preparation completed successfully!")
    except Exception as e:
        logging.error(f"Error preparing dataset: {str(e)}")
        raise

if __name__ == "__main__":
    main() 