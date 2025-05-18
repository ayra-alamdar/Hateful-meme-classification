import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
from transformers import AutoTokenizer
import pandas as pd
import numpy as np
from pathlib import Path
import nltk
from nltk.corpus import wordnet
import random
from typing import Optional, Dict, List, Tuple

from config import (
    IMAGE_SIZE, MAX_TEXT_LENGTH, 
    IMAGE_AUGMENTATION, TEXT_AUGMENTATION
)

class HatefulMemesDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        text_model: str = "bert",
        transform: Optional[transforms.Compose] = None,
        augment: bool = False
    ):
        """
        Args:
            data_dir: Path to the data directory
            split: train/val/test
            text_model: bert/lstm
            transform: Optional transform to be applied on images
            augment: Whether to apply augmentation
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.text_model = text_model
        self.augment = augment and split == "train"
        
        # Load annotations
        self.data = pd.read_json(
            self.data_dir / f"{split}.jsonl",
            lines=True
        )
        
        # Initialize tokenizer
        if text_model == "bert":
            self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        else:
            self.tokenizer = nltk.word_tokenize
            
        # Set up image transforms
        if transform is not None:
            self.transform = transform
        else:
            self.transform = self._get_default_transforms()
            
        if self.augment:
            self.aug_transform = self._get_augmentation_transforms()
            
    def _get_default_transforms(self) -> transforms.Compose:
        """Get default image transforms."""
        return transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
    def _get_augmentation_transforms(self) -> transforms.Compose:
        """Get augmentation transforms for images."""
        return transforms.Compose([
            transforms.RandomRotation(IMAGE_AUGMENTATION["random_rotate"]),
            transforms.RandomHorizontalFlip(IMAGE_AUGMENTATION["random_flip"]),
            transforms.ColorJitter(brightness=0.2, contrast=0.2)
        ])
        
    def _augment_text(self, text: str) -> str:
        """Apply synonym replacement augmentation to text."""
        if not self.augment:
            return text
            
        words = text.split()
        num_replacements = min(
            len(words),
            TEXT_AUGMENTATION["max_replacements"]
        )
        
        for _ in range(num_replacements):
            if random.random() < TEXT_AUGMENTATION["synonym_replace_prob"]:
                idx = random.randint(0, len(words) - 1)
                word = words[idx]
                
                # Find synonyms
                synonyms = []
                for syn in wordnet.synsets(word):
                    for lemma in syn.lemmas():
                        if lemma.name() != word:
                            synonyms.append(lemma.name())
                            
                if synonyms:
                    words[idx] = random.choice(synonyms)
                    
        return " ".join(words)
        
    def _process_text(self, text: str) -> Dict[str, torch.Tensor]:
        """Process text based on the model type."""
        text = self._augment_text(text)
        
        if self.text_model == "bert":
            encoding = self.tokenizer(
                text,
                max_length=MAX_TEXT_LENGTH,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
            return {
                "input_ids": encoding["input_ids"].squeeze(0),
                "attention_mask": encoding["attention_mask"].squeeze(0)
            }
        else:
            # For LSTM
            tokens = self.tokenizer(text.lower())
            tokens = tokens[:MAX_TEXT_LENGTH]
            return {"tokens": tokens}
            
    def __len__(self) -> int:
        return len(self.data)
        
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.data.iloc[idx]
        
        # Load and process image
        image = Image.open(self.data_dir / row["img"]).convert("RGB")
        if self.augment:
            image = self.aug_transform(image)
        image = self.transform(image)
        
        # Process text
        text_data = self._process_text(row["text"])
        
        # Combine everything
        item = {
            "image": image,
            **text_data,
            "label": torch.tensor(row["label"], dtype=torch.long)
        }
        
        return item

def get_dataloader(
    data_dir: str,
    split: str,
    batch_size: int,
    text_model: str = "bert",
    num_workers: int = 2,
    augment: bool = False
) -> torch.utils.data.DataLoader:
    """Create a DataLoader for the dataset."""
    dataset = HatefulMemesDataset(
        data_dir=data_dir,
        split=split,
        text_model=text_model,
        augment=augment
    )
    
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=num_workers,
        pin_memory=True
    ) 