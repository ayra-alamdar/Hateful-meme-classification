# Hateful Memes Detection

This project implements multimodal (image + text) classification for hateful memes detection using PyTorch. It includes both early and late fusion approaches combining BERT/LSTM for text and CNN/ResNet for images.

## Google Colab Setup

1. Mount your Google Drive:

```python
from google.colab import drive
drive.mount('/content/drive')
```

2. Clone this repository:

```bash
!git clone https://github.com/YOUR_USERNAME/hateful-memes-detection.git
!cd hateful-memes-detection
```

3. Install dependencies:

```bash
!pip install -r requirements.txt
```

4. Download required NLTK data:

```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
```

## Project Structure

```
├── data/                    # Data directory (create after cloning)
├── models/                  # Model implementations
│   ├── text_models.py      # LSTM and BERT models
│   ├── image_models.py     # CNN and ResNet models
│   ├── fusion_models.py    # Early and late fusion implementations
├── utils/
│   ├── dataset.py          # Custom dataset and data loading
│   ├── preprocessing.py     # Text and image preprocessing
│   ├── augmentation.py     # Data augmentation techniques
│   ├── visualization.py    # Visualization utilities
├── train.py                # Training script
├── evaluate.py             # Evaluation script
├── config.py              # Configuration parameters
└── requirements.txt       # Project dependencies
```

## Training

Run the training script with desired parameters:

```bash
!python train.py --model [early/late] --text_model [lstm/bert] --image_model [cnn/resnet] --batch_size 32 --epochs 10
```

## Evaluation

Evaluate trained models:

```bash
!python evaluate.py --model_path checkpoints/best_model.pth
```

## TensorBoard Visualization

Launch TensorBoard:

```python
%load_ext tensorboard
%tensorboard --logdir runs/
```
