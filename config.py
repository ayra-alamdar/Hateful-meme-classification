from pathlib import Path

# Data configuration
DATA_DIR = Path("data")
IMAGE_SIZE = 224
MAX_TEXT_LENGTH = 128
BATCH_SIZE = 32
NUM_WORKERS = 2

# Model configuration
TEXT_MODELS = {
    "lstm": {
        "embedding_dim": 300,
        "hidden_dim": 256,
        "num_layers": 2,
        "dropout": 0.3,
        "bidirectional": True
    },
    "bert": {
        "model_name": "bert-base-uncased",
        "freeze_layers": 8
    }
}

IMAGE_MODELS = {
    "cnn": {
        "channels": [32, 64, 128],
        "kernel_size": 3,
        "dropout": 0.3
    },
    "resnet": {
        "model_name": "resnet50",
        "pretrained": True
    }
}

FUSION_CONFIG = {
    "early": {
        "hidden_dims": [512, 256],
        "dropout": 0.3
    },
    "late": {
        "hidden_dim": 256,
        "dropout": 0.3
    }
}

# Training configuration
LEARNING_RATE = 2e-5
NUM_EPOCHS = 10
EARLY_STOPPING_PATIENCE = 3
WARMUP_STEPS = 100

# Augmentation configuration
IMAGE_AUGMENTATION = {
    "random_rotate": 15,  # degrees
    "random_flip": 0.5    # probability
}

TEXT_AUGMENTATION = {
    "synonym_replace_prob": 0.1,
    "max_replacements": 2
}

# Paths
CHECKPOINT_DIR = Path("checkpoints")
TENSORBOARD_DIR = Path("runs")

# Create directories if they don't exist
for directory in [DATA_DIR, CHECKPOINT_DIR, TENSORBOARD_DIR]:
    directory.mkdir(exist_ok=True, parents=True) 