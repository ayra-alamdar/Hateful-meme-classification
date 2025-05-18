import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional

from config import FUSION_CONFIG
from models.text_models import LSTMTextProcessor, BERTTextProcessor
from models.image_models import CNNImageProcessor, ResNetImageProcessor

class LateFusionModel(nn.Module):
    def __init__(
        self,
        text_model: str = "bert",
        image_model: str = "resnet",
        hidden_dim: int = FUSION_CONFIG["late"]["hidden_dim"],
        dropout: float = FUSION_CONFIG["late"]["dropout"],
        num_classes: int = 2,
        vocab_size: Optional[int] = None
    ):
        super().__init__()
        
        # Initialize text processor
        if text_model == "bert":
            self.text_processor = BERTTextProcessor()
            text_out_dim = 768  # BERT hidden size
        else:
            assert vocab_size is not None, "vocab_size required for LSTM"
            self.text_processor = LSTMTextProcessor(vocab_size)
            text_out_dim = 256  # LSTM hidden size
            
        # Initialize image processor
        if image_model == "resnet":
            self.image_processor = ResNetImageProcessor()
        else:
            self.image_processor = CNNImageProcessor()
        image_out_dim = 256  # Both CNN and ResNet output 256-dim
        
        # Fusion classifier
        self.classifier = nn.Sequential(
            nn.Linear(text_out_dim + image_out_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
        
    def forward(
        self,
        image: torch.Tensor,
        text_data: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Args:
            image: Image tensor [batch_size, 3, 224, 224]
            text_data: Dict containing text tensors (varies by model type)
        Returns:
            Tensor of shape [batch_size, num_classes]
        """
        # Process text and image separately
        text_features = self.text_processor(**text_data)
        image_features = self.image_processor(image)
        
        # Concatenate features
        combined = torch.cat([text_features, image_features], dim=1)
        
        # Final classification
        output = self.classifier(combined)
        
        return output

class EarlyFusionModel(nn.Module):
    def __init__(
        self,
        text_model: str = "bert",
        image_model: str = "resnet",
        hidden_dims: list = FUSION_CONFIG["early"]["hidden_dims"],
        dropout: float = FUSION_CONFIG["early"]["dropout"],
        num_classes: int = 2,
        vocab_size: Optional[int] = None,
        use_attention: bool = True
    ):
        super().__init__()
        
        # Initialize text processor
        if text_model == "bert":
            self.text_processor = BERTTextProcessor()
            text_out_dim = 768  # BERT hidden size
        else:
            assert vocab_size is not None, "vocab_size required for LSTM"
            self.text_processor = LSTMTextProcessor(vocab_size)
            text_out_dim = 256
            
        # Initialize image processor
        if image_model == "resnet":
            self.image_processor = ResNetImageProcessor()
        else:
            self.image_processor = CNNImageProcessor()
        image_out_dim = 256
        
        # Cross-modal attention (optional)
        self.use_attention = use_attention
        if use_attention:
            self.attention = CrossModalAttention(
                text_dim=text_out_dim,
                image_dim=image_out_dim
            )
            
        # Fusion MLP - Adjust input dimension based on concatenated features
        combined_dim = text_out_dim + image_out_dim  # 768 + 256 = 1024 for BERT, 256 + 256 = 512 for LSTM
        
        # Create MLP layers with proper dimensions
        layers = []
        prev_dim = combined_dim
        
        for dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout)
            ])
            prev_dim = dim
            
        # Final classification layer
        layers.append(nn.Linear(prev_dim, num_classes))
        self.fusion_mlp = nn.Sequential(*layers)
        
    def forward(
        self,
        image: torch.Tensor,
        text_data: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Args:
            image: Image tensor [batch_size, 3, 224, 224]
            text_data: Dict containing text tensors (varies by model type)
        Returns:
            Tensor of shape [batch_size, num_classes]
        """
        # Get features from both modalities
        text_features = self.text_processor(**text_data)  # [batch_size, text_dim]
        image_features = self.image_processor(image)      # [batch_size, image_dim]
        
        # Apply cross-modal attention if enabled
        if self.use_attention:
            text_features, image_features = self.attention(
                text_features,
                image_features
            )
            
        # Concatenate features along feature dimension
        combined = torch.cat([text_features, image_features], dim=1)
        
        # Final classification
        output = self.fusion_mlp(combined)
        
        return output

class CrossModalAttention(nn.Module):
    def __init__(self, text_dim: int, image_dim: int):
        super().__init__()
        
        # Project both modalities to same dimension for attention
        self.text_proj = nn.Linear(text_dim, 256)
        self.image_proj = nn.Linear(image_dim, 256)
        
        # Attention layers
        self.attention = nn.MultiheadAttention(
            embed_dim=256,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
    def forward(
        self,
        text_features: torch.Tensor,
        image_features: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            text_features: Text features [batch_size, text_dim]
            image_features: Image features [batch_size, image_dim]
        Returns:
            Tuple of attended features [batch_size, text_dim], [batch_size, image_dim]
        """
        # Project features
        text_proj = self.text_proj(text_features).unsqueeze(1)
        image_proj = self.image_proj(image_features).unsqueeze(1)
        
        # Cross attention: text attending to image
        text_attended, _ = self.attention(
            text_proj,
            image_proj,
            image_proj
        )
        
        # Cross attention: image attending to text
        image_attended, _ = self.attention(
            image_proj,
            text_proj,
            text_proj
        )
        
        return text_attended.squeeze(1), image_attended.squeeze(1) 