import torch
import torch.nn as nn
from transformers import BertModel, BertConfig
from typing import Dict, Tuple

from config import TEXT_MODELS, MAX_TEXT_LENGTH

class LSTMTextProcessor(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = TEXT_MODELS["lstm"]["embedding_dim"],
        hidden_dim: int = TEXT_MODELS["lstm"]["hidden_dim"],
        num_layers: int = TEXT_MODELS["lstm"]["num_layers"],
        dropout: float = TEXT_MODELS["lstm"]["dropout"],
        bidirectional: bool = TEXT_MODELS["lstm"]["bidirectional"]
    ):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True
        )
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(
            hidden_dim * (2 if bidirectional else 1),
            hidden_dim
        )
        
    def forward(
        self,
        tokens: torch.Tensor,
        lengths: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            tokens: Tensor of token indices [batch_size, seq_len]
            lengths: Tensor of sequence lengths [batch_size]
        Returns:
            Tensor of shape [batch_size, hidden_dim]
        """
        # Embed tokens
        embedded = self.dropout(self.embedding(tokens))
        
        # Pack sequence for LSTM
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False
        )
        
        # Process with LSTM
        output, (hidden, _) = self.lstm(packed)
        
        if self.lstm.bidirectional:
            # Concatenate the final hidden states from both directions
            hidden = torch.cat(
                (hidden[-2,:,:], hidden[-1,:,:]),
                dim=1
            )
        else:
            hidden = hidden[-1]
            
        # Final projection
        output = self.fc(self.dropout(hidden))
        
        return output

class BERTTextProcessor(nn.Module):
    def __init__(
        self,
        model_name: str = TEXT_MODELS["bert"]["model_name"],
        freeze_layers: int = TEXT_MODELS["bert"]["freeze_layers"]
    ):
        super().__init__()
        
        # Load pre-trained BERT
        self.bert = BertModel.from_pretrained(model_name)
        
        # Freeze specified number of layers
        if freeze_layers > 0:
            for param in self.bert.embeddings.parameters():
                param.requires_grad = False
                
            for i in range(freeze_layers):
                for param in self.bert.encoder.layer[i].parameters():
                    param.requires_grad = False
                    
        # Add projection layer
        config = self.bert.config
        self.fc = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            input_ids: Tensor of token ids [batch_size, seq_len]
            attention_mask: Tensor of attention masks [batch_size, seq_len]
        Returns:
            Tensor of shape [batch_size, hidden_size]
        """
        # Get BERT outputs
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Use [CLS] token representation
        pooled_output = outputs.pooler_output
        
        # Project through final layer
        output = self.fc(self.dropout(pooled_output))
        
        return output 