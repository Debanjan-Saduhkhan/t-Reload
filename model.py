from __future__ import annotations
import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer

from .config import ModelConfig

logger = logging.getLogger(__name__)


class TReloadModel(nn.Module):
    """
    Implementation of the t-reload method described in the research paper.
    
    This model implements the core architecture and methodology
    for the t-reload approach to model training and optimization.
    """
    
    def __init__(self, config: ModelConfig):
        """
        Initialize the t-reload model.
        
        Args:
            config: Model configuration
        """
        super().__init__()
        self.config = config
        
        # Embedding layer
        self.embedding = nn.Embedding(
            num_embeddings=config.vocab_size if hasattr(config, 'vocab_size') else 30522,
            embedding_dim=config.hidden_size,
            padding_idx=0
        )
        
        # Positional encoding
        self.pos_encoding = nn.Parameter(
            torch.randn(1, 1024, config.hidden_size)
        )
        
        # Transformer encoder layers
        encoder_layer = TransformerEncoderLayer(
            d_model=config.hidden_size,
            nhead=config.num_heads,
            dim_feedforward=config.hidden_size * 4,
            dropout=config.dropout,
            activation=config.activation,
            batch_first=True
        )
        
        self.transformer = TransformerEncoder(
            encoder_layer,
            num_layers=config.num_layers
        )
        
        # Output layers
        self.dropout = nn.Dropout(config.dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_classes if hasattr(config, 'num_classes') else 2)
        
        # Initialize weights
        self._init_weights()
        
        logger.info(f"Initialized t-reload model with {config.num_layers} layers")
    
    def _init_weights(self) -> None:
        """Initialize model weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the model.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            labels: Ground truth labels
            
        Returns:
            Dictionary containing logits and loss (if labels provided)
        """
        batch_size, seq_len = input_ids.shape
        
        # Create attention mask if not provided
        if attention_mask is None:
            attention_mask = (input_ids != 0).long()
        
        # Embeddings
        embeddings = self.embedding(input_ids)
        
        # Add positional encoding
        if seq_len <= self.pos_encoding.size(1):
            pos_enc = self.pos_encoding[:, :seq_len, :]
        else:
            # Extend positional encoding if needed
            pos_enc = F.interpolate(
                self.pos_encoding.transpose(1, 2),
                size=seq_len,
                mode='linear'
            ).transpose(1, 2)
        
        embeddings = embeddings + pos_enc
        
        # Transformer encoding
        # Create proper attention mask for transformer
        transformer_mask = self._create_transformer_mask(attention_mask)
        
        encoded = self.transformer(
            embeddings,
            src_key_padding_mask=transformer_mask
        )
        
        # Global average pooling
        # Use attention mask to exclude padding tokens
        mask_expanded = attention_mask.unsqueeze(-1).float()
        pooled = (encoded * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1)
        
        # Classification
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)
        
        outputs = {"logits": logits}
        
        # Calculate loss if labels are provided
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
            outputs["loss"] = loss
        
        return outputs
    
    def _create_transformer_mask(self, attention_mask: torch.Tensor) -> torch.Tensor:
        """Create proper attention mask for transformer."""
        # Convert from [batch_size, seq_len] to [batch_size, seq_len]
        # where True means the token should be masked (ignored)
        return attention_mask == 0
    
    def get_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Get embeddings for input tokens."""
        return self.embedding(input_ids)
    
    def reload_parameters(self, checkpoint_path: str) -> None:
        """
        Reload model parameters from checkpoint.
        
        This implements the core t-reload functionality where
        the model can reload its parameters during training.
        
        Args:
            checkpoint_path: Path to checkpoint file
        """
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint
            
            # Load state dict
            missing_keys, unexpected_keys = self.load_state_dict(state_dict, strict=False)
            
            if missing_keys:
                logger.warning(f"Missing keys: {missing_keys}")
            if unexpected_keys:
                logger.warning(f"Unexpected keys: {unexpected_keys}")
            
            logger.info(f"Successfully reloaded parameters from {checkpoint_path}")
            
        except Exception as e:
            logger.error(f"Failed to reload parameters: {e}")
            raise
    
    def save_checkpoint(self, path: str, optimizer=None, epoch: int = 0) -> None:
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.state_dict(),
            'config': self.config.to_dict()
        }
        
        if optimizer:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        
        torch.save(checkpoint, path)
        logger.info(f"Checkpoint saved to {path}")


class TReloadTrainer:
    """Training wrapper for the t-reload model."""
    
    def __init__(self, model: TReloadModel, config: Dict[str, Any]):
        """
        Initialize the trainer.
        
        Args:
            model: The t-reload model
            config: Training configuration
        """
        self.model = model
        self.config = config
        self.device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
        
        # Move model to device
        self.model.to(self.device)
        
        # Setup optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.get('learning_rate', 1e-4),
            weight_decay=config.get('weight_decay', 0.01)
        )
        
        # Setup scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.get('num_epochs', 100)
        )
        
        logger.info(f"Initialized trainer on device: {self.device}")
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Single training step."""
        self.model.train()
        
        # Move batch to device
        batch = {k: v.to(self.device) for k, v in batch.items()}
        
        # Forward pass
        outputs = self.model(**batch)
        loss = outputs['loss']
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.get('gradient_clip', 1.0))
        
        self.optimizer.step()
        
        return {'loss': loss.item()}
    
    def evaluate_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Single evaluation step."""
        self.model.eval()
        
        with torch.no_grad():
            batch = {k: v.to(self.device) for k, v in batch.items()}
            outputs = self.model(**batch)
            
            if 'labels' in batch:
                # Calculate accuracy
                logits = outputs['logits']
                predictions = torch.argmax(logits, dim=-1)
                accuracy = (predictions == batch['labels']).float().mean().item()
                
                return {
                    'loss': outputs.get('loss', 0.0).item(),
                    'accuracy': accuracy
                }
            
            return {'loss': outputs.get('loss', 0.0).item()}
    
    def reload_and_continue(self, checkpoint_path: str) -> None:
        """Reload model and continue training."""
        self.model.reload_parameters(checkpoint_path)
        logger.info("Model reloaded, continuing training...")


def create_model(config: ModelConfig) -> TReloadModel:
    """Create a t-reload model instance."""
    return TReloadModel(config)
