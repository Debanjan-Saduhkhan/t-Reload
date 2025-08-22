from __future__ import annotations
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

logger = logging.getLogger(__name__)


class TReloadDataset(Dataset):
    """
    Dataset class for t-reload method implementation.
    
    This dataset handles the specific data format and preprocessing
    required for the t-reload approach described in the paper.
    """
    
    def __init__(
        self,
        data_path: Union[str, Path],
        max_length: int = 512,
        tokenizer=None,
        split: str = "train"
    ):
        """
        Initialize the dataset.
        
        Args:
            data_path: Path to the data file
            max_length: Maximum sequence length
            tokenizer: Tokenizer for text processing
            split: Dataset split (train/val/test)
        """
        self.data_path = Path(data_path)
        self.max_length = max_length
        self.tokenizer = tokenizer
        self.split = split
        
        # Load data
        self.data = self._load_data()
        logger.info(f"Loaded {len(self.data)} samples for {split} split")
    
    def _load_data(self) -> List[Dict[str, Any]]:
        """Load data from file."""
        try:
            with open(self.data_path, 'r', encoding='utf-8') as f:
                if self.data_path.suffix == '.json':
                    data = json.load(f)
                else:
                    # Handle other formats if needed
                    data = [line.strip() for line in f if line.strip()]
            
            # Ensure data is in list format
            if isinstance(data, dict):
                data = data.get(self.split, [])
            elif not isinstance(data, list):
                data = [data]
                
            return data
        except Exception as e:
            logger.error(f"Failed to load data from {self.data_path}: {e}")
            return []
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single data sample."""
        sample = self.data[idx]
        
        # Process the sample based on its format
        if isinstance(sample, str):
            # Simple text sample
            processed = self._process_text(sample)
        elif isinstance(sample, dict):
            # Structured sample
            processed = self._process_dict(sample)
        else:
            raise ValueError(f"Unsupported sample type: {type(sample)}")
        
        return processed
    
    def _process_text(self, text: str) -> Dict[str, torch.Tensor]:
        """Process text sample."""
        if self.tokenizer:
            # Use provided tokenizer
            encoding = self.tokenizer(
                text,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            return {
                'input_ids': encoding['input_ids'].squeeze(0),
                'attention_mask': encoding['attention_mask'].squeeze(0)
            }
        else:
            # Simple character-level processing as fallback
            chars = list(text[:self.max_length])
            input_ids = torch.tensor([ord(c) for c in chars], dtype=torch.long)
            attention_mask = torch.ones_like(input_ids)
            
            # Pad to max_length
            if len(input_ids) < self.max_length:
                padding = torch.zeros(self.max_length - len(input_ids), dtype=torch.long)
                input_ids = torch.cat([input_ids, padding])
                attention_mask = torch.cat([attention_mask, padding])
            
            return {
                'input_ids': input_ids,
                'attention_mask': attention_mask
            }
    
    def _process_dict(self, sample: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Process dictionary sample."""
        # Extract text fields
        text_fields = ['text', 'input', 'content', 'sentence']
        text = None
        
        for field in text_fields:
            if field in sample:
                text = sample[field]
                break
        
        if text is None:
            raise ValueError(f"No text field found in sample: {sample}")
        
        # Process text
        processed = self._process_text(text)
        
        # Add labels if available
        if 'label' in sample:
            processed['labels'] = torch.tensor(sample['label'], dtype=torch.long)
        
        return processed


class TReloadDataLoader:
    """Data loader factory for t-reload experiments."""
    
    @staticmethod
    def create_dataloaders(
        train_path: Union[str, Path],
        val_path: Optional[Union[str, Path]] = None,
        test_path: Optional[Union[str, Path]] = None,
        config: Optional[Dict[str, Any]] = None,
        tokenizer=None
    ) -> Tuple[DataLoader, Optional[DataLoader], Optional[DataLoader]]:
        """
        Create data loaders for training, validation, and testing.
        
        Args:
            train_path: Path to training data
            val_path: Path to validation data
            test_path: Path to test data
            config: Configuration dictionary
            tokenizer: Tokenizer for text processing
            
        Returns:
            Tuple of (train_loader, val_loader, test_loader)
        """
        if config is None:
            config = {}
        
        # Extract configuration
        batch_size = config.get('batch_size', 32)
        max_length = config.get('max_length', 512)
        num_workers = config.get('num_workers', 4)
        
        # Create datasets
        train_dataset = TReloadDataset(
            train_path, max_length=max_length, tokenizer=tokenizer, split="train"
        )
        
        val_loader = None
        if val_path:
            val_dataset = TReloadDataset(
                val_path, max_length=max_length, tokenizer=tokenizer, split="val"
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True
            )
        
        test_loader = None
        if test_path:
            test_dataset = TReloadDataset(
                test_path, max_length=max_length, tokenizer=tokenizer, split="test"
            )
            test_loader = DataLoader(
                test_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True
            )
        
        # Create train loader
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True
        )
        
        return train_loader, val_loader, test_loader


def create_sample_data(output_dir: Union[str, Path]) -> None:
    """Create sample data files for testing."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Sample training data
    train_data = [
        {"text": "This is a sample sentence for training.", "label": 0},
        {"text": "Another example text for the dataset.", "label": 1},
        {"text": "Sample data to test the implementation.", "label": 0}
    ]
    
    with open(output_dir / "train.json", 'w') as f:
        json.dump(train_data, f, indent=2)
    
    # Sample validation data
    val_data = [
        {"text": "Validation sample text.", "label": 1},
        {"text": "Another validation example.", "label": 0}
    ]
    
    with open(output_dir / "val.json", 'w') as f:
        json.dump(val_data, f, indent=2)
    
    # Sample test data
    test_data = [
        {"text": "Test sample for evaluation.", "label": 0},
        {"text": "Final test example.", "label": 1}
    ]
    
    with open(output_dir / "test.json", 'w') as f:
        json.dump(test_data, f, indent=2)
    
    logger.info(f"Sample data created in {output_dir}")
