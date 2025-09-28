"""DataLoader utilities for sequential recommendation."""

from torch.utils.data import DataLoader, Sampler
import torch
import numpy as np
import random
from dataset import SequentialDataset, NegativeSamplingDataset, create_train_test_split, collate_fn
from config import Config


class SequentialSampler(Sampler):
    """Custom sampler that groups sequences by similar lengths for efficient batching."""
    
    def __init__(self, dataset, batch_size, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        # Calculate sequence lengths
        self.lengths = []
        for i in range(len(dataset)):
            sample = dataset[i]
            seq_len = sample['seq_len'].item()
            self.lengths.append((i, seq_len))
        
        # Sort by length
        self.lengths.sort(key=lambda x: x[1])
        
    def __iter__(self):
        if self.shuffle:
            # Shuffle within length groups
            batch_indices = []
            for i in range(0, len(self.lengths), self.batch_size):
                batch = self.lengths[i:i + self.batch_size]
                random.shuffle(batch)
                batch_indices.extend(batch)
            self.lengths = batch_indices
        
        indices = [idx for idx, _ in self.lengths]
        return iter(indices)
    
    def __len__(self):
        return len(self.dataset)


class DataLoaderManager:
    """Manager class for creating and handling data loaders."""
    
    def __init__(self, data_dict, config=None):
        """
        Initialize the data loader manager.
        
        Args:
            data_dict: Dictionary containing preprocessed data
            config: Configuration object
        """
        self.data_dict = data_dict
        self.config = config or Config()
        
        self.sequences = data_dict['sequences']
        self.num_items = data_dict['num_items']
        
        # Create train/test split
        self.train_sequences, self.test_sequences = create_train_test_split(
            self.sequences, self.config.TRAIN_RATIO
        )
        
        # Create datasets
        self.train_dataset = NegativeSamplingDataset(
            self.train_sequences, 
            self.num_items, 
            max_len=self.config.MAX_SEQ_LEN,
            mode='train',
            num_negatives=1
        )
        
        self.test_dataset = SequentialDataset(
            self.test_sequences,
            self.num_items,
            max_len=self.config.MAX_SEQ_LEN,
            mode='test'
        )
        
        print(f"DataLoaderManager initialized:")
        print(f"  Train dataset: {len(self.train_dataset)} samples")
        print(f"  Test dataset: {len(self.test_dataset)} samples")
        print(f"  Number of items: {self.num_items}")
        print(f"  Max sequence length: {self.config.MAX_SEQ_LEN}")
    
    def get_train_loader(self, batch_size=None, shuffle=True, num_workers=0):
        """Get training data loader."""
        batch_size = batch_size or self.config.BATCH_SIZE
        
        if shuffle:
            sampler = SequentialSampler(self.train_dataset, batch_size, shuffle=True)
            shuffle = False  # Don't use DataLoader's shuffle when using custom sampler
        else:
            sampler = None
        
        return DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            collate_fn=collate_fn,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available()
        )
    
    def get_test_loader(self, batch_size=None, num_workers=0):
        """Get testing data loader."""
        batch_size = batch_size or self.config.BATCH_SIZE
        
        return DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available()
        )
    
    def get_validation_loader(self, batch_size=None, num_workers=0):
        """Get validation data loader (same as test for now)."""
        return self.get_test_loader(batch_size, num_workers)
    
    def get_data_info(self):
        """Get information about the data."""
        return {
            'num_users': self.data_dict['num_users'],
            'num_items': self.num_items,
            'train_size': len(self.train_dataset),
            'test_size': len(self.test_dataset),
            'max_seq_len': self.config.MAX_SEQ_LEN,
            'avg_seq_len': np.mean([len(seq) for seq in self.sequences]),
            'user_to_idx': self.data_dict['user_to_idx'],
            'item_to_idx': self.data_dict['item_to_idx'],
            'idx_to_user': self.data_dict['idx_to_user'],
            'idx_to_item': self.data_dict['idx_to_item']
        }


def create_dataloaders(data_dict, config=None, batch_size=None, num_workers=0):
    """
    Convenience function to create train and test data loaders.
    
    Args:
        data_dict: Dictionary containing preprocessed data
        config: Configuration object
        batch_size: Batch size (optional)
        num_workers: Number of worker processes
    
    Returns:
        train_loader, test_loader, data_info
    """
    manager = DataLoaderManager(data_dict, config)
    
    train_loader = manager.get_train_loader(batch_size, num_workers=num_workers)
    test_loader = manager.get_test_loader(batch_size, num_workers=num_workers)
    data_info = manager.get_data_info()
    
    return train_loader, test_loader, data_info


if __name__ == "__main__":
    from data_utils import preprocess_data
    
    print("Loading and preprocessing data...")
    data = preprocess_data()
    
    print("Creating data loaders...")
    train_loader, test_loader, data_info = create_dataloaders(data, batch_size=32)
    
    print("Data info:")
    for key, value in data_info.items():
        if not isinstance(value, dict):  # Skip the mapping dictionaries
            print(f"  {key}: {value}")
    
    print("\nTesting train loader...")
    for i, batch in enumerate(train_loader):
        print(f"Batch {i}:")
        for key, value in batch.items():
            print(f"  {key}: {value.shape if hasattr(value, 'shape') else type(value)}")
        
        if i >= 2:  # Only show first few batches
            break
    
    print("\nTesting test loader...")
    for i, batch in enumerate(test_loader):
        print(f"Test Batch {i}:")
        for key, value in batch.items():
            print(f"  {key}: {value.shape if hasattr(value, 'shape') else type(value)}")
        
        if i >= 1:  # Only show first batch
            break