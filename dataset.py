"""Sequential Recommendation Dataset for Amazon Beauty."""

import torch
from torch.utils.data import Dataset
import numpy as np
import random
from config import Config


class SequentialDataset(Dataset):
    """Dataset for sequential recommendation."""
    
    def __init__(self, sequences, num_items, max_len=None, mode='train'):
        """
        Initialize the dataset.
        
        Args:
            sequences: List of user interaction sequences
            num_items: Total number of items
            max_len: Maximum sequence length
            mode: 'train' or 'test'
        """
        self.sequences = sequences
        self.num_items = num_items
        self.max_len = max_len or Config.MAX_SEQ_LEN
        self.mode = mode
        
        # Filter sequences that are too short
        self.valid_sequences = [seq for seq in sequences if len(seq) >= 3]
        
        print(f"Dataset initialized with {len(self.valid_sequences)} valid sequences")
        print(f"Mode: {mode}, Max length: {self.max_len}")
    
    def __len__(self):
        return len(self.valid_sequences)
    
    def __getitem__(self, idx):
        sequence = self.valid_sequences[idx].copy()
        
        if self.mode == 'train':
            # For training, we create multiple samples from one sequence
            # by using different subsequences as input and next item as target
            
            # Randomly select a subsequence end position
            end_pos = random.randint(2, len(sequence))  # At least 2 items for input
            input_seq = sequence[:end_pos-1]
            target_item = sequence[end_pos-1]
            
        else:
            # For testing, use all but last item as input, last item as target
            input_seq = sequence[:-1]
            target_item = sequence[-1]
        
        # Pad or truncate sequence
        if len(input_seq) > self.max_len:
            input_seq = input_seq[-self.max_len:]
        else:
            # Pad with zeros (0 is reserved for padding)
            input_seq = [0] * (self.max_len - len(input_seq)) + input_seq
        
        # Create attention mask
        mask = [1 if item != 0 else 0 for item in input_seq]
        
        return {
            'input_seq': torch.tensor(input_seq, dtype=torch.long),
            'target_item': torch.tensor(target_item, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.float),
            'seq_len': torch.tensor(sum(mask), dtype=torch.long)
        }


class NegativeSamplingDataset(SequentialDataset):
    """Dataset with negative sampling for sequential recommendation."""
    
    def __init__(self, sequences, num_items, max_len=None, mode='train', num_negatives=1):
        """
        Initialize the dataset with negative sampling.
        
        Args:
            sequences: List of user interaction sequences
            num_items: Total number of items
            max_len: Maximum sequence length
            mode: 'train' or 'test'
            num_negatives: Number of negative samples per positive sample
        """
        super().__init__(sequences, num_items, max_len, mode)
        self.num_negatives = num_negatives
        
        # Create set of all items for negative sampling
        self.all_items = set(range(1, num_items + 1))  # Exclude 0 (padding)
        
        # Create user item sets for negative sampling
        self.user_items = {}
        for i, seq in enumerate(self.valid_sequences):
            self.user_items[i] = set(seq)
    
    def __getitem__(self, idx):
        data = super().__getitem__(idx)
        
        if self.mode == 'train':
            # Generate negative samples
            user_items = self.user_items[idx]
            negative_items = []
            
            while len(negative_items) < self.num_negatives:
                neg_item = random.randint(1, self.num_items)
                if neg_item not in user_items:
                    negative_items.append(neg_item)
            
            data['negative_items'] = torch.tensor(negative_items, dtype=torch.long)
        
        return data


def create_train_test_split(sequences, train_ratio=None):
    """
    Create train/test split for sequential recommendation.
    
    Args:
        sequences: List of user interaction sequences
        train_ratio: Ratio of training data
    
    Returns:
        train_sequences, test_sequences
    """
    train_ratio = train_ratio or Config.TRAIN_RATIO
    
    train_sequences = []
    test_sequences = []
    
    for seq in sequences:
        if len(seq) < 3:  # Skip sequences that are too short
            continue
            
        # Split each user's sequence
        split_point = max(2, int(len(seq) * train_ratio))
        
        train_seq = seq[:split_point]
        test_seq = seq  # For test, we use the full sequence
        
        if len(train_seq) >= 2:  # Ensure train sequence has at least 2 items
            train_sequences.append(train_seq)
            test_sequences.append(test_seq)
    
    print(f"Created {len(train_sequences)} train sequences and {len(test_sequences)} test sequences")
    return train_sequences, test_sequences


def collate_fn(batch):
    """Custom collate function for DataLoader."""
    keys = batch[0].keys()
    collated = {}
    
    for key in keys:
        if key == 'negative_items':
            # Handle variable length negative items
            collated[key] = torch.stack([item[key] for item in batch if key in item])
        else:
            collated[key] = torch.stack([item[key] for item in batch])
    
    return collated


if __name__ == "__main__":
    # Test the dataset
    from data_utils import preprocess_data
    
    print("Loading data...")
    data = preprocess_data()
    sequences = data['sequences']
    num_items = data['num_items']
    
    print("Creating train/test split...")
    train_sequences, test_sequences = create_train_test_split(sequences)
    
    print("Creating datasets...")
    train_dataset = NegativeSamplingDataset(train_sequences, num_items, mode='train')
    test_dataset = SequentialDataset(test_sequences, num_items, mode='test')
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")
    
    # Test a sample
    sample = train_dataset[0]
    print("Sample from training dataset:")
    for key, value in sample.items():
        print(f"{key}: {value}")