"""
Simple tests for the sequential recommendation dataset and dataloader.
Run this to validate the implementation works correctly.
"""

import sys
import os
import unittest
import torch
import numpy as np
from unittest.mock import patch

# Add the current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import Config
from dataset import SequentialDataset, NegativeSamplingDataset, create_train_test_split
from dataloader import DataLoaderManager, create_dataloaders


class TestSequentialRecommendation(unittest.TestCase):
    
    def setUp(self):
        """Set up test data."""
        # Create mock data
        self.num_users = 100
        self.num_items = 500
        
        # Generate mock sequences
        np.random.seed(42)
        self.sequences = []
        for _ in range(self.num_users):
            seq_len = np.random.randint(5, 20)
            sequence = np.random.randint(1, self.num_items + 1, seq_len).tolist()
            self.sequences.append(sequence)
        
        self.data_dict = {
            'sequences': self.sequences,
            'num_users': self.num_users,
            'num_items': self.num_items,
            'user_to_idx': {f'user_{i}': i for i in range(self.num_users)},
            'item_to_idx': {f'item_{i}': i + 1 for i in range(self.num_items)},
            'idx_to_user': {i: f'user_{i}' for i in range(self.num_users)},
            'idx_to_item': {i + 1: f'item_{i}' for i in range(self.num_items)},
            'df': None
        }
    
    def test_config(self):
        """Test configuration class."""
        self.assertIsInstance(Config.MAX_SEQ_LEN, int)
        self.assertIsInstance(Config.BATCH_SIZE, int)
        self.assertIsInstance(Config.TRAIN_RATIO, float)
        self.assertGreater(Config.MAX_SEQ_LEN, 0)
        self.assertGreater(Config.BATCH_SIZE, 0)
        self.assertTrue(0 < Config.TRAIN_RATIO < 1)
    
    def test_sequential_dataset(self):
        """Test basic sequential dataset."""
        dataset = SequentialDataset(self.sequences, self.num_items, max_len=10, mode='train')
        
        # Test dataset length
        self.assertGreater(len(dataset), 0)
        self.assertLessEqual(len(dataset), len(self.sequences))
        
        # Test sample structure
        sample = dataset[0]
        self.assertIn('input_seq', sample)
        self.assertIn('target_item', sample)
        self.assertIn('mask', sample)
        self.assertIn('seq_len', sample)
        
        # Test tensor types and shapes
        self.assertIsInstance(sample['input_seq'], torch.Tensor)
        self.assertIsInstance(sample['target_item'], torch.Tensor)
        self.assertEqual(sample['input_seq'].shape[0], 10)  # max_len
        self.assertEqual(len(sample['target_item'].shape), 0)  # scalar
    
    def test_negative_sampling_dataset(self):
        """Test dataset with negative sampling."""
        dataset = NegativeSamplingDataset(
            self.sequences, 
            self.num_items, 
            max_len=10, 
            mode='train',
            num_negatives=3
        )
        
        # Test sample structure
        sample = dataset[0]
        self.assertIn('negative_items', sample)
        self.assertEqual(sample['negative_items'].shape[0], 3)  # num_negatives
    
    def test_train_test_split(self):
        """Test train/test split functionality."""
        train_sequences, test_sequences = create_train_test_split(self.sequences, train_ratio=0.8)
        
        self.assertGreater(len(train_sequences), 0)
        self.assertGreater(len(test_sequences), 0)
        self.assertEqual(len(train_sequences), len(test_sequences))  # Same number of users
    
    def test_dataloader_manager(self):
        """Test DataLoaderManager functionality."""
        manager = DataLoaderManager(self.data_dict)
        
        # Test data loaders creation
        train_loader = manager.get_train_loader(batch_size=32)
        test_loader = manager.get_test_loader(batch_size=32)
        
        self.assertIsNotNone(train_loader)
        self.assertIsNotNone(test_loader)
        
        # Test data info
        data_info = manager.get_data_info()
        self.assertIn('num_users', data_info)
        self.assertIn('num_items', data_info)
        self.assertEqual(data_info['num_users'], self.num_users)
        self.assertEqual(data_info['num_items'], self.num_items)
    
    def test_dataloader_batches(self):
        """Test that data loader produces valid batches."""
        train_loader, test_loader, data_info = create_dataloaders(
            self.data_dict, 
            batch_size=16
        )
        
        # Test train batch
        train_batch = next(iter(train_loader))
        self.assertIn('input_seq', train_batch)
        self.assertIn('target_item', train_batch)
        self.assertIn('mask', train_batch)
        self.assertIn('negative_items', train_batch)
        
        # Check batch dimensions
        batch_size = train_batch['input_seq'].shape[0]
        self.assertLessEqual(batch_size, 16)
        self.assertEqual(train_batch['target_item'].shape[0], batch_size)
        self.assertEqual(train_batch['mask'].shape[0], batch_size)
        
        # Test test batch
        test_batch = next(iter(test_loader))
        self.assertIn('input_seq', test_batch)
        self.assertIn('target_item', test_batch)
        self.assertIn('mask', test_batch)
        self.assertNotIn('negative_items', test_batch)  # No negatives in test
    
    def test_sequence_lengths(self):
        """Test that sequence lengths are handled correctly."""
        dataset = SequentialDataset(self.sequences, self.num_items, max_len=15, mode='test')
        
        for i in range(min(10, len(dataset))):  # Test first 10 samples
            sample = dataset[i]
            seq_len = sample['seq_len'].item()
            mask = sample['mask']
            
            # Check that sequence length matches mask
            self.assertEqual(seq_len, mask.sum().item())
            
            # Check that non-masked positions come before masked ones
            mask_np = mask.numpy()
            if seq_len < len(mask_np):
                # There should be zeros (padding) at the beginning
                padding_end = len(mask_np) - seq_len
                self.assertTrue(all(mask_np[:padding_end] == 0))
                self.assertTrue(all(mask_np[padding_end:] == 1))
    
    def test_item_indices(self):
        """Test that item indices are in valid range."""
        dataset = SequentialDataset(self.sequences, self.num_items, mode='train')
        
        for i in range(min(10, len(dataset))):
            sample = dataset[i]
            input_seq = sample['input_seq']
            target_item = sample['target_item']
            
            # Check target item is in valid range
            self.assertGreaterEqual(target_item.item(), 1)
            self.assertLessEqual(target_item.item(), self.num_items)
            
            # Check input sequence items (non-zero) are in valid range
            non_zero_items = input_seq[input_seq != 0]
            if len(non_zero_items) > 0:
                self.assertTrue(torch.all(non_zero_items >= 1))
                self.assertTrue(torch.all(non_zero_items <= self.num_items))


def run_basic_tests():
    """Run basic functionality tests without unittest framework."""
    print("Running Basic Functionality Tests...")
    print("=" * 50)
    
    # Test 1: Configuration
    print("✓ Testing configuration...")
    config = Config()
    assert hasattr(config, 'MAX_SEQ_LEN')
    assert hasattr(config, 'BATCH_SIZE')
    print(f"  Max sequence length: {config.MAX_SEQ_LEN}")
    print(f"  Batch size: {config.BATCH_SIZE}")
    
    # Test 2: Mock data creation
    print("\n✓ Creating mock data...")
    np.random.seed(42)
    num_users, num_items = 50, 200
    sequences = []
    for _ in range(num_users):
        seq_len = np.random.randint(5, 15)
        sequence = np.random.randint(1, num_items + 1, seq_len).tolist()
        sequences.append(sequence)
    
    data_dict = {
        'sequences': sequences,
        'num_users': num_users,
        'num_items': num_items,
        'user_to_idx': {f'user_{i}': i for i in range(num_users)},
        'item_to_idx': {f'item_{i}': i + 1 for i in range(num_items)},
        'idx_to_user': {i: f'user_{i}' for i in range(num_users)},
        'idx_to_item': {i + 1: f'item_{i}' for i in range(num_items)},
        'df': None
    }
    print(f"  Created {num_users} users with {num_items} items")
    
    # Test 3: Dataset creation
    print("\n✓ Testing dataset creation...")
    dataset = SequentialDataset(sequences, num_items, max_len=10, mode='train')
    print(f"  Dataset size: {len(dataset)}")
    
    sample = dataset[0]
    print(f"  Sample keys: {list(sample.keys())}")
    print(f"  Input sequence shape: {sample['input_seq'].shape}")
    print(f"  Target item: {sample['target_item'].item()}")
    
    # Test 4: DataLoader creation
    print("\n✓ Testing dataloader creation...")
    try:
        train_loader, test_loader, data_info = create_dataloaders(
            data_dict, batch_size=8
        )
        print(f"  Train loader created successfully")
        print(f"  Test loader created successfully")
        print(f"  Data info: {len(data_info)} fields")
        
        # Test batch
        batch = next(iter(train_loader))
        print(f"  First batch shape: {batch['input_seq'].shape}")
        
    except Exception as e:
        print(f"  Error creating dataloaders: {e}")
        return False
    
    print("\n" + "=" * 50)
    print("✓ All basic tests passed!")
    return True


if __name__ == "__main__":
    # Run basic tests first
    success = run_basic_tests()
    
    if success:
        print("\n" + "=" * 50)
        print("Running Detailed Unit Tests...")
        print("=" * 50)
        
        # Run unittest suite
        unittest.main(argv=[''], exit=False, verbosity=2)
    else:
        print("Basic tests failed. Skipping detailed tests.")
        sys.exit(1)