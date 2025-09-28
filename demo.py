"""Demo script showing how to use the dataset and dataloader for sequential recommendation."""

import torch
import numpy as np
from data_utils import preprocess_data
from dataloader import create_dataloaders, DataLoaderManager
from dataset import SequentialDataset, NegativeSamplingDataset
from config import Config


def demo_basic_usage():
    """Demonstrate basic usage of the dataset and dataloader."""
    
    print("="*50)
    print("BGCN Sequential Recommendation Demo")
    print("="*50)
    
    # Step 1: Load and preprocess data
    print("\n1. Loading and preprocessing Amazon Beauty data...")
    try:
        data = preprocess_data()
        print("✓ Data loaded successfully!")
    except Exception as e:
        print(f"✗ Error loading data: {e}")
        print("This might be due to network issues. The system is designed to handle this gracefully.")
        # For demo purposes, create mock data
        print("Creating mock data for demonstration...")
        data = create_mock_data()
    
    # Step 2: Create data loaders
    print("\n2. Creating data loaders...")
    train_loader, test_loader, data_info = create_dataloaders(
        data, 
        batch_size=32, 
        num_workers=0
    )
    
    print("✓ Data loaders created successfully!")
    print(f"  Number of users: {data_info['num_users']}")
    print(f"  Number of items: {data_info['num_items']}")
    print(f"  Training samples: {data_info['train_size']}")
    print(f"  Test samples: {data_info['test_size']}")
    print(f"  Average sequence length: {data_info['avg_seq_len']:.2f}")
    
    # Step 3: Demonstrate training data
    print("\n3. Examining training data batches...")
    for i, batch in enumerate(train_loader):
        print(f"\nBatch {i+1}:")
        print(f"  Input sequences shape: {batch['input_seq'].shape}")
        print(f"  Target items shape: {batch['target_item'].shape}")
        print(f"  Attention masks shape: {batch['mask'].shape}")
        print(f"  Sequence lengths shape: {batch['seq_len'].shape}")
        if 'negative_items' in batch:
            print(f"  Negative items shape: {batch['negative_items'].shape}")
        
        # Show sample data from first batch
        if i == 0:
            print(f"\n  Sample from batch:")
            print(f"    Input sequence: {batch['input_seq'][0][:10]}...")  # First 10 items
            print(f"    Target item: {batch['target_item'][0].item()}")
            print(f"    Sequence length: {batch['seq_len'][0].item()}")
            print(f"    Attention mask: {batch['mask'][0][:10]}...")  # First 10 mask values
        
        if i >= 2:  # Show only first 3 batches
            break
    
    # Step 4: Demonstrate test data
    print("\n4. Examining test data batches...")
    for i, batch in enumerate(test_loader):
        print(f"\nTest Batch {i+1}:")
        print(f"  Input sequences shape: {batch['input_seq'].shape}")
        print(f"  Target items shape: {batch['target_item'].shape}")
        print(f"  Attention masks shape: {batch['mask'].shape}")
        print(f"  Sequence lengths shape: {batch['seq_len'].shape}")
        
        if i >= 1:  # Show only first 2 batches
            break
    
    # Step 5: Show data statistics
    print("\n5. Data Statistics:")
    sequences = data['sequences']
    seq_lengths = [len(seq) for seq in sequences]
    
    print(f"  Total sequences: {len(sequences)}")
    print(f"  Min sequence length: {min(seq_lengths)}")
    print(f"  Max sequence length: {max(seq_lengths)}")
    print(f"  Median sequence length: {np.median(seq_lengths):.2f}")
    print(f"  Std sequence length: {np.std(seq_lengths):.2f}")
    
    # Step 6: Show configuration
    print("\n6. Configuration:")
    config_vars = [attr for attr in dir(Config) if not attr.startswith('_') and not callable(getattr(Config, attr))]
    for var in config_vars:
        print(f"  {var}: {getattr(Config, var)}")
    
    print("\n" + "="*50)
    print("Demo completed successfully!")
    print("="*50)


def create_mock_data():
    """Create mock data for demonstration when real data is unavailable."""
    print("Creating mock Amazon Beauty dataset...")
    
    # Mock parameters
    num_users = 1000
    num_items = 5000
    min_seq_len = 5
    max_seq_len = 30
    
    # Generate mock sequences
    sequences = []
    for user_id in range(num_users):
        seq_len = np.random.randint(min_seq_len, max_seq_len + 1)
        # Create a sequence with some repeated items (realistic user behavior)
        sequence = []
        for _ in range(seq_len):
            if len(sequence) > 0 and np.random.random() < 0.3:  # 30% chance to repeat
                item = np.random.choice(sequence)
            else:
                item = np.random.randint(1, num_items + 1)
            sequence.append(item)
        sequences.append(sequence)
    
    # Create mappings
    user_to_idx = {f"user_{i}": i for i in range(num_users)}
    item_to_idx = {f"item_{i}": i + 1 for i in range(num_items)}
    idx_to_user = {i: f"user_{i}" for i in range(num_users)}
    idx_to_item = {i + 1: f"item_{i}" for i in range(num_items)}
    
    return {
        'sequences': sequences,
        'num_users': num_users,
        'num_items': num_items,
        'user_to_idx': user_to_idx,
        'item_to_idx': item_to_idx,
        'idx_to_user': idx_to_user,
        'idx_to_item': idx_to_item,
        'df': None  # Not needed for demo
    }


def demonstrate_custom_usage():
    """Show how to use the components with custom parameters."""
    print("\n" + "="*50)
    print("Custom Usage Demonstration")
    print("="*50)
    
    # Create mock data
    data = create_mock_data()
    
    # Create custom configuration
    class CustomConfig(Config):
        MAX_SEQ_LEN = 20
        BATCH_SIZE = 64
        TRAIN_RATIO = 0.7
    
    # Use DataLoaderManager for more control
    print("\n1. Using DataLoaderManager with custom config...")
    manager = DataLoaderManager(data, CustomConfig())
    
    # Get data loaders with custom parameters
    train_loader = manager.get_train_loader(batch_size=128, shuffle=True)
    test_loader = manager.get_test_loader(batch_size=64)
    
    print(f"✓ Custom loaders created:")
    print(f"  Train loader batch size: 128")
    print(f"  Test loader batch size: 64")
    print(f"  Max sequence length: {CustomConfig.MAX_SEQ_LEN}")
    
    # Test a batch
    batch = next(iter(train_loader))
    print(f"\n2. Sample batch:")
    for key, value in batch.items():
        print(f"  {key}: {value.shape if hasattr(value, 'shape') else type(value)}")


if __name__ == "__main__":
    # Run the main demo
    demo_basic_usage()
    
    # Run custom usage demo
    demonstrate_custom_usage()