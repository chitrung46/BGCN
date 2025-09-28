#!/usr/bin/env python3
"""
Quick start example for BGCN Sequential Recommendation on Amazon Beauty dataset.

This script demonstrates the minimal code needed to get started with the dataset and dataloader.
"""

from data_utils import preprocess_data
from dataloader import create_dataloaders
import torch


def main():
    """Main function demonstrating basic usage."""
    print("🚀 BGCN Sequential Recommendation - Quick Start")
    print("=" * 55)
    
    # Step 1: Load the data
    print("📊 Loading Amazon Beauty dataset...")
    try:
        data = preprocess_data()
        print(f"✅ Dataset loaded: {data['num_users']} users, {data['num_items']} items")
    except Exception as e:
        print(f"⚠️  Could not download data: {e}")
        print("📝 This is normal in restricted environments. Using mock data for demo...")
        
        # Create simple mock data for demonstration
        import numpy as np
        np.random.seed(42)
        sequences = [[np.random.randint(1, 101) for _ in range(np.random.randint(5, 15))] 
                    for _ in range(100)]
        
        data = {
            'sequences': sequences,
            'num_users': 100,
            'num_items': 100,
            'user_to_idx': {f'user_{i}': i for i in range(100)},
            'item_to_idx': {f'item_{i}': i + 1 for i in range(100)},
            'idx_to_user': {i: f'user_{i}' for i in range(100)},
            'idx_to_item': {i + 1: f'item_{i}' for i in range(100)},
            'df': None
        }
        print(f"✅ Mock dataset created: {data['num_users']} users, {data['num_items']} items")
    
    # Step 2: Create data loaders
    print("\n🔄 Creating data loaders...")
    train_loader, test_loader, data_info = create_dataloaders(
        data, 
        batch_size=32,  # Smaller batch size for demo
        num_workers=0   # No multiprocessing for simplicity
    )
    print(f"✅ Data loaders ready!")
    print(f"   📈 Training batches: {len(train_loader)}")
    print(f"   📊 Test batches: {len(test_loader)}")
    print(f"   📏 Average sequence length: {data_info['avg_seq_len']:.1f}")
    
    # Step 3: Show sample batch
    print("\n🔍 Sample training batch:")
    sample_batch = next(iter(train_loader))
    
    print(f"   🔢 Input sequences: {sample_batch['input_seq'].shape}")
    print(f"   🎯 Target items: {sample_batch['target_item'].shape}")
    print(f"   👁️  Attention masks: {sample_batch['mask'].shape}")
    print(f"   ➖ Negative samples: {sample_batch['negative_items'].shape}")
    
    # Step 4: Show how to use in a training loop
    print("\n⚙️  Example training loop structure:")
    print("""
    for epoch in range(num_epochs):
        for batch in train_loader:
            # Get data
            input_seq = batch['input_seq']        # [batch_size, seq_len]
            target = batch['target_item']         # [batch_size]
            mask = batch['mask']                  # [batch_size, seq_len]
            negatives = batch['negative_items']   # [batch_size, num_neg]
            
            # Forward pass
            # logits = model(input_seq, mask)
            
            # Compute loss
            # loss = compute_loss(logits, target, negatives)
            
            # Backward pass
            # loss.backward()
            # optimizer.step()
    """)
    
    # Step 5: Quick data validation
    print("✅ Data validation:")
    batch = sample_batch
    seq_lens = batch['seq_len']
    masks = batch['mask']
    
    # Check that sequence lengths match masks
    mask_sums = masks.sum(dim=1)
    length_match = torch.all(seq_lens == mask_sums)
    print(f"   ✓ Sequence lengths match masks: {length_match}")
    
    # Check target items are in valid range
    targets = batch['target_item']
    valid_targets = torch.all((targets >= 1) & (targets <= data['num_items']))
    print(f"   ✓ Target items in valid range: {valid_targets}")
    
    # Check negative samples are different from targets
    negatives = batch['negative_items']
    different_negatives = torch.all(negatives != targets.unsqueeze(1))
    print(f"   ✓ Negative samples different from targets: {different_negatives}")
    
    print("\n🎉 Quick start completed successfully!")
    print("\n📚 Next steps:")
    print("   1. Run 'python demo.py' for a detailed demonstration")
    print("   2. Run 'python test_implementation.py' to validate the setup")
    print("   3. Check the README.md for comprehensive documentation")
    print("   4. Implement your sequential recommendation model!")


if __name__ == "__main__":
    main()