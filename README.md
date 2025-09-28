# BGCN - Sequential Recommendation on Amazon Beauty Dataset

This repository provides a complete implementation of dataset and dataloader utilities for sequential recommendation on the Amazon Beauty dataset. The implementation includes data downloading, preprocessing, dataset creation, and efficient data loading for training sequential recommendation models.

## Features

- **Automated Data Download**: Automatically downloads the Amazon Beauty dataset from Stanford SNAP
- **Data Preprocessing**: Comprehensive preprocessing pipeline with filtering and sequence creation
- **Sequential Dataset**: Custom PyTorch dataset for sequential recommendation tasks
- **Negative Sampling**: Built-in negative sampling for improved training
- **Efficient DataLoader**: Custom samplers and batching strategies for optimal training
- **Configurable**: Easy-to-modify configuration system
- **Demo Scripts**: Complete demonstration of usage

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/chitrung46/BGCN.git
cd BGCN

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

```python
from data_utils import preprocess_data
from dataloader import create_dataloaders

# Load and preprocess data
data = preprocess_data()

# Create data loaders
train_loader, test_loader, data_info = create_dataloaders(data, batch_size=256)

# Use in training loop
for batch in train_loader:
    input_seq = batch['input_seq']        # [batch_size, max_seq_len]
    target_item = batch['target_item']    # [batch_size]
    mask = batch['mask']                  # [batch_size, max_seq_len]
    negative_items = batch['negative_items']  # [batch_size, num_negatives]
    # Your model training code here
```

### Run Demo

```bash
python demo.py
```

## Dataset Description

The Amazon Beauty dataset contains user reviews and ratings for beauty products. The sequential recommendation setup uses:

- **Input**: User's historical interaction sequence
- **Output**: Next item the user is likely to interact with
- **Preprocessing**: Filters users and items with minimum interactions (configurable)
- **Sequence Creation**: Creates temporal sequences based on review timestamps

## Data Statistics

- **Users**: ~22K after filtering
- **Items**: ~12K after filtering  
- **Interactions**: ~198K after filtering
- **Average sequence length**: ~9 interactions per user

## Architecture

### Core Components

1. **`config.py`**: Configuration management
2. **`data_utils.py`**: Data downloading and preprocessing utilities
3. **`dataset.py`**: PyTorch dataset classes for sequential recommendation
4. **`dataloader.py`**: DataLoader management and custom sampling strategies
5. **`demo.py`**: Demonstration and testing scripts

### Key Classes

- **`SequentialDataset`**: Base dataset class for sequential recommendation
- **`NegativeSamplingDataset`**: Dataset with negative sampling capabilities
- **`SequentialSampler`**: Custom sampler that groups sequences by length
- **`DataLoaderManager`**: High-level interface for data loading

## Configuration

Modify `config.py` to adjust parameters:

```python
class Config:
    # Data parameters
    MIN_INTERACTIONS = 5    # Minimum interactions per user/item
    MAX_SEQ_LEN = 50       # Maximum sequence length
    TRAIN_RATIO = 0.8      # Train/test split ratio
    
    # Training parameters
    BATCH_SIZE = 256       # Batch size for training
    
    # Other parameters...
```

## Advanced Usage

### Custom Configuration

```python
from config import Config
from dataloader import DataLoaderManager

class CustomConfig(Config):
    MAX_SEQ_LEN = 100
    BATCH_SIZE = 512
    MIN_INTERACTIONS = 10

# Use custom config
manager = DataLoaderManager(data, CustomConfig())
train_loader = manager.get_train_loader()
```

### Custom Negative Sampling

```python
from dataset import NegativeSamplingDataset

# Create dataset with more negative samples
dataset = NegativeSamplingDataset(
    sequences, 
    num_items, 
    num_negatives=5  # 5 negative samples per positive
)
```

### Sequence Length Grouping

The dataloader automatically groups sequences by similar lengths for efficient batching, reducing padding and improving training speed.

## Data Format

### Input Batch Format
```python
batch = {
    'input_seq': torch.Tensor,      # [batch_size, max_seq_len] - padded sequences
    'target_item': torch.Tensor,    # [batch_size] - next item to predict
    'mask': torch.Tensor,           # [batch_size, max_seq_len] - attention mask
    'seq_len': torch.Tensor,        # [batch_size] - actual sequence lengths
    'negative_items': torch.Tensor  # [batch_size, num_negatives] - negative samples
}
```

## Performance Optimizations

- **Length-based batching**: Groups similar length sequences
- **Efficient padding**: Minimal padding with attention masks
- **Memory pinning**: Enabled for GPU training
- **Custom sampling**: Reduces computational overhead

## Testing

Run the demo script to test the implementation:

```bash
python demo.py
```

The script will:
1. Download and preprocess the data
2. Create train/test splits
3. Initialize data loaders
4. Show sample batches
5. Display data statistics

## Requirements

- Python 3.7+
- PyTorch 1.8+
- NumPy
- Pandas
- Scipy
- Scikit-learn
- tqdm
- requests

## License

This project is available under the MIT License.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Citation

If you use this code in your research, please cite:

```
@misc{bgcn-sequential-rec,
  title={BGCN: Sequential Recommendation Dataset and DataLoader for Amazon Beauty},
  author={Your Name},
  year={2024},
  url={https://github.com/chitrung46/BGCN}
}
```