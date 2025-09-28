"""Configuration file for BGCN sequential recommendation system."""

import os

class Config:
    # Data paths
    DATA_DIR = "data"
    BEAUTY_DATA_URL = "http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Beauty_5.json.gz"
    BEAUTY_DATA_FILE = "reviews_Beauty_5.json.gz"
    PROCESSED_DATA_FILE = "beauty_processed.pkl"
    
    # Dataset parameters
    MIN_INTERACTIONS = 5  # Minimum interactions per user
    MAX_SEQ_LEN = 50     # Maximum sequence length
    TRAIN_RATIO = 0.8    # Train/test split ratio
    
    # Model parameters
    EMBEDDING_DIM = 64
    HIDDEN_DIM = 128
    NUM_LAYERS = 2
    DROPOUT = 0.1
    
    # Training parameters
    BATCH_SIZE = 256
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 100
    
    # Evaluation
    TOP_K = [5, 10, 20]
    
    @classmethod
    def create_data_dir(cls):
        """Create data directory if it doesn't exist."""
        os.makedirs(cls.DATA_DIR, exist_ok=True)