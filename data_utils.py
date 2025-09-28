"""Data utilities for downloading and preprocessing Amazon Beauty dataset."""

import os
import gzip
import json
import pickle
import requests
import pandas as pd
import numpy as np
from collections import Counter, defaultdict
from tqdm import tqdm
from config import Config


def download_data():
    """Download Amazon Beauty dataset."""
    Config.create_data_dir()
    
    data_path = os.path.join(Config.DATA_DIR, Config.BEAUTY_DATA_FILE)
    
    if os.path.exists(data_path):
        print(f"Data file already exists at {data_path}")
        return data_path
    
    print("Downloading Amazon Beauty dataset...")
    response = requests.get(Config.BEAUTY_DATA_URL, stream=True)
    response.raise_for_status()
    
    total_size = int(response.headers.get('content-length', 0))
    
    with open(data_path, 'wb') as f, tqdm(
        desc="Downloading",
        total=total_size,
        unit='B',
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
                pbar.update(len(chunk))
    
    print(f"Downloaded data to {data_path}")
    return data_path


def parse_json_gz(file_path):
    """Parse gzipped JSON file."""
    data = []
    with gzip.open(file_path, 'rt', encoding='utf-8') as f:
        for line in tqdm(f, desc="Parsing JSON"):
            try:
                data.append(json.loads(line.strip()))
            except json.JSONDecodeError:
                continue
    return data


def preprocess_data():
    """Preprocess the Amazon Beauty dataset for sequential recommendation."""
    processed_path = os.path.join(Config.DATA_DIR, Config.PROCESSED_DATA_FILE)
    
    if os.path.exists(processed_path):
        print(f"Loading preprocessed data from {processed_path}")
        with open(processed_path, 'rb') as f:
            return pickle.load(f)
    
    # Download data if not exists
    data_path = download_data()
    
    # Parse the data
    print("Parsing dataset...")
    raw_data = parse_json_gz(data_path)
    
    # Convert to DataFrame
    df = pd.DataFrame(raw_data)
    print(f"Original data shape: {df.shape}")
    
    # Keep only necessary columns
    df = df[['reviewerID', 'asin', 'unixReviewTime']].copy()
    df.columns = ['user_id', 'item_id', 'timestamp']
    
    # Convert timestamp to integer
    df['timestamp'] = df['timestamp'].astype(int)
    
    # Remove users and items with too few interactions
    print("Filtering users and items...")
    user_counts = df['user_id'].value_counts()
    item_counts = df['item_id'].value_counts()
    
    valid_users = user_counts[user_counts >= Config.MIN_INTERACTIONS].index
    valid_items = item_counts[item_counts >= Config.MIN_INTERACTIONS].index
    
    df = df[df['user_id'].isin(valid_users) & df['item_id'].isin(valid_items)]
    print(f"After filtering: {df.shape}")
    
    # Sort by user and timestamp
    df = df.sort_values(['user_id', 'timestamp'])
    
    # Create user and item mappings
    unique_users = df['user_id'].unique()
    unique_items = df['item_id'].unique()
    
    user_to_idx = {user: idx for idx, user in enumerate(unique_users)}
    item_to_idx = {item: idx + 1 for idx, item in enumerate(unique_items)}  # +1 for padding
    
    # Add reverse mappings
    idx_to_user = {idx: user for user, idx in user_to_idx.items()}
    idx_to_item = {idx: item for item, idx in item_to_idx.items()}
    
    # Map IDs to indices
    df['user_idx'] = df['user_id'].map(user_to_idx)
    df['item_idx'] = df['item_id'].map(item_to_idx)
    
    # Group by user to create sequences
    user_sequences = defaultdict(list)
    user_timestamps = defaultdict(list)
    
    for _, row in tqdm(df.iterrows(), desc="Creating sequences", total=len(df)):
        user_idx = row['user_idx']
        item_idx = row['item_idx']
        timestamp = row['timestamp']
        
        user_sequences[user_idx].append(item_idx)
        user_timestamps[user_idx].append(timestamp)
    
    # Convert to lists
    sequences = []
    for user_idx in range(len(unique_users)):
        if user_idx in user_sequences:
            sequences.append(user_sequences[user_idx])
    
    # Statistics
    seq_lengths = [len(seq) for seq in sequences]
    print(f"Number of users: {len(unique_users)}")
    print(f"Number of items: {len(unique_items)}")
    print(f"Number of interactions: {len(df)}")
    print(f"Average sequence length: {np.mean(seq_lengths):.2f}")
    print(f"Max sequence length: {max(seq_lengths)}")
    
    # Prepare data dictionary
    data_dict = {
        'sequences': sequences,
        'num_users': len(unique_users),
        'num_items': len(unique_items),
        'user_to_idx': user_to_idx,
        'item_to_idx': item_to_idx,
        'idx_to_user': idx_to_user,
        'idx_to_item': idx_to_item,
        'df': df
    }
    
    # Save preprocessed data
    with open(processed_path, 'wb') as f:
        pickle.dump(data_dict, f)
    
    print(f"Saved preprocessed data to {processed_path}")
    return data_dict


if __name__ == "__main__":
    data = preprocess_data()
    print("Data preprocessing completed!")