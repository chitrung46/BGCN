import torch
import pandas as pd
from torch.utils.data import Dataset

class SRDataset(Dataset):
    def __init__(self, seqs, masks):
        super(SRDataset).__init__()
        self.seqs = seqs
        self.masks = masks

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, index):
        seq = self.seqs[index]
        mask = self.masks[index]
        data = seq[:-1][0]
        rating = seq[:-1][1]
        review = seq[:-1][2]
        label = [seq[-1]][0]
        return torch.LongTensor(data), torch.LongTensor(label), torch.LongTensor(mask)

def normalize_seq(seqs, min_seq_len=0, max_seq_len=None):
    seq_len = [len(seq) for seq in seqs.values()]
    if max_seq_len is not None:
        max_len = max(seq_len)
    else:
        max_len = max_seq_len 
    
    filtered_seqs = dict()
    masks = dict()
    padding = [(0, '', pd.Timestamp(0))]

    for uidx, seq in seqs.items():
        if len(seq) >= min_seq_len:
            l = len(seq)
            filtered_seqs[uidx] = [padding*(max_len-l) + seq[-max_len:] if (max_len > l) 
                                    else seq[-max_len:]]
            masks[uidx] = [[0]*(max_len-l) + [1]*max_len if (max_len > l) 
                                    else [1]*max_len]

    
    print(f"Average length of sequence: {sum(seq_len)/len(seq_len)}")
    print(f"Maximum length of sequence: {max_len}")
    print(f"Number of users before filtering: {len(seqs)}")
    print(f"Number of users after filtering: {len(filtered_seqs)}")

    return filtered_seqs, masks, max_len

def process_All_Beauty(args):
    if args.dataset == 'All_Beauty':
        df = pd.read_json('./data/All_Beauty.jsonl.gz', compression='gzip', lines=True)
        df.drop(columns=['title', 'images', 'asin'], inplace=True)
        df.rename(columns={'parent_asin': 'item_id'}, inplace=True)
        df = df[df['verified_purchase'] == True]

    seqs = dict()
    user_map = dict()
    item_map = dict()
    user_num = 0
    item_num = 0

    for index, row in df.iterrows():
        user_id = row['user_id']
        item_id = row['item_id']
        rating = row['rating']
        review = row['text']
        timestamp = row['timestamp']

        # mapping
        if user_id not in user_map:
            user_map[user_id] = user_num
            user_num += 1
        if item_id not in item_map:
            item_map[item_id] = item_num
            item_num += 1

        uidx = user_map[user_id]
        iidx = item_map[item_id]

        if uidx in seqs.keys():
            seqs[uidx].append([iidx, rating, review, timestamp])
        else:
            seqs[uidx] = [[iidx, rating, review, timestamp]]
    
    print(f"Number of item: {len(item_map)}")

    for seq in seqs.values():
        seq.sort(key=lambda x: x[3])     

    filtered_seqs, masks, max_len = normalize_seq(seqs, args.min_seq_len, args.max_seq_len)   

    return filtered_seqs, masks, max_len

def load_dataset(args):
    if args.dataset == 'All_Beauty': 
        seqs, masks, max_len = process_All_Beauty(args)
    return seqs, masks
            