import argparse
import torch
import random
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description="Training Configuration")
    parser.add_argument('--log_dir', type=str, default='./logs', help='Directory to save logs and models')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--model_name', type=str, default='model', help='Name of the model')
    parser.add_argument('--dataset', type=str, default='All_Beauty', help='Dataset name')
    
    parser.add_argument('--min_seq_length', type=int, default=2, help='Minimum length of a user behavior\'s sequence')
    parser.add_argument('--max_seq_length', type=int, default=50, help='Maximum length of a user behavior\'s sequence')
    
    
    # model
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    return parser.parse_args()

def init_seed(seed):
    '''
    Disable cudnn to maximize reproducibility
    '''
    torch.cuda.cudnn_enabled = False
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def init_device(opt):
    if torch.cuda.is_available():
        opt.cuda = True
        torch.cuda.set_device(int(opt.device[5]))
    else:
        opt.cuda = False
        opt.device = 'cpu'
    return opt

def init_optim(model, opt):
    return torch.optim.Adam(params=model.parameters(),lr=opt.lr_init)

def get_memory_usage(device):
    allocated_memory = torch.cuda.memory_allocated(device) / (1024*1024.)
    cached_memory = torch.cuda.memory_cached(device) / (1024*1024.)
    return allocated_memory, cached_memory