import argparse
import torch
import random
import configparser
import numpy as np

def parse_args():
    args = argparse.ArgumentParser(description="Training Configuration")
    args.add_argument('--log_dir', type=str, default='./logs', help='Directory to save logs and models')
    args.add_argument('--model_name', type=str, default='BGCN', help='Name of the model')
    args.add_argument('--mode', type=str, default='train', help='Mode: train or test')
    args.add_argument('--device', default='cuda:0', type=str, help='indices of GPUs')
    args.add_argument('--dataset', type=str, default='All_Beauty', help='Dataset name')
    
    args1 = args.parse_args()
    
    # get configuration
    config_file = './config/{}_{}.conf'.format(args1.model_name, args1.dataset)
    config = configparser.ConfigParser()
    config.read(config_file)

    # data
    args.add_argument('--min_seq_len', type=int, default=config['data']['min_seq_len'], help='Minimum length of a user behavior\'s sequence')
    args.add_argument('--max_seq_len', type=int, default=config['data']['max_seq_len'], help='Maximum length of a user behavior\'s sequence')
    args.add_argument('--k_transition', type=int, default=config['data']['k_transition'], help='Number of transitions for constructing global graph')
    args.add_argument('--data_processed', type=bool, default=config['data'].getboolean('data_processed'), help='Whether the dataset has been processed before')
    # model
    args.add_argument('--batch_size', type=int, default=config['model']['batch_size'], help='Batch size for training')
    args.add_argument('--hidden_dim', type=int, default=config['model']['hidden_dim'], help='Hidden dim of model layers')
    args.add_argument('--gcn_layers', type=int, default=config['model']['gcn_layers'], help='Number of GCN layers')
    args.add_argument('--layer_norm_eps', type=float, default=config['model']['layer_norm_eps'], help='Epsilon for layer normalization')
    
    # bertreviewencoder
    args.add_argument('--bre_model_name', type=str, default=config['bre']['model_name'], help='Pretrained BERT model name')
    args.add_argument('--bre_max_len', type=int, default=config['bre']['max_len'], help='Maximum length of review text')
    args.add_argument('--bre_out_dim', type=int, default=config['bre']['out_dim'],  help='Output dimension of BERT review encoder')
    args.add_argument('--bre_dropout', type=float, default=config['bre']['dropout'], help='Dropout rate in BERT review encoder')
    args.add_argument('--bre_freeze_bert', type=bool, default=config['bre'].getboolean('freeze_bert'), help='Whether to freeze BERT parameters during training')
    args.add_argument('--bre_pooling', type=str, default=config['bre']['pooling'], help='Pooling method for BERT outputs: cls or mean')
    args.add_argument('--bre_normalize', type=bool, default=config['bre'].getboolean('normalize'), help='Whether to normalize BERT output embeddings')  
    
    # train
    args.add_argument('--epochs', type=int, default=config['train']['epochs'], help='Number of training epochs')

    args = args.parse_args()
    return args

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

def construct_global_graph(seqs, item_num, k):
    G = np.zeros((item_num, item_num), dtype=int)
    print("Started constructing global graph...")

    for idx, seq in enumerate(seqs.values()):
        # print(seq)
        for i in range(len(seq)):
            item_i = seq[i][0]
            for j in range(k):
                if i-j-1 >= 0:
                    item_j = seq[i-j-1][0]
                    G[item_j, item_i] += 1
    print(f"Global graph size: {np.array(G).shape}")
    # normalization
    for row in G:
        row_sum = sum(row)
        row /= row_sum
    print("Normalized global graph constructed")
    return G

def construct_session_graph(seqs, standard_seq_len, item_num):
    S = [np.zeros((standard_seq_len, 3*standard_seq_len), dtype=int) for seq in seqs]

    for idx, seq in enumerate(seqs):
        for i in range (len(seq)):
            item_i = seq[i]
            for j in range(i):
                item_j = seq[j]
                S[idx][item_i][item_j] = 1 # in
                S[idx][item_j][standard_seq_len+item_i] = 1 # out

                if item_i == item_j:
                    S[idx][item_i][2*standard_seq_len+item_i] = 1 # self-loop
    return S
