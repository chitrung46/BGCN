import torch
import torch.nn as nn
from util import pos_encoding, GCN, GatedGNN, LAM, BERTReviewEncoder
from lib.func import construct_session_graph

class LAM(nn.Module):
    def __init__(self):
        super(LAM, self).__init__()

    def forward(self, h, review, mask):

        return h

class BGCN(nn.Module):
    def __init__(self, args, global_graph, item_num):
        super(BGCN, self).__init__()
        self.hidden_dim = args.hidden_dim
        self.item_num = item_num
        self.in_dim = args.in_dim
        self.out_dim = args.out_dim
        max_len = args.max_seq_len
        # self.embedding = nn.Embedding(max_len, self.hidden_dim)
        # self.pos = pos_encoding(max_len, self.hidden_dim)
        self.global_graph = global_graph # [N, N]
        # self.layer_norm_eps = args.layer_norm_eps
        # self.layer_norm = nn.LayerNorm(self.hidden_dim, eps=self.layer_norm_eps)
        self.gcn = GCN(self.in_dim, self.out_dim, self.hidden_dim, num_layers=args.gcn_layers, dropout=args.dropout)
        self.ggnn = GatedGNN(self.in_dim, self.out_dim, self.hidden_dim, num_layers=2, dropout=args.dropout)
        self.bre = BERTReviewEncoder(args.bre_model_name, 
                                     args.bre_max_length, 
                                     args.bre_out_dim, 
                                     args.bre_dropout,
                                     args.bre_freeze_bert, 
                                     args.bre_pooling, 
                                     args.bre_normalize)
        self.LAM = LAM()
        self.pos_encoding = nn.Embedding(item_num, self.hidden_dim)
        self.linear1 = nn.Linear(self.hidden_dim, 3*self.hidden_dim, bias=True)
        self.linear2 = nn.Linear(self.hidden_dim, self.hidden_dim, bias=True)
        self.linear3 = nn.Linear(self.hidden_dim, self.hidden_dim, bias=True)
        self.linear4 = nn.Linear(self.hidden_dim, self.hidden_dim, bias=True)

    def forward(self, seq, review, mask):
        seq_len = torch.sum(mask, dim=1)  # [B, 1]
        h = self.embedding(seq)
        h_global = self.gcn(h, self.global_graph) # [B, N, d]
        session_graph = construct_session_graph(seq, mask) # adj_in, adj_out, adj_self [B, 3, N, N]
        h_local = self.ggnn(h, session_graph)
        h_hybrid = self.LAM(h_global, h_local, mask)
        h_review = self.bre(review)
        pos = self.pos_encoding(seq)
        z = torch.tanh(self.linear1(torch.cat([h_hybrid, h_review, pos], dim=-1))) # [B, N, d]
        s = torch.sum(h_hybrid, dim=-1) / seq_len
        z_proj = self.linear2(z)
        s_proj = self.linear3(s)
        beta = self.linear4(torch.sigmoid(z_proj + s_proj))
        S = torch.sum(beta * h_hybrid, dim=1)
        scores = torch.softmax(S*h, dim=1)
        return scores
    
    def compute_scores(hidden, mask):
        scores = 0
        return scores