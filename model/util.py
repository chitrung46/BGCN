import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

def pos_encoding(seq_len, dim):
    pe = torch.zeros(seq_len, dim)
    for pos in range(seq_len):
        for i in range(0, dim, 2):
            pe[pos, i] = torch.sin(pos / (10000 ** ((2 * i)/dim)))
            if i + 1 < dim:
                pe[pos, i + 1] = torch.cos(pos / (10000 ** ((2 * (i + 1))/dim)))
    return pe

def normalized_laplacian(adj):
    # adj: [N, N]
    I = torch.eye(adj.shape[0]).to(adj.device)
    A_tilde = adj + I # A + I
    D_tilde = torch.diag(torch.sum(A_tilde, dim=1))
    D_inv = D_tilde**(-0.5)
    D_inv[torch.isinf(D_inv)] = 0
    A_hat = torch.matmul(D_inv, torch.matmul(A_tilde, D_inv)) # D_tilde^(-1/2) * A_tilde * D_tilde^(-1/2)   
    return A_hat

class GConv(nn.Module):
    def __init__(self, in_dim, out_dim, bias=False):
        super(GConv, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.weight = nn.Parameter(torch.FloatTensor(in_dim, out_dim))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_dim))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        self.weight.data.kaiming_uniform_()
        if self.bias is not None:
            self.bias.data.kaiming_uniform_()

    def forward(self, x, adj):
        support = torch.matmul(x, self.weight)
        h = torch.matmul(adj, support)
        h = torch.relu(h)
        if self.bias is not None:
            return h + self.bias
        else:
            return h

class GCN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers, dropout=0.5):
        super(GCN, self).__init__()
        self.layers = nn.ModuleList()
        self.dropout = dropout
        if num_layers == 1:
            self.layers.append(GConv(in_dim, out_dim))
        else:
            self.layers.append(GConv(in_dim, hidden_dim))
            for _ in range(num_layers - 2):
                self.layers.append(GConv(hidden_dim, hidden_dim))
            self.layers.append(GConv(hidden_dim, out_dim))

    def forward(self, x, adj):
        h = x
        for layer in self.layers:
            h = layer(h, adj)
            h = torch.dropout(h, self.dropout, training=self.training)
        return h
    
class GatedGNN(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(GatedGNN, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.linear_z = nn.Linear(in_dim + out_dim, out_dim)
        self.linear_r = nn.Linear(in_dim + out_dim, out_dim)
        self.linear_h = nn.Linear(in_dim + out_dim, out_dim)

    def GatedGNNCell(self, x, h):
        a_in = 
        a_out = 
        combined = torch.cat([x, h], dim=1)
        z = torch.sigmoid(self.linear_z(combined))
        r = torch.sigmoid(self.linear_r(combined))
        h_tilde = torch.tanh(self.linear_h(torch.cat([x, r * h], dim=1)))
        h_new = (1 - z) * h + z * h_tilde
        return h_new

    def forward(self, x, adj):
        
        return h
    
class BERTReviewEncoder(nn.Module):
    def __init__(self, model_name, max_length, out_dim, dropout, freeze_bert, pooling, normalize):
        super(BERTReviewEncoder, self).__init__()
        self.max_length = max_length
        self.pooling = pooling
        self.normalize = normalize
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.bert = AutoModel.from_pretrained(model_name)
        
        hidden = self.bert.config.hidden_size
        
        self.proj = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
        )
        self.ln = nn.LayerNorm(out_dim)

        if freeze_bert:
            for p in self.bert.parameters():
                p.requires_grad = False

    def _pool(self, last_hidden, attn_mask):
        if self.pooling == "cls":
            sent = last_hidden[:, 0]
        else:
            mask = attn_mask.unsqueeze(-1).float()  # [B, T, 1]
            summed = (last_hidden * mask).sum(dim=1)  # [B, H]
            counts = mask.sum(dim=1).clamp(min=1.0)  # [B, 1]
            sent = summed / counts
        return sent

    def forward(self, texts, device=None):
        toks = self.tokenizer(
            texts, padding=True, truncation=True,
            max_length=self.max_length, return_tensors="pt"
        )

        if device:
            toks = {k: v.to(device) for k, v in toks.items()}

        with torch.no_grad() if self.args.freeze_bert else torch.enable_grad():
            out = self.bert(**toks)
            last_hidden = out.last_hidden_state  # [B, T, H]
            attn_mask = toks['attention_mask']  # [B, T]
            emb = self._pool(last_hidden, attn_mask)  # [B, H]

        proj = self.proj(emb)  # [B, d]
        proj = self.ln(proj)

        if self.normalize:
            proj = torch.normalize(proj, p=2, dim=-1)

        return proj
