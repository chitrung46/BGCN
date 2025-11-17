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

class LAM(nn.Module):
    def __init__(self, hidden_dim):
        super(LAM, self).__init__()
        self.linear = nn.Linear(hidden_dim, hidden_dim, bias=True)

    def forward(self, h_global, h_local, seq_len, max_len):
        alpha = torch.sigmoid(self.linear(seq_len * max_len)) # [B]
        alpha = alpha.unsqueeze(-1).unsqueeze(-1)  # [B, 1, 1]
        h_hybrid = (1 - alpha) * h_global + alpha * h_local
        return h_hybrid

class GatedGNN(nn.Module):
    def __init__(self, hidden_dim, n_steps, item_num):
        super(GatedGNN, self).__init__()
        self.n_steps = n_steps
        self.item_num = item_num
        self.linear_in = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.linear_out = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.linear_self = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.linear_za = nn.Linear(3*hidden_dim, hidden_dim, bias=True)
        self.linear_zh = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.linear_ra = nn.Linear(3*hidden_dim, hidden_dim, bias=True)
        self.linear_rh = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.linear_hhat_a = nn.Linear(3*hidden_dim, hidden_dim, bias=True)
        self.linear_hhat_rh = nn.Linear(hidden_dim, hidden_dim, bias=True)

    def GatedGNNCell(self, hidden, adj):
        A_in = adj[:, :, :self.item_num]
        A_out = adj[:, :, self.item_num:2*self.item_num]
        A_self = adj[:, :, 2*self.item_num:]

        a_in = torch.matmul(A_in, self.linear_in(hidden))
        a_out = torch.matmul(A_out, self.linear_out(hidden))
        a_self = torch.matmul(A_self, self.linear_self(hidden))
        a = torch.cat([a_in, a_out, a_self], dim=2)
        z = torch.sigmoid(self.linear_za(a) + self.linear_zh(hidden))
        r = torch.sigmoid(self.linear_ra(a) + self.linear_rh(hidden))
        h_hat = torch.tanh(self.linear_hhat_a(a) + self.linear_hhat_rh(r * hidden))
        output = (1-z) * hidden + z * h_hat

        return output

    def forward(self, adj, hidden):
        for _ in range(self.n_steps):
            hidden = self.GatedGNNCell(hidden, adj)
        return hidden
    
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
