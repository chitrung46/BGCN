import numpy as np
import torch

def recall(scores, labels, k):
    scores = scores.cpu()
    labels = labels.cpu()
    rank = (-scores).argsort(dim=1)
    cut = rank[:, :k]
    hits = labels.gather(1, cut)
    return hits.sum() / labels.sum() if labels.sum() > 0 else 0.

def precision(scores, labels, k):
    scores = scores.cpu()
    labels = labels.cpu()
    rank = (-scores).argsort(dim=1)
    cut = rank[:, :k]
    hits = labels.gather(1, cut)
    return hits.sum() / k if k > 0 else 0.

def cross_entropy(scores, labels):
    logp = torch.nn.functional.log_softmax(scores, dim=1)
    loss = - (logp * labels).sum(dim=1).mean()
    return loss.cpu().item()

def mrr(scores, labels, k):
    scores = scores.cpu()
    labels = labels.cpu()
    rank = (-scores).argsort(dim=1)
    cut = rank[:, :k]
    hits = labels.gather(1, cut)
    position = torch.arange(1, k + 1).float().to(labels.device)
    rr = hits / position
    return rr.mean().cpu().item() if rr.numel() > 0 else 0.

def ndcg(scores, labels, k):
    scores = scores.cpu()
    labels = labels.cpu()
    rank = (-scores).argsort(dim=1)
    cut = rank[:, :k]
    hits = labels.gather(1, cut)
    position = torch.arange(2, 2+k)
    weights =  1.0 / torch.log2(position.float())
    dcg = (hits * weights).sum(1)
    idcg = torch.tensor([ (weights[:min(n, k)]).sum() for n in labels.sum(1)]).to(labels.device)
    ndcg = dcg / idcg
    return ndcg.mean().cpu().item()

def Metrics(scores, labels, ks=[]):
    assert type(scores) == type(labels)
    metrics = {}

    for k in sorted(ks, reverse=True):
        metrics = {
            'MRR@{}'.format(k): mrr(scores, labels, k),
            'NDCG@{}'.format(k): ndcg(scores, labels, k)
        }   

    return metrics