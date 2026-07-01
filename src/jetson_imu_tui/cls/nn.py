"""Vendored LIMU-BERT + GRU model definitions (BERT-finetune inference path only).

Copied VERBATIM from ``LIMU-BERT-Public/models.py`` (and ``split_last`` / ``merge_last``
from ``utils.py``) so a saved ``BERTClassifier`` checkpoint's ``state_dict`` keys line up
exactly. Do not rename attributes or reorder module creation — that would break loading.

The two config objects (``BASE_V3`` bert config, ``GRU_V3`` classifier config) hardcode
the ``base_v3`` / ``gru_v3`` entries from the repo's ``config/limu_bert.json`` and
``config/classifier.json``; we don't ship the JSON loader.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# --- config (hardcoded base_v3 / gru_v3) -------------------------------------
@dataclass
class BertConfig:
    hidden: int
    hidden_ff: int
    feature_num: int
    n_layers: int
    n_heads: int
    seq_len: int
    emb_norm: bool = True


@dataclass
class ClassifierConfig:
    seq_len: int
    input: int
    num_rnn: int
    num_layers: list
    rnn_io: list
    num_linear: int
    linear_io: list
    activ: bool = False
    dropout: bool = False


# base_v3 in config/limu_bert.json
BASE_V3 = BertConfig(hidden=72, hidden_ff=144, feature_num=6, n_layers=4, n_heads=4, seq_len=20)
# gru_v3 in config/classifier.json
GRU_V3 = ClassifierConfig(
    seq_len=20, input=6, num_rnn=2, num_layers=[2, 1], rnn_io=[[6, 20], [20, 10]],
    num_linear=2, linear_io=[[10, 10], [10, 3]], activ=False, dropout=True,
)


# --- tensor helpers (from utils.py) ------------------------------------------
def split_last(x, shape):
    "split the last dimension to given shape"
    shape = list(shape)
    assert shape.count(-1) <= 1
    if -1 in shape:
        shape[shape.index(-1)] = x.size(-1) // -np.prod(shape)
    return x.view(*x.size()[:-1], *shape)


def merge_last(x, n_dims):
    "merge the last n_dims to a dimension"
    s = x.size()
    assert n_dims > 1 and n_dims < len(s)
    return x.view(*s[:-n_dims], -1)


# --- model (verbatim from models.py) -----------------------------------------
def gelu(x):
    "Implementation of the gelu activation function by Hugging Face"
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class LayerNorm(nn.Module):
    "A layernorm module in the TF style (epsilon inside the square root)."
    def __init__(self, cfg, variance_epsilon=1e-12):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(cfg.hidden), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros(cfg.hidden), requires_grad=True)
        self.variance_epsilon = variance_epsilon

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.gamma * x + self.beta


class Embeddings(nn.Module):

    def __init__(self, cfg, pos_embed=None):
        super().__init__()

        # factorized embedding
        self.lin = nn.Linear(cfg.feature_num, cfg.hidden)
        if pos_embed is None:
            self.pos_embed = nn.Embedding(cfg.seq_len, cfg.hidden)  # position embedding
        else:
            self.pos_embed = pos_embed

        self.norm = LayerNorm(cfg)
        self.emb_norm = cfg.emb_norm

    def forward(self, x):
        seq_len = x.size(1)
        pos = torch.arange(seq_len, dtype=torch.long, device=x.device)
        pos = pos.unsqueeze(0).expand(x.size(0), seq_len)  # (S,) -> (B, S)

        # factorized embedding
        e = self.lin(x)
        if self.emb_norm:
            e = self.norm(e)
        e = e + self.pos_embed(pos)
        return self.norm(e)


class MultiHeadedSelfAttention(nn.Module):
    """ Multi-Headed Dot Product Attention """
    def __init__(self, cfg):
        super().__init__()
        self.proj_q = nn.Linear(cfg.hidden, cfg.hidden)
        self.proj_k = nn.Linear(cfg.hidden, cfg.hidden)
        self.proj_v = nn.Linear(cfg.hidden, cfg.hidden)
        self.scores = None  # for visualization
        self.n_heads = cfg.n_heads

    def forward(self, x):
        q, k, v = self.proj_q(x), self.proj_k(x), self.proj_v(x)
        q, k, v = (split_last(x, (self.n_heads, -1)).transpose(1, 2)
                   for x in [q, k, v])
        scores = q @ k.transpose(-2, -1) / np.sqrt(k.size(-1))
        scores = F.softmax(scores, dim=-1)
        h = (scores @ v).transpose(1, 2).contiguous()
        h = merge_last(h, 2)
        self.scores = scores
        return h


class PositionWiseFeedForward(nn.Module):
    """ FeedForward Neural Networks for each position """
    def __init__(self, cfg):
        super().__init__()
        self.fc1 = nn.Linear(cfg.hidden, cfg.hidden_ff)
        self.fc2 = nn.Linear(cfg.hidden_ff, cfg.hidden)

    def forward(self, x):
        return self.fc2(gelu(self.fc1(x)))


class Transformer(nn.Module):
    """ Transformer with Self-Attentive Blocks"""
    def __init__(self, cfg):
        super().__init__()
        self.embed = Embeddings(cfg)
        self.n_layers = cfg.n_layers
        self.attn = MultiHeadedSelfAttention(cfg)
        self.proj = nn.Linear(cfg.hidden, cfg.hidden)
        self.norm1 = LayerNorm(cfg)
        self.pwff = PositionWiseFeedForward(cfg)
        self.norm2 = LayerNorm(cfg)

    def forward(self, x):
        h = self.embed(x)
        for _ in range(self.n_layers):
            h = self.attn(h)
            h = self.norm1(h + self.proj(h))
            h = self.norm2(h + self.pwff(h))
        return h


class ClassifierGRU(nn.Module):
    def __init__(self, cfg, input=None, output=None, feats=False):
        super().__init__()
        for i in range(cfg.num_rnn):
            if input is not None and i == 0:
                self.__setattr__('gru' + str(i), nn.GRU(input, cfg.rnn_io[i][1], num_layers=cfg.num_layers[i], batch_first=True))
            else:
                self.__setattr__('gru' + str(i),
                                 nn.GRU(cfg.rnn_io[i][0], cfg.rnn_io[i][1], num_layers=cfg.num_layers[i],
                                        batch_first=True))
        for i in range(cfg.num_linear):
            if output is not None and i == cfg.num_linear - 1:
                self.__setattr__('lin' + str(i), nn.Linear(cfg.linear_io[i][0], output))
            else:
                self.__setattr__('lin' + str(i), nn.Linear(cfg.linear_io[i][0], cfg.linear_io[i][1]))
        self.activ = cfg.activ
        self.dropout = cfg.dropout
        self.num_rnn = cfg.num_rnn
        self.num_linear = cfg.num_linear

    def forward(self, input_seqs, training=False):
        h = input_seqs
        for i in range(self.num_rnn):
            rnn = self.__getattr__('gru' + str(i))
            h, _ = rnn(h)
            if self.activ:
                h = F.relu(h)
        h = h[:, -1, :]
        if self.dropout:
            h = F.dropout(h, training=training)
        for i in range(self.num_linear):
            linear = self.__getattr__('lin' + str(i))
            h = linear(h)
            if self.activ:
                h = F.relu(h)
        return h


class BERTClassifier(nn.Module):

    def __init__(self, bert_cfg, classifier=None, frozen_bert=False):
        super().__init__()
        self.transformer = Transformer(bert_cfg)
        if frozen_bert:
            for p in self.transformer.parameters():
                p.requires_grad = False
        self.classifier = classifier

    def forward(self, input_seqs, training=False):
        h = self.transformer(input_seqs)
        h = self.classifier(h, training)
        return h
