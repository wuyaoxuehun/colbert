import string

import torch
import torch.nn as nn
from torch import einsum
from transformers import BertPreTrainedModel, BertModel

from colbert.parameters import DEVICE

from torch.autograd import Variable


def row_pairwise_distances(x: torch.tensor, y: torch.tensor):
    dist_mat = Variable(torch.Tensor(x.size()[0], y.size()[0]).type(x.dtype).to(x.device), requires_grad=True)

    for i, row in enumerate(x.split(1)):
        r_v = row.expand_as(y)
        sq_dist = torch.sum((r_v - y) ** 2, 1)
        # print(dist_mat.size())
        # print(sq_dist.view(1, -1).size())
        dist_mat[i] = sq_dist.view(1, -1)
    return dist_mat


def expanded_pairwise_distances(x, y=None):
    differences = x.unsqueeze(1) - y.unsqueeze(0)
    distances = torch.sum(differences * differences, -1)
    return distances


def pairwise_distances(x, y=None):
    y_t = torch.transpose(y, 0, 1)
    dist = 2.0 - 2.0 * torch.mm(x, y_t)
    dist[dist != dist] = 0
    return dist


from zhon.hanzi import punctuation

puncts = ''.join(punctuation.split())


class ColBERT_List(BertPreTrainedModel):
    def __init__(self, config, query_maxlen, doc_maxlen, dim=128, similarity_metric='cosine'):

        super(ColBERT_List, self).__init__(config)

        self.query_maxlen = query_maxlen
        self.doc_maxlen = doc_maxlen
        self.similarity_metric = similarity_metric
        self.dim = dim

        self.bert: nn.Module = BertModel(config)
        # self.linear_q = nn.Linear(config.hidden_size, dim, bias=False)
        # self.linear_d = nn.Linear(config.hidden_size, dim, bias=False)
        self.linear = nn.Linear(config.hidden_size, dim, bias=False)
        self.init_weights()

    def forward(self, Q, D, q_mask=None, d_mask=None, output_prebatch=False, pre_batch_input=None):
        Q = self.query(*Q)
        D = self.doc(*D)
        TD = D
        TD_mask = d_mask
        if pre_batch_input:
            pre_D = [_[0] for _ in pre_batch_input]
            pre_d_mask = [_[1] for _ in pre_batch_input]
            TD = torch.cat([D] + [_ for _ in pre_D])
            TD_mask = torch.cat([d_mask] + [_.to(d_mask.device) for _ in pre_d_mask])
        score = self.score(Q, TD, q_mask=q_mask, d_mask=TD_mask)
        if output_prebatch:
            return score, D
        else:
            return score

    def query(self, input_ids, attention_mask, **kwargs):
        input_ids, attention_mask = input_ids.to(DEVICE), attention_mask.to(DEVICE)
        Q = self.bert(input_ids, attention_mask=attention_mask)[0]
        # Q = self.linear_q(Q)
        Q = self.linear(Q)
        Q = torch.nn.functional.normalize(Q, p=2, dim=2)
        return Q

    def doc(self, input_ids, attention_mask, **kwargs):
        # input_ids, attention_mask = input_ids.to(DEVICE), attention_mask.to(DEVICE)
        D = self.bert(input_ids, attention_mask=attention_mask)[0]
        # D = self.linear_d(D)
        D = self.linear(D)
        D = torch.nn.functional.normalize(D, p=2, dim=2)
        return D

    def score(self, Q, D, q_mask=None, d_mask=None):
        if d_mask is not None and q_mask is not None:
            D = D * d_mask[..., None]
            Q = Q * q_mask[..., None]
        scores = einsum("qmh,dnh->qdmn", Q, D).max(-1)[0].sum(-1)
        return scores
