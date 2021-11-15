import string

import torch
import torch.nn as nn
from transformers import BertPreTrainedModel, BertModel

from colbert.parameters import DEVICE
from zhon.hanzi import punctuation
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


puncts = ''.join(punctuation.split())


class ColBERT_Distill_QA(BertPreTrainedModel):
    def __init__(self, config, query_maxlen, doc_maxlen, mask_punctuation, dim=128, similarity_metric='cosine', tokenizer=None):

        super(ColBERT_Distill_QA, self).__init__(config)

        self.query_maxlen = query_maxlen
        self.doc_maxlen = doc_maxlen
        self.similarity_metric = similarity_metric
        self.dim = dim

        self.mask_punctuation = mask_punctuation
        self.skiplist = {}

        if self.mask_punctuation:
            self.tokenizer = tokenizer
            self.skiplist = {w: True
                             for symbol in puncts
                             for w in [symbol, self.tokenizer.encode(symbol, add_special_tokens=False)[0]]}

        self.bert = BertModel(config)
        self.linear = nn.Linear(config.hidden_size, dim, bias=False)

        self.init_weights()

    def forward(self, Q, D, mask_q=False):
        return self.score(self.query(*Q), self.doc(*D), mask_q=mask_q)

    def query(self, input_ids, attention_mask):
        input_ids, attention_mask = input_ids.to(DEVICE), attention_mask.to(DEVICE)
        Q = self.bert(input_ids, attention_mask=attention_mask)[0]
        Q = self.linear(Q)
        # return torch.nn.functional.normalize(Q, p=2, dim=2)
        # return torch.nn.functional.normalize(Q, p=2, dim=2), torch.tensor(self.mask(attention_mask), device=DEVICE).float()
        return Q, torch.tensor(self.mask(attention_mask), device=DEVICE).float()

    def doc(self, input_ids, attention_mask, keep_dims=True):
        input_ids, attention_mask = input_ids.to(DEVICE), attention_mask.to(DEVICE)
        D = self.bert(input_ids, attention_mask=attention_mask)[0]
        D = self.linear(D)
        mask = torch.tensor(self.mask(input_ids), device=DEVICE).unsqueeze(2).float()
        D = D * mask

        # D = torch.nn.functional.normalize(D, p=2, dim=2)
        if not keep_dims:
            D, mask = D.cpu().to(dtype=torch.float16), mask.cpu().bool().squeeze(-1)
            D = [d[mask[idx]] for idx, d in enumerate(D)]
        return D

    def score(self, Q, D, mask_q=False):
        Q, q_mask = Q
        scores = None
        if self.similarity_metric == 'cosine':
            Q = Q.unsqueeze(1)
            D = D.permute(0, 2, 1)
            # Q @ D -> Q*D*seqQ*seqD
            # return (Q @ D).max(-1).values.sum(-1)
            scores = (Q @ D).max(-1).values
            if mask_q:
                scores = scores.permute(1, 0, 2)
                # print(scores.size())
                # print(q_mask.size())
                # exit()
                scores = (scores * q_mask).sum(-1).T
            else:
                scores = scores.sum(-1)

        elif self.similarity_metric == 'l2':
            QQ = Q.view(Q.size(0) * Q.size(1), -1)
            DD = D.view(D.size(0) * D.size(1), -1)
            dist = expanded_pairwise_distances(QQ, DD).view(Q.size(0), Q.size(1), D.size(0), D.size(1))
            # dist = pairwise_distances(QQ, DD).view(Q.size(0), Q.size(1), D.size(0), D.size(1))
            scores = (-1.0 * dist).max(-1).values
            if mask_q:
                scores = scores.permute(2, 0, 1)
                scores = (scores * q_mask).sum(-1).T
            else:
                scores = scores.sum(1)

        return scores

    def mask(self, input_ids):
        # mask = [[(x not in self.skiplist) and (x != 0) for x in d] for d in input_ids.cpu().tolist()]
        mask = [[idx == 0 for idx, x in enumerate(d)] for d in input_ids.cpu().tolist()]
        return mask
