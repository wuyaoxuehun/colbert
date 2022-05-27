import json

import math
from torch import nn, einsum
import torch
from transformers import BertPreTrainedModel

from colbert.modeling.model_utils import span_mean, max_pool_by_mask, avg_pool_by_mask
import torch.nn.functional as F

from conf import use_part_weight, dim, QView, DView
from colbert.modeling.submodels import Comatch
from allennlp.modules.span_extractors import MaxPoolingSpanExtractor, EndpointSpanExtractor


class BaseModel(nn.Module):
    # def __init__(self, config, colbert_config, reader_config, load_old=True):
    def __init__(self):
        super().__init__()
        self.model = None
        self.linear = None
        self.word_weight_linear = None
        self.q_word_weight_linear = None
        self.d_word_weight_linear = None
        self.part_linear = None
        # self.get_representation = self.get_representation_whole if not use_part_weight else self.get_representation_part
        self.get_representation = self.get_representation_whole
        # kernel_sizes = [1]
        # n_kernels = 128
        # self.convs = nn.ModuleList([
        #     nn.Conv2d(
        #         in_channels=1,
        #         out_channels=n_kernels,
        #         kernel_size=(size, 768)
        #     )
        #     for size in kernel_sizes
        # ])
        # self.conv_fc = nn.Linear(len(kernel_sizes) * n_kernels, 1)

        # self.dropout = nn.Dropout(0.1)
        # self.part_linear = nn.Sequential(
        #     # nn.Linear(len(kernel_sizes) * n_kernels, 768),
        #     nn.Linear(768, 768),
        #     nn.ReLU(),
        #     nn.Linear(768, 1, bias=False),
        # )
        # dim = 128
        # self.esim_linear_42 = nn.Sequential(
        #     # nn.Linear(len(kernel_sizes) * n_kernels, 768),
        #     nn.Linear(4 * dim, 2 * dim),
        #     nn.ReLU(),
        #     nn.Linear(2 * dim, 2 * dim),
        # )
        # self.esim_linear_80 = nn.Sequential(
        #     # nn.Linear(len(kernel_sizes) * n_kernels, 768),
        #     nn.Linear(8 * dim, 1 * dim),
        #     nn.ReLU(),
        #     nn.Linear(1 * dim, 1),
        # )
        # self.proj = nn.Linear(dim, 768)
        # self.comatch = Comatch(dim=dim)
        # self.extractor = EndpointSpanExtractor(input_dim=768, combination="x+y")
        # self.miu_linear = nn.Linear(768, 128)
        # self.logv_linear = nn.Linear(768, 128)

    @property
    def encoder(self):
        return self.model

    def conv(self, embeddings):
        embeddings = embeddings.unsqueeze(1)
        conved = [F.relu(conv(embeddings)).squeeze(3) for conv in self.convs]
        pooled = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in conved]
        flattened = self.dropout(torch.cat(pooled, dim=1))
        # output = self.conv_fc(flattened)
        return flattened

    # def get_representation_whole(self, t, active_indices, active_padding=None, multiply_span_len=False,
    #                              use_word=True, l2norm=True, with_word_weight=False, embd_type="query", output_ori=False):
    #     if use_word:
    #         # proj first, span mean second, won't hurt?
    #         # t_ = self.linear(t_)
    #         # t_, span_len = span_mean(t_, active_indices)
    #         t_, span_len = span_mean(t, active_indices)
    #         # t_, span_len = t, None
    #         t = self.linear(t_)
    #         # if l2norm:
    #         t_norm = torch.nn.functional.normalize(t, p=2, dim=2)
    #         # t = t * span_len[:, :, None]
    #         # t = t + self.span_len_embedding(span_len)
    #         # t = t * self.span_len_embedding[span_len][:, :, None]
    #         # t = t * self.span_len_embedding[span_len][:, :, None]
    #         if multiply_span_len:
    #             # t = t * span_len[:, :, None]
    #             # t = t * torch.log2(span_len[:, :, None] + 1)
    #             # t = t * (span_len[:, :, None])
    #             t_norm = t_norm
    #         if with_word_weight:
    #             if embd_type == "query":
    #                 word_weight = self.q_word_weight_linear(t_).squeeze(-1)
    #                 word_weight[active_padding == 0] = -1e4
    #                 softmax_word_weight = F.softmax(word_weight, dim=-1)
    #                 t_norm = t_norm * softmax_word_weight[..., None]
    #             # else:
    #             #     word_weight = self.d_word_weight_linear(t_).squeeze(-1)
    #             #     word_weight[active_padding == 0] = -1e4
    #             #     softmax_word_weight = F.softmax(word_weight, dim=-1)
    #             #     t = t * softmax_word_weight[..., None]
    #         #
    #         #     return t
    #         # else:
    #         #     return t
    #     # else:
    #     #     t = self.linear(t)
    #     #     if l2norm:
    #     #         t = torch.nn.functional.normalize(t, p=2, dim=2)
    #     if output_ori:
    #         return t_norm, t
    #     return t

    # def get_representation_part(self, t, active_indices, active_padding=None, multiply_span_len=True,
    #                             use_word=True, l2norm=True, with_word_weight=False, embd_type="query"):
    #     t_, span_len = span_mean(t, active_indices)
    #     # t_, span_len = t, None
    #     t = self.linear(t_)
    #     if l2norm:
    #         t = torch.nn.functional.normalize(t, p=2, dim=2)
    #     if with_word_weight:
    #         if embd_type == "query":
    #             word_weight = self.q_word_weight_linear(t_).squeeze(-1)
    #             # word_weight[active_padding == 0] = -1e4
    #             part_reprs = []
    #             softmax_part_word_weights = []
    #             for i in [1, 2, 3]:
    #                 part_word_weight = word_weight.clone()
    #                 part_word_weight[active_padding != i] = -1e4
    #                 softmax_part_word_weight = F.softmax(part_word_weight, dim=-1)
    #                 softmax_part_word_weights.append(softmax_part_word_weight)
    #                 # input(softmax_part_word_weights[0])
    #
    #                 part_t = (t_ * softmax_part_word_weight[..., None]).sum(1)
    #                 part_reprs.append(part_t)
    #             part_reprs = torch.stack(part_reprs, dim=1)
    #             part_weight = self.part_linear(part_reprs).squeeze(-1)
    #             softmax_part_weight = F.softmax(part_weight, dim=-1)
    #             softmax_part_word_weights_ = []
    #             for i in [1, 2, 3]:
    #                 t_ = softmax_part_word_weights[i - 1] + softmax_part_weight[:, i - 1][:, None]
    #                 t_[active_padding != i] = 0
    #                 softmax_part_word_weights_.append(t_)
    #             # print(softmax_part_word_weights_)
    #             softmax_part_word_weights = sum(softmax_part_word_weights_)
    #             # input(softmax_part_weight)
    #             t = t * softmax_part_word_weights[..., None]
    #
    #             # if part_spans is not None:
    #             #     parts_mean, parts_mean_len, = span_mean(t_, part_spans)
    #     return t

    # def get_representation_whole(self, t, active_indices, active_padding=None, multiply_span_len=False,
    #                               use_word=True, l2norm=True, with_word_weight=False, embd_type="query", output_ori=False):
    #     # t_, span_len = span_mean(t, active_indices)
    #     # t_ = self.extractor(t, active_indices)
    #     # t = self.linear(t_)
    #     t = self.linear(t[:])
    #     # t = self.comatch.proj_linear(t_)
    #     t_norm = torch.nn.functional.normalize(t, p=2, dim=2)
    #     # if multiply_span_len:
    #     #     t_norm = t_norm
    #     if with_word_weight:
    #         if embd_type == "query":
    #             word_weight = self.q_word_weight_linear(t_).squeeze(-1)
    #             if True:
    #                 scale_factor = torch.log(active_padding.sum(-1)) / math.log(256) / math.sqrt(dim)
    #                 # print(word_weight[0])
    #                 word_weight = word_weight * scale_factor[:, None]
    #                 word_weight[active_padding == 0] = -1e4
    #                 # print(scale_factor)
    #                 # print(word_weight[0])
    #             softmax_word_weight = F.softmax(word_weight, dim=-1)
    #             # t_norm = t_norm * softmax_word_weight[..., None]
    #             t_norm = t_norm * softmax_word_weight[..., None]
    #     if output_ori:
    #         return t_norm, t_
    #     return t_norm

    def get_representation_whole(self, t, active_indices, active_padding=None, multiply_span_len=False,
                                 use_word=True, l2norm=True, with_word_weight=False, embd_type="query", output_ori=False):
        if embd_type == 'query':
            # t = self.linear(t[:])
            t = t[:, :QView, ...].contiguous()
        else:
            t = t[:, :DView, ...].contiguous()
        t = torch.nn.functional.normalize(t, p=2, dim=2)
        return t

    def get_representation_whole_(self, t, active_indices, *args, **kwargs):
        t_, span_len = span_mean(t, active_indices)
        mius, logvs = self.miu_linear(t_), self.logv_linear(t_)
        return torch.stack([mius, logvs], dim=1)

    def get_representation_part_(self, t, active_indices, active_padding=None, multiply_span_len=True,
                                 use_word=True, l2norm=True, with_word_weight=False, embd_type="query"):
        t_, span_len = span_mean(t, active_indices)
        # t_, span_len = t, None
        t = self.linear(t_)
        if l2norm:
            t = torch.nn.functional.normalize(t, p=2, dim=2)
        if with_word_weight:
            if embd_type == "query":
                word_weight = self.q_word_weight_linear(t_).squeeze(-1)
                # word_weight[active_padding == 0] = -1e4
                part_reprs = []
                softmax_part_word_weights = []
                for i in [1, 2, 3]:
                    part_word_weight = word_weight.clone()
                    part_word_weight[active_padding != i] = -1e4
                    softmax_part_word_weight = F.softmax(part_word_weight, dim=-1)
                    softmax_part_word_weights.append(softmax_part_word_weight)
                    # input(softmax_part_word_weights[0])

                    part_t = (t_ * softmax_part_word_weight[..., None]).sum(1)
                    part_reprs.append(part_t)
                part_reprs = torch.stack(part_reprs, dim=1)
                part_weight = self.part_linear(part_reprs).squeeze(-1)
                softmax_part_weight = F.softmax(part_weight, dim=-1)
                softmax_part_word_weights_ = []
                for i in [1, 2, 3]:
                    t_ = softmax_part_word_weights[i - 1] * (softmax_part_weight[:, i - 1][:, None])  # multiply, not add
                    t_[active_padding != i] = 0
                    softmax_part_word_weights_.append(t_)
                # print(softmax_part_word_weights_)
                # tlen = sum(active_padding.bool()[0])
                softmax_part_word_weights_ = sum(softmax_part_word_weights_)

                # print(sum(softmax_part_word_weights)[0][:tlen])
                # print(softmax_part_word_weights_[0][:tlen])
                # print(softmax_part_weight[0])
                # input()

                t = t * softmax_part_word_weights_[..., None]

                # if part_spans is not None:
                #     parts_mean, parts_mean_len, = span_mean(t_, part_spans)
        return t

    def get_representation_part(self, t, active_indices, active_padding=None, multiply_span_len=True,
                                use_word=True, l2norm=True, with_word_weight=False, embd_type="query"):
        t_, span_len = span_mean(t, active_indices)
        # t_, span_len = t, None
        t = self.linear(t_)
        if l2norm:
            t = torch.nn.functional.normalize(t, p=2, dim=2)
        if with_word_weight:
            if embd_type == "query":
                word_weight = self.q_word_weight_linear(t_).squeeze(-1)
                word_weight[active_padding == 0] = -1e4
                part_reprs = []
                softmax_part_word_weights = []
                for i in [1, 2, 3]:
                    part_word_weight = word_weight.clone()
                    # part_word_weight = torch.ones_like(word_weight)
                    part_word_weight[active_padding != i] = -1e4
                    softmax_part_word_weight = F.softmax(part_word_weight, dim=-1)
                    softmax_part_word_weights.append(softmax_part_word_weight)
                    part_t = (t_ * softmax_part_word_weight[..., None]).sum(1)
                    part_reprs.append(part_t)
                part_reprs = torch.stack(part_reprs, dim=1)
                part_weight = self.part_linear(part_reprs).squeeze(-1)
                softmax_part_weight = F.softmax(part_weight, dim=-1)

                softmax_word_weights = sum(softmax_part_word_weights)
                for i in [1, 2, 3]:
                    # temp = softmax_part_weight[:, i - 1][:, None].expand_as(softmax_word_weights)
                    # print(softmax_word_weights.size())
                    # input(softmax_part_weight.size())
                    temp = softmax_part_weight[:, i - 1][:, None].expand_as(softmax_word_weights)
                    temp1 = temp.clone()
                    temp1[active_padding != i] = 1
                    softmax_word_weights = softmax_word_weights * temp1
                    # print(temp1)
                    # print(softmax_word_weights)
                    # input()
                    # input()
                # print(sum(softmax_part_word_weights)[0][:tlen])
                # print(softmax_part_word_weights_[0][:tlen])
                # print(softmax_part_weight[0])
                # print(softmax_word_weights[0])
                # input()

                t = t * softmax_word_weights[..., None]
        return t

    def get_representation_part___(self, t, active_indices, active_padding=None, multiply_span_len=True,
                                   use_word=True, l2norm=True, with_word_weight=False, embd_type="query"):
        t_, span_len = span_mean(t, active_indices)
        # t_, span_len = t, None
        t = self.linear(t_)
        if l2norm:
            t = torch.nn.functional.normalize(t, p=2, dim=2)
        if with_word_weight:
            if embd_type == "query":
                word_weight = self.q_word_weight_linear(t_).squeeze(-1)
                word_weight[active_padding == 0] = -1e4
                part_reprs = []
                softmax_part_word_weights = []
                # softmax_word_weights = F.softmax(word_weight, dim=-1)
                for i in [1, 2, 3]:
                    part_word_weight = word_weight.clone()
                    # part_word_weight = torch.ones_like(word_weight)
                    part_word_weight[active_padding != i] = -1e4
                    softmax_part_word_weight = F.softmax(part_word_weight, dim=-1)
                    softmax_part_word_weights.append(softmax_part_word_weight)
                    # input(softmax_part_word_weights[0])
                    # print(part_word_weight)
                    temp = t_.clone()
                    # temp[active_padding != i] = 0
                    # part_t = self.conv(temp)
                    # temp[active_padding == i] = -1e4
                    # part_t = temp.max(1).values
                    temp[active_padding != i] = 0
                    part_t = temp.sum(1) / ((active_padding == i).sum(-1)[..., None])
                    # input(temp)

                    part_reprs.append(part_t)
                part_reprs = torch.stack(part_reprs, dim=1)
                part_weight = self.part_linear(part_reprs).squeeze(-1)
                softmax_part_weight = F.softmax(part_weight, dim=-1)
                # softmax_part_word_weights_ = []
                # active_padding_valid = active_padding.clone()
                # active_padding_valid[active_padding == 0] =
                for i in [1, 2, 3]:
                    # t_ = softmax_part_word_weights[i - 1] * (softmax_part_weight[:, i - 1][:, None])  # multiply, not add
                    # t_[active_padding != i] = 0
                    # softmax_part_word_weights_.append(t_)
                    temp = softmax_part_weight[:, i - 1][:, None].expand_as(softmax_word_weights)
                    temp1 = temp.clone()
                    temp1 = temp1 / ((active_padding == i).sum(-1)[..., None])
                    temp1[active_padding != i] = 1
                    # print(part_weight)
                    # print(softmax_part_weight[0])
                    # print(temp[0])
                    # print(temp1[0])
                    # print(active_padding[0])
                    softmax_word_weights = softmax_word_weights * temp1
                    # input()
                # print(softmax_part_weight[0])
                # print(softmax_word_weights[0])
                # input()
                # print(softmax_part_word_weights_)
                # tlen = sum(active_padding.bool()[0])
                # softmax_part_word_weights_ = sum(softmax_part_word_weights_)

                # print(sum(softmax_part_word_weights)[0][:tlen])
                # print(softmax_part_word_weights_[0][:tlen])
                print(softmax_part_weight[0])
                print(softmax_word_weights[0])
                input()

                t = t * softmax_word_weights[..., None]

                # if part_spans is not None:
                #     parts_mean, parts_mean_len, = span_mean(t_, part_spans)
        return t

    def get_dpr_representation(self, t, l2norm=True):
        # t = self.linear(t)
        t_ = t[:, :1, :]
        if l2norm:
            t = torch.nn.functional.normalize(t_, p=2, dim=2)
        return t

    def query(self, input_ids, attention_mask, active_indices=None, active_padding=None, output_ori=False, with_word_weight=True, dpr=False, **kwargs):
        # output = self.encoder(input_ids, attention_mask=attention_mask, return_dict=True).last_hidden_state
        output = self.encoder(input_ids, attention_mask=attention_mask, return_dict=True, output_hidden_states=True).hidden_states[-1]
        if not dpr:
            Q = self.get_representation(output, active_indices, active_padding, with_word_weight=with_word_weight, output_ori=output_ori)
        else:
            Q = self.get_dpr_representation(output)
        # if output_ori:
        # output = self.get_representation(output, active_indices, active_padding, with_word_weight=False)
        # return Q, output
        return Q

    def doc(self, input_ids, attention_mask, active_indices=None, active_padding=None, output_ori=False, with_word_weight=False, dpr=False, **kwargs):
        # output = self.encoder(input_ids, attention_mask=attention_mask, return_dict=True).last_hidden_state
        output = self.encoder(input_ids, attention_mask=attention_mask, return_dict=True, output_hidden_states=True).hidden_states[-1]
        if not dpr:
            D = self.get_representation(output, active_indices, multiply_span_len=False,
                                        with_word_weight=with_word_weight, output_ori=output_ori,
                                        embd_type="doc")
        else:
            D = self.get_dpr_representation(output)
        # if output_ori:
        # output = self.get_representation(output, active_indices, active_padding, with_word_weight=False)
        # return D, output
        # D = self.comatch.proj_linear(D)
        return D

    # def score_esim(self, Q, D, q_mask=None, d_mask=None):
    #     Q = F.normalize(Q, p=2, dim=-1)
    #     q, d, m, n, h = Q.size(0), D.size(0), Q.size(1), D.size(1), Q.size(2)
    #     sim_mat = einsum("qmh,dnh->qdmn", Q, D)
    #     sim_mat = sim_mat * q_mask[:, None, :, None] * d_mask[None, :, None, :]
    #     q_softmax, d_softmax = F.softmax(sim_mat, -1), F.softmax(sim_mat, -2)
    #     q_align, d_align = einsum("qdmn,dnh->qdmh", q_softmax, D), einsum("qdmn,qmh->qdnh", d_softmax, Q)
    #     Q_expand, D_expand = Q.unsqueeze(1).expand(q, d, m, h), D.unsqueeze(0).expand(q, d, n, h)
    #     # print(q_align.size(), Q_expand.size())
    #     # input()
    #     q_enhanced = self.esim_linear_42(torch.cat([q_align, Q_expand, q_align - Q_expand, q_align * Q_expand], dim=-1))
    #     d_enhanced = self.esim_linear_42(torch.cat([d_align, D_expand, d_align - D_expand, d_align * D_expand], dim=-1))
    #     q_max, d_max = max_pool_by_mask(q_enhanced, q_mask.unsqueeze(1)), max_pool_by_mask(d_enhanced, d_mask.unsqueeze(0))
    #     q_mean, d_mean = avg_pool_by_mask(q_enhanced, q_mask.unsqueeze(1)), avg_pool_by_mask(d_enhanced, d_mask.unsqueeze(0))
    #     scores = self.esim_linear_80(torch.cat([q_max, d_max, q_mean, d_mean], dim=-1)).squeeze(-1)
    #     # return scores / 0.05
    #     return scores

    def score_comatch(self, Q, D, q_mask=None, d_mask=None):
        return self.comatch(Q, D, q_mask, d_mask)
        # Q = F.normalize(Q, p=2, dim=-1)
        q, d, m, n, h = Q.size(0), D.size(0), Q.size(1), D.size(1), Q.size(2)
        Q_expand, D_expand = Q.unsqueeze(1).expand(q, d, m, h), D.unsqueeze(0).expand(q, d, n, h)
        q_mask, d_mask = q_mask.unsqueeze(1).expand(q, d, m).reshape(q * d, m), d_mask.unsqueeze(0).expand(q, d, n).reshape(q * d, n)
        Q, D = Q_expand.reshape(q * d, m, h), D_expand.reshape(q * d, n, h)
        # print(Q.size(), D.size(), q_mask.size(), d_mask.size())
        scores = self.comatch(Q, q_mask, D, d_mask)
        scores = scores.view(q, d)
        # return scores / 0.05
        return scores

    @staticmethod
    def score(Q, D, lce=False, *args, **kwargs):
        # print(q_mask[0])
        # q_mask = q_mask.bool().to(dtype=torch.int32)
        # d_mask = d_mask.bool().to(dtype=torch.int32)
        # if d_mask is not None and q_mask is not None:
        # Q_norm, D_norm = [F.normalize(_, p=2, dim=-1) for _ in [Q, D]]
        # Q_norm, D_norm = Q, D
        # D = D_norm * d_mask[..., None]
        # Q = Q_norm * q_mask[..., None]
        # if q_word_weight:
        # print(Q[0].norm(p=2, dim=-1))
        # print(D[0].norm(p=2, dim=-1))
        # input()
        simmat = einsum("qmh,dnh->qdmn", Q, D)
        scores_match, indices = simmat.max(-1)
        # print(scores[0][0])
        # input()
        scores = scores_match.sum(-1)
        # if output_match_weight:
        #     return scores, einsum("qmh,dnh->qdmn", F.normalize(Q, p=2, dim=-1), D).max(-1)[0]
        # scores = scores / (q_mask.bool().sum(-1)[:, None])
        # scores = F.relu(einsum("qmh,dnh->qdmn", Q, D)).max(-1)[0].sum(-1)
        if lce:
            from colbert.training.losses import BiEncoderNllLoss
            simmat = simmat.contiguous()
            lce_scores = simmat.view(Q.size(0) * Q.size(1) * D.size(0), -1) / 2e-2
            lce_labels = indices.view(-1)
            # print(lce_scores.size(), lce_labels.size())
            # input()
            lce_loss = BiEncoderNllLoss(lce_scores, positive_idx_per_question=lce_labels)
            return scores, lce_loss
        return scores

    @staticmethod
    def score_(Q, D, q_mask=None, d_mask=None, q_word_weight=None, output_match_weight=False):
        # print(Q.size(), D.size())
        # input()
        Q_mius, Q_logvs = Q[:, 0, ...], Q[:, 1, ...]
        D_mius, D_logvs = D[:, 0, ...], D[:, 1, ...]
        K, L = q_mask.bool().sum(-1), d_mask.bool().sum(-1)
        q, d, m, n, h = Q_mius.size(0), D_mius.size(0), Q_mius.size(1), D_mius.size(1), Q_mius.size(2)
        Q_mius_expand, D_mius_expand = Q_mius.unsqueeze(1).unsqueeze(3).expand(q, d, m, n, h), D_mius.unsqueeze(0).unsqueeze(2).expand(q, d, m, n, h)
        Q_logvs_expand, D_logvs_expand = Q_logvs.unsqueeze(1).unsqueeze(3).expand(q, d, m, n, h), D_logvs.unsqueeze(0).unsqueeze(2).expand(q, d, m, n, h)
        q_mask, d_mask = q_mask.unsqueeze(1).expand(q, d, m).reshape(q * d, m), d_mask.unsqueeze(0).expand(q, d, n).reshape(q * d, n)
        Q_mius, D_mius = Q_mius_expand.reshape(q * d * m * n, h), D_mius_expand.reshape(q * d * m * n, h)
        Q_logvs, D_logvs = Q_logvs_expand.reshape(q * d * m * n, h), D_logvs_expand.reshape(q * d * m * n, h)
        # print(Q_mius.size(), D_mius.size(), D_logvs.size(), Q_logvs.size())
        kl = 0.5 * (D_logvs - Q_logvs + (Q_logvs.exp() + (Q_mius - D_mius) ** 2) / (D_logvs.exp())).sum(-1)
        kl = kl.view(q * d, m, n)
        L_ = L[None, :, None, None].expand(q, d, m, n).reshape(q * d, m, n)
        kl = kl - 0.5 * h * L_
        d_mask = d_mask[:, None, :].expand(q * d, m, n)
        kl = kl * d_mask + 1e4 * (1 - d_mask)
        L, K = L[None, :].expand(q, d), K[:, None].expand(q, d)
        scores = kl.min(-1).values.mean(-1).view(q, d) + torch.log(L / K) / K
        return - scores

    @staticmethod
    def score__(Q, D, q_mask=None, d_mask=None, q_word_weight=None):
        # print(q_mask[0])
        # q_mask = q_mask.bool().to(dtype=torch.int32)
        # d_mask = d_mask.bool().to(dtype=torch.int32)
        ww = Q.norm(p=2, dim=-1)
        Q = F.normalize(Q, p=2, dim=-1)
        if d_mask is not None and q_mask is not None:
            D = D * d_mask[..., None]
            Q = Q * q_mask[..., None]
        # if q_word_weight:
        # print(Q[0].norm(p=2, dim=-1))
        # input()
        sim_mat = einsum("qmh,dnh->qdmn", Q, D)
        if (not Q.requires_grad) or True:
            scores = sim_mat.max(-1)[0]
            # input(Q.requires_grad)
        else:
            sim_mat = sim_mat * d_mask[None, :, None, :]
            # scores = torch.einsum("qdmn,qdmn->qdm", F.gumbel_softmax(sim_mat, tau=0.1, hard=True), sim_mat)
            scores = (F.gumbel_softmax(sim_mat, tau=0.1, hard=True) * sim_mat).sum(-1)
        scores = scores * ww[:, None, :]
        scores = scores * q_mask[:, None, :]
        scores = scores.sum(-1)
        # scores = scores / (q_mask.bool().sum(-1)[:, None])
        # scores = F.relu(einsum("qmh,dnh->qdmn", Q, D)).max(-1)[0].sum(-1)
        return scores

    @staticmethod
    def score_(Q, D, q_mask=None, d_mask=None, q_word_weight=None):
        if d_mask is not None and q_mask is not None:
            D = D * d_mask[..., None]
            Q = Q * q_mask[..., None]
        # if q_word_weight:
        normalized_D = F.normalize(D, p=2, dim=-1)
        values, indices = einsum("qmh,dnh->qdmn", Q, normalized_D).max(-1)
        d_word_weight = torch.norm(D, p=2, dim=-1).unsqueeze(0).expand((indices.size(0), indices.size(1), D.size(1)))

        selected_d_word_weight = torch.gather(d_word_weight, dim=-1, index=indices)
        assert selected_d_word_weight.size() == values.size()
        weighted_scores = values * selected_d_word_weight
        scores = weighted_scores.sum(-1)
        scores = scores / (q_mask.bool().sum(-1)[:, None])
        # print(Q)
        # print(D)
        # print(d_word_weight)
        # print(values)
        # print(indices)
        # print(selected_d_word_weight)
        # print(weighted_scores)

        # scores = F.relu(einsum("qmh,dnh->qdmn", Q, D)).max(-1)[0].sum(-1)
        return scores

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def save(self: BertPreTrainedModel, save_dir: str):
        save_info = self.save_info
        if save_info is None:
            save_info = {}
        to_save = self.state_dict()
        if isinstance(self, nn.DataParallel):
            to_save = self.module
        import os
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
        print('*' * 20 + "saving checkpoint to " + save_dir + '*' * 20)

        torch.save(to_save, os.path.join(save_dir, "pytorch.bin"))
        # args = {
        #     'colbert_config': self.colbert_config,
        #     'reader_config': self.reader_config
        # }
        args = save_info
        json.dump(args, open(os.path.join(save_dir, "training_args.bin"), 'w', encoding='utf8'), ensure_ascii=False, indent=4)

    def load(self: BertPreTrainedModel, checkpoint: str):
        # logger.info('*' * 20 + "loading checkpoint from " + checkpoint + '*' * 20)
        print('*' * 20 + "loading checkpoint from " + checkpoint + '*' * 20)
        return self.load_state_dict(torch.load(checkpoint, map_location=lambda storage, loc: storage), strict=True)


def test_score():
    Q = torch.tensor([[[1, 5, 4], [2, 8, 1]]]).float()
    D = torch.tensor([[[0, 0, 0], [1, 1, 1]], [[3, 2, 1], [1, 1, 3]]]).float()
    # Q = torch.randn(2, 4, 8)
    # D = torch.randn(3, 8, 8)
    # q_mask = torch.tensor([[1, 1]])
    # d_mask = torch.tensor([[1, 1, ], [1, 1, ]])
    q_mask, d_mask = torch.ones(Q.size()[:2]), torch.ones(D.size()[:2])
    score = BaseModel.score(Q, D, q_mask, d_mask)
    print(score)


def test_model():
    Q = torch.randn(2, 2, 2, 2)
    D = torch.randn(2, 3, 2, 2)
    q_mask = torch.tensor([[1, 1], [1, 1]])
    d_mask = torch.tensor([[1, 1], [1, 1], [1, 1]])
    Q_mius, Q_logvs = Q[0], Q[1]
    D_mius, D_logvs = D[0], D[1]
    K, L = q_mask.bool().sum(-1), d_mask.bool().sum(-1)
    q, d, m, n, h = Q_mius.size(0), D_mius.size(0), Q_mius.size(1), D_mius.size(1), Q_mius.size(2)
    Q_mius_expand, D_mius_expand = Q_mius.unsqueeze(1).unsqueeze(3).expand(q, d, m, n, h), D_mius.unsqueeze(0).unsqueeze(2).expand(q, d, m, n, h)
    Q_logvs_expand, D_logvs_expand = Q_logvs.unsqueeze(1).unsqueeze(3).expand(q, d, m, n, h), D_logvs.unsqueeze(0).unsqueeze(2).expand(q, d, m, n, h)
    q_mask, d_mask = q_mask.unsqueeze(1).expand(q, d, m).reshape(q * d, m), d_mask.unsqueeze(0).expand(q, d, n).reshape(q * d, n)
    Q_mius, D_mius = Q_mius_expand.reshape(q * d * m * n, h), D_mius_expand.reshape(q * d * m * n, h)
    Q_logvs, D_logvs = Q_logvs_expand.reshape(q * d * m * n, h), D_logvs_expand.reshape(q * d * m * n, h)
    # print(Q_mius.size(), D_mius.size(), D_logvs.size(), Q_logvs.size())
    kl = 0.5 * (D_logvs - Q_logvs + (Q_logvs.exp() + (Q_mius - D_mius) ** 2) / (D_logvs.exp())).sum(-1)
    kl = kl.view(q * d, m, n)
    # print(L[None, None, :].size(), q*d, m, n)
    L_ = L[None, :, None, None].expand(q, d, m, n).reshape(q * d, m, n)
    kl = kl - 0.5 * h * L_
    d_mask = d_mask[:, None, :].expand(q * d, m, n)
    # d_mask_ = 1 - d_mask
    kl = kl * d_mask + 1e4 * (1 - d_mask)
    L, K = L[None, :].expand(q, d), K[:, None].expand(q, d)
    # print(kl.min(-1).values.mean(1).size(), L.size(), K.size())
    scores = kl.min(-1).values.mean(1).view(q, d) + torch.log(L / K)
    print(scores)
    # print(kl.size())


if __name__ == '__main__':
    # test_score()
    test_model()
