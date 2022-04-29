import dgl
import math
import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import RelGraphConv
from torch.nn import CrossEntropyLoss
from torch_geometric.nn import RGCNConv
from transformers import BertModel, BertPreTrainedModel
# from transformers import Multi
# from colbert.modeling.layers import MultiHeadAttention
from colbert.modeling.model_utils import collect_pbqo, batch_index_select, max_pool_by_mask, avg_pool_by_mask
from conf import *
from colbert.training.losses import listMLEWeighted
from torch import nn
from torch.nn import MultiheadAttention
from torch_geometric.nn import GATv2Conv, GATConv

from colbert.modeling.submodels import MyMHA


def getmask(p, q):
    return (p.unsqueeze(-1)) @ (q.unsqueeze(-2))


# class BertForDHC(BertPreTrainedModel):
#     def __init__(self, config):
#         super().__init__(config)
#         self.bert = BertModel(config)
#         self.n_head = 8
#         self.bqo_p_mha = MultiHeadAttentionWithBias(n_head=self.n_head, d_model=config.hidden_size,
#                                                     d_k=config.hidden_size // self.n_head,
#                                                     d_v=config.hidden_size // self.n_head,
#                                                     dropout=0, bias_weight=1)
#         self.p_bqo_mha = MultiHeadAttention(n_head=self.n_head, d_model=config.hidden_size,
#                                             d_k=config.hidden_size // self.n_head,
#                                             d_v=config.hidden_size // self.n_head,
#                                             dropout=0)
#         self.dropout = nn.Dropout(0.1)
#         self.linear = nn.Sequential(
#             nn.Linear(2 * config.hidden_size, 1),
#         )
#         self.init_weights()
#
#     def forward(self, input_ids=None, token_type_ids=None, attention_mask=None, segment_lens=None, retriever_bias=None, labels=None, is_evaluating=False, **kwargs):
#         B, _, PN = input_ids.size()[:3]
#         input_ids = input_ids.view(-1, input_ids.size(3))
#         token_type_ids = token_type_ids.view(-1, token_type_ids.size(3))
#         attention_mask = attention_mask.view(-1, attention_mask.size(3))
#         segment_lens = segment_lens.view(-1, segment_lens.size(3))
#
#         seq_output, pooled_output = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, return_dict=False)[:2]
#         p, p_mask, bqo, bqo_mask = collect_p_bqo(seq_output, segment_lens)
#         # pooled_output = pooled_output.view(-1, 4, pooled_output.size(1))
#
#         P = p.size(1)
#         Q = bqo.size(1)
#         p1 = p.reshape(B * 4, PN * P, -1)
#         p_mask1 = p_mask.reshape(B * 4, PN * P)
#         # bqo1 = bqo.reshape(B * 4, PN, Q, -1).mean(dim=1)
#         # bqo1 = einsum('abc,ab->ac', bqo.reshape(B * 4, PN, Q, -1), retriever_bias.view(B * 4, PN))
#         # input(retriever_bias.size())
#         # retriever_bias = torch.zeros_like(retriever_bias)
#
#         retriever_bias_soft = F.softmax(retriever_bias, dim=-1)
#         bqo1 = einsum('abcd,ab->acd', bqo.reshape(B * 4, PN, Q, -1), retriever_bias_soft)
#         retriever_bias1 = repeat(retriever_bias, 'a b -> a c (b d)', c=Q, d=P)
#
#         bqo_mask1 = bqo_mask[0::PN]
#         bqo_p_mask = getmask(bqo_mask1, p_mask1)
#         bqo_p_output = self.bqo_p_mha(q=bqo1, k=p1, v=p1, mask=bqo_p_mask, bias=retriever_bias1)
#
#         p_bqo_mask = getmask(p_mask, bqo_mask)
#         p_bqo_output = self.p_bqo_mha(q=p, k=bqo, v=bqo, mask=p_bqo_mask)
#         # add residual connection, may also near layer norm #todo
#         bqo_p_output += bqo1
#         p_bqo_output += p
#         bqo_p_output[bqo_mask1 == 0] = -1e4
#         p_bqo_output[p_mask == 0] = -1e4
#
#         pooled_bqo_p = bqo_p_output.max(1)[0]
#         # retriever_bias = retriever_bias.view(B, 4, PN)
#         pooled_p_bqo = p_bqo_output.max(1)[0].view(B * 4, PN, -1)
#         pooled_p_bqo = einsum('abc,ab->ac', pooled_p_bqo, retriever_bias_soft)
#         res_output = torch.cat([pooled_bqo_p, pooled_p_bqo], dim=-1)
#
#         res_output = self.dropout(res_output)
#         res_output = res_output.view(B, 4, res_output.size(-1))
#         preds = self.linear(res_output).squeeze(-1)
#         if is_evaluating:
#             output = preds, labels
#             return output
#         loss_fun = CrossEntropyLoss()
#         loss = loss_fun(preds, labels)
#         return loss, preds, labels

class Object(object):
    pass


# class MyMHA(nn.Module):
#     def __init__(self, d_model, n_head, dropout=0., bias=True, add_bias_kv=False, add_zero_attn=False, d_k=None, d_v=None, num_layers=None):
#         super().__init__()
#         self.mhas = nn.ModuleList([MultiheadAttention(d_model, n_head, dropout, bias, add_bias_kv, add_zero_attn, d_k, d_v)
#                                    for _ in range(num_layers)])
#         self.layer_norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(num_layers)])
#
#
#     def forward(self, s, s_mask, t, t_mask):
#         s = s.permute(1, 0, 2)
#         t = t.permute(1, 0, 2)
#         s_mask = s_mask == 0
#         t_mask = t_mask == 0
#         for mha, ln in zip(self.mhas, self.layer_norms):
#             s_ = mha(query=s, key=t, value=t, key_padding_mask=t_mask)[0]
#             t_ = mha(query=t, key=s, value=s, key_padding_mask=s_mask)[0]
#             s, t = ln(s_ + s), ln(t_ + t)
#             # s, t = s_, t_
#         return s.permute(1, 0, 2), t.permute(1, 0, 2)


# class MyMHA_(nn.Module):
#     def __init__(self, d_model, n_head, dropout=0., bias=True, add_bias_kv=False, add_zero_attn=False, d_k=None, d_v=None, num_layers=None):
#         super().__init__()
#         self.mhas = nn.ModuleList([MultiHeadAttention(n_head=self.n_head, d_model=config.hidden_size,
#                                                       d_k=config.hidden_size // self.n_head,
#                                                       d_v=config.hidden_size // self.n_head,
#                                                       dropout=dropout_rate) for _ in range(n_mha_layers)])
#         self.layer_norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(num_layers)])
#
#     def forward(self, s, s_mask, t, t_mask):
#         s = s.permute(1, 0, 2)
#         t = t.permute(1, 0, 2)
#         for mha, ln in zip(self.mhas, self.layer_norms):
#             s_ = mha(query=s, key=t, value=t, key_padding_mask=t_mask)[0]
#             t_ = mha(query=t, key=s, value=s, key_padding_mask=s_mask)[0]
#             s, t = ln(s_), ln(t_)
#         return s.permute(1, 0, 2), t.permute(1, 0, 2)




class MultiGatedRGCN(nn.Module):
    def __init__(self, in_channels, out_channels, n_relations, n_layers=1):
        super().__init__()
        self.rgcns = nn.ModuleList([RGCNConv(in_channels=in_channels, out_channels=out_channels, num_relations=n_relations) for _ in range(n_layers)])
        # self.rgcn_lns = nn.ModuleList([nn.LayerNorm(in_channels) for _ in range(n_layers)])
        self.rgcn_gate_linear = nn.Sequential(
            nn.Linear(2 * in_channels, out_channels),
            nn.Sigmoid()
        )

    def forward(self, nodes, edges, edge_types):
        for rgcn, ln in zip(self.rgcns, self.rgcn_lns):
            nodes_ = rgcn(nodes, edges, edge_types)
            gate = self.rgcn_gate_linear(torch.cat([nodes_, nodes], dim=-1))
            nodes = gate * F.tanh(nodes_) + (1 - gate) * nodes
            # nodes = ln(nodes + nodes_)
            # nodes = self.layer_norm(nodes)
        return nodes


class BertForMC(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel.from_pretrained(reader_pretrained, config=config)
        self.n_head = 12
        self.config = config
        dropout_rate = 0.1
        n_mha_layers = 1
        n_rgcn_layers = 3

        # self.mhas = nn.ModuleList([MultiHeadAttention(n_head=self.n_head, d_model=config.hidden_size,
        #                                               d_k=config.hidden_size // self.n_head,
        #                                               d_v=config.hidden_size // self.n_head,
        #                                               dropout=dropout_rate) for _ in range(n_mha_layers)])
        self.mhas = MyMHA(n_head=self.n_head, d_model=config.hidden_size,
                          d_k=config.hidden_size,
                          d_v=config.hidden_size,
                          dropout=dropout_rate, num_layers=n_mha_layers)
        # config.attention_probs_dropout_prob = dropout_rate
        # self.mhas = MyMHA(config, num_layers=n_mha_layers)
        # self.mha.w_qs.weight.data.copy_(self.bert.encoder.layer[-1].attention.self.query.weight.data)
        # self.mha.w_ks.weight.data.copy_(self.bert.encoder.layer[-1].attention.self.key.weight.data)
        # self.mha.w_vs.weight.data.copy_(self.bert.encoder.layer[-1].attention.self.value.weight.data)
        # self.mha.fc.weight.data.copy_(self.bert.encoder.layer[-1].attention.output.dense.weight.data)

        self.score_linear_proj = nn.Sequential(
            nn.Linear(4 * config.hidden_size, config.hidden_size),
            nn.Tanh(),
            nn.Linear(config.hidden_size, config.hidden_size),
        )

        self.match_proj = nn.Sequential(
            nn.Linear(2 * config.hidden_size, config.hidden_size),
            nn.Tanh(),
            nn.Linear(config.hidden_size, config.hidden_size),
        )

        self.score_linear_proj_8 = nn.Sequential(
            nn.Linear(8 * config.hidden_size, config.hidden_size),
            nn.Tanh(),
            nn.Linear(config.hidden_size, config.hidden_size),
        )
        self.score_linear_3 = nn.Sequential(
            nn.Linear(3 * config.hidden_size, config.hidden_size),
            nn.Tanh(),
            nn.Linear(config.hidden_size, 1),
        )

        self.score_linear_6 = nn.Sequential(
            nn.Linear(6 * config.hidden_size, config.hidden_size),
            nn.Tanh(),
            nn.Linear(config.hidden_size, 1),
        )

        # self.mha_output_proj = self.linear = nn.Sequential(
        #     nn.Linear(4 * config.hidden_size, config.hidden_size),
        #     nn.ReLU(),
        #     nn.Linear(config.hidden_size, config.hidden_size),
        # )
        self.dropout = nn.Dropout(dropout_rate)
        self.choice_linear = nn.Sequential(
            nn.Linear(config.hidden_size, 1),
        )
        self.para_choice_linear = nn.Sequential(
            nn.Linear(reader_p_num, 1)
        )

        self.fine_match_linear = nn.Sequential(
            nn.Linear(4 * config.hidden_size, config.hidden_size),
            nn.Tanh(),
            nn.Linear(config.hidden_size, config.hidden_size),
        )

        self.fine_match_linear_3 = nn.Sequential(
            nn.Linear(3 * config.hidden_size, config.hidden_size),
            nn.Tanh(),
            nn.Linear(config.hidden_size, config.hidden_size),
        )

        self.fine_match_linear_1 = nn.Sequential(
            nn.Linear(1 * config.hidden_size, config.hidden_size),
            nn.Tanh(),
            nn.Linear(config.hidden_size, config.hidden_size),
        )

        self.attention_linear = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.Tanh(),
            nn.Linear(config.hidden_size, 1),
        )
        self.rgcns = nn.ModuleList([RGCNConv(config.hidden_size, config.hidden_size, 14) for _ in range(n_rgcn_layers)])
        self.rgcn_lns = nn.ModuleList([nn.LayerNorm(config.hidden_size) for _ in range(n_rgcn_layers)])
        self.rgcn_linear = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.Tanh(),
            nn.Linear(config.hidden_size, 1),
        )
        # self.rgcn_gate_linear = nn.Sequential(
        #     nn.Linear(2 * config.hidden_size, config.hidden_size),
        #     nn.Tanh(),
        #     nn.Linear(config.hidden_size, 1),
        # )
        # self.rgcns = MultiGatedRGCN(in_channels=config.hidden_size, out_channels=config.hidden_size, n_relations=14, n_layers=n_rgcn_layers)
        self.init_weights()

    def match_one_(self, s, s_mask, t, t_mask, proj=False):
        # st_mask = getmask(s_mask, t_mask)
        # ts_mask = getmask(t_mask, s_mask)
        # for mha in self.mhas:
        #     st_output = mha(q=s, k=t, v=t, mask=st_mask)
        #     ts_output = mha(q=t, k=s, v=s, mask=ts_mask)
        #     s, t = self.layer_norm(st_output), self.layer_norm(ts_output)
        st_output, ts_output = self.mhas(s, s_mask, t, t_mask)
        # add residual connection, may also near layer norm #todo
        # st_output += s
        # ts_output += t
        # st_output = self.layer_norm(st_output)
        # ts_output = self.layer_norm(ts_output)
        st_output[s_mask == 0] = -1e4
        ts_output[t_mask == 0] = -1e4

        pooled_st = st_output.max(1)[0]
        # retriever_bias = retriever_bias.view(B, 4, PN)
        pooled_ts = ts_output.max(1)[0]
        # pooled_output = torch.cat([pooled_st, pooled_ts, ], dim=-1)
        # if proj:
        #     pooled_output = torch.cat([pooled_st, pooled_ts, pooled_st - pooled_ts, pooled_st * pooled_ts], dim=-1)
        #     res_output = self.dropout(self.mha_output_proj(pooled_output))
        #     return res_output
        # else:
        return pooled_st, pooled_ts



    def match_one(self, s, s_mask, t, t_mask, proj=False):
        st_output, ts_output = self.mhas(s, s_mask, t, t_mask)
        # add residual connection, may also near layer norm #todo
        # st_output += s
        # ts_output += t
        # st_output = self.layer_norm(st_output)
        # ts_output = self.layer_norm(ts_output)
        # st_output[s_mask == 0] = -1e4
        # ts_output[t_mask == 0] = -1e4
        #
        # max_pooled_st = st_output.max(1)[0]
        # max_pooled_ts = ts_output.max(1)[0]
        #
        # st_output[s_mask == 0] = 0
        # ts_output[t_mask == 0] = 0
        #
        # mean_pooled_st = st_output.sum(1) / (s_mask.sum(-1)[..., None])
        # mean_pooled_ts = ts_output.sum(1) / (s_mask.sum(-1)[..., None])
        max_pooled_st = max_pool_by_mask(st_output, s_mask)
        mean_pooled_st = avg_pool_by_mask(st_output, s_mask)

        max_pooled_ts = max_pool_by_mask(ts_output, t_mask)
        mean_pooled_ts = avg_pool_by_mask(ts_output, t_mask)

        # retriever_bias = retriever_bias.view(B, 4, PN)
        pooled_st = self.match_proj(torch.cat([max_pooled_st, mean_pooled_st], dim=-1))
        pooled_ts = self.match_proj(torch.cat([max_pooled_ts, mean_pooled_ts], dim=-1))

        # pooled_output = torch.cat([pooled_st, pooled_ts, ], dim=-1)
        # if proj:
        #     pooled_output = torch.cat([pooled_st, pooled_ts, pooled_st - pooled_ts, pooled_st * pooled_ts], dim=-1)
        #     res_output = self.dropout(self.mha_output_proj(pooled_output))
        #     return res_output
        # else:
        return pooled_st, pooled_ts

    def pool_mix_two(self, pooled_st, pooled_ts):
        pooled_output = torch.cat([pooled_st, pooled_ts, pooled_st - pooled_ts, pooled_st * pooled_ts], dim=-1)
        res_output = self.score_linear_proj(self.dropout(pooled_output))
        return res_output

    def uni_match(self, s, s_mask, t, t_mask):
        st_mask = getmask(s_mask, t_mask)
        st_output = self.mha(q=s, k=t, v=t, mask=st_mask)

        # st_output += s
        # st_output = self.layer_norm(st_output)
        st_output[s_mask == 0] = -1e4
        pooled_st = st_output.max(1)[0]
        return pooled_st

    def fine_match(self, pbqo, pbqo_mask):
        output = []
        for i in range(4):
            t1, t1_mask = pbqo[i], pbqo_mask[i]
            for j in range(i + 1, 4):
                t2, t2_mask = pbqo[j], pbqo_mask[j]
                output.append(self.match_one(t1, t1_mask, t2, t2_mask))

        res = torch.cat(output, dim=1)
        res = self.dropout(self.fine_match_linear(res))
        return res

    def triple_match_(self, pbqo, pbqo_mask):
        output = []
        for i in range(4):
            t1, t1_mask = pbqo[i], pbqo_mask[i]
            t1[t1_mask == 0] = -1e4
            t = [t1.max(1).values]
            # t = []
            for j in range(4):
                if i == j:
                    continue
                t2, t2_mask = pbqo[j], pbqo_mask[j]
                # t.append(self.uni_match(t1, t1_mask, t2, t2_mask))
                t.append(self.match_one(t1, t1_mask, t2, t2_mask, proj=False)[0])
            res_t = self.dropout(self.layer_norm(self.fine_match_linear(torch.cat(t, dim=-1))))
            # res_t = self.dropout(self.layer_norm(torch.stack(t, dim=1).sum(1)))
            output.append(res_t)
        res = output
        return res

    def triple_match__(self, match_mat):
        output = []
        for i in range(4):
            t = []
            for j in range(4):
                # t.append(self.uni_match(t1, t1_mask, t2, t2_mask))
                t.append(match_mat[i][j])
            res_t = self.fine_match_linear(torch.cat(t, dim=-1))
            # res_t = self.dropout(self.layer_norm(torch.stack(t, dim=1).sum(1)))
            output.append(res_t)
        res = output
        return res

    def triple_match(self, match_mat):
        output = []
        for i in range(4):
            t = []
            for j in range(4):
                # t.append(self.uni_match(t1, t1_mask, t2, t2_mask))
                t.append(match_mat[i][j])
                # if j != i:
                #     t.append(match_mat[i][j])
            # res_t = self.fine_match_linear_1(torch.cat(t, dim=-1))
            # res_t = self.fine_match_linear(torch.cat(t, dim=-1))
            res_t = self.fine_match_linear(torch.cat(t, dim=-1))
            # res_t = self.dropout(self.layer_norm(torch.stack(t, dim=1).sum(1)))
            output.append(res_t)
        res = output
        return res

    def p_bqo_match(self, match_mat):
        output = []
        for i in range(1):
            for j in range(i + 1, 4):
                # matched_output = self.pool_mix_two(match_mat[i][j], match_mat[j][i])
                matched_output = self.pool_mix_two(match_mat[i][j] + match_mat[i][i], match_mat[j][i] + match_mat[j][j])
                output.append(matched_output)
        res = self.score_linear_3(self.dropout(torch.cat(output, dim=-1)))
        return res

    def pbqo_match_residual(self, match_mat):
        output = []
        for i in range(4):
            for j in range(i + 1, 4):
                matched_output = self.pool_mix_two(match_mat[i][j], match_mat[j][i])
                output.append(matched_output)
        res = self.score_linear_6(self.dropout(torch.cat(output, dim=-1)))
        return res

    def pbqo_match(self, match_mat):
        output = []
        for i in range(4):
            for j in range(i + 1, 4):
                matched_output = self.pool_mix_two(match_mat[i][j], match_mat[j][i])
                output.append(matched_output)
        res = self.score_linear_6(self.dropout(torch.cat(output, dim=-1)))
        return res

    def score_paras(self, match_mat, B, PN, retriever_bias=None, n_keep=2, labels=None, must_keep=False):
        # res = self.p_bqo_match(match_mat)
        res = self.pbqo_match(match_mat)
        res = res.view(B * 4, PN)
        loss_fn = nn.CrossEntropyLoss()
        if hasattr(self, "para_choice_linear"):
            rank_preds = self.para_choice_linear(res).view(B, 4)
        else:
            rank_preds = res.max(-1).values.view(B, 4)
        # if labels:
        choice_para_loss = loss_fn(rank_preds, labels)
        indices = res.sort(descending=True, dim=-1).indices
        rank_loss = listMLEWeighted(y_pred=res / retriever_bias_temperature, y_true=retriever_bias) / PN
        # input(rank_loss)
        if must_keep:
            indices = torch.tensor([list(range(n_keep))] * (B * 4), device=indices.device)
        for i in range(4):
            match_mat[i][i] = batch_index_select(match_mat[i][i].view(B * 4, PN, -1), dim=1, inds=indices[..., :n_keep])
            for j in range(i + 1, 4):
                match_mat[i][j] = batch_index_select(match_mat[i][j].view(B * 4, PN, -1), dim=1, inds=indices[..., :n_keep])
                match_mat[j][i] = batch_index_select(match_mat[j][i].view(B * 4, PN, -1), dim=1, inds=indices[..., :n_keep])
        return match_mat, choice_para_loss, rank_loss, rank_preds, indices

    def vanilla_match(self, match_mat):
        output = []
        for i in range(4):
            # res_t = self.dropout(self.layer_norm(torch.stack(t, dim=1).sum(1)))
            output.append(match_mat[i][i])
        res = output
        return res

    def forward_rgcn(self, nodes, edges, edge_types):
        # nodes = self.layer_norm(nodes)
        for rgcn, ln in zip(self.rgcns, self.rgcn_lns):
            nodes_ = rgcn(nodes, edges, edge_types)
            nodes = ln(nodes_)
            # nodes = ln(nodes + nodes_)
            # nodes = self.layer_norm(nodes)
        return nodes
        # return self.rgcns(nodes, edges, edge_types)

    def rank_docs_(self, pbqo, pbqo_mask, B=None, PN=None, H=768, n_keep=2):
        edges, edge_types = [torch.tensor(_).to(pbqo[0].device) for _ in get_edges(PN)]
        pbqo_new = [_.clone() for _ in pbqo]
        for i in range(1, 4):
            pbqo_new[i][pbqo_mask[i] == 0] = -1e4
            pbqo_new[i] = pbqo_new[i].view(B * 4, PN, -1, H).max(2).values.mean(1)
        pbqo_new[0] = pbqo_new[0].view(B * 4, PN, -1, H).max(2).values
        output = []
        # pbqo_output = [_.view(B * 4, PN, -1, H) for _ in pbqo]
        # pbqo_mask_output = [_.view(B * 4, PN, -1) for _ in pbqo_mask]
        # pbqo_res = []
        # pbqo_mask_res = []
        for i in range(B * 4):
            nodes = torch.stack([pbqo_new[_][i] for _ in range(1, 4)], dim=0)
            # print(nodes.size(), pbqo_new[0][i].size())
            # input()
            nodes = torch.cat([pbqo_new[0][i], nodes], dim=0)
            # node_output = self.rgcn(nodes, edges, edge_types)
            node_output = self.forward_rgcn(nodes, edges, edge_types)
            # input(node_output.size())
            # pnodes = node_output[3:]
            # weights = self.rgcn_linear(pnodes).squeeze(-1)
            # sorted_indices = weights.sort(descending=True).indices
            # pbqo_res.append([pbqo_output[i][sorted_indices[:n_keep]]])
            # t = self.weighted_sum(pnodes, weights)
            # output.append(t)
            output.append(node_output[2])

        res = torch.stack(output, dim=0)
        return res

    def rank_docs__(self, pbqo, pbqo_mask, B=None, PN=None, H=768, n_keep=2):
        edges, edge_types = [torch.tensor(_).to(pbqo[0].device) for _ in get_edges(PN)]
        pbqo_new = [_.clone() for _ in pbqo]
        for i in range(1, 4):
            pbqo_new[i][pbqo_mask[i] == 0] = -1e4
            pbqo_new[i] = pbqo_new[i].view(B * 4, PN, -1, H).max(2).values.mean(1).view(B, 4, -1)
        pbqo_new[0] = pbqo_new[0].view(B, 4, PN, -1, H).max(-2).values
        output = []
        # pbqo_output = [_.view(B * 4, PN, -1, H) for _ in pbqo]
        # pbqo_mask_output = [_.view(B * 4, PN, -1) for _ in pbqo_mask]
        # pbqo_res = []
        # pbqo_mask_res = []
        for i in range(B):
            nodes = torch.stack([pbqo_new[_][i] for _ in range(1, 4)], dim=1)
            # print(nodes.size(), pbqo_new[0][i].size())
            # input()
            nodes = torch.cat([pbqo_new[0][i], nodes], dim=1).view((3 + PN) * 4, H)
            # node_output = self.rgcn(nodes, edges, edge_types)
            node_output = self.forward_rgcn(nodes, edges, edge_types)
            # input(node_output.size())
            # pnodes = node_output[3:]
            # weights = self.rgcn_linear(pnodes).squeeze(-1)
            # sorted_indices = weights.sort(descending=True).indices
            # pbqo_res.append([pbqo_output[i][sorted_indices[:n_keep]]])
            # t = self.weighted_sum(pnodes, weights)
            # output.append(t)
            total = 3 + PN
            opt_index = [2 + _ * total for _ in range(4)]
            output.append(node_output[opt_index])

        res = torch.stack(output, dim=0).view(B, 4, H)
        return res

    def rank_docs___(self, pbqo, B=None, PN=None, H=768):
        edges, edge_types = [torch.tensor(_).to(pbqo[0].device) for _ in get_edges(PN)]
        pbqo_new = [_.clone() for _ in pbqo]
        for i in range(1, 4):
            pbqo_new[i] = pbqo_new[i].view(B * 4, PN, H).mean(1).view(B, 4, -1)
        pbqo_new[0] = pbqo_new[0].view(B, 4, PN, H)
        output = []
        for i in range(B):
            nodes = torch.stack([pbqo_new[_][i] for _ in range(1, 4)], dim=1)
            nodes = torch.cat([pbqo_new[0][i], nodes], dim=1).view((3 + PN) * 4, H)
            node_output = self.forward_rgcn(nodes, edges, edge_types)
            # input(node_output.size())
            # pnodes = node_output[3:]
            # weights = self.rgcn_linear(pnodes).squeeze(-1)
            # sorted_indices = weights.sort(descending=True).indices
            # pbqo_res.append([pbqo_output[i][sorted_indices[:n_keep]]])
            # t = self.weighted_sum(pnodes, weights)
            # output.append(t)
            total = 3 + PN
            opt_index = [2 + _ * total for _ in range(4)]
            output.append(node_output[opt_index])

        res = self.dropout(torch.stack(output, dim=0).view(B, 4, H))
        return res

    def match_matrix(self, pbqo, pbqo_mask):
        match_mat = [[0] * 4 for _ in range(4)]
        for i in range(4):
            t1, t1_mask = pbqo[i], pbqo_mask[i]
            # t1[t1_mask == 0] = -1e4
            # match_mat[i][i] = t1.max(1).values
            # max_pooled_ts = t1.max(1)[0]
            # t1[t1_mask == 0] = -0
            # mean_pooled_ts = t1.sum(1)
            match_mat[i][i] = self.match_proj(torch.cat([max_pool_by_mask(t1, t1_mask), avg_pool_by_mask(t1, t1_mask)], dim=-1))

            for j in range(i + 1, 4):
                if i == j:
                    continue
                t2, t2_mask = pbqo[j], pbqo_mask[j]
                # t.append(self.uni_match(t1, t1_mask, t2, t2_mask))
                pooled_st, pooled_ts = self.match_one(t1, t1_mask, t2, t2_mask, proj=False)
                match_mat[i][j] = pooled_st
                match_mat[j][i] = pooled_ts
        return match_mat

    def rank_docs(self, pbqo, B, PN, H=768):
        edges, edge_types = [torch.tensor(_).to(pbqo[0].device) for _ in get_edges(PN)]
        pbqo_new = [_.clone() for _ in pbqo]
        for i in range(1, 4):
            pbqo_new[i] = pbqo_new[i].view(B * 4, PN, H).mean(1).view(B, 4, -1)
        pbqo_new[0] = pbqo_new[0].view(B, 4, PN, H)
        output = []
        for i in range(B):
            nodes = torch.stack([pbqo_new[_][i] for _ in range(1, 4)], dim=1)
            nodes = torch.cat([nodes, pbqo_new[0][i]], dim=1).view((3 + PN) * 4, H)
            node_output = self.forward_rgcn(nodes, edges, edge_types)
            total = 3 + PN
            opt_index = [2 + _ * total for _ in range(4)]
            output.append(node_output[opt_index])

        res = self.dropout(torch.stack(output, dim=0).view(B, 4, H))
        return res

    def weighted_sum(self, embs, weights):
        if weights is None:
            weights = self.attention_linear(embs).squeeze(-1)
        softmax_weigths = F.softmax(weights, -1)
        res = embs * softmax_weigths[..., None]
        res = res.sum(-2)
        res_output = self.dropout(res)
        return res_output

    def vanilla_forward(self, pooled_output, B=None, labels=None, is_evaluating=False):
        preds = self.choice_linear(pooled_output.view(B, 4, -1, pooled_output.size(-1)).max(2).values).view(B, 4)
        loss = nn.CrossEntropyLoss()(preds, labels)
        if is_evaluating:
            output = (loss,), preds, labels
            return output
        return (loss,),

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, segment_lens=None, labels=None, retriever_bias=None, is_evaluating=False, is_testing=False, args=None, **kwargs):
        B, ON, PN, SL = input_ids.size()[:4]
        input_ids = input_ids.view(-1, SL)
        token_type_ids = token_type_ids.view(-1, SL)
        attention_mask = attention_mask.view(-1, SL)
        segment_lens = segment_lens.view(-1, segment_lens.size(3))

        seq_output, pooled_output = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, return_dict=False)[:2]

        # return self.vanilla_forward(pooled_output, B, labels, is_evaluating)

        pbqo, pbqo_mask = collect_pbqo(seq_output, segment_lens)
        # pooled_output = pooled_output.view(-1, 4, pooled_output.size(1))

        # res_output = self.fine_match(pbqo, pbqo_mask)
        # res_output = res_output.view(B * ON, PN, self.config.hidden_size)
        # res_output = self.weighted_sum(res_output, torch.ones((B * ON, PN)))
        if retriever_bias is None:
            # assert is_evaluating == True
            retriever_bias = torch.ones((B * ON, PN), device=input_ids.device)
        else:
            retriever_bias = torch.tensor(retriever_bias, device=input_ids.device)
            retriever_bias = retriever_bias.view(B * ON, PN)
        match_mat = self.match_matrix(pbqo, pbqo_mask)
        match_mat, choice_para_loss, rank_loss, rank_preds, indices = \
            self.score_paras(match_mat, B=B, PN=PN, retriever_bias=retriever_bias, n_keep=n_keep, labels=labels, must_keep=False)  # args.cur_epoch < epoch_thre)
        # pbqo_matched = self.vanilla_match(match_mat)
        # res_output = self.rank_docs(pbqo, pbqo_mask, B=B, PN=PN)

        if training_rank:
            if is_evaluating:
                output = (rank_loss,), rank_preds, labels
                return output
            return (rank_loss,),

        pbqo_matched = self.triple_match(match_mat)
        res_output = self.rank_docs(pbqo_matched, B=B, PN=n_keep)

        # res_output = self.weighted_sum(res_output, retriever_bias / retriever_bias_temperature)
        # res_output = self.weighted_sum(res_output, retriever_bias)

        res_output = res_output.view(B, 4, res_output.size(-1))
        preds = self.choice_linear(res_output).squeeze(-1)
        loss_fun = CrossEntropyLoss()
        loss = loss_fun(preds, labels)
        if is_evaluating:
            # output = (loss, rank_loss), preds, labels
            output = (loss,), preds, labels
            # output = (rank_loss,), rank_preds, labels
            return output
        if is_testing:
            output = (loss,), preds, labels, indices
            return output
        # return (rank_loss,),
        return (loss,),
        if args.cur_epoch >= epoch_thre:
            coef = 0.2 * (0.5 ** (args.cur_epoch - 1))
            return (rank_loss * coef, loss * (1 - coef)),
        else:
            return (loss * 0.8, rank_loss * 0.2),


def test_reader():
    sl = 8
    input_ids = torch.randint(0, 100, (2, 4, 2, sl))
    attention_mask = torch.tensor([[[[1, 1, 1, 1, 1, 1, 1, 0]] * 2] * 4] * 2)
    segment_lens = torch.tensor([[[[1, 1, 1, 1]] * 2] * 4] * 2)
    token_type_ids = torch.ones_like(attention_mask)
    print(input_ids.size(), attention_mask.size(), segment_lens.size())
    model = BertForMC.from_pretrained(pretrain)
    output = model(input_ids, token_type_ids, attention_mask, segment_lens)


def test_dataset():
    from dgl.data.rdf import AIFBDataset  # , MUTAGDataset, BGSDataset, AMDataset

    def load_data(data_name, get_norm=False, inv_target=False):
        if data_name == 'aifb':
            dataset = AIFBDataset()
        elif data_name == 'mutag':
            dataset = MUTAGDataset()
        elif data_name == 'bgs':
            dataset = BGSDataset()
        else:
            dataset = AMDataset()

        # Load hetero-graph
        hg = dataset[0]

        num_rels = len(hg.canonical_etypes)
        category = dataset.predict_category
        num_classes = dataset.num_classes
        labels = hg.nodes[category].data.pop('labels')
        train_mask = hg.nodes[category].data.pop('train_mask')
        test_mask = hg.nodes[category].data.pop('test_mask')
        train_idx = th.nonzero(train_mask, as_tuple=False).squeeze()
        test_idx = th.nonzero(test_mask, as_tuple=False).squeeze()

        if get_norm:
            # Calculate normalization weight for each edge,
            # 1. / d, d is the degree of the destination node
            for cetype in hg.canonical_etypes:
                hg.edges[cetype].data['norm'] = dgl.norm_by_dst(hg, cetype).unsqueeze(1)
            edata = ['norm']
        else:
            edata = None

        # get target category id
        category_id = hg.ntypes.index(category)

        g = dgl.to_homogeneous(hg, edata=edata)
        # Rename the fields as they can be changed by for example NodeDataLoader
        g.ndata['ntype'] = g.ndata.pop(dgl.NTYPE)
        g.ndata['type_id'] = g.ndata.pop(dgl.NID)
        node_ids = th.arange(g.num_nodes())

        # find out the target node ids in g
        loc = (g.ndata['ntype'] == category_id)
        target_idx = node_ids[loc]

        if inv_target:
            # Map global node IDs to type-specific node IDs. This is required for
            # looking up type-specific labels in a minibatch
            inv_target = th.empty((g.num_nodes(),), dtype=th.int64)
            inv_target[target_idx] = th.arange(0, target_idx.shape[0],
                                               dtype=inv_target.dtype)
            return g, num_rels, num_classes, labels, train_idx, test_idx, target_idx, inv_target
        else:
            return g, num_rels, num_classes, labels, train_idx, test_idx, target_idx

    g, num_rels, num_classes, labels, train_idx, test_idx, target_idx = load_data(data_name="aifb")
    g = g.int()
    print(g[0])
    print(g[0].srcdata[dgl.NID])
    print(g[0].edata[dgl.ETYPE], g[0].edata['norm'])


def test_pg():
    from torch_geometric.nn import RGCNConv
    in_channels = 768
    out_channels = 4
    num_relations = 3
    model = RGCNConv(in_channels, out_channels, num_relations)
    x = torch.randn(8, 768)
    edge_index = torch.tensor([[1, 2, 3], [4, 5, 6]])
    edge_type = torch.tensor([0, 1, 2])
    # mask = edge_type == 1
    # res = edge_index[:, mask]
    # print(res)
    output = model(x, edge_index, edge_type)
    print(output)


def get_edges(PN, concat=True):
    edge_types = ["bq", "qb", "bo", "ob", "qo", "oq", "pb", "bp", "pq", "qp", "po", "op", "pp", "self"]
    if concat:
        edge_types.append("oo")
    edge_type_map = {_: idx for idx, _ in enumerate(edge_types)}
    edge_types = []
    edges = []
    for s, sidx in zip("bqo", [0, 1, 2]):
        for t, tidx in zip("bqo", [0, 1, 2]):
            edges.append((sidx, tidx))
            if s != t:
                edge_type = edge_type_map[s + t]
            else:
                edge_type = edge_type_map['self']
            edge_types.append(edge_type)
        for i in range(PN):
            tidx = 3 + i
            edges.append((sidx, tidx))
            edge_type = edge_type_map[s + 'p']
            edge_types.append(edge_type)
    for i in range(PN):
        sidx = 3 + i
        for t in range(3):
            edges.append((sidx, t))
            edge_types.append(edge_type_map['p' + "bqo"[t]])
        for j in range(PN):
            tidx = 3 + j
            edges.append((sidx, tidx))
            if sidx != tidx:
                edge_type = edge_type_map["pp"]
            else:
                edge_type = edge_type_map['self']
            edge_types.append(edge_type)

    if concat:
        res_edges = edges[:]
        res_edge_types = edge_types[:] * 4
        total = 3 + PN
        for i in range(1, 4):
            start = i * total
            new_edges = [(start + s, start + t) for s, t in edges]
            res_edges += new_edges
        for i in range(4):
            for j in range(4):
                if i == j:
                    continue
                s = 2 + i * total
                t = 2 + j * total
                res_edges.append((s, t))
                res_edge_types.append(edge_type_map["oo"])
        edges = res_edges
        edge_types = res_edge_types
    edges = list(zip(*edges))
    return edges, edge_types


def test_edges():
    PN = 2
    edges, edge_types = get_edges(PN)
    for i, j in zip(edges, edge_types):
        print(i, j)


def testmha():
    d_model, n_head = 2, 1
    mha = MultiheadAttention(d_model, n_head)
    torch.random.manual_seed(100)
    s = torch.randn((1, 3, 2))
    # t = torch.randint(0, 2, (1, 4, 2)).permute(1, 0, 2).float()
    t = s.clone()[:, [2, 1, 0], ...]
    s = s.permute(1, 0, 2).float()
    t = t.permute(1, 0, 2).float()
    s = F.normalize(s, p=1, dim=-1)
    t = F.normalize(t, p=1, dim=-1)
    mask = torch.tensor([[1, 1, 0]])
    output = mha(s, t, t, key_padding_mask=mask == 0)
    res = output[0].permute(1, 0, 2)
    res = res - s
    print(s)
    print(t)
    print(mask)
    print(res)
    print(output[1].permute(1, 0, 2))


def test_load_pretrain():
    model = BertModel.from_pretrained(reader_pretrained)


def test_rgat():
    from torch_geometric.nn import RGCNConv, GATConv


# class SubsetOperator(torch.nn.Module):
#     def __init__(self, k, tau=1.0, hard=False):
#         super(SubsetOperator, self).__init__()
#         self.k = k
#         self.hard = hard
#         self.tau = tau
#
#     def forward(self, scores, EPSILON=1e-4):
#         m = torch.distributions.gumbel.Gumbel(torch.zeros_like(scores), torch.ones_like(scores))
#         g = m.sample()
#         scores = scores + g
#
#         # continuous top k
#         khot = torch.zeros_like(scores)
#         onehot_approx = torch.zeros_like(scores)
#         for i in range(self.k):
#             khot_mask = torch.max(1.0 - onehot_approx, torch.tensor([EPSILON]))
#             print('mask', khot_mask)
#             scores = scores + torch.log(khot_mask)
#             onehot_approx = torch.nn.functional.softmax(scores / self.tau, dim=1)
#             khot = khot + onehot_approx
#         print('khot', khot)
#         if self.hard:
#             # will do straight through estimation if training
#             khot_hard = torch.zeros_like(khot)
#             val, ind = torch.topk(khot, self.k, dim=1)
#             khot_hard = khot_hard.scatter_(1, ind, 1)
#             res = khot_hard - khot.detach() + khot
#         else:
#             res = khot
#
#         return res

class SubsetOperator(torch.nn.Module):
    def __init__(self, k, tau=1.0, hard=False):
        super(SubsetOperator, self).__init__()
        self.k = k
        self.hard = hard
        self.tau = tau

    def forward(self, scores, EPSILON=1e-4):
        m = torch.distributions.gumbel.Gumbel(torch.zeros_like(scores), torch.ones_like(scores))
        g = m.sample()
        scores = scores + g
        topk_indices = torch.topk(scores, self.k, dim=1)[1]
        print("gumbel", scores)
        # continuous top k
        khot = torch.zeros_like(scores)
        onehot_approx = torch.zeros_like(scores)
        khot_list = []
        for i in range(self.k):
            khot_mask = torch.max(1.0 - onehot_approx, torch.tensor([EPSILON]))
            # print('mask', khot_mask)
            scores = scores + torch.log(khot_mask)
            onehot_approx = torch.nn.functional.softmax(scores / self.tau, dim=1)
            khot = khot + onehot_approx
            khot_list.append(onehot_approx)

        # print('khot', khot)
        # print(khot_list)
        if self.hard:
            res = []
            # will do straight through estimation if training
            # topk_indices = torch.topk(khot, self.k, dim=1)[1]
            for i in range(self.k):
                khot = khot_list[i]
                # input(topk_indices[..., i:i + 1])
                khot_hard = torch.zeros_like(khot)
                # val, ind = torch.topk(khot, 1, dim=1)
                ind = topk_indices[..., i:i+1]
                # print(ind)
                khot_hard = khot_hard.scatter_(1, ind, 1)
                # print(khot_hard)
                res.append(khot_hard - khot.detach() + khot)
            # input(res)
            # print(res)
            res = torch.stack(res, 1)
            # if self.k == 1:
            #     res = res.unsqueeze(1)
        else:
            res = khot

        return res


def test_gumbel():
    bs, k, d = 2, 4, 3
    torch.manual_seed(2)
    h = torch.randn(bs, k, d, requires_grad=True)
    linear = torch.nn.Linear(d, 1)
    res = linear(h).view(bs, k)
    # res = torch.tensor([[1.1, 1.2, 1.3, 1.4], [2, 2, 5, 1]]).float()
    n_keep = 2
    print(res)
    if False:
        indices = res.sort(descending=True, dim=-1).indices
        print(indices)
        selected = batch_index_select(h, dim=1, inds=indices[..., :n_keep])
    else:
        ind = SubsetOperator(k=n_keep, tau=1, hard=True)(res)
        if True:
            # print(ind.size(), h.size())
            h_ = ind @ h
            selected = h_
        else:
            # h_ = torch.einsum("bk,bkd->bkd", ind, h)
            # h_ = h * ind[..., None]
            h_ = ind @ h
            selected = h_
            # indices = ind.sort(descending=True, dim=-1).indices
            # print(indices)
            # selected = batch_index_select(h_, dim=1, inds=indices[..., :n_keep])
        # print(h)
        print(selected)
        print(ind)
    linear2 = torch.nn.Linear(d, 2)
    F.cross_entropy(linear2(selected.mean(1)), torch.tensor([1, 0])).backward()
    print(h.grad)


if __name__ == '__main__':
    # test_reader()
    # test_dataset()
    # test_pg()
    # test_edges()
    # test_load_pretrain()
    # testmha()
    test_gumbel()
