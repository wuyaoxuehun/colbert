# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch.nn.functional as F

from transformers import BertModel, BertPreTrainedModel
from transformers.modeling_bert import BertEncoder

'''
Graph Attention network component
'''
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss


class GraphAttention(nn.Module):
    def __init__(self, in_dim=64,
                 out_dim=64,
                 num_heads=12,
                 feat_drop=0.6,
                 attn_drop=0.6,
                 alpha=0.2,
                 residual=True):

        super(GraphAttention, self).__init__()
        self.num_heads = num_heads
        self.fc = nn.Linear(in_dim, out_dim, bias=False)
        if feat_drop:
            self.feat_drop = nn.Dropout(feat_drop)
        else:
            self.feat_drop = lambda x: x
        if attn_drop:
            self.attn_drop = nn.Dropout(attn_drop)
        else:
            self.attn_drop = lambda x: x
        self.attn_l = nn.Parameter(torch.Tensor(size=(1, num_heads, out_dim)))
        self.attn_r = nn.Parameter(torch.Tensor(size=(1, num_heads, out_dim)))
        nn.init.xavier_normal_(self.fc.weight.data, gain=1.414)
        nn.init.xavier_normal_(self.attn_l.data, gain=1.414)
        nn.init.xavier_normal_(self.attn_r.data, gain=1.414)
        self.leaky_relu = nn.LeakyReLU(alpha)
        self.residual = residual
        self.activation = nn.ReLU()
        self.res_fc = nn.Linear(in_dim, out_dim, bias=False)
        nn.init.xavier_normal_(self.res_fc.weight.data, gain=1.414)

    ### this is Gragh attention network part, we follow standard inplementation from DGL library
    def forward(self, g):
        self.g = g
        h = g.ndata['h']
        h = h.reshape((h.shape[0], self.num_heads, -1))
        ft = self.fc(h)
        a1 = (ft * self.attn_l).sum(dim=-1).unsqueeze(-1)
        a2 = (ft * self.attn_r).sum(dim=-1).unsqueeze(-1)
        g.ndata.update({'ft': ft, 'a1': a1, 'a2': a2})
        g.apply_edges(self.edge_attention)
        g.update_all(self.message_func, self.reduce_func)
        ret = g.ndata['ft']
        if self.residual:
            if self.res_fc is not None:
                resval = self.res_fc(h)
            else:
                resval = torch.unsqueeze(h, 1)
            ret = resval + ret
        g.ndata['h'] = self.activation(ret.flatten(1))

    def message_func(self, edges):
        return {'z': edges.src['ft'], 'a': edges.data['a']}

    def reduce_func(self, nodes):
        alpha = F.softmax(nodes.mailbox['a'], dim=1)
        h = torch.sum(alpha * nodes.mailbox['z'], dim=1)
        return {'ft': h}

    def edge_attention(self, edges):
        a = self.leaky_relu(edges.src['a1'] + edges.dst['a2'])
        return {'a': a}


'''
Transformer-XH Encoder, we apply on last three BERT layers 

'''


class TransformerXHEncoder(BertEncoder):
    def __init__(self, config):
        super(TransformerXHEncoder, self).__init__(config)
        self.heads = ([8] * 1) + [1]
        self.config = config
        self.build_model()
        ### Here we apply on the last three layers, but it's ok to try different layers here.
        self.linear_layer1 = nn.Linear(config.hidden_size * 2, config.hidden_size)
        self.linear_layer2 = nn.Linear(config.hidden_size * 2, config.hidden_size)
        self.linear_layer3 = nn.Linear(config.hidden_size * 2, config.hidden_size)
        self.linear_layer1.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        self.linear_layer2.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        self.linear_layer3.weight.data.normal_(mean=0.0, std=self.config.initializer_range)

    def build_model(self):
        self.graph_layers = nn.ModuleList()
        # input to hidden
        device = torch.device("cuda")

        i2h = self.build_input_layer().to(device)
        self.graph_layers.append(i2h)
        # hidden to hidden
        h2h = self.build_hidden_layer().to(device)
        self.graph_layers.append(h2h)
        h2h = self.build_hidden_layer().to(device)
        self.graph_layers.append(h2h)

    ### here the graph has dimension 64, with 12 heads, the dropout rates are 0.6
    def build_input_layer(self):
        return GraphAttention()

    def build_hidden_layer(self):
        return GraphAttention()

    def forward(self, graph, hidden_states, attention_mask, gnn_layer_num, output_all_encoded_layers=True):
        all_encoder_layers = []
        for i, layer_module in enumerate(self.layer):
            hidden_states = layer_module(hidden_states, attention_mask)[0]
            pooled_output = hidden_states[:, 0]
            graph.ndata['h'] = pooled_output

            if i >= gnn_layer_num:
                if i == 9:
                    g_layer = self.graph_layers[0]
                    g_layer(graph)
                    graph_outputs = graph.ndata.pop('h')
                    ht_ori = hidden_states.clone()
                    ht_ori[:, 0] = self.linear_layer1(torch.cat((graph_outputs, pooled_output), -1))
                elif i == 10:
                    g_layer = self.graph_layers[1]
                    g_layer(graph)
                    graph_outputs = graph.ndata.pop('h')
                    ht_ori = hidden_states.clone()
                    ht_ori[:, 0] = self.linear_layer2(torch.cat((graph_outputs, pooled_output), -1))
                else:
                    g_layer = self.graph_layers[2]
                    g_layer(graph)
                    graph_outputs = graph.ndata.pop('h')
                    ht_ori = hidden_states.clone()
                    ht_ori[:, 0] = self.linear_layer3(torch.cat((graph_outputs, pooled_output), -1))
                hidden_states = ht_ori
                if output_all_encoded_layers:
                    all_encoder_layers.append(ht_ori)
            else:
                if output_all_encoded_layers:
                    all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers


'''
Transformer-XH main class
'''


class Transformer_xh(BertModel):
    def __init__(self, config):
        super(Transformer_xh, self).__init__(config)

        self.encoder = TransformerXHEncoder(config)

    def forward(self: BertModel, graph, input_ids, token_type_ids=None, attention_mask=None, position_ids=None, gnn_layer=11):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        # extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = extended_attention_mask.to(dtype=self.dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        embedding_output = self.embeddings(input_ids, position_ids=position_ids, token_type_ids=token_type_ids)
        encoder_outputs = self.encoder(graph, embedding_output,
                                       extended_attention_mask, gnn_layer)
        sequence_output = encoder_outputs[-1]
        pooled_output = self.pooler(sequence_output)
        outputs = sequence_output, pooled_output  # add hidden_states and attentions if they are here

        return outputs  # sequence_output, pooled_output, (hidden_states), (attentions)


class ScaledDotProductAttention(nn.Module):
    """ Scaled Dot-Product Attention """

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):
        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))
        if mask is not None:
            # attn = attn.masked_fill(mask == 0, -10000)
            attn = attn.masked_fill(mask == 0, -1e4)
        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)

        return output, attn


class MultiHeadAttention(nn.Module):
    """ Multi-Head Attention module """

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)

        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, q, k, v, mask=None, output_heads=False):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        residual = q

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)  # For head axis broadcasting.

        q, attn = self.attention(q, k, v, mask=mask)

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        # q_heads = None
        # if output_heads:
        # q_heads = q.transpose(1, 2).contiguous().view(sz_b, len_q, self.n_head, self.d_v)
        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, self.n_head * self.d_v)
        q = self.dropout(self.fc(q))
        q += residual
        q = self.layer_norm(q)
        # return q, q_heads, attn
        return q


def seperate_seq(sequence_output, context_len, bg_len, ques_len, opt_len, model_type="bert"):
    context_len = context_len.squeeze(-1)
    bg_len = bg_len.squeeze(-1)
    ques_len = ques_len.squeeze(-1)
    opt_len = opt_len.squeeze(-1)
    context_max_len = max(context_len) + 1
    bg_max_len = max(bg_len) + 1
    ques_max_len = max(ques_len) + 1
    opt_max_len = max(opt_len) + 1

    # p_seq_output = sequence_output.new(sequence_output.size()).zero_()
    p_seq_output = torch.zeros((sequence_output.size(0), context_max_len, sequence_output.size(2)), device=sequence_output.device)
    p_seq_mask = torch.zeros((sequence_output.size(0), context_max_len), device=sequence_output.device)
    # bg_seq_output = sequence_output.new(sequence_output.size()).zero_()
    bg_seq_output = torch.zeros((sequence_output.size(0), bg_max_len, sequence_output.size(2)), device=sequence_output.device)
    bg_seq_mask = torch.zeros((sequence_output.size(0), bg_max_len), device=sequence_output.device)

    q_seq_output = torch.zeros((sequence_output.size(0), ques_max_len, sequence_output.size(2)), device=sequence_output.device)
    q_seq_mask = torch.zeros((sequence_output.size(0), ques_max_len), device=sequence_output.device)

    o_seq_output = torch.zeros((sequence_output.size(0), opt_max_len, sequence_output.size(2)), device=sequence_output.device)
    o_seq_mask = torch.zeros((sequence_output.size(0), opt_max_len), device=sequence_output.device)
    # print(context_len)
    # input()
    context_len = context_len.squeeze(-1)
    bg_len = bg_len.squeeze(-1)
    ques_len = ques_len.squeeze(-1)
    opt_len = opt_len.squeeze(-1)
    for i in range(context_len.size(0)):
        p_seq_output[i, :context_len[i] + 1] = sequence_output[i, 1:context_len[i] + 2]
        bg_seq_output[i, :bg_len[i] + 1] = sequence_output[i, context_len[i] + 2
                                                              :context_len[i] + 2 + bg_len[i] + 1]
        q_seq_output[i, :ques_len[i] + 1] = sequence_output[i, context_len[i] + 2 + bg_len[i] + 1
                                                               :context_len[i] + 2 + bg_len[i] + 1 + ques_len[i] + 1]
        o_seq_output[i, :opt_len[i] + 1] = sequence_output[i, context_len[i] + 2 + bg_len[i] + 1 + ques_len[i] + 1
                                                              :context_len[i] + 2 + bg_len[i] + 1 + ques_len[i] + 1 + opt_len[i] + 1]
        # print('...' * 199)

        p_seq_mask[i, :context_len[i] + 1] = 1
        bg_seq_mask[i, :bg_len[i] + 1] = 1
        q_seq_mask[i, :ques_len[i] + 1] = 1
        o_seq_mask[i, :opt_len[i] + 1] = 1

    return p_seq_output, bg_seq_output, q_seq_output, o_seq_output, p_seq_mask, bg_seq_mask, q_seq_mask, o_seq_mask


def get_mask(hp_mask, hq_mask):
    a = hp_mask.unsqueeze(-1)
    b = hq_mask.unsqueeze(-2)
    mask_mat = torch.matmul(a, b)
    return mask_mat


import dgl


class self_attn(nn.Module):
    def __init__(self, d_model):
        super().__init__()

        self.linear = nn.Sequential(
            nn.Linear(d_model * 6, d_model * 3),
            nn.Tanh(),
            nn.Linear(d_model * 3, 1),
            nn.Tanh()
        )

    def forward(self, ctx):
        attn = self.linear(ctx).squeeze(-1)
        softmax_attn = F.softmax(attn, dim=-1).unsqueeze(-1)
        weighted_sum = (ctx * softmax_attn).sum(2)
        return weighted_sum


class ModelXH(BertPreTrainedModel):
    def __init__(self, config, p_num=2, p_topk=2, gnn_layer=9, n_head=8):
        super().__init__(config)
        ### node_encoder -> Transformer-XH
        self.bert = Transformer_xh(config)
        self.p_num = p_num
        self.p_topk = p_topk
        self.gnn_layer = gnn_layer
        self.node_dropout = nn.Dropout(config.hidden_dropout_prob)

        d_model = config.hidden_size
        self.d_model = d_model
        # d_k, d_v = (d_model,) * 2
        d_k, d_v = (d_model // n_head,) * 2
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # self.layers = nn.ModuleList([nn.ModuleList([MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout),
        #                                             # MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout),
        #                                             ])
        self.p_score_linear = nn.Linear(d_model, 1)

        self.layer_num = 1
        self.layer = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=0.1)
        self.fuse_match = nn.Sequential(
            nn.Linear(d_model * 2, d_model * 1),
            nn.ReLU()
        )

        self.self_attn = self_attn(d_model=d_model)
        # self.linear = nn.Linear(d_v * 4, 1)

        self.pred_final_layer = nn.Linear(self.d_model * 6, 1)

        self.init_weights()

    def forward(self, batch, device):
        ### Transformer-XH for node representations
        preds = None
        # print(batch[1:])
        g = batch[0]
        g = dgl.batch(sum(g, []))
        bsize = len(g) // 4 // self.p_num

        g.ndata['encoding'] = g.ndata['encoding'].to(device)
        # print(g.ndata['encoding'].size())
        g.ndata['encoding_mask'] = g.ndata['encoding_mask'].to(device)
        g.ndata['segment_id'] = g.ndata['segment_id'].to(device)
        outputs = self.bert(g, g.ndata['encoding'], g.ndata['segment_id'], g.ndata['encoding_mask'], gnn_layer=self.gnn_layer)
        # outputs = self.node_encoder(input_ids=g.ndata['encoding'], token_type_ids=g.ndata['segment_id'], attention_mask=g.ndata['encoding_mask'])
        # node_sequence_output = outputs[0]
        sequence_output, node_pooled_output = outputs
        # print(g.ndata['context_len'].size())
        context_len, bg_len, ques_len, opt_len = g.ndata['context_len'].squeeze(-1), \
                                                 g.ndata['background_len'].squeeze(-1), g.ndata['question_len'].squeeze(-1), g.ndata[
                                                     'opt_len'].squeeze(-1)

        # context_len, bg_len, ques_len, opt_len = g.ndata['context_len'].view(bsize, 4, self.p_num), \
        #                                          g.ndata['background_len'].view(bsize, 4, self.p_num), \
        #                                          g.ndata['question_len'].view(bsize, 4, self.p_num), \
        #                                          g.ndata['opt_len'].view(bsize, 4, self.p_num)

        # seqlen = sequence_output.size(1)
        ## sequence_output = sequence_output.view(bsize, 4, self.p_num, seqlen, self.d_model)
        # sequence_output = sequence_output.view(bsize, 4, self.p_num, -1)

        # p_sorted = self.p_score_linear(node_pooled_output).squeeze(-1).view(bsize, 4, self.p_num).argsort(-1, descending=True)[:, :, :self.p_topk]
        # t = p_sorted.unsqueeze(-1).repeat_interleave(seqlen * self.d_model, -1)
        # sequence_output = torch.gather(sequence_output, 2, t).view(-1, seqlen, self.d_model)
        #
        # # print(context_len.size())
        # # print(p_sorted.size())
        # # input()
        # context_len = torch.gather(context_len, 2, p_sorted).view(-1)
        # bg_len = torch.gather(bg_len, 2, p_sorted).view(-1)
        # ques_len = torch.gather(ques_len, 2, p_sorted).view(-1)
        # opt_len = torch.gather(opt_len, 2, p_sorted).view(-1)
        # print(context_len)
        # print(max(context_len) + 2)
        # input()
        p_seq_output, bg_seq_output, q_seq_output, o_seq_output, p_seq_mask, bg_seq_mask, q_seq_mask, o_seq_mask = \
            seperate_seq(sequence_output, context_len, bg_len, ques_len, opt_len)

        # match_cand = [p_seq_output[:, :max(context_len) + 2, :],
        #               bg_seq_output[:, :max(bg_len) + 2, :],
        #               q_seq_output[:, :max(ques_len) + 2, :],
        #               o_seq_output[:, :max(opt_len) + 2, :]]
        # macth_mask = [p_seq_mask[:, :max(context_len) + 2],
        #               bg_seq_mask[:, :max(bg_len) + 2],
        #               q_seq_mask[:, :max(ques_len) + 2],
        #               o_seq_mask[:, :max(opt_len) + 2]]
        match_cand = [p_seq_output, bg_seq_output, q_seq_output, o_seq_output]
        macth_mask = [p_seq_mask, bg_seq_mask, q_seq_mask, o_seq_mask]
        all_matches = []

        for i in range(4):
            for j in range(i + 1, 4):
                Q, Q_mask = match_cand[i], macth_mask[i]
                K, K_mask = match_cand[j], macth_mask[j]
                Q_K_mask = get_mask(Q_mask, K_mask)
                K_Q_mask = get_mask(K_mask, Q_mask)

                for i in range(self.layer_num):
                    Q_ = self.layer(Q, K, K, Q_K_mask)
                    K_ = self.layer(K, Q, Q, K_Q_mask)
                    Q, K = Q_, K_

                Q[Q_mask == 0] = -1e4
                K[K_mask == 0] = -1e4
                Q_pooled = Q.max(1)[0]
                K_pooled = K.max(1)[0]
                matched = self.fuse_match(torch.cat([Q_pooled, K_pooled], dim=-1))

                all_matches.append(matched)  # b*H

        # all_matches = torch.stack(all_matches, dim=1)
        all_matches = torch.cat(all_matches, dim=-1).view(bsize, 4, self.p_num, self.d_model * 6)
        # all_matches = torch.cat(all_matches, dim=-1).view(bsize, 4, self.p_topk, self.d_model * 6)
        weighted_match = self.self_attn(all_matches)  # bs * 4 * d_model

        preds = self.pred_final_layer(weighted_match).squeeze(-1)

        # node_pooled_output = self.node_dropout(node_pooled_output)
        # node_pooled_output = node_pooled_output.view(len(g) // 4 // self.p_num, 4, self.p_num * self.config.hidden_size)

        #### Task specific layer (last layer)
        # logits_score = self.final_layer(node_pooled_output).squeeze(-1)

        # logits_pred = self.pred_final_layer(node_pooled_output).squeeze(-1)
        # logits_pred = logits_pred.view(bsize, 4)
        # preds = logits_pred
        labels = batch[1]
        if labels.squeeze(0).size():
            batch[1].squeeze(0)

        labels = batch[1].to(device)
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(preds, labels)
            return loss, preds
        else:
            return preds


def test_sort():
    bsize = 2
    p_num = 4
    seqlen = 128
    h = 128
    a = torch.randn(bsize, 4, p_num, seqlen, h)
    plen = torch.randn(bsize, 4, p_num)
    b = torch.randint(0, 3, size=(bsize, 4, 2))
    tb = b.unsqueeze(-1).repeat_interleave(seqlen, -1).unsqueeze(-1).repeat_interleave(h, -1)
    print(a.size(), tb.size())
    res = torch.gather(a, 2, tb)

    plenres = torch.gather(plen, 2, b)
    print(plen)
    print(b)
    print(plenres)
    input()

    gold = torch.zeros(size=(2, 2, 1, 2))
    tb = b.squeeze(-1)
    for i in range(2):
        for j in range(2):
            for k in range(1):
                gold[i, j, k, :] = a[i, j, tb[i, j, k], :]

    assert torch.allclose(gold, res)
    print(a)
    print(t)
    print(res)


def test_attention():
    attn = MultiHeadAttention(n_head=4, d_model=128, d_k=32, d_v=32)
    k = torch.randn(4, 16, 128)
    v = torch.randn(4, 4, 128)
    a = attn(k, v, v)
    print(a.size())


if __name__ == '__main__':
    # Q = torch.tensor([[[1, 1, 1], [2, 2, 2]]]).float()
    # Q = torch.tensor([[[1, 1, 1], [6, 6, 6], [20, 20, 20], [0, 0, 0]]]).float()
    # K = torch.tensor([[[5, 5, 5], [8, 8, 8], [0, 0, 0]]]).float()
    # Q = torch.randn(2, 5, 3)
    # K = torch.randn(2, 3, 3)
    # # K = torch.tensor([[[5, 5, 5], [0, 0, 0]], [[7, 7, 7], [8, 8, 8]]]).float()
    #
    # Q_mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 0, 0]])
    # K_mask = torch.tensor([[1, 1, 0], [1, 0, 0]])
    #
    # print(Q.size())
    # print(K.size())
    # attn = MultiHeadAttention(n_head=1, d_model=3, d_k=3, d_v=3, dropout=0)
    # Q_K_mask = get_mask(Q_mask, K_mask)
    # print(Q_K_mask)
    # a = attn(q=Q, k=K, v=K, mask=Q_K_mask, output_heads=False)
    # a[Q_mask == 0] = -1e4
    # print(a)
    # a = a.max(1)[0]
    # print(a)
    # test_sort()
    test_attention()
    pass
