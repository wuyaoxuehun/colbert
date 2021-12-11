import torch
from einops import repeat
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import BertModel, BertPreTrainedModel

from colbert.modeling.layers import MultiHeadAttentionWithBias, MultiHeadAttention
from colbert.modeling.model_utils import collect_p_bqo
from torch import einsum
import torch.nn.functional as F


def getmask(p, q):
    return (p.unsqueeze(-1)) @ (q.unsqueeze(-2))


class BertForDHC(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config)
        self.n_head = 8
        self.bqo_p_mha = MultiHeadAttentionWithBias(n_head=self.n_head, d_model=config.hidden_size,
                                                    d_k=config.hidden_size // self.n_head,
                                                    d_v=config.hidden_size // self.n_head,
                                                    dropout=0, bias_weight=1)
        self.p_bqo_mha = MultiHeadAttention(n_head=self.n_head, d_model=config.hidden_size,
                                            d_k=config.hidden_size // self.n_head,
                                            d_v=config.hidden_size // self.n_head,
                                            dropout=0)
        self.dropout = nn.Dropout(0.1)
        self.linear = nn.Sequential(
            nn.Linear(2 * config.hidden_size, 1),
        )
        self.init_weights()

    def forward(self, input_ids=None, token_type_ids=None, attention_mask=None, segment_lens=None, retriever_bias=None, labels=None, is_evaluating=False, **kwargs):
        B, _, PN = input_ids.size()[:3]
        input_ids = input_ids.view(-1, input_ids.size(3))
        token_type_ids = token_type_ids.view(-1, token_type_ids.size(3))
        attention_mask = attention_mask.view(-1, attention_mask.size(3))
        segment_lens = segment_lens.view(-1, segment_lens.size(3))

        seq_output, pooled_output = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, return_dict=False)[:2]
        p, p_mask, bqo, bqo_mask = collect_p_bqo(seq_output, segment_lens)
        # pooled_output = pooled_output.view(-1, 4, pooled_output.size(1))

        P = p.size(1)
        Q = bqo.size(1)
        p1 = p.reshape(B * 4, PN * P, -1)
        p_mask1 = p_mask.reshape(B * 4, PN * P)
        # bqo1 = bqo.reshape(B * 4, PN, Q, -1).mean(dim=1)
        # bqo1 = einsum('abc,ab->ac', bqo.reshape(B * 4, PN, Q, -1), retriever_bias.view(B * 4, PN))
        # input(retriever_bias.size())
        # retriever_bias = torch.zeros_like(retriever_bias)

        retriever_bias_soft = F.softmax(retriever_bias, dim=-1)
        bqo1 = einsum('abcd,ab->acd', bqo.reshape(B * 4, PN, Q, -1), retriever_bias_soft)
        retriever_bias1 = repeat(retriever_bias, 'a b -> a c (b d)', c=Q, d=P)

        bqo_mask1 = bqo_mask[0::PN]
        bqo_p_mask = getmask(bqo_mask1, p_mask1)
        bqo_p_output = self.bqo_p_mha(q=bqo1, k=p1, v=p1, mask=bqo_p_mask, bias=retriever_bias1)

        p_bqo_mask = getmask(p_mask, bqo_mask)
        p_bqo_output = self.p_bqo_mha(q=p, k=bqo, v=bqo, mask=p_bqo_mask)
        # add residual connection, may also near layer norm #todo
        bqo_p_output += bqo1
        p_bqo_output += p
        bqo_p_output[bqo_mask1 == 0] = -1e-4
        p_bqo_output[p_mask == 0] = -1e-4

        pooled_bqo_p = bqo_p_output.max(1)[0]
        # retriever_bias = retriever_bias.view(B, 4, PN)
        pooled_p_bqo = p_bqo_output.max(1)[0].view(B * 4, PN, -1)
        pooled_p_bqo = einsum('abc,ab->ac', pooled_p_bqo, retriever_bias_soft)
        res_output = torch.cat([pooled_bqo_p, pooled_p_bqo], dim=-1)

        res_output = self.dropout(res_output)
        res_output = res_output.view(B, 4, res_output.size(-1))
        preds = self.linear(res_output).squeeze(-1)
        if is_evaluating:
            output = preds, labels
            return output
        loss_fun = CrossEntropyLoss()
        loss = loss_fun(preds, labels)
        return loss, preds, labels
