import torch
from torch import nn, einsum
from torch.nn import MultiheadAttention

from colbert.modeling.model_utils import max_pool_by_mask, avg_pool_by_mask
import torch.nn.functional as F


class MyMHA(nn.Module):
    def __init__(self, d_model, n_head, dropout=0., bias=True, add_bias_kv=False, add_zero_attn=False, d_k=None, d_v=None, num_layers=None):
        super().__init__()
        self.mhas = nn.ModuleList([MultiheadAttention(d_model, n_head, dropout, bias, add_bias_kv, add_zero_attn, d_k, d_v)
                                   for _ in range(num_layers)])
        self.layer_norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(num_layers)])
        self.match_proj = nn.Sequential(
            nn.Linear(4 * d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )
        # self.score_linear_proj = nn.Sequential(
        #     nn.Linear(2 * d_model, d_model),
        #     nn.Tanh(),
        #     nn.Linear(d_model, d_model)
        # )
        # self.gat = GATConv(in_channels=d_model, out_channels=d_model, dropout=dropout)

    def forward(self, s, s_mask, t, t_mask):
        s = s.permute(1, 0, 2)
        t = t.permute(1, 0, 2)
        s_mask = s_mask == 0
        t_mask = t_mask == 0
        for mha, ln in zip(self.mhas, self.layer_norms):
            s_ = mha(query=s, key=t, value=t, key_padding_mask=t_mask)[0]
            t_ = mha(query=t, key=s, value=s, key_padding_mask=s_mask)[0]
            s, t = ln(s_ + s), ln(t_ + t)
            # s = ln(self.match_proj(torch.cat([s_, s, s_ - s, s_ * s], dim=-1)))
            # t = ln(self.match_proj(torch.cat([t_, t, t_ - t, t_ * t], dim=-1)))
            # s, t = s_, t_
            # s, t = s_, t_
        return s.permute(1, 0, 2), t.permute(1, 0, 2)


class Comatch(nn.Module):
    def __init__(self, dim=128, n_head=1, dropout=0, n_layers=1):
        super().__init__()
        d_model = 768
        self.mhas = MyMHA(n_head=n_head,
                          d_model=d_model,
                          d_k=d_model,
                          d_v=d_model,
                          dropout=dropout,
                          num_layers=n_layers)
        self.match_proj = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )

        self.q_upproj = nn.Sequential(
            nn.Linear(768, d_model),
            # nn.ReLU(),
            # nn.Linear(d_model, dim),
        )
        self.d_upproj = nn.Sequential(
            nn.Linear(dim, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )

        self.score_linear_proj = nn.Sequential(
            nn.Linear(4 * d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1)
        )
        self.proj_linear = nn.Linear(d_model, dim, bias=False)
        self.dropout = nn.Dropout(0)

    def forward(self, Q, D, q_mask, d_mask):
        # Q, D = [F.normalize(self.upproj(_), p=2, dim=-1) for _ in [Q, D]]
        # Q, D = F.normalize(self.q_upproj(Q)), F.normalize(self.d_upproj(D))
        # Q, D = F.normalize(Q, p=2, dim=-1), F.normalize(self.d_upproj(D), p=2, dim=-1)
        # Q, D = F.normalize(self.q_upproj(Q), p=2, dim=-1), F.normalize(self.d_upproj(D), p=2, dim=-1)
        Q, D = F.normalize(self.proj_linear(Q), p=2, dim=-1), F.normalize(self.proj_linear(D), p=2, dim=-1)
        # return self.max_forward(Q, D, q_mask, d_mask)
        return self.max_forward_gumbel(Q, D, q_mask, d_mask)
        # s = torch.cat([s, s.mean(1).unsqueeze(1)], dim=1)
        # s_mask = torch.cat([s_mask, torch.zeros(s_mask.size(0), 1, dtype=s_mask.dtype, device=s_mask.device)], dim=-1)
        # t = torch.cat([t, t.mean(1).unsqueeze(1)], dim=1)
        # t_mask = torch.cat([t_mask, torch.zeros(t_mask.size(0), 1, dtype=t_mask.dtype, device=t_mask.device)], dim=-1)
        s, t = self.upproj(s), self.upproj(t)
        st_output, ts_output = self.mhas(s, s_mask, t, t_mask)
        max_pooled_st = max_pool_by_mask(st_output, s_mask)
        mean_pooled_st = avg_pool_by_mask(st_output, s_mask)
        max_pooled_ts = max_pool_by_mask(ts_output, t_mask)
        mean_pooled_ts = avg_pool_by_mask(ts_output, t_mask)

        pooled_st = self.match_proj(torch.cat([max_pooled_st, mean_pooled_st], dim=-1))
        pooled_ts = self.match_proj(torch.cat([max_pooled_ts, mean_pooled_ts], dim=-1))
        pooled_output = torch.cat([pooled_st, pooled_ts, pooled_st - pooled_ts, pooled_st * pooled_ts], dim=-1)
        # pooled_output = torch.cat([max_pooled_st, max_pooled_ts], dim=-1)
        # pooled_output = torch.cat([max_pooled_st, max_pooled_ts, max_pooled_st - max_pooled_ts, max_pooled_st * max_pooled_ts], dim=-1)
        # pooled_output = torch.cat([mean_pooled_st, mean_pooled_ts, mean_pooled_st - mean_pooled_ts, mean_pooled_st * mean_pooled_ts], dim=-1)
        scores = self.score_linear_proj(self.dropout(pooled_output)).squeeze(-1)
        return scores

    def max_forward_gumbel(self, Q, D, q_mask, d_mask):
        # if d_mask is not None and q_mask is not None:
        #     D = D * d_mask[..., None]
        #     Q = Q * q_mask[..., None]
        # if q_word_weight:
        # print(Q[0].norm(p=2, dim=-1))
        # input()
        sim_mat = einsum("qmh,dnh->qdmn", Q, D)
        if (not Q.requires_grad) or False:
            scores = sim_mat.max(-1)[0]
            # input(Q.requires_grad)
        else:
            # sim_mat = sim_mat * d_mask[None, :, None, :]
            d_mask_ = d_mask.clone()[None, :, None, :]
            sim_mat = sim_mat * d_mask_ + (1 - d_mask_) * -1e4
            sim_mat_ = F.log_softmax(sim_mat / 0.1, dim=-1)
            # input(sim_mat[0][0][0].sort().values)
            # sim_mat = sim_mat / 1
            # scores = torch.einsum("qdmn,qdmn->qdm", F.gumbel_softmax(sim_mat, tau=0.1, hard=True), sim_mat)
            scores = (F.gumbel_softmax(sim_mat_, tau=0.5, hard=True, dim=-1) * sim_mat).sum(-1)
        scores = scores * q_mask[:, None, :]
        scores = scores.sum(-1)
        # scores = scores / (q_mask.bool().sum(-1)[:, None])
        # scores = F.relu(einsum("qmh,dnh->qdmn", Q, D)).max(-1)[0].sum(-1)
        return scores

    def max_forward(self, Q, D, q_mask, d_mask):
        D = D * d_mask[..., None]
        Q = Q * q_mask[..., None]
        scores_match = einsum("qmh,dnh->qdmn", Q, D).max(-1)[0]
        # scores_match = einsum("qmh,dnh->qdmn", Q, D).mean(-1)
        scores = scores_match.sum(-1)
        return scores


# def testnograd():