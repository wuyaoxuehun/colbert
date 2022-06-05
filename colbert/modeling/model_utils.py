import torch


# Batched index_select
def batch_index_select(t, dim, inds):
    dummy = inds.unsqueeze(2).expand(inds.size(0), inds.size(1), t.size(2))
    out = t.gather(dim, dummy)  # b x e x f
    return out


def span_mean(Q, spans, *args, **kwargs):
    presum_q = Q.cumsum(1)
    presum_q = torch.cat([torch.zeros([Q.size(0), 1, Q.size(2)], device=Q.device), presum_q], dim=1)
    start_idx, end_idx = spans[:, :, 0], spans[:, :, 1]
    # print(presum_q.size(), start_idx.size(), end_idx.size())
    start_q, end_q = [batch_index_select(presum_q, 1, _) for _ in [start_idx, end_idx]]
    res_q = end_q - start_q
    span_len = end_idx - start_idx
    # span_len[span_len == 0] = 1
    res_q = res_q / span_len[:, :, None]
    return res_q, span_len


def to_real_input_all(t):
    # real_inputs = [to_real_input(_) for _ in t]
    real_inputs = t
    t = list(zip(*real_inputs))
    # t[1] = np.array(t[1])
    return [torch.tensor(_) for _ in t]


def max_pool_by_mask(s, s_mask, value=-1e4, dim=-2):
    # s_mask = s_mask.unsqueeze(dim)
    reverse_mask = 1.0 - s_mask
    values_to_add = value * reverse_mask
    res = (s * s_mask[..., None]) + values_to_add[..., None]
    # s[s_mask == 0] = value
    # res = s
    return res.max(dim).values


def avg_pool_by_mask(s, s_mask, dim=-2):
    return (s * s_mask[..., None]).sum(dim) / (s_mask.sum(-1)[..., None])


if __name__ == '__main__':
    # test_collect()
    pass
