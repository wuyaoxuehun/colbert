import torch


def collect_p_bqo(seq_output: torch.tensor, segment_lens):
    bs, seqlen, h = seq_output.size()
    p = torch.zeros((bs, seqlen, h), device=seq_output.device)
    p_mask = torch.zeros((bs, seqlen), device=seq_output.device)
    bqo = torch.zeros((bs, seqlen, h), device=seq_output.device)
    bqo_mask = torch.zeros((bs, seqlen), device=seq_output.device)
    # print(seq_output.size(), segment_lens.size())
    # input()
    p_len_max = 0
    bqo_len_max = 0
    for idx, (output, segment_len) in enumerate(zip(seq_output, segment_lens)):
        p[idx, :segment_len[0], ...] = seq_output[idx, 1:
                                                       1 + segment_len[0], ...]
        # part_len =
        p_mask[idx, :segment_len[0]] = 1
        p_len_max = max(p_len_max, segment_len[0])
        bqo[idx, :segment_len[1] + segment_len[2] + segment_len[3] + 2, ...] = \
            seq_output[idx, 1 + segment_len[0] + 1:
                            1 + segment_len[0] + 1 + segment_len[1] + segment_len[2] + segment_len[3] + 2]
        bqo_mask[idx, :segment_len[1] + segment_len[2] + segment_len[3] + 2] = 1
        bqo_len_max = max(bqo_len_max, segment_len[1] + segment_len[2] + segment_len[3] + 2)

    p, p_mask, bqo, bqo_mask = p[:, :p_len_max, ...], p_mask[:, :p_len_max, ...], bqo[:, :bqo_len_max, ...], bqo_mask[:, :bqo_len_max, ...]
    return p, p_mask, bqo, bqo_mask


def collect_pbqo(seq_output: torch.tensor, segment_lens):
    bs, seqlen, h = seq_output.size()
    t, t_mask = [torch.zeros((bs, seqlen, h), device=seq_output.device) for _ in range(4)], [torch.zeros((bs, seqlen), device=seq_output.device) for _ in range(4)]
    parts_total = 4
    max_lens = [0] * 4
    for idx, (output, segment_len) in enumerate(zip(seq_output, segment_lens)):
        segment_len_presum = torch.cat([torch.tensor([0], device=seq_output.device), (segment_len + 1).cumsum(-1)])
        # print(segment_len_presum)
        # print(segment_len)
        for i in range(segment_len_presum.size(0) - 1):
            # print(segment_len_presum[i] + 1, segment_len_presum[i + 1], segment_len[i])
            # print(t[i][idx][0][:segment_len[i]].size(), output[segment_len_presum[i] + 1: segment_len_presum[i + 1], ...].size())
            t[i][idx, :segment_len[i], ...] = output[segment_len_presum[i] + 1:segment_len_presum[i + 1], ...]
            t_mask[i][idx, :segment_len[i], ...] = 1
            max_lens[i] = max(max_lens[i], segment_len[i])

    t, t_mask = [t[i][:, :max_lens[i], ...] for i in range(parts_total)], [t_mask[i][:, :max_lens[i], ...] for i in range(parts_total)]
    return t, t_mask


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


def span_mean_2(Q=None, spans=None):
    torch.set_printoptions(precision=2, threshold=100)
    Q = torch.randn(5, 2, requires_grad=True)
    # spans = torch.tensor([[0, 3], [4, 7], [7, 8], [8, 10]])
    spans = torch.tensor([[0, 3], [4, 5]])
    # padded = torch.nn.functional.pad(
    #     Q.cumsum(dim=0), (0, 0, 1, 0)
    # )
    #
    # pools = torch.diff(
    #     padded[spans], dim=1
    # ).squeeze() / torch.diff(spans, dim=1)
    span1 = Q[0:3, ...].mean(0, keepdim=True)
    span2 = Q[4:5, ...].mean(0, keepdim=True)
    torch.cat([span1, span2], dim=0).sum().sum().backward()
    # pools.sum().sum().backward()
    print(Q.grad)


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


def test_collect():
    output = torch.randn(2, 16, 2)
    segment_len = torch.tensor([[1, 2, 3, 5], [2, 1, 3, 3]])
    t, t_mask = collect_pbqo(output, segment_len)
    idx = 0
    print(output[idx])
    print([t[_][idx] for _ in range(4)])
    print([t_mask[_][idx] for _ in range(4)])
    # print(t_mask)


if __name__ == '__main__':
    # test_collect()
    span_mean_2()
