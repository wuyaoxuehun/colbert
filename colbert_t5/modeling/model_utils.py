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
