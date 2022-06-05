import logging

import torch
from torch.distributed import get_rank

logger = logging.getLogger("__main__")


def split_parameters(module):
    params_decay = []
    params_no_decay = []
    for name, param in module.named_parameters():
        if 'bias' in name:
            params_no_decay.append(param)
        else:
            params_decay.append(param)

    assert len(list(module.parameters())) == len(params_decay) + len(params_no_decay), (len(list(module.parameters())), len(params_decay) + len(params_no_decay))
    return params_decay, params_no_decay


def distributed_concat(tensor, num_total_examples=None, concat=True):
    output_tensors = [tensor.clone() for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(output_tensors, tensor)
    # input(output_tensors[0].grad_fn)
    res = output_tensors
    if concat:
        res = torch.cat(res, dim=0)
        if num_total_examples:
            res = res[:num_total_examples]
    # truncate the dummy elements added by SequentialDistributedSampler
    return res


def collection_qd_masks(data):
    output = []
    rank = get_rank()
    for t in data:
        all_t = distributed_concat(t, concat=False)
        # all_t[rank], all_t[0] = all_t[0], t
        all_t[rank] = t
        all_t = torch.cat(all_t, dim=0)
        output.append(all_t)

    return output


def keep_nonzero(Q, q_word_mask):
    assert len(Q.size()) == 2
    q_word_mask_bool = q_word_mask.bool()
    real_Q = Q[q_word_mask_bool]
    real_q_word_mask = q_word_mask[q_word_mask_bool]
    return real_Q, real_q_word_mask


# def qd_mask_to_realinput(Q=None, D=None, q_word_mask=None, d_word_mask=None, keep_dim=True):
#     output = []
#     if Q is not None:
#         if len(Q.size()) == 3:
#             Q = Q[:, :Q_TOPK, ...]
#             q_word_mask = q_word_mask[:, :Q_TOPK, ...]
#         else:
#             Q = Q[:Q_TOPK, ...]
#             q_word_mask = q_word_mask[:Q_TOPK, ...]
#
#         # q_word_mask = torch.ones_like(q_word_mask)
#         if not keep_dim:
#             Q, q_word_mask = keep_nonzero(Q, q_word_mask)
#         output.extend([Q, q_word_mask])
#     if D is not None:
#         if len(D.size()) == 3:
#             D = D[:, :D_TOPK, ...]
#             d_word_mask = d_word_mask[:, :D_TOPK, ...]
#         else:
#             D = D[:D_TOPK, ...]
#             d_word_mask = d_word_mask[:D_TOPK, ...]
#         # d_word_mask = torch.ones_like(d_word_mask)
#         if not keep_dim:
#             D, d_word_mask = keep_nonzero(D, d_word_mask)
#         output.extend([D, d_word_mask])
#
#     return output

def qd_mask_to_realinput(t=None, t_mask=None, max_length=None, keep_dim=True):
    if len(t.size()) == 3:
        t = t[:, :max_length, ...]
        t_mask = t_mask[:, :max_length, ...]
    else:
        t = t[:max_length, ...]
        t_mask = t_mask[:max_length, ...]
    if not keep_dim:
        t, t_mask = keep_nonzero(t, t_mask)
    return t, t_mask


if __name__ == '__main__':
    pass
