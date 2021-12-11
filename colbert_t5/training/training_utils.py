import math
from functools import wraps
import torch
import numpy as np

from conf import Q_TOPK, D_TOPK


class SequentialDistributedSampler(torch.utils.data.sampler.Sampler):
    """
    Distributed Sampler that subsamples indicies sequentially,
    making it easier to collate all results at the end.
    Even though we only use this sampler for eval and predict (no training),
    which means that the model params won't have to be synced (i.e. will not hang
    for synchronization even if varied number of forward passes), we still add extra
    samples to the sampler to make it evenly divisible (like in `DistributedSampler`)
    to make it easy to `gather` or `reduce` resulting tensors at the end of the loop.
    """

    def __init__(self, dataset, batch_size, rank=None, num_replicas=None):
        if num_replicas is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = torch.distributed.get_world_size()
        if rank is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = torch.distributed.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.batch_size = batch_size
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.batch_size / self.num_replicas)) * self.batch_size
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        indices = list(range(len(self.dataset)))
        # add extra samples to make it evenly divisible
        indices += [indices[-1]] * (self.total_size - len(indices))
        # subsample
        indices = indices[self.rank * self.num_samples: (self.rank + 1) * self.num_samples]
        return iter(indices)

    def __len__(self):
        return self.num_samples


#
# def scheduler_neg(epoch, total_epoch):
#     if epoch < int(total_epoch * 1 / 5):
#         return [1, 1, 1]
#     if epoch < int(total_epoch * 3 / 5):
#         return [5, 5, 1]
#     else:
#         return [10, 5, 1]

def scheduler_neg(epoch, total_epoch):
    # epoch += 1
    min_val = [1, 1, 1]
    # return min_val
    max_val = [10, 5, 1]
    return softmax([(ma - mi) / (total_epoch - 1) * epoch + mi for ma, mi in zip(max_val, min_val)], T=5)
    # return [(ma - mi) / (total_epoch - 1) * epoch + mi for ma, mi in zip(max_val, min_val)]


def softmax(x, T=1):
    x = np.array(x) / T
    y = np.exp(x - np.max(x))
    f_x = y / sum(y)
    return f_x


def sample_T_scheduler(epoch, total_epoch):
    min_T, max_T = 1, 20
    # return max_T
    return max_T - (max_T - min_T) / (total_epoch - 1) * epoch


def coef_scheduler(epoch, total_epoch):
    min_T, max_T = 0.1, 0.9
    # # return max_T
    # return max_T - (max_T - min_T) / (total_epoch - 1) * epoch
    start_epoch, end_epoch = total_epoch // 4, total_epoch // 4 * 3
    if epoch <= start_epoch:
        return min_T
    elif epoch > end_epoch:
        return max_T
    else:
        return (max_T - min_T) / (end_epoch - start_epoch) * (epoch - start_epoch + 1)


def test_coef_scheduler():
    for i in range(30):
        print(coef_scheduler(i, 30))


def pre_activate_coroutine(func):
    @wraps(func)
    def primer(*args, **kwargs):
        gen = func(*args, **kwargs)
        next(gen)
        return gen

    return primer


@pre_activate_coroutine
def moving_average():
    total = 0.0
    count = 0
    average = 0
    while True:
        term = yield average
        total += term
        count += 1
        average = total / count


class MAverage:
    def __init__(self):
        self.count = 0
        self.summ = 0

    def add(self, t):
        self.summ += t
        self.count += 1
        return self.get_average()

    def get_average(self):
        if self.count == 0:
            return 0
        return self.summ / self.count


def test_sample_T():
    E = 10
    for i in range(E):
        print(sample_T_scheduler(i, total_epoch=E))


def split_parameters_(module):
    params_decay = []
    params_no_decay = []
    for m in module.modules():
        if isinstance(m, torch.nn.Linear):
            params_decay.append(m.weight)
            if m.bias is not None:
                params_no_decay.append(m.bias)
        elif isinstance(m, torch.nn.modules.conv._ConvNd):
            params_decay.append(m.weight)
            if m.bias is not None:
                params_no_decay.append(m.bias)
        elif isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
            params_no_decay.extend([*m.parameters()])
        elif len(list(m.children())) == 0:
            params_decay.extend([*m.parameters()])
    assert len(list(module.parameters())) == len(params_decay) + len(params_no_decay), (len(list(module.parameters())), len(params_decay) + len(params_no_decay))
    return params_decay, params_no_decay


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


def qd_mask_to_realinput_(Q, D, q_word_mask=None, d_word_mask=None):
    Q = Q[:, :Q_TOPK, ...]
    q_word_mask = torch.ones(Q.size()[:2])
    D = D[:, :D_TOPK, ...]
    d_word_mask = torch.ones(D.size()[:2])
    return Q, q_word_mask, D, d_word_mask


def distributed_concat(tensor, num_total_examples=None, concat=True):
    output_tensors = [tensor.clone() for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(output_tensors, tensor)
    # input(output_tensors[0].grad_fn)
    res = output_tensors
    if concat:
        res = torch.cat(res, dim=0)
    # truncate the dummy elements added by SequentialDistributedSampler
    return res


def collection_qd_masks(Q, q_word_mask, D, d_word_mask, rank):
    data = [Q, q_word_mask, D, d_word_mask]
    output = []
    for t in data:
        all_t = distributed_concat(t, concat=False)
        # all_t[rank], all_t[0] = all_t[0], t
        all_t[rank] = t
        all_t = torch.cat(all_t, dim=0)
        output.append(all_t)

    return output


def pre_batch_enable(epoch, epoch_total):
    return False
    return epoch_total // 4 <= epoch <= epoch_total - 5


def mix_qd(Q, q_word_mask, D, d_word_mask, aug_num=1, p_num=2, alpha=None):
    lam = 0.5
    q1 = Q[:2 * aug_num:2]
    q1_word_mask = q_word_mask[:2 * aug_num:2]
    q2 = Q[1: 2 * aug_num:2]
    q2_word_mask = q_word_mask[1:2 * aug_num:2]
    aug_q = lam * q1 + (1 - lam) * q2
    aug_q_word_mask = q1_word_mask & q2_word_mask
    fq = torch.cat([Q, aug_q], dim=0)
    fq_word_mask = torch.cat([q_word_mask, aug_q_word_mask])

    pos_d = D[:2 * aug_num * p_num:p_num]
    pos_d_word_mask = d_word_mask[:2 * aug_num * p_num:2]
    neg_d = D[1:2 * aug_num * p_num:2]
    neg_d_word_mask = d_word_mask[1:2 * aug_num * p_num:2]

    aug_pos_d = lam * pos_d[:2 * aug_num:2] + (1 - lam) * pos_d[1:2 * aug_num:2]
    aug_pos_d_word_mask = pos_d_word_mask[:2 * aug_num:2] & pos_d_word_mask[1:2 * aug_num:2]

    aug_neg_d = lam * neg_d[:2 * aug_num:2] + (1 - lam) * neg_d[1:2 * aug_num:2]
    aug_neg_d_word_mask = neg_d_word_mask[:2 * aug_num:2] & neg_d_word_mask[1:2 * aug_num:2]

    fd = torch.cat([D, aug_pos_d, aug_neg_d])
    fd_word_mask = torch.cat([d_word_mask, aug_pos_d_word_mask, aug_neg_d_word_mask])

    qn, dn = Q.size(0), D.size(0)
    positive_idxes = torch.tensor([_ * p_num for _ in range(qn)])
    aug_pos_idxes = torch.tensor([qn * p_num + _ for _ in range(aug_num)])
    fpositive_idxes = torch.cat([positive_idxes, aug_pos_idxes])

    fqn, fdn = qn + aug_num, dn + 2 * aug_num
    aug_mask = torch.zeros((fqn, fdn), dtype=torch.bool).cuda()
    for i in range(aug_num):
        aug_mask[i * 2, dn + i] = True
        aug_mask[i * 2 + 1, dn + i] = True
    # print(aug_mask.size())
    for i in range(aug_num):
        # print(qn + i, p_num * 2 + i)
        aug_mask[qn + i, p_num * 2 * i] = True
        aug_mask[qn + i, p_num * 2 * i + p_num] = True
    # print(Q)
    # print(fq)
    #
    # print(D)
    # print(fd)
    # print(fpositive_idxes)
    # print(aug_mask.to(dtype=torch.int32))
    return fq, fq_word_mask, fd, fd_word_mask, aug_mask, fpositive_idxes


def test_mix_qd():
    qn, dn, dim = 4, 8, 2
    Q = torch.tensor([[[i] * dim] for i in range(qn)])
    D = torch.tensor([[[i] * dim] for i in range(dn)])
    q_word_mask = torch.tensor([[1] for _ in range(qn)])
    d_word_mask = torch.tensor([[1] for _ in range(dn)])
    mix_qd(Q, q_word_mask, D, d_word_mask, aug_num=2)


def keep_nonzero(Q, q_word_mask):
    assert len(Q.size()) == 2
    q_word_mask_bool = q_word_mask.bool()
    real_Q = Q[q_word_mask_bool]
    real_q_word_mask = q_word_mask[q_word_mask_bool]
    return real_Q, real_q_word_mask


def qd_mask_to_realinput(Q=None, D=None, q_word_mask=None, d_word_mask=None, keep_dim=True):
    output = []
    if Q is not None:
        if len(Q.size()) == 3:
            Q = Q[:, :Q_TOPK, ...]
            q_word_mask = q_word_mask[:, :Q_TOPK, ...]
        else:
            Q = Q[:Q_TOPK, ...]
            q_word_mask = q_word_mask[:Q_TOPK, ...]

        # q_word_mask = torch.ones_like(q_word_mask)
        if not keep_dim:
            Q, q_word_mask = keep_nonzero(Q, q_word_mask)
        output.extend([Q, q_word_mask])
    if D is not None:
        D = D[:, :D_TOPK, ...]
        d_word_mask = d_word_mask[:, :D_TOPK, ...]
        # d_word_mask = torch.ones_like(d_word_mask)
        if not keep_dim:
            D, d_word_mask = keep_nonzero(D, d_word_mask)
        output.extend([D, d_word_mask])

    return output


def qd_answer_mask_to_realinput(Q, D, answer_cls_idx):
    # Q = Q[:, :Q_TOPK, ...]
    # print(Q.size())
    # input(answer_cls_idx)
    Q = Q.gather(dim=1, index=answer_cls_idx[:, None, None].repeat((1, 1, 768)))
    q_word_mask = torch.ones(Q.size()[:2])
    D = D[:, :1, ...]
    d_word_mask = torch.ones(D.size()[:2])
    return Q, q_word_mask, D, d_word_mask


if __name__ == '__main__':
    # test_sample_T()
    # for i in range(10):
    #     print(scheduler_neg(i, 10))

    # test_coef_scheduler()
    test_mix_qd()
