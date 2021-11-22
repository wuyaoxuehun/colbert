import math
from functools import wraps
import torch
import numpy as np


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
    return max_T - (max_T - min_T) / (total_epoch - 1) * epoch


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


def test_sample_T():
    E = 10
    for i in range(E):
        print(sample_T_scheduler(i, total_epoch=E))


def split_parameters(module):
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
    assert len(list(module.parameters())) == len(params_decay) + len(params_no_decay)
    return params_decay, params_no_decay


if __name__ == '__main__':
    # test_sample_T()
    for i in range(10):
        print(scheduler_neg(i, 10))
