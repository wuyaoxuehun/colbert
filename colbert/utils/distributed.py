import os
import random
import torch
import numpy as np
from contextlib import contextmanager
import logging
logger = logging.getLogger("__main__")

def init(rank):
    nranks = 'WORLD_SIZE' in os.environ and int(os.environ['WORLD_SIZE'])
    nranks = max(1, nranks)
    is_distributed = nranks > 1
    # num_gpus = torch.cuda.device_count()
    # is_distributed = num_gpus > 1
    if rank == 0:
        print('nranks =', nranks, '\t num_gpus =', torch.cuda.device_count())

    if is_distributed:
        # num_gpus = torch.cuda.device_count()
        torch.cuda.set_device(rank % nranks)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')

    return nranks, is_distributed


def barrier(rank):
    if rank >= 0:
        torch.distributed.barrier()


@contextmanager
def torch_distributed_zero_first(rank: int):
    """Decorator to make all processes in distributed training wait for each local_master to do something.
    """
    logger.info(f"current rank {rank}")
    if rank not in [-1, 0]:
        torch.distributed.barrier()
    # 这里的用法其实就是协程的一种哦。
    if rank == 0:
        yield
        torch.distributed.barrier()

    logger.info(f"current rank passed {rank}")


