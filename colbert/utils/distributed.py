import os
import torch
import logging

logger = logging.getLogger("__main__")


def init(rank):
    nranks = 'WORLD_SIZE' in os.environ and int(os.environ['WORLD_SIZE'])
    nranks = max(1, nranks)
    is_distributed = nranks > 1

    if is_distributed:
        torch.cuda.set_device(rank % nranks)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
    return nranks, is_distributed


def init_dist():
    rank = int(os.environ['LOCAL_RANK'])
    nranks = int(os.environ['WORLD_SIZE'])
    is_distributed = nranks > 1
    if is_distributed:
        torch.cuda.set_device(rank % nranks)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
    return rank, nranks, is_distributed


def barrier(rank):
    if rank >= 0:
        torch.distributed.barrier()
