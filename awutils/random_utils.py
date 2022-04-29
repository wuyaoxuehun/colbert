import random
import numpy as np
import torch


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def train_set_seed(seed, rank=0):
    # rank = 0
    torch.manual_seed(seed + rank)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.benchmark = True
    random.seed(seed + rank)
    np.random.seed(seed + rank)
    torch.cuda.manual_seed_all(seed + rank)
