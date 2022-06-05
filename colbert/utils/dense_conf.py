from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

pretrain_base = Path("/home/awu/experiments/pretrain")

pretrain_map = {
    'bert': str(pretrain_base / "chinese-bert-wwm-ext"),
}
OmegaConf.register_new_resolver("pretrainMap", lambda x: pretrain_map[x])

data_dir_dic = {
    "dureader": lambda dstype, idx: f"/home2/awu/testcb/data/dureader/{dstype}.json"
}

context_random = np.random.default_rng(1234)


def load_dense_conf():
    args = OmegaConf.load("proj_conf/dense.yaml")
    args.dense_training_args.pretrain = pretrain_map[args.dense_training_args.pretrain_name]
    return args


if __name__ == '__main__':
    pass
