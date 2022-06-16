from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

pretrain_base = Path("/home/awu/experiments/pretrain")

pretrain_map = {
    'bert': str(pretrain_base / "chinese-bert-wwm-ext"),
    'macbert': str(pretrain_base / "macbert_large"),
}
OmegaConf.register_new_resolver("pretrainMap", lambda x: pretrain_map[x])

data_base_dir = "/home/awu/experiments/geo/others/testcb/data/dureader_dataset/"
data_dir_dic = {
    "dureader": lambda dstype, idx: f"{data_base_dir}/{dstype}.json",
    "dureader1": lambda dstype, idx: f"{data_base_dir}/{dstype}_ce.json",
    "dureader_iter": lambda dstype, idx: f"{data_base_dir}/{dstype}_iter.json",
    "dureaderCE": lambda dstype, idx: f"{data_base_dir}/{dstype}_ce.json",
    "dureaderCETest": lambda dstype, idx: f"{data_base_dir}/{dstype}_ce_rerank.json",
}

context_random = np.random.default_rng(1234)


def load_dense_conf():
    args = OmegaConf.load("proj_conf/dense.yaml")
    args.dense_training_args.pretrain = pretrain_map[args.dense_training_args.pretrain_name]
    return args


if __name__ == '__main__':
    pass
