import torch
from transformers import HfArgumentParser

from proj_conf.training_arguments import MyTraniningArgs

print(torch.cuda.device_count())
from colbert.training.colbert_trainer import *
from colbert.utils.dense_conf import *


def main():
    parser = HfArgumentParser((MyTraniningArgs,))
    trainer_args, = parser.parse_args_into_dataclasses()

    args = OmegaConf.load("proj_conf/dense.yaml")
    args.dense_training_args.pretrain = pretrain_map[args.dense_training_args.pretrain_name]
    if args.dense_training_args.do_train:
        train(args, trainer_args)
    if args.dense_training_args.do_eval:
        eval_retrieval_for_model(args)


if __name__ == "__main__":
    main()
