from argparse import ArgumentParser

import torch.cuda

from colbert.utils import distributed


class CBQAArguments:
    def __init__(self, description):
        self.parser = ArgumentParser(description=description)

        self.add_argument('--batch_size', dest='batch_size', default=4, type=int)
        self.add_argument('--amp', dest='amp', action='store_true')
        self.add_argument('--do_train', dest='do_train', action='store_true')
        self.add_argument('--do_eval', dest='do_eval', action='store_true')
        self.add_argument('--do_test', dest='do_test', action='store_true')
        self.add_argument('--epoch', dest='epoch', default=6, type=int)
        self.add_argument('--retriever_lr', dest='retriever_lr', default="1e-5", type=float)
        self.add_argument('--lr', dest='lr', default="2e-5", type=float)
        self.add_argument('--train_files', dest='train_files', type=str)
        self.add_argument('--dev_files', dest='dev_files', type=str)
        self.add_argument('--test_files', dest='test_files', type=str)
        self.add_argument('--output_dir', dest='output_dir', default="output", type=str)
        self.add_argument('--logging_steps', dest='logging_steps', type=int, default=1)
        self.add_argument('--gradient_accumulation_steps', dest='gradient_accumulation_steps', type=int, default=1)
        self.add_argument('--local_rank', dest='rank', default=0, type=int)

        self.add_argument('--checkpoint', default="", type=str)
        self.add_argument('--index_path', default="", type=str)

        # self.add_argument('--gpu_ids', type=str, default='-1', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
    def add_argument(self, *args, **kw_args):
        return self.parser.add_argument(*args, **kw_args)

    def parse(self):
        args = self.parser.parse_args()
        # input(torch.cuda.device_count())
        args.nranks, args.distributed = distributed.init(args.rank)

        return args
