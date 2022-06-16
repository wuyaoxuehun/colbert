import copy
import logging
import random

import numpy as np
from torch.utils.data import Dataset

from awutils.file_utils import load_json
from colbert.utils.dense_conf import data_dir_dic

logger = logging.getLogger("__main__")


def collate_fun(batch):
    return {"batch": batch}


def load_data(task):
    data = []
    for file_abbr in task.split(','):
        temp = file_abbr.split('-')
        ds, ds_type, fold = temp
        file = data_dir_dic.get(ds)(ds_type, fold)
        print(f"******* loading data from {file} ***********")
        data += load_json(file)
        # import random
        # random.seed(0)
        # random.shuffle(data)
        if ds_type == 'train':
            data = data[:]
        else:
            data = data[:]

    return data[:]


def merge_to_reader_input(batch_examples, batch_paras):
    para_idx = 0
    batch_paras, batch_scores, batch_pids = batch_paras
    assert len(batch_paras) == len(batch_examples)
    for example in batch_examples:
        example[f'res'] = [{"p_id": pid,
                            "paragraph": '',
                            "paragraph_cut": _,
                            "colbert_score": score}
                           for _, score, pid in zip(batch_paras[para_idx], batch_scores[para_idx], batch_pids[para_idx])]
        para_idx += 1
    assert para_idx == len(batch_paras)


class ColbertDataset(Dataset):
    def __init__(self, task, *args, **kwargs):
        super().__init__()
        # self.tokenizer: CostomTokenizer = CostomTokenizer(args)
        self.data = load_data(task)
        output_data = []
        # t_random = np.random.default_rng(1234)
        # random.seed(42)
        # if task.find('train') != -1 or task.find('dev') != -1:
        #     for t in self.data:
        #         assert all([_ not in t['positive_ctxs'] for _ in t["hard_negative_ctxs"]])
        #         for pos in t['positive_ctxs']:
        #             t_ = copy.deepcopy(t)
        #             t_['positive_ctxs'] = [pos]
        #             # t_['hard_negative_ctxs'] = list(t_random.choice(t["hard_negative_ctxs"][:], 20, replace=False))
        #             random.shuffle(t_['hard_negative_ctxs'])
        #             output_data.append(t_)
        #
        #     self.data = output_data

    def __getitem__(self, index):
        index = index % len(self.data)
        return self.data[index]

    def __len__(self):
        return len(self.data)
