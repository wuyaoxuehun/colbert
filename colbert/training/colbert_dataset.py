import logging

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
    def __init__(self, args, task):
        super().__init__()
        # self.tokenizer: CostomTokenizer = CostomTokenizer(args)
        self.data = load_data(task)

    def __getitem__(self, index):
        index = index % len(self.data)
        return self.data[index]

    def __len__(self):
        return len(self.data)
