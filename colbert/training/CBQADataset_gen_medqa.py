import json
from collections import namedtuple
from functools import lru_cache
from typing import List, Any, Dict

# import jieba
import torch
from torch.utils.data import Dataset, RandomSampler, SequentialSampler, DataLoader
from tqdm import tqdm
import logging
import numpy as np

import conf
from colbert.modeling.tokenization.doc_tokenization import DocTokenizer
from colbert.modeling.tokenization.query_tokenization import QueryTokenizer
from colbert.modeling.tokenization.utils import CostomTokenizer
# from conf import pretrain_map, Enable_tqdm, data_dir_dic, ir_topk
from conf import *
from corpus.sort_dataset import max_span_rouge_str, max_span_rouge12_str
from colbert.training.training_utils import softmax
# from colbert.modeling.inference import to_real_input_all
from colbert.modeling.model_utils import to_real_input_all

logger = logging.getLogger("__main__")


def load_data(file, task=0):
    import json
    data = []
    with open(file, encoding='utf8') as f:
        all_data = json.load(f)
        logger.info(len(all_data))
        for instance in all_data:
            if 'answer' in instance and instance['answer'] not in list('ABCD'):
                print(instance['answer'])
                continue
            data.append(instance)
    logger.info(len(data))
    if file.find('dev') != -1:
        data = data[:]
    return data


def load_webq(file):
    import json
    with open(file, encoding='utf8') as f:
        data = json.load(f)
        if file.find('test') != -1:
            return data
        output = []
        for t in data:
            if len(t['pos_contexts']) == 0:
                continue
            output.append(t)

        return output


def get_real_files(file):
    files = file.split(',')
    print(files)
    real_files = []
    for file in files:
        ir_type, mode, fold = file.split("-")
        real_files.append(data_dir_dic[ir_type](mode, fold))

    return real_files


def load_data_(files):
    real_files = get_real_files(files)
    data = []
    for path in real_files:
        print(path)
        with open(path, 'r', encoding='utf8') as f:
            data += json.load(f)

    return data


def collate_fun():
    def fun(batch):
        return batch

    return fun


def get_score_rouge12(inst, para, part):
    # refs = inst[part + '_cut']['tok']
    # hyps = para['paragraph_cut']['tok']
    hyps = ' '.join(list(para['paragraph']))
    refs = ' '.join(list(inst[part]))
    return max_span_rouge12_str(hyps=hyps, refs=refs, rouge2_weight=rouge2_weight)


def get_final_score_for_inst(inst, para, opt):
    para['background_score'] = 0
    if partweights[0] > 0:
        para['background_score'] = get_score_rouge12(inst, para, 'background') if inst['background'] else 1
    para['question_score'] = get_score_rouge12(inst, para, 'question') if inst['question'] else 1
    para['opt_score'] = get_score_rouge12(inst, para, opt.upper())
    para['score'] = sum([score * weight for score, weight in
                         zip([para['background_score'], para['question_score'], para['opt_score']], partweights)])
    return para['score']


class CBQADataset(Dataset):
    doc_pre_tensorize = None
    if use_word and False:
        print('loading pre tensorize')
        doc_pre_tensorize = torch.load(doc_pre_tensorize_file)
        # doc_pre_tensorize = to_real_input_all(tensorized_docs)
        # query_pre_tensorize = torch.load(train_dev_query_pre_tensorize_file)
        print('loaded pre tensorize')

    def __init__(self, file_abbrs, tokenizer=None, doc_maxlen=None, query_maxlen=None, reader_max_seq_length=256, eager=False, mode='test'):
        super().__init__()
        self.tokenizer: CostomTokenizer = tokenizer
        if self.tokenizer is None:
            self.tokenizer = CostomTokenizer.from_pretrained(pretrain_map[pretrain_choose])

        self.query_tokenizer = QueryTokenizer(query_maxlen)
        self.doc_tokenizer = DocTokenizer(doc_maxlen)
        self.file_abbrs = file_abbrs
        self.reader_max_seq_length = reader_max_seq_length
        self.query_maxlen = query_maxlen
        self.doc_maxlen = doc_maxlen
        self.eager = eager
        self.eager_data = None
        self.all_len = []
        # self.prepare_eager_data()
        self.mode = mode
        self.cache_data = {}
        self.sample_T = None
        self.data = self.load_data()
        # random.shuffle(self.data)
        # if mode == 'train':
        #     self.data = self.data[:len(self.data) // 64]
        # else:
        #     self.data = self.data[:len(self.data) // 8]

        # self.all_paras = load_all_paras()
        # self.data = [self.data[0]] * 50
        # self.pre_tensorize =
        # self.load_pretensorize()

    def load_data(self):
        data = []

        for file_abbr in self.file_abbrs.split(','):
            temp = file_abbr.split('-')
            if len(temp) == 2:
                ds, ds_type = temp
                file = data_dir_dic.get(ds)(ds_type)
            else:
                ds, ds_type, fold = temp
                file = data_dir_dic.get(ds)(ds_type, fold)
            # if ds == 'c3':
            #     data += load_c3()
            # else:
            print(f"******* loading data from {file} ***********")

            data += load_data(file, 0)
            # data += load_webq(file)
        # import random
        # random.seed(0)
        # random.shuffle(data)
        if self.mode == 'train':
            data = data[:]
        else:
            data = data[:]

        return data[:]

    @staticmethod
    def merge_to_reader_input(batch_examples, batch_paras, extra=None):
        para_idx = 0
        # print(len(batch_paras), len(batch_examples))
        batch_paras, batch_scores = batch_paras
        assert len(batch_paras) == len(batch_examples)
        for example in batch_examples:
            example[f'res'] = [{"p_id": _['p_id'],
                                "paragraph": ''.join(_['paragraph_cut']['tok'].split()),
                                "paragraph_cut": _['paragraph_cut'],
                                "colbert_score": score}
                               for _, score in zip(batch_paras[para_idx], batch_scores[para_idx])]
            para_idx += 1
        assert para_idx == len(batch_paras)

    def tokenize_for_retriever(self, batch):
        obj = self.query_tokenizer.tensorize_noopt_dict(batch_text=batch)
        return obj

    def tokenize_for_train_retriever(self, batch: List[Dict], eval_p_num=None, is_evaluating=False, only_true_answer=False):
        if eval_p_num is None:
            eval_p_num = p_num
        scores, docs = [], []
        for t in batch:
            # cur_docs = t['positive_ctxs'][:1] + t['hard_negative_ctxs'][:1]
            if not is_evaluating:
                # if len(t['positive_ctxs']) == 1:
                #     t['positive_ctxs'].append(t['positive_ctxs'][-1])
                cur_pos_docs = list(context_random.choice(t['positive_ctxs'][:], 1))
                cur_neg_docs = list(context_random.choice(t['hard_negative_ctxs'][:], 1))
            else:
                # cur_pos_docs = list(context_random.choice(t['positive_ctxs'][:10], 1))
                cur_pos_docs = t['positive_ctxs'][:2]
                if len(cur_pos_docs) < 2:
                    cur_pos_docs.append(cur_pos_docs[-1])
                assert len(t['hard_negative_ctxs']) >= 1
                cur_neg_docs = list(t['hard_negative_ctxs'][10:18])
            cur_docs = cur_pos_docs + cur_neg_docs
            docs += cur_docs
        # if padded_negs:
        #     for para in padded_negs:
        #         para['paragraph'] = ''.join(para['paragraph_cut']['tok'].split())
        #     docs += padded_negs
        # if padding_neg_num:
        #     sampled_docs = context_random.choice(list(self.all_paras.values()), padding_neg_num)
        #     docs += list(sampled_docs)
        D = self.doc_tokenizer.tensorize_dict(docs)
        # D = None
        # print(len(set([doc["p_id"] for doc in docs])), len(docs))
        # input()
        # input(scores)
        # D = [torch.tensor(_) for _ in list(zip(*[self.doc_pre_tensorize[int(doc["p_id"])] for doc in docs]))]
        scores = torch.tensor(scores)
        # print(len(batch[0]['paragraph_a']))
        # D_scores = torch.tensor([j for _ in batch for j in self.get_score_for_inst(_, padded_negs, eval_p_num=eval_p_num)])
        return D, scores

    def __getitem__(self, index):
        index = index % len(self.data)
        return self.data[index]

    def __len__(self):
        return len(self.data)
