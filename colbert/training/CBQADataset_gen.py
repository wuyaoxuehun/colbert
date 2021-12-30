import json
from collections import namedtuple
from typing import List, Any, Dict

# import jieba
import torch
from torch.utils.data import Dataset, RandomSampler, SequentialSampler, DataLoader
from tqdm import tqdm
import logging
import numpy as np

from colbert.modeling.tokenization.doc_tokenization import DocTokenizer
from colbert.modeling.tokenization.query_tokenization import QueryTokenizer
from colbert.modeling.tokenization.utils import CostomTokenizer
# from conf import pretrain_map, Enable_tqdm, data_dir_dic, ir_topk
from conf import *
from corpus.sort_dataset import max_span_rouge_str, max_span_rouge12_str
from colbert.training.training_utils import softmax
from colbert.modeling.inference import to_real_input_all

logger = logging.getLogger(name='__main__')


def load_data(file, task=0):
    import json
    data = []
    with open(file, encoding='utf8') as f:
        all_data = json.load(f)
        logger.info(len(all_data))
        for instance in all_data:
            if instance['answer'] not in list('ABCD'):
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


class CBQADataset(Dataset):
    # print('loading pre tensorize')
    # pre_tensorize = torch.load(train_dev_pre_tensorize_file)
    # print('loaded pre tensorize')

    def __init__(self, file_abbrs, tokenizer=None, doc_maxlen=None, query_maxlen=None, reader_max_seq_length=256, eager=False, mode='dev'):
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
        self.prepare_eager_data()
        self.mode = mode
        self.cache_data = {}
        self.sample_T = None
        self.data = self.load_data()[:]
        # self.pre_tensorize =
        # self.load_pretensorize()

    def load_data(self):
        data = []
        logger.info(f"loading data from {self.file_abbrs}")

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
            # data += load_data(file, 0)
            data += load_webq(file)
        # import random
        # random.seed(0)
        # random.shuffle(data)
        if self.mode == 'train':
            data = data[:]
        else:
            data = data[:]

        return data[:]

    def merge_to_reader_input_(self, batch_examples, batch_paras):
        para_idx = 0
        assert len(batch_paras) == len(batch_examples)
        for example in batch_examples:
            example['paragraph_abcd'] = [{"p_id": _['p_id'],
                                          "paragraph": ''.join(_['paragraph_cut']['tok'].split()),
                                          "paragraph_cut": _['paragraph_cut']} for _ in batch_paras[para_idx]]
            para_idx += 1

    def merge_to_reader_input(self, batch_examples, batch_paras):
        para_idx = 0
        assert len(batch_paras) == len(batch_examples)
        for example in batch_examples:
            example['contexts'] = batch_paras[para_idx]
            para_idx += 1

    def prepare_eager_data(self):
        if self.eager:
            logger.info("preparing eager data!")
            from utils import cache_and_load
            def gen_data(d):
                all_data = []
                for t in tqdm(d, disable=not Enable_tqdm):
                    all_data.append(self.transform_inst(t))
                logger.info(f'''\n
                    mean: {np.mean(self.all_len)}
                    std: {np.std(self.all_len)}
                    99.9%: {np.percentile(self.all_len, 99.9)}
                ''')
                return all_data

            self.eager_data = cache_and_load(self.file_abbrs, method=gen_data, overwrite=False,
                                             json=False, cache_dir=None, d=self.data)
            logger.info("eager data preprared!")

    def get_score_rouge12_(self, ending, para):
        refs = ' '.join(list(ending))
        hyps = ' '.join(list(para['paragraph']))
        return max_span_rouge_str(hyps=hyps, refs=refs)

    def get_score_rouge12(self, inst, para, part):
        # refs = inst[part + '_cut']['tok']
        # hyps = para['paragraph_cut']['tok']
        hyps = ' '.join(list(para['paragraph']))
        refs = ' '.join(list(inst[part]))
        return max_span_rouge12_str(hyps=hyps, refs=refs, rouge2_weight=rouge2_weight)

    def get_final_score_for_inst(self, inst, para, opt):
        para['background_score'] = 0
        if partweights[0] > 0:
            para['background_score'] = self.get_score_rouge12(inst, para, 'background') * partweights[0] if inst['background'] else 1
        para['question_score'] = self.get_score_rouge12(inst, para, 'question') * partweights[1] if inst['question'] else 1
        para['opt_score'] = self.get_score_rouge12(inst, para, opt.upper()) * partweights[2]
        para['score'] = para['background_score'] + para['question_score'] + para['opt_score']
        return para['score']

    def get_score_for_inst_every_ending(self, inst: Dict, eval_p_num=None, is_evaluating=False):
        contexts = [inst[f'paragraph_{_}'] for _ in 'abcd']
        endings = [inst[_] for _ in 'ABCD']
        for opt, paras, ending in (zip(list('abcd'), contexts, endings)):
            if opt.upper() != inst['answer'].upper():
                continue
            pos = []
            neg = []
            for para in paras[:ir_topk]:
                self.get_final_score_for_inst(inst, para, opt)
                # input('hhh')
                if para['score'] >= pos_thre:
                    pos.append(para)
                elif para['score'] < pos_thre:
                    neg.append(para)
            if is_evaluating:
                paras.sort(key=lambda x: x['score'], reverse=True)
                inst[f'paragraph_{opt}'] = paras[:eval_p_num]
                continue

            pos.sort(key=lambda x: x['score'], reverse=True)
            neg.sort(key=lambda x: x['score'], reverse=True)

            while len(pos) < pos_num:
                pos.append(neg.pop(0))
            while len(neg) < neg_num:
                neg.insert(0, pos.pop())
            # paras.sort(key=lambda x:x['score'], reverse=True)
            # np.random.seed(0)
            if self.sample_T:
                # self.sample_T = 1
                pos_dist = softmax([_['score'] / self.sample_T for _ in pos[:max_pos]])
                neg_dist = softmax([_['score'] / self.sample_T for _ in neg[:max_neg]])
                # print(pos_dist, neg_dist)
                # print(sum(pos_dist), sum(neg_dist))
                # input()
                pos = np.random.choice(pos[:max_pos], replace=False, size=pos_num, p=pos_dist).tolist()
                neg = np.random.choice(neg[:max_neg], replace=False, size=neg_num, p=neg_dist).tolist()
            else:
                pos = np.random.choice(pos[:max_pos], replace=False, size=pos_num).tolist()
                neg = np.random.choice(neg[:max_neg], replace=False, size=neg_num).tolist()
            inst[f'paragraph_{opt}'] = pos + neg
            # assert len(inst[f'paragraph_{opt}']) == eval_p_num == 2

    def get_score_for_inst_(self, inst: Dict, padded_negs=None, eval_p_num=None):
        contexts = [inst[f'paragraph_{_}'] for _ in 'abcd']
        # endings = [inst[_] for _ in 'ABCD']
        scores = []
        for i in range(4):
            t = []
            for paras in contexts:
                for para in paras[:eval_p_num]:
                    t.append(self.get_final_score_for_inst(inst, para, "ABCD"[i]))
            if padded_negs:
                for para in padded_negs:
                    # t.append(self.get_final_score_for_inst(inst, para, "ABCD"[i]))
                    t.append(0)
            scores.append(t)
            # if len(t) != p_num * 4 + len(padded_negs):
            #     print(len(t))
            #     input()
        return scores

    def get_score_for_inst(self, inst: Dict, padded_negs=None, eval_p_num=None):
        # contexts = [inst[f'paragraph_{_}'] for _ in 'abcd']
        # endings = [inst[_] for _ in 'ABCD']
        scores = []
        t = []
        for para in inst['paragraph_' + inst['answer'].lower()][:eval_p_num]:
            t.append(self.get_final_score_for_inst(inst, para, inst['answer']))
        if padded_negs:
            for para in padded_negs:
                # t.append(self.get_final_score_for_inst(inst, para, "ABCD"[i]))
                t.append(0)
        scores.append(t)
        # if len(t) != p_num * 4 + len(padded_negs):
        #     print(len(t))
        #     input()
        return scores

    def tokenize_for_retriever(self, batch):
        obj = self.query_tokenizer.tensorize_noopt_dict(batch_text=batch)
        return obj

    def tokenize_for_train_retriever_(self, batch: List[Dict], padded_negs=None, eval_p_num=None, is_evaluating=False):
        docs = []
        # if eval_p_num is None:
        #     eval_p_num = p_num

        for line_idx, example in enumerate(batch):
            # docs += example['pos_contexts'][:pos_num]
            t_pos = example['pos_contexts'][:max_pos]
            t_neg = example['pos_contexts'][:max_neg]
            if is_evaluating:
                docs += t_pos[:pos_num]
            else:
                context_random.shuffle(t_pos)
                docs += t_pos[:pos_num]
            if len(example['neg_contexts']) == 0:
                example['neg_contexts'] = np.random.choice(batch[line_idx - 1]['pos_contexts'][:neg_num], replace=True, size=neg_num).tolist()
            if is_evaluating:
                docs += t_neg[:eval_neg_num]
            else:
                context_random.shuffle(t_neg)
                docs += t_neg[:neg_num]

        # if padded_negs and False:
        #     docs += padded_negs

        D_ids, D_mask, D_word_mask = self.doc_tokenizer.tensorize_dict(docs)

        # D_scores = torch.tensor([j for _ in batch for j in self.get_score_for_inst(_, padded_negs, eval_p_num=eval_p_num)])
        # return D_ids, D_mask, D_word_mask, None
        return D_ids, D_mask, D_word_mask

    def tokenize_for_train_retriever(self, batch: List[Dict], padded_negs=None, eval_p_num=None, is_evaluating=False):
        docs = []
        # if eval_p_num is None:
        #     eval_p_num = p_num

        for line_idx, example in enumerate(batch):
            if is_evaluating:
                docs += example['pos_contexts'][:pos_num]
            else:
                docs += np.random.choice(example['pos_contexts'][:max_pos], replace=False, size=pos_num).tolist()
            if len(example['neg_contexts']) == 0:
                example['neg_contexts'] = np.random.choice(batch[line_idx - 1]['pos_contexts'][:neg_num], replace=True, size=neg_num).tolist()
            if is_evaluating:
                docs += example['neg_contexts'][:neg_num]
            else:
                docs += np.random.choice(example['neg_contexts'][:max_neg], replace=False, size=neg_num).tolist()

        # if padded_negs and False:
        #     docs += padded_negs

        # tensorized_docs = []
        # for doc in docs:
        #     tensorized_docs.append(self.pre_tensorize[doc['idx']])
        # output = to_real_input_all(tensorized_docs)

        output = self.doc_tokenizer.tensorize_dict(docs)

        # D_scores = torch.tensor([j for _ in batch for j in self.get_score_for_inst(_, padded_negs, eval_p_num=eval_p_num)])
        # return D_ids, D_mask, D_word_mask, None
        return output

    def tokenize_for_reader(self, batch: List[Dict]):
        all_choice_features = []
        for inst in batch:
            choices_features = []
            # sentence_index = [0] * 100
            all_contexts = [inst['paragraph_' + opt] for opt in 'abcd']
            endings = [inst[opt] for opt in 'ABCD']
            token_nums = []
            for ending_index, (contexts, ending) in enumerate(zip(all_contexts, endings)):
                input_ids_opt = []
                input_mask_opt = []
                segment_ids_opt = []
                segment_lens_opt = []
                for context in contexts[:p_num]:
                    context_tokens = self.tokenizer.tokenize(context['paragraph'])
                    background_tokens = self.tokenizer.tokenize(inst['background'])
                    start_ending_tokens = self.tokenizer.tokenize(inst['question'])
                    ending_tokens = self.tokenizer.tokenize(ending)
                    token_nums.append(len(context_tokens) + len(background_tokens) + len(start_ending_tokens) + len(ending_tokens))

                    # ending_tokens = start_ending_tokens + ending_tokens
                    self.tokenizer.truncate_seq([context_tokens, background_tokens, start_ending_tokens, ending_tokens],
                                                self.reader_max_seq_length - 5, [3])
                    doc_len = len(context_tokens)
                    background_len = len(background_tokens)
                    question_len = len(start_ending_tokens)
                    option_len = len(ending_tokens)
                    input_tokens = ['[CLS]'] + context_tokens + ['[SEP]'] + background_tokens + ['[SEP]'] \
                                   + start_ending_tokens + ['[SEP]'] + ending_tokens + ['[SEP]']
                    padding_len = self.reader_max_seq_length - len(input_tokens)

                    input_ids = self.tokenizer.convert_tokens_to_ids(input_tokens) + [0] * padding_len
                    input_mask = [1] * len(input_tokens) + [0] * padding_len
                    segment_ids = [0] * (len(context_tokens) + 2) + [1] * (len(background_tokens) + 1) + [0] * (len(start_ending_tokens) + 1) + \
                                  [1] * (len(ending_tokens) + 1) + [0] * padding_len

                    assert len(input_ids) == self.reader_max_seq_length
                    assert len(input_mask) == self.reader_max_seq_length
                    assert len(segment_ids) == self.reader_max_seq_length

                    input_ids_opt.append(input_ids)
                    input_mask_opt.append(input_mask)
                    segment_ids_opt.append(segment_ids)
                    segment_lens_opt.append([doc_len, background_len, question_len, option_len])

                choices_features.append([input_ids_opt, input_mask_opt,
                                         segment_ids_opt, segment_lens_opt])
            all_choice_features.append(choices_features)
        all_input_ids = [[_[0] for _ in i] for i in all_choice_features]
        all_input_mask = [[_[1] for _ in i] for i in all_choice_features]
        all_segment_ids = [[_[2] for _ in i] for i in all_choice_features]
        all_segment_lens = [[_[3] for _ in i] for i in all_choice_features]
        all_labels = [ord(inst['answer']) - ord('A') for inst in batch]
        return torch.tensor(all_input_ids), torch.tensor(all_input_mask), \
               torch.tensor(all_segment_ids), torch.tensor(all_segment_lens), torch.tensor(all_labels)

    def __getitem__(self, index):
        index = index % len(self.data)
        return self.data[index]

    def __len__(self):
        return len(self.data)
