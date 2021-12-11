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

logger = logging.getLogger(name='__main__')


# keys = ["background", "question", "A", "B", "C", "D"]
# ekeys = ["ID"]
# for k in keys:
#     ekeys.append(k)
#     ekeys.append(k + "_cut")
# ekeys.append("answer")

# Example = namedtuple("Example", ["ID", "background", "background_cut", "question", "question_cut", "A", "A_cut", "paragraph_a", "B", "B_cut", "paragraph_b",
#                                  "C", "C_cut", "paragraph_c", "D", "D_cut", "paragraph_d", "answer"])


def load_data(file, task=0):
    import json
    data = []
    with open(file, encoding='utf8') as f:
        all_data = json.load(f)
        logger.info(len(all_data))
        for instance in all_data:
            # contexts = []
            # background = filterBackground(instance['background'])
            # question_text = filterQuestion(instance['question'])

            if instance['answer'] not in list('ABCD'):
                print(instance['answer'])
                continue
            # data.append(Example(
            #     ID=instance['id'],
            #     background=instance['background'],
            #     background_cut=instance['background_cut'],
            #     question=instance['question'],
            #     question_cut=instance['question_cut'],
            #     A=instance['A'],
            #     A_cut=instance['A_cut'],
            #     paragraph_a=None,
            #     B=instance['B'],
            #     B_cut=instance['B_cut'],
            #     paragraph_b=None,
            #     C=instance['C'],
            #     C_cut=instance['C_cut'],
            #     paragraph_c=None,
            #     D=instance['D'],
            #     D_cut=instance['D_cut'],
            #     paragraph_d=None,
            #     answer=ord(instance['answer']) - ord('A')
            # ))
            data.append(instance)
    logger.info(len(data))
    if file.find('dev') != -1:
        data = data[:]
    return data


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


# def collate_fun(device, RD=True):
#     def collate(batch: List[Dict[str, Any]]):
#         keys = batch[0].keys()
#         input_dict = {}
#         # max_labels_len = max([len(_['labels']) for _ in batch])
#         # print(batch[0]['attention_mask'])
#         # print(batch[0]['attention_mask'][0])
#         max_seq_len_batch = max([sum(k) for _ in batch for j in _['attention_mask'] for k in j])
#         max_seq_len_batch = 512
#         aug_num = len(batch[0]['attention_mask'])
#         # print(f"augnum: {aug_num}")
#         # max_seq_len_batch = 512
#         # max_seq_len_batch = len(batch[0]['attention_mask'])
#         # print(batch)
#         # input()
#         for k in keys:
#             if k == "segment_lens":
#                 # input_dict[k] = [j[0] for _ in batch for j in _[k]]
#                 input_dict[k] = [j for _ in batch for j in _[k]]
#                 # input(input_dict[k])
#             elif k in ["labels", "task"]:
#                 # print([_[k] for _ in batch])
#                 input_dict[k] = [_[k] * aug_num for _ in batch]
#                 # input(input_dict[k])
#             else:
#                 input_dict[k] = [[s[:max_seq_len_batch] for s in j] for _ in batch for j in _[k]]
#
#         # input_dict['sent_nums'] = [len(_['labels']) for _ in batch for t in range(aug_num)]
#         for k in list(keys):
#             # print(k)
#             # if RD:
#             #     if k == 'input_ids':
#             #         input_dict[k] = torch.tensor([mappings[i % mapping_len](input_dict[k][i // mapping_len]) for i in range(mapping_len * len(input_dict[k]))]).to(device)
#             #     else:
#             #         input_dict[k] = torch.tensor([input_dict[k][i // mapping_len] for i in range(mapping_len * len(input_dict[k]))]).to(device)
#             #
#             #     # print(input_dict[k][:3])
#             #     # input()
#             # else:
#             input_dict[k] = torch.tensor(input_dict[k]).to(device)
#             # print(k, input_dict[k].size())
#             # if k == "labels":
#             #     print(input_dict[k])
#             # input()
#             # print(k, '\n', input_dict[k][0])
#             # input()
#         # print(input_dict['input_ids'][0][0])
#         # input()
#         return input_dict
# return collate

def collate_fun():
    def fun(batch):
        return batch

    return fun


class CBQADataset(Dataset):
    def __init__(self, file_abbrs, tokenizer=None, doc_maxlen=None, query_maxlen=None, reader_max_seq_length=256, eager=False, mode='dev'):
        super().__init__()
        self.tokenizer: CostomTokenizer = tokenizer
        if self.tokenizer is None:
            self.tokenizer = CostomTokenizer.from_pretrained(pretrain_map['bert'])

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
            data += load_data(file, 0)
        # import random
        # random.seed(0)
        # random.shuffle(data)
        if self.mode == 'train':
            data = data[:]
        else:
            data = data[:]

        return data

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
            # print(len(pos), len(neg))
            # print([_['score'] for _ in pos])
            # print([_['score'] for _ in neg])
            # input()
            # if len(neg) == 0:
            #     neg.append({
            #         'score': 0.0,
            #         'paragraph': '#',
            #         'paragraph_cut': {
            #             'tok': '#'
            #         }
            #     })
            # if len(pos) == 0:
            #     if neg[0]['score'] < -1:
            #         # ending_cut = inst[opt.upper() + '_cut']
            #         # pos.append({
            #         #     'score': 1,
            #         #     'paragraph': ending,
            #         #     'paragraph_cut': ending_cut
            #         # })
            #         # pos.append()
            #         assert False
            #     else:
            #         while len(pos) < pos_num:
            #             pos.append(neg.pop(0))
            # print([_['score'] for _ in pos])
            # print([_['score'] for _ in neg])
            # input()
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

    def get_score_for_inst(self, inst: Dict, padded_negs=None, eval_p_num=None):
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

    def tokenize_for_retriever(self, batch):
        ids, mask, word_mask = self.query_tokenizer.tensorize_allopt_dict(batch_text=batch)
        return ids, mask, word_mask

    def tokenize_for_train_retriever(self, batch: List[Dict], padded_negs=None, eval_p_num=None, is_evaluating=False):
        docs = []
        if eval_p_num is None:
            eval_p_num = p_num
        for t in batch:
            self.get_score_for_inst_every_ending(t, eval_p_num, is_evaluating)
        # import json
        # json.dump(batch, open("temp.json", 'w', encoding='utf8'), ensure_ascii=False, indent=2)
        # input()

        for line_idx, example in enumerate(batch):
            contexts = [example['paragraph_' + opt] for opt in 'abcd']
            for paras in contexts:
                assert len(paras) >= eval_p_num
                docs += paras[:eval_p_num]
        if padded_negs:
            for para in padded_negs:
                para['paragraph'] = ''.join(para['paragraph_cut']['tok'].split())
            docs += padded_negs

        D_ids, D_mask, D_word_mask = self.doc_tokenizer.tensorize_dict(docs)

        D_scores = torch.tensor([j for _ in batch for j in self.get_score_for_inst(_, padded_negs, eval_p_num=eval_p_num)])
        return D_ids, D_mask, D_word_mask, D_scores

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

    # def transform_inst(self, inst: Dict):
    #     input_ids, token_type_ids, attention_mask, segment_lens = self.tokenize(inst)
    #     return {
    #         "input_ids": input_ids,
    #         "token_type_ids": token_type_ids,
    #         "attention_mask": attention_mask,
    #         "segment_lens": segment_lens,
    #         "labels": ord(inst['answer']) - ord('A'),
    #         'task': 0
    #     }

    def __getitem__(self, index):
        index = index % len(self.data)
        return self.data[index]

    def __len__(self):
        return len(self.data)
