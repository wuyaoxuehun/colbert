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
        self.all_paras = load_all_paras()

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

    def merge_to_reader_input_allopt(self, batch_examples, batch_paras, extra=None):
        para_idx = 0
        # print(len(batch_paras), len(batch_examples))
        assert len(batch_paras) == len(batch_examples) * 4
        for example in batch_examples:
            example['paragraph_abcd'] = [{"p_id": _['p_id'],
                                          "paragraph": ''.join(_['paragraph_cut']['tok'].split()),
                                          "paragraph_cut": _['paragraph_cut']} for _ in batch_paras[para_idx]]
            para_idx += 1
        # assert para_idx == len(batch_paras)

    @staticmethod
    def merge_to_reader_input(batch_examples, batch_paras, extra=None):
        para_idx = 0
        # print(len(batch_paras), len(batch_examples))
        batch_paras, batch_scores = batch_paras
        assert len(batch_paras) == len(batch_examples) * 4
        for example in batch_examples:
            for opt in "abcd":
                example[f'paragraph_{opt}'] = [{"p_id": _['p_id'],
                                                "paragraph": ''.join(_['paragraph_cut']['tok'].split()),
                                                "paragraph_cut": _['paragraph_cut'],
                                                "colbert_score": score}
                                               for _, score in zip(batch_paras[para_idx], batch_scores[para_idx])]
                para_idx += 1
        assert para_idx == len(batch_paras)

    def get_score_for_inst_every_ending(self, inst: Dict, eval_p_num=None, is_evaluating=False, only_true_answer=False):
        """
        ? 为什么要区分pos和neg呢，直接取全部得分的topk不可以吗？原因是这里是随机采样，所以我们希望采样到的段落不要全是完全无关的段落， 也不要是完全相关的段落（难区分）
        :param inst:
        :param eval_p_num:
        :param is_evaluating:
        :param only_true_answer:
        :return:
        """
        contexts = [inst[f'paragraph_{_}'] for _ in 'abcd']
        cand_paras = [para for docs in contexts for para in docs]

        endings = [inst[_] for _ in 'ABCD']
        selected_docs = []

        for opt, paras, ending in (zip(list('abcd'), contexts, endings)):
            # if opt.upper() != inst['answer'].upper() and only_true_answer:
            #     continue
            pos = []
            neg = []
            for para in paras[:ir_topk]:
                get_final_score_for_inst(inst, para, opt)
                if para['score'] >= pos_thre:
                    pos.append(para)
                elif para['score'] < pos_thre:
                    neg.append(para)
            if is_evaluating:
                paras = paras[:ir_topk]
                paras.sort(key=lambda x: x['score'], reverse=True)
                inst[f'paragraph_{opt}'] = paras[:eval_p_num]
                selected_docs += inst[f'paragraph_{opt}']
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
                # print(pos_dist, neg_dist)
                # print(sum(pos_dist), sum(neg_dist))
                # input()
                pos = context_random.choice(pos[:max_pos], replace=False, size=pos_num, p=pos_dist).tolist()
                if neg:
                    neg_dist = softmax([_['score'] / self.sample_T for _ in neg[:max_neg]])
                    neg = context_random.choice(neg[:max_neg], replace=False, size=neg_num, p=neg_dist).tolist()
            else:
                pos = context_random.choice(pos[:max_pos], replace=False, size=pos_num).tolist()
                if neg_num:
                    neg = context_random.choice(neg[:max_neg], replace=False, size=neg_num).tolist()
                else:
                    neg = []

            inst[f'paragraph_{opt}'] = pos + neg
            assert len(inst[f'paragraph_{opt}']) == eval_p_num
            selected_docs += inst[f'paragraph_{opt}']

        # input(len(cand_paras))
        # selected_doc_dic = {doc["p_id"]: doc for doc in selected_docs}
        # selected_docs = list(selected_doc_dic.values())
        # # print(selected_doc_dic.keys())
        # inst['extra_docs'] = list({para['p_id']: para for para in cand_paras if para['p_id'] not in selected_doc_dic}.values())
        # # assert (set([_['p_id'] for _ in inst['extra_docs']]) & set(selected_doc_dic.keys())) == set()
        # np.random.shuffle(inst['extra_docs'])
        # docs = selected_docs + inst['extra_docs'][:4 * eval_p_num - len(selected_docs)]
        #
        docs = selected_docs
        # assert len(docs) == eval_p_num * 4, (len(selected_docs), eval_p_num)
        # print(len(selected_docs), len(inst['extra_docs']), len(docs))
        # assert (len(set([doc["p_id"] for doc in docs])) == len(docs)), (len(set([doc["p_id"] for doc in docs])), len(docs))

        return docs
        # assert len(inst[f'paragraph_{opt}']) == eval_p_num == 2

    def get_score_for_inst(self, inst: Dict, docs, padded_negs=None, eval_p_num=None, only_true_answer=False):
        # if only_true_answer:
        #     contexts = [inst[f"paragraph_{inst['answer'].lower()}"]]
        #     opts = inst['answer'].upper()
        # else:
        #     contexts = [inst[f"paragraph_{_.lower()}"] for _ in "ABCD"]
        #     opts = "ABCD"
        # endings = [inst[_] for _ in 'ABCD']
        scores = []
        for idx, opt in enumerate("ABCD"):
            t = []
            # for paras in contexts:
            #     for para in paras[:eval_p_num]:
            #         t.append(self.get_final_score_for_inst(inst, para, opt))
            for para in docs:
                t.append(get_final_score_for_inst(inst, para, opt))
            scores.append(t)
            # if len(t) != p_num * 4 + len(padded_negs):
            #     print(len(t))
            #     input()
        return scores

    def get_score_for_batch(self, batch: List[Dict], padded_negs=None, eval_p_num=None, only_true_answer=False):

        # endings = [inst[_] for _ in 'ABCD']
        # docs = []
        scores = []
        docs = [_ for t in batch for opt in "abcd" for _ in t['paragraph_' + opt][:eval_p_num]]
        for inst in batch:
            for idx, opt in enumerate("ABCD"):
                cur_scores = [self.get_final_score_for_inst(inst, para, opt) for para in docs]
                scores.append(cur_scores)
        return scores, docs

    def tokenize_for_retriever(self, batch):
        obj = self.query_tokenizer.tensorize_noopt_dict(batch_text=batch)
        return obj

    def tokenize_for_train_retriever(self, batch: List[Dict], eval_p_num=None, is_evaluating=False, only_true_answer=False):
        if eval_p_num is None:
            eval_p_num = p_num
        scores, docs = [], []
        for t in batch:
            # assert len(t["paragraph_a"]) > eval_p_num, (len(t["paragraph_a"]), eval_p_num)
            cur_docs = self.get_score_for_inst_every_ending(t, eval_p_num, is_evaluating)
            cur_scores = self.get_score_for_inst(t, docs=cur_docs, eval_p_num=eval_p_num)
            scores.append(cur_scores)
            docs += cur_docs
        # if padded_negs:
        #     for para in padded_negs:
        #         para['paragraph'] = ''.join(para['paragraph_cut']['tok'].split())
        #     docs += padded_negs
        if padding_neg_num:
            sampled_docs = context_random.choice(list(self.all_paras.values()), padding_neg_num)
            docs += list(sampled_docs)

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

    def tokenize_for_train_retriever_(self, batch: List[Dict], padded_negs=None, eval_p_num=None, is_evaluating=False, only_true_answer=False):
        """
        计算batch内所有para与所有query的相似度
        :param batch:
        :param padded_negs:
        :param eval_p_num:
        :param is_evaluating:
        :param only_true_answer:
        :return:
        """
        if eval_p_num is None:
            eval_p_num = p_num
        scores, docs = [], []
        for t in batch:
            assert len(t["paragraph_a"]) > eval_p_num, (len(t["paragraph_a"]), eval_p_num)
            self.get_score_for_inst_every_ending(t, eval_p_num, is_evaluating)
        scores, cur_docs = self.get_score_for_batch(batch, eval_p_num=eval_p_num)
        docs += cur_docs

        D = self.doc_tokenizer.tensorize_dict(docs)
        scores = torch.tensor(scores)
        # print(len(batch[0]['paragraph_a']))

        # D_scores = torch.tensor([j for _ in batch for j in self.get_score_for_inst(_, padded_negs, eval_p_num=eval_p_num)])
        return D, scores

    # def tokenize_for_reader_(self, batch: List[Dict]):
    #     all_choice_features = []
    #     for inst in batch:
    #         choices_features = []
    #         # sentence_index = [0] * 100
    #         all_contexts = [inst['paragraph_' + opt] for opt in 'abcd']
    #         endings = [inst[opt] for opt in 'ABCD']
    #         token_nums = []
    #         for ending_index, (contexts, ending) in enumerate(zip(all_contexts, endings)):
    #             input_ids_opt = []
    #             input_mask_opt = []
    #             segment_ids_opt = []
    #             segment_lens_opt = []
    #             for context in contexts[:p_num]:
    #                 context_tokens = self.tokenizer.tokenize(context['paragraph'])
    #                 background_tokens = self.tokenizer.tokenize(inst['background'])
    #                 start_ending_tokens = self.tokenizer.tokenize(inst['question'])
    #                 ending_tokens = self.tokenizer.tokenize(ending)
    #                 token_nums.append(len(context_tokens) + len(background_tokens) + len(start_ending_tokens) + len(ending_tokens))
    #
    #                 # ending_tokens = start_ending_tokens + ending_tokens
    #                 self.tokenizer.truncate_seq([context_tokens, background_tokens, start_ending_tokens, ending_tokens],
    #                                             self.reader_max_seq_length - 5, [3])
    #                 doc_len = len(context_tokens)
    #                 background_len = len(background_tokens)
    #                 question_len = len(start_ending_tokens)
    #                 option_len = len(ending_tokens)
    #                 input_tokens = ['[CLS]'] + context_tokens + ['[SEP]'] + background_tokens + ['[SEP]'] \
    #                                + start_ending_tokens + ['[SEP]'] + ending_tokens + ['[SEP]']
    #                 padding_len = self.reader_max_seq_length - len(input_tokens)
    #
    #                 input_ids = self.tokenizer.convert_tokens_to_ids(input_tokens) + [0] * padding_len
    #                 input_mask = [1] * len(input_tokens) + [0] * padding_len
    #                 segment_ids = [0] * (len(context_tokens) + 2) + [1] * (len(background_tokens) + 1) + [0] * (len(start_ending_tokens) + 1) + \
    #                               [1] * (len(ending_tokens) + 1) + [0] * padding_len
    #
    #                 assert len(input_ids) == self.reader_max_seq_length
    #                 assert len(input_mask) == self.reader_max_seq_length
    #                 assert len(segment_ids) == self.reader_max_seq_length
    #
    #                 input_ids_opt.append(input_ids)
    #                 input_mask_opt.append(input_mask)
    #                 segment_ids_opt.append(segment_ids)
    #                 segment_lens_opt.append([doc_len, background_len, question_len, option_len])
    #
    #             choices_features.append([input_ids_opt, input_mask_opt,
    #                                      segment_ids_opt, segment_lens_opt])
    #         all_choice_features.append(choices_features)
    #     all_input_ids = [[_[0] for _ in i] for i in all_choice_features]
    #     all_input_mask = [[_[1] for _ in i] for i in all_choice_features]
    #     all_segment_ids = [[_[2] for _ in i] for i in all_choice_features]
    #     all_segment_lens = [[_[3] for _ in i] for i in all_choice_features]
    #     all_labels = [ord(inst['answer']) - ord('A') for inst in batch]
    #     return torch.tensor(all_input_ids), torch.tensor(all_input_mask), \
    #            torch.tensor(all_segment_ids), torch.tensor(all_segment_lens), torch.tensor(all_labels)

    @lru_cache(maxsize=None)
    def tokenize_one(self, context_paragraph, background, question, ending):
        context_tokens = self.tokenizer.tokenize(context_paragraph)
        background_tokens = self.tokenizer.tokenize(background)
        start_ending_tokens = self.tokenizer.tokenize(question)
        ending_tokens = self.tokenizer.tokenize(ending)
        # token_nums.append(len(context_tokens) + len(background_tokens) + len(start_ending_tokens) + len(ending_tokens))
        # ending_tokens = start_ending_tokens + ending_tokens
        self.tokenizer.truncate_seq([context_tokens, background_tokens, start_ending_tokens, ending_tokens],
                                    reader_max_seq_length - 5, [3])

        doc_len = len(context_tokens)
        background_len = len(background_tokens)
        question_len = len(start_ending_tokens)
        option_len = len(ending_tokens)

        assert min(background_len, question_len, option_len) > 0, (background_len, question_len, option_len)
        input_tokens = ['[CLS]'] + context_tokens + ['[SEP]'] + background_tokens + ['[SEP]'] \
                       + start_ending_tokens + ['[SEP]'] + ending_tokens + ['[SEP]']
        padding_len = reader_max_seq_length - len(input_tokens)

        input_ids = self.tokenizer.convert_tokens_to_ids(input_tokens) + [0] * padding_len
        input_mask = [1] * len(input_tokens) + [0] * padding_len
        segment_ids = [0] * (len(context_tokens) + 2) + [1] * (len(background_tokens) + 1) + [0] * (len(start_ending_tokens) + 1) + \
                      [1] * (len(ending_tokens) + 1) + [0] * padding_len

        assert len(input_ids) == reader_max_seq_length
        assert len(input_mask) == reader_max_seq_length
        assert len(segment_ids) == reader_max_seq_length
        return input_ids, input_mask, segment_ids, (doc_len, background_len, question_len, option_len)

    def tokenize_for_reader(self, batch: List[Dict]):
        all_choice_features = []
        for inst in batch:
            choices_features = []
            # sentence_index = [0] * 100
            all_contexts = [inst['paragraph_' + opt] for opt in 'abcd']
            endings = [inst[opt] for opt in 'ABCD']
            # token_nums = []
            for ending_index, (contexts, ending) in enumerate(zip(all_contexts, endings)):
                input_ids_opt = []
                input_mask_opt = []
                segment_ids_opt = []
                segment_lens_opt = []
                for context in contexts[:reader_p_num]:
                    if len(inst['question']) == 0:
                        question = "请回答下列问题。"
                    else:
                        question = inst["question"]
                    if len(inst['background']) == 0:
                        background = "背景。"
                    else:
                        background = inst['background']

                    input_ids, input_mask, segment_ids, (doc_len, background_len, question_len, option_len) = \
                        self.tokenize_one(context['paragraph'], background, question, ending)
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
