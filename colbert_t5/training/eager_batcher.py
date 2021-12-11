import itertools
import json
import random
from functools import partial
from operator import itemgetter

import numpy as np

from colbert.base_config import pos_num, neg_num, segmenter, max_neg_sample, max_pos_sample
from colbert.modeling.tokenization import QueryTokenizer, DocTokenizer, tensorize_triples
from colbert.utils.runs import Run
from colbert.utils.utils import print_message

from colbert.base_config import pos_threshold, neg_threshold

class EagerBatcher():
    def __init__(self, args, rank=0, nranks=1):
        self.rank, self.nranks = rank, nranks
        self.bsize, self.accumsteps = args.bsize, args.accumsteps

        self.query_tokenizer = QueryTokenizer(args.query_maxlen, segmenter)
        self.doc_tokenizer = DocTokenizer(args.doc_maxlen, segmenter)
        self.tensorize_triples = partial(tensorize_triples, self.query_tokenizer, self.doc_tokenizer)

        self.triples_path = args.triples
        self._reset_and_shuffle_triples()
        # print_message(f"totally {len(self.data)} examples")

    def shuffle_dataset_by_id(self, data):
        random.shuffle(data)
        out_data = []
        for opt in 'ABCD':
            for t in data:
                pos_ctx, neg_ctxs = [], []
                for p in t[f'paragraph_{opt.lower()}']:
                    p['score'] = p['rouge_o']
                    if p['score'] > pos_threshold:
                        pos_ctx.append(p)
                    elif p['score'] < pos_threshold:
                        neg_ctxs.append(p)
                if not pos_ctx:
                    continue
                out_data.append({
                    "question_cut": t['question_cut'],
                    "background_cut": t['background_cut'],
                    "option_cut": t[f'{opt}_cut'],
                    "positive_ctxs": pos_ctx,
                    "hard_negative_ctxs": neg_ctxs
                })

        # assert len(out_data) == len(data) * 4
        return out_data

    def _reset_and_shuffle_triples(self):
        self.data = json.load(open(self.triples_path, mode='r', encoding="utf-8"))
        # self.data = [_ for _ in self.data if _['positive_ctxs'] and _['hard_negative_ctxs']] #only count if has pos as well as negs
        # self.data = [_ for _ in self.data if _['positive_ctxs']] #only count if has pos as well as negs
        # self.data = [_ for _ in self.data if _['positive_ctxs'] and _['hard_negative_ctxs']] #only count if has pos as well as negs
        # random.shuffle(self.data)
        self.data = self.shuffle_dataset_by_id(self.data)
        if self.rank == 0:
            print_message("total data size " + str(len(self.data)))

        self.length = len(self.data)
        self.reader = iter(self.data)
        self.position = 0

    def __iter__(self):
        return self

    def __len__(self):
        return self.length // self.bsize // self.nranks

    def __next__(self):
        queries, positives, negatives = [], [], []

        for line_idx, example in zip(range(self.bsize * self.nranks), self.reader):
            if (self.position + line_idx) % self.nranks != self.rank:
                continue

            # query = example['background'] + ['SEP'] + example['question'] + ['SEP'] + example['option']
            query = example
            # discount_rate = 1
            pos = example['positive_ctxs'][:max_pos_sample]
            assert pos
            while len(pos) < pos_num:
                pos.append(pos[-1])
            pos = np.random.choice(pos, replace=False, size=pos_num).tolist()

            neg = example['hard_negative_ctxs'][:max_neg_sample]
            if len(neg) == 0:
                neg = [{
                    'score': 0.0,
                    'paragraph_cut': {
                        'tok': '#'
                    }
                }]
            while len(neg) < neg_num:
                neg.append(neg[-1])
            neg = np.random.choice(neg, replace=False, size=neg_num).tolist()
            queries.append(query)
            positives.append(pos)
            negatives.append(neg)
        # if len(queries) < self.bsize:
        # self.position += line_idx + 1
        if len(queries) < 1:
            raise StopIteration

        return self.collate(queries, positives, negatives)

    # def __next__(self):
    #     queries, positives, negatives = [], [], []
    #
    #     for line_idx, example in zip(range(self.bsize * self.nranks), self.reader):
    #         if (self.position + line_idx) % self.nranks != self.rank:
    #             continue
    #
    #         # query = example['background'] + ['SEP'] + example['question'] + ['SEP'] + example['option']
    #         query = example
    #
    #         pos = example['positive_ctxs'][:pos_num]
    #         if len(pos) == 0:
    #             pos = [{
    #                 'paragraph_cut':{
    #                     'tok':'正确 段落'
    #                 }
    #             }]
    #             print('pos:', pos)
    #
    #         while len(pos) < pos_num:
    #             pos.append(pos[-1])
    #
    #         neg = example['hard_negative_ctxs'][:neg_num]
    #         if len(neg) == 0 and neg_num != 0:
    #             neg = [{
    #                 'paragraph_cut': {
    #                     'tok': '错误 段落'
    #                 }
    #             }]
    #             print('neg:', neg)
    #
    #         while len(neg) < neg_num:
    #             neg.append(neg[-1])
    #
    #         queries.append(query)
    #         positives.append(pos)
    #         negatives.append(neg)
    #     # if len(queries) < self.bsize:
    #     if len(queries) < 1:
    #         raise StopIteration
    #
    #     return self.collate(queries, positives, negatives)

    def collate(self, queries, positives, negatives):
        assert len(queries) == len(positives) == len(negatives)  # == self.bsize

        return self.tensorize_triples(queries, positives, negatives, self.bsize // self.accumsteps)

    def skip_to_batch(self, batch_idx, intended_batch_size):
        self._reset_and_shuffle_triples()
        if self.rank < 1:
            Run.warn(f'Skipping to batch #{batch_idx} (with intended_batch_size = {intended_batch_size}) for training.')

        # _ = [self.reader.readline() for _ in range(batch_idx * intended_batch_size)]

        return None
