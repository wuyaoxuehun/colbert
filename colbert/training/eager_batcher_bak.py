import os
import ujson

from functools import partial
from colbert.utils.utils import print_message
from colbert.modeling.tokenization import QueryTokenizer, DocTokenizer, tensorize_triples

from colbert.utils.runs import Run
import json
from colbert.base_config import pos_num, neg_num, segmenter
import random

class EagerBatcher():
    def __init__(self, args, rank=0, nranks=1):
        self.rank, self.nranks = rank, nranks
        self.bsize, self.accumsteps = args.bsize, args.accumsteps

        self.query_tokenizer = QueryTokenizer(args.query_maxlen, segmenter)
        self.doc_tokenizer = DocTokenizer(args.doc_maxlen, segmenter)
        self.tensorize_triples = partial(tensorize_triples, self.query_tokenizer, self.doc_tokenizer)

        self.triples_path = args.triples
        self._reset_and_shuffle_triples()

    def _reset_and_shuffle_triples(self):
        self.data = json.load(open(self.triples_path, mode='r', encoding="utf-8"))
        random.shuffle(self.data)
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
            query = example['question']
            pos = [_['text'] for _ in example['positive_ctxs'][:pos_num]]
            while len(pos) < pos_num:
                pos.append(pos[-1])
            # neg = [_['text'] for _ in example['negative_ctxs'][neg_num]]
            neg = [_['text'] for _ in example['hard_negative_ctxs'][1:1+neg_num]]
            while len(neg) < neg_num:
                neg.append(neg[-1])
            queries.append(query)
            positives.append(pos)
            negatives.append(neg)
        # if len(queries) < self.bsize:
        if len(queries) < 1:
            raise StopIteration

        return self.collate(queries, positives, negatives)

    def collate(self, queries, positives, negatives):
        assert len(queries) == len(positives) == len(negatives) #== self.bsize

        return self.tensorize_triples(queries, positives, negatives, self.bsize // self.accumsteps)

    def skip_to_batch(self, batch_idx, intended_batch_size):
        self._reset_and_shuffle_triples()

        Run.warn(f'Skipping to batch #{batch_idx} (with intended_batch_size = {intended_batch_size}) for training.')

        _ = [self.reader.readline() for _ in range(batch_idx * intended_batch_size)]

        return None
