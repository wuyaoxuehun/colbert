import torch
from pyserini.eval.evaluate_dpr_retrieval import has_answers, SimpleTokenizer

# from colbert import base_config

from transformers import BertTokenizerFast, T5TokenizerFast
from functools import reduce
from tqdm import tqdm
# from colbert.base_config import part_weight, puncts, GENERATION_ENDING_TOK
from typing import List, Any
import numpy as np
import string
import nltk
from colbert.utils.func_utils import cache_decorator
from conf import Q_marker_token, D_marker_token, encoder_tokenizer, CLS, SEP, pretrain_choose, answer_SEP, answer_prefix, title_prefix
from nltk.tokenize import sent_tokenize
from colbert.modeling.cpy.hello_world import get_real_inputs2, get_tok_avg_inputs
from conf import *


def cache_function(*args, **kwargs):
    parts = kwargs.get('parts')
    weights = kwargs.get('weights')
    max_seq_length = kwargs.get('max_seq_length')
    if weights is None:
        weights = [1] * len(parts)
    if type(parts[0]) == list:
        parts = sum(parts, [])
    key = [str(hash(''.join(parts)))] + [str(_) for _ in weights] + [str(max_seq_length)]
    return '-'.join(key)


# class CostomTokenizer(BertTokenizerFast):
class CostomTokenizer:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # self.dpr_tokenizer = SimpleTokenizer()
        self.ignore_words = {SEP} | puncts
        # self.word_tokenizer = nltk.word_tokenize
        # if pretrain_choose.find("t5") != -1:
        #     self.ignore_words |= {CLS, SEP, answer_SEP}
        self.tokenize_q_noopt_segmented_dict = self.tokenize_q
        self.tokenize_d_segmented_dict = self.tokenize_d
        self.tokenize_parts = self.tokenize_seqs
        self.tokenizer = BertTokenizerFast.from_pretrained(pretrain)
        if pretrain_choose.find("bert") != -1:
            self.tokenizer.add_special_tokens({"additional_special_tokens": [f"[unused1]", "[unused2]"]})
            self.q_makers = "[unused1]"
            self.d_makers = "[unused2]"

    def tokenize_seqs(self, seqs=None, max_seq_length=None, is_query=False, is_indexing=False):
        # words = CLS + (self.q_makers if is_query else self.d_makers) + parts + SEP
        input_seqs = ["[CLS]" + ((self.q_makers if is_query else self.d_makers) + _ + SEP) for _ in seqs]
        tokens = [self.tokenizer.tokenize(_) for _ in input_seqs]
        active_padding = [[(1 if j not in self.ignore_words else 0) for j in i] + [0] * (max_seq_length - len(i))
                          for i in tokens]
        input_ids = [self.tokenizer.convert_tokens_to_ids(_) + [0] * (max_seq_length - len(_)) for _ in tokens]
        attention_mask = [[1] * len(_) + [0] * (max_seq_length - len(_)) for _ in tokens]
        return input_ids, attention_mask, active_padding

    def tokenize_q(self, batch_examples, max_seq_length):
        questions = []
        for t in batch_examples:
            question = t['question']
            questions.append(question)
        q = self.tokenize_parts(seqs=questions, max_seq_length=max_seq_length, is_query=True)
        return [torch.tensor(_) for _ in q]

    def tokenize_d(self, batch_text, max_seq_length, to_tensor=True):
        docs = []
        for t in batch_text:
            doc = t
            docs.append(doc)
        d = self.tokenize_parts(seqs=docs, max_seq_length=max_seq_length, is_query=False, is_indexing=not to_tensor)
        if to_tensor:
            return [torch.tensor(_) for i, _ in enumerate(d)]
        else:
            return d
