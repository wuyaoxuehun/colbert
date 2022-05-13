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
class CostomTokenizer(encoder_tokenizer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.doc_cache = {}
        self.query_cache = {}
        if pretrain_choose.find("bert") != -1:
            self.add_special_tokens({"additional_special_tokens": ["[unused1]", "[unused2]"]})
        self.dpr_tokenizer = SimpleTokenizer()
        # self.ignore_words = set(list(string.punctuation) + [Q_marker_token, D_marker_token])
        self.ignore_words = {SEP} | puncts
        # self.ignore_words = {}
        # self.ignore_words = {SEP} | puncts
        # nlp = spacy.load('en_core_web_sm', disable=['ner', 'tagger', 'parser', 'textcat', 'tok2vec', 'attribute_ruler', 'lemmatizer'])
        # print(nlp.pipe_names)
        # self.word_tokenizer = lambda s: [str(i) for i in nlp(s)]
        # nlp = nltk.ToktokTokenizer()
        # self.word_tokenizer = nlp.tokenize
        # self.word_tokenizer = lambda s: s.split()
        self.word_tokenizer = nltk.word_tokenize

        if pretrain_choose.find("t5") != -1:
            self.ignore_words |= {CLS, SEP, answer_SEP}
        # self.tokenize_q_noopt_segmented_dict = self.tokenize_q_noopt_segmented_dict_geo
        # self.tokenize_d_segmented_dict = self.tokenize_d_segmented_dict_geo
        # self.tokenize_q_noopt_segmented_dict = self.tokenize_q_noopt_segmented_dict_medqa
        # self.tokenize_d_segmented_dict = self.tokenize_d_segmented_dict_medqa
        self.tokenize_q_noopt_segmented_dict = self.tokenize_q_noopt_segmented_dict_medqa
        self.tokenize_d_segmented_dict = self.tokenize_d_segmented_dict_medqa
        # self.tokenize_parts =

    @property
    def tokenize_parts(self):
        return self.tokenize_multiple_parts_tok if not use_word else self.tokenize_multiple_parts_tok_avg

    def truncate_seq(self, tokens_list: List[List[Any]], max_len, keep=None, stategy="longest"):
        if stategy != "longest":
            raise NotImplemented
        all_len = [len(_) for _ in tokens_list]
        while sum(all_len) > max_len:
            if keep:
                t = [all_len[i] for i in range(len(tokens_list)) if i not in keep]
                if sum(t) != 0:
                    all_len = [0 if i in keep else all_len[i] for i in range(len(tokens_list))]
            max_len_idx = int(np.argmax(all_len))
            tokens_list[max_len_idx].pop()
            all_len = [len(_) for _ in tokens_list]

    # @cache_decorator(cache_fun=cache_function)
    def tokenize_multiple_parts__(self, parts=None, max_seq_length=None, weights=None, generation_ending_token_idx=None, keep=(), marker=None):
        assert len(weights) == len(parts)
        if weights is None:
            weights = [1] * len(parts)
        if type(parts[0]) == str:
            parts = [_.strip().split() for _ in parts]

        nparts = len(parts)
        doc_tokens = [[] for _ in range(nparts)]
        word_pos0_mask = [[] for _ in range(nparts)]
        cur_segments = [[] for _ in range(nparts)]
        for part_id, text in enumerate(parts):
            segments = text
            for word_idx, segment in enumerate(segments):
                segment = segment.strip()
                tokens = super().tokenize(segment)
                if not tokens:
                    # print(tokens)
                    continue
                part_mask = weights[part_id] if segment not in puncts else 0
                # word_pos0_mask[part_id].extend([part_mask] + [0] * (len(tokens) - 1))
                word_pos0_mask[part_id].extend([part_mask] + [1] * (len(tokens) - 1))
                # word_pos0_mask.extend([part_mask] + [1] * (len(tokens) - 1))
                doc_tokens[part_id].extend(tokens)
                cur_segments[part_id].append(segment)
        self.truncate_seq(tokens_list=doc_tokens, max_len=max_seq_length - 2 - len(parts) - int((generation_ending_token_idx is not None)),
                          keep=keep)
        for i in range(nparts):
            part_len = len(doc_tokens[i])
            word_pos0_mask[i] = word_pos0_mask[i][:part_len]
            cur_segments[i] = cur_segments[i][:len([_ for _ in word_pos0_mask[i] if _ != 0])]

        doc_tokens = ['[CLS]', '.'] + reduce(lambda a, b: a + ['[SEP]'] + b, doc_tokens) + ['[SEP]'] + \
                     ([generation_ending_token_idx] if generation_ending_token_idx is not None else [])
        # word_pos0_mask = [1, 0] + reduce(lambda a, b: a + [0] + b, word_pos0_mask) + [0] + \
        #                  ([0] if generation_ending_token_idx is not None else [])
        word_pos0_mask = [1, 1] + reduce(lambda a, b: a + [1] + b, word_pos0_mask) + [1] + \
                         ([1] if generation_ending_token_idx is not None else [])

        cur_segments = ['[CLS]'] + sum(cur_segments, [])

        padding_length = max_seq_length - len(doc_tokens)
        input_ids = self.convert_tokens_to_ids(doc_tokens) + [0] * padding_length
        if marker is not None:
            input_ids[1] = marker
        attention_mask = [1] * (max_seq_length - padding_length) + [0] * padding_length

        # word_pos0_mask = [1] + word_pos0_mask[:(max_seq_length - padding_length)] + [0] + [0] * padding_length
        word_pos0_mask = word_pos0_mask + [0] * padding_length
        assert len(input_ids) == len(attention_mask) == len(word_pos0_mask) == max_seq_length, \
            (len(input_ids), len(attention_mask), len(word_pos0_mask))
        return input_ids, attention_mask, word_pos0_mask, cur_segments

    @cache_decorator(cache_fun=cache_function)
    def tokenize_multiple_parts_(self, parts=None, max_seq_length=None, weights=None, generation_ending_token_idx=None, keep=()):
        if weights is None:
            weights = [1] * len(parts)
        assert len(weights) == len(parts)
        # if type(parts[0]) == str:
        #     parts = [_.strip().split() for _ in parts]

        nparts = len(parts)
        doc_tokens = [[] for _ in range(nparts)]
        word_pos0_mask = [[] for _ in range(nparts)]
        cur_segments = [[] for _ in range(nparts)]
        for part_id, text in enumerate(parts):
            tokens = super().tokenize(text)
            if not tokens:
                # print(tokens)
                continue
            doc_tokens[part_id].extend(tokens)
        self.truncate_seq(tokens_list=doc_tokens, max_len=max_seq_length - 2 - len(parts) - int((generation_ending_token_idx is not None)),
                          keep=keep)
        for i in range(nparts):
            part_len = len(doc_tokens[i])
            word_pos0_mask[i] = [1] * part_len
            cur_segments[i] = [' '] * part_len

        doc_tokens = ['[CLS]', '.'] + reduce(lambda a, b: a + ['[SEP]'] + b, doc_tokens) + ['[SEP]'] + \
                     ([generation_ending_token_idx] if generation_ending_token_idx is not None else [])
        word_pos0_mask = [1, 0] + reduce(lambda a, b: a + [0] + b, word_pos0_mask) + [0] + \
                         ([0] if generation_ending_token_idx is not None else [])

        cur_segments = ['[CLS]'] + sum(cur_segments, [])

        padding_length = max_seq_length - len(doc_tokens)
        input_ids = self.convert_tokens_to_ids(doc_tokens) + [0] * padding_length
        attention_mask = [1] * (max_seq_length - padding_length) + [0] * padding_length

        # word_pos0_mask = [1] + word_pos0_mask[:(max_seq_length - padding_length)] + [0] + [0] * padding_length
        word_pos0_mask = word_pos0_mask + [0] * padding_length
        assert len(input_ids) == len(attention_mask) == len(word_pos0_mask) == max_seq_length, \
            (len(input_ids), len(attention_mask), len(word_pos0_mask))
        return input_ids, attention_mask, word_pos0_mask, cur_segments

    @cache_decorator(cache_fun=cache_function)
    def tokenize_multiple_parts___(self, parts=None, max_seq_length=None, weights=None, generation_ending_token_idx=None, keep=(),
                                   marker_token=None, add_special_tokens=True, prefix=None):
        assert len(weights) == len(parts)
        if weights is None:
            weights = [1] * len(parts)
        # if add_special_tokens:
        #     self.padding_side = "left"
        #     self.pad_token = self.tokenizer.eos_token  # to avoid an error
        # else:
        #     self.padding_side = "right"
        #     self.pad_token = self.tokenizer.pad_token  # to avoid an error

        nparts = len(parts)
        if add_special_tokens:
            input_sequence = CLS + marker_token + SEP.join(parts) + SEP
        else:
            input_sequence = (prefix if prefix is not None else "") + answer_SEP.join(parts) + SEP
        # convert_tokens_to_ids, batch_encode_plus
        encoding = self.encode_plus(input_sequence,
                                    padding='max_length',
                                    max_length=max_seq_length,
                                    truncation=True,
                                    add_special_tokens=False)
        input_ids = encoding['input_ids']
        attention_mask = encoding['attention_mask']
        # print(sum(attention_mask))
        # print(input_sequence)
        # print(input_ids)
        # print(attention_mask)
        # input()

        # word_pos0_mask = [1] + word_pos0_mask[:(max_seq_length - padding_length)] + [0] + [0] * padding_length
        word_pos0_mask = attention_mask
        cur_segments = [' '] * len(input_ids)
        assert len(input_ids) == len(attention_mask) == len(word_pos0_mask) == max_seq_length, \
            (len(input_ids), len(attention_mask), len(word_pos0_mask))
        return input_ids, attention_mask, word_pos0_mask, cur_segments

    # @cache_decorator(cache_fun=cache_function)
    def tokenize_multiple_parts_word(self, parts=None, max_seq_length=None, weights=None, generation_ending_token_idx=None, keep=(),
                                     marker_token=None, add_special_tokens=True, prefix=None, output_real=True):
        assert len(weights) == len(parts)
        word_parts = parts
        # for part in parts:
        #     # words = nltk.word_tokenize(part)
        #     words = self.word_tokenizer(part)
        #     word_parts.append(words)
        # words = ["query: "] + words + ['</s>']

        if add_special_tokens:
            words = [CLS, marker_token] + join_list(SEP, word_parts) + [SEP]
        else:
            words = ([prefix] if prefix else []) + join_list(SEP, word_parts) + [SEP]

        ignore_word_indices = [i for i, w in enumerate(words) if w in self.ignore_words]
        inputs = self.encode_plus(words, is_split_into_words=True, add_special_tokens=False, max_length=max_seq_length, truncation=True)
        input_ids = inputs['input_ids']
        word_ids = inputs.word_ids()
        if output_real:
            input_ids, attention_mask, active_indices, active_padding = get_real_inputs2(input_ids, word_ids, ignore_word_indices, max_seq_length)
            return input_ids, attention_mask, active_indices, active_padding
        # assert len(input_ids) == len(attention_mask) == len(active_padding) == len(active_indices) == max_seq_length, \
        #     (len(input_ids), len(attention_mask), len(active_padding), len(active_indices))
        return input_ids, word_ids, ignore_word_indices

    def tokenize_multiple_parts_tok(self, parts=None, max_seq_length=None, weights=None, generation_ending_token_idx=None, keep=(),
                                    marker_token=None, add_special_tokens=True, prefix=None, output_real=True, **kwargs):
        assert len(weights) == len(parts)
        word_parts = parts
        # for part in parts:
        #     # words = nltk.word_tokenize(part)
        #     words = self.word_tokenizer(part)
        #     word_parts.append(words)
        # words = ["query: "] + words + ['</s>']

        if add_special_tokens:
            words = CLS + marker_token + SEP.join(word_parts) + SEP
        else:
            assert False
            words = ([prefix] if prefix else []) + join_list(SEP, word_parts) + [SEP]

        # ignore_word_indices = [i for i, w in enumerate(words) if w in self.ignore_words]
        inputs = self.encode_plus(words,
                                  padding='max_length',
                                  max_length=max_seq_length,
                                  truncation=True,
                                  add_special_tokens=False)

        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        active_len = sum(attention_mask)
        active_indices = list(range(active_len)) + [0] * (max_seq_length - active_len)
        active_padding = [1] * active_len + [0] * (max_seq_length - active_len)
        # word_ids = inputs.word_ids()
        return input_ids, attention_mask, active_indices, active_padding

    def tokenize_multiple_parts_tok_avg(self, parts=None, max_seq_length=None, weights=None, generation_ending_token_idx=None, keep=(),
                                        marker_token=None, add_special_tokens=True, prefix=None, output_real=True, max_span_length=None):
        assert len(weights) == len(parts)
        # is_query = len(parts) > 1
        word_parts = parts
        if add_special_tokens:
            words = [CLS, marker_token] + join_list(SEP, word_parts) + [SEP]
            # ignore marker token
            # words = [CLS] + join_list(SEP, word_parts) + [SEP]
        else:
            words = ([prefix] if prefix else []) + join_list(SEP, word_parts) + [SEP]
        # input(words)
        ignore_word_indices = [i for i, w in enumerate(words) if w in self.ignore_words]

        is_query = marker_token == Q_marker_token

        if is_query and use_part_weight:
            part_spans = []
            cur_idx = 0
            word_weight_parts = []
            cur_part = 0
            # part_weight = [1, 5, 20]
            part_weight = [1, 2, 3]
            for word in words:
                if word not in self.ignore_words:
                    word_weight_parts.append(part_weight[cur_part])
                if word == SEP:
                    cur_part += 1
                    part_spans.append((cur_idx, len(word_weight_parts)))
                    cur_idx = len(word_weight_parts)
        # print(word_weight_parts, word_parts)
        # input()
        inputs = self.encode_plus(words, is_split_into_words=True, add_special_tokens=False,
                                  max_length=max_seq_length, padding='max_length', truncation=True)
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        word_ids = inputs.word_ids()

        assert len(input_ids) == len(attention_mask) == max_seq_length
        if True or output_real:
            active_spans, active_padding = get_tok_avg_inputs(word_ids, ignore_word_indices, max_span_length=max_span_length)
            if is_query and use_part_weight:
                active_padding = word_weight_parts[:max_span_length] + [0] * (max(0, max_span_length - len(word_weight_parts)))
            #     return input_ids, attention_mask, active_spans, active_padding, part_spans
            # print(active_padding)
            # print(part_spans)
            # input()
            # print(active_padding)
            # input()
            # assert sum(active_padding) == len(word_weight_parts)
            # word_weight_parts += [0] * (max_seq_length - len(word_weight_parts))
            # input(active_spans)
            # return input_ids, attention_mask, active_spans, active_padding, word_weight_parts
            return input_ids, attention_mask, active_spans, active_padding
        # assert len(input_ids) == len(attention_mask) == len(active_padding) == len(active_indices) == max_seq_length, \
        #     (len(input_ids), len(attention_mask), len(active_padding), len(active_indices))
        # return input_ids, attention_mask, word_ids, ignore_word_indices

    def tokenize_multiple_parts_tok_avg_(self, parts=None, max_seq_length=None, weights=None, generation_ending_token_idx=None, keep=(),
                                         marker_token=None, add_special_tokens=True, prefix=None, output_real=True, max_span_length=None):
        assert len(weights) == len(parts)
        # is_query = len(parts) > 1
        word_parts = parts
        if add_special_tokens:
            words = [CLS, marker_token] + join_list(SEP, word_parts) + [SEP]
            # ignore marker token
            # words = [CLS] + join_list(SEP, word_parts) + [SEP]
        else:
            words = ([prefix] if prefix else []) + join_list(SEP, word_parts) + [SEP]
        # input(words)
        ignore_word_indices = [i for i, w in enumerate(words) if w in self.ignore_words]
        inputs = self.encode_plus(words, is_split_into_words=True, add_special_tokens=False,
                                  max_length=max_seq_length, padding='max_length', truncation=True)
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        word_ids = inputs.word_ids()

        assert len(input_ids) == len(attention_mask) == max_seq_length
        active_spans = [(i, i + 1) for i in range(max_seq_length)]
        active_padding = attention_mask
        return input_ids, attention_mask, active_spans, active_padding

        if True or output_real:
            active_spans, active_padding = get_tok_avg_inputs(word_ids, ignore_word_indices, max_span_length=max_span_length)
            # assert sum(active_padding) == len(word_weight_parts)
            # word_weight_parts += [0] * (max_seq_length - len(word_weight_parts))
            # input(active_spans)
            # return input_ids, attention_mask, active_spans, active_padding, word_weight_parts
            return input_ids, attention_mask, active_spans, active_padding

    def tokenize_multiple_parts_bak(self, parts=None, max_seq_length=None, weights=None, generation_ending_token_idx=None, keep=(),
                                    marker_token=None, add_special_tokens=True, prefix=None):
        assert len(weights) == len(parts)
        word_parts = []
        for part in parts:
            words = self.word_tokenizer(part)
            word_parts.append(words)
        # words = ["query: "] + words + ['</s>']

        if add_special_tokens:
            words = [CLS, marker_token] + join_list(SEP, word_parts) + [SEP]
        else:
            words = ([prefix] if prefix else []) + join_list(SEP, word_parts) + [SEP]

        ignore_word_indices = [i for i, w in enumerate(words) if w in self.ignore_words]
        token_to_word, word_to_tokens = [], {}
        input_ids = []
        for word_id, word in enumerate(words):
            tokens = self.encode(word, add_special_tokens=False)
            token_to_word += [word_id] * len(tokens)
            word_to_tokens[word_id] = (len(input_ids), len(input_ids) + len(tokens))
            input_ids += tokens

        max_active_len = len(input_ids)
        input_ids += [0] * (max_seq_length - len(input_ids))
        attention_mask = np.zeros((max_seq_length, max_seq_length)).astype(int)
        attention_mask[:max_active_len, :max_active_len] = 1
        active_indices = []

        i = 0
        while i < max_active_len:
            word_idx = token_to_word[i]
            start, end = word_to_tokens[word_idx]
            attention_mask[start:end - 1, :start] = 0
            attention_mask[start:end - 1, end:] = 0
            attention_mask[:start, start:end - 1] = 0
            attention_mask[end:, start:end - 1] = 0
            if word_idx not in ignore_word_indices:
                active_indices.append(end - 1)
            i = end
        # for i in range(max_active_len):
        #     idx = inputs.token_to_word(i)
        #     print(i, idx, words[idx], i in active_indices)
        # input()

        active_indices += [0] * (max_seq_length - len(active_indices))
        active_padding = [1] * len(active_indices) + [0] * (max_seq_length - len(active_indices))
        attention_mask = attention_mask.tolist()
        # word_pos0_mask = [1] + word_pos0_mask[:(max_seq_length - padding_length)] + [0] + [0] * padding_length
        # word_pos0_mask = attention_mask
        # cur_segments = [' '] * len(input_ids)

        # assert len(input_ids) == len(attention_mask) == len(active_padding) == len(active_indices) == max_seq_length, \
        #     (len(input_ids), len(attention_mask), len(active_padding), len(active_indices))
        return input_ids, attention_mask, active_indices, active_padding

    def tokenize_q_noopt_segmented_dict_geo(self, batch_examples, max_seq_length, answer_max_seq_length=64, marker_token=None):
        q = []

        for t in batch_examples:
            question = t['question_cut']['tok'].split()
            background = t['background_cut']['tok'].split()
            for opt in "ABCD":
                option = t[opt + '_cut']['tok'].split()
                # enum = [background, question, answer]
                # enum = [question, option, background]
                enum = [question, option, background]
                output = self.tokenize_parts(parts=enum, max_seq_length=max_seq_length, weights=[1] * len(enum),
                                             generation_ending_token_idx=None, keep=[2], marker_token=Q_marker_token, max_span_length=query_max_span_length)
                q.append(output)

            # return doc_tokens, doc_words, tok_to_orig_seg_map, word_pos0_mask
        # for t in [q, ans, title]:
        #     t[1] = np.array(t[1])
        # return [[torch.tensor(_) for _ in t] for t in [q, ans, title]]
        q = list(zip(*q))
        return [torch.tensor(_) for i, _ in enumerate(q)]

    def tokenize_d_segmented_dict_geo(self, batch_text, max_seq_length, tqdm_enable=False, to_tensor=True, word=False):
        # num = 4
        d = []
        # batch_text = tqdm(batch_text, disable=len(batch_text) < 1000)
        # all_doc_segments = []
        for t in batch_text:
            doc = [t['paragraph_cut']['tok'].split()]
            output = self.tokenize_parts(parts=doc, max_seq_length=max_seq_length, weights=[1] * len(doc),
                                         generation_ending_token_idx=None, keep=[], marker_token=D_marker_token, output_real=to_tensor,
                                         max_span_length=doc_max_span_length)
            d.append(output)
        # d[1] = np.array(d[1])
        if to_tensor:
            # return [torch.tensor(_) for _ in d]
            d = list(zip(*d))
            # return [(torch.tensor(_) if i != 100 else _) for i, _ in enumerate(d)]
            return [torch.tensor(_) for i, _ in enumerate(d)]
        else:
            return d

    def tokenize_q_noopt_segmented_dict_medqa(self, batch_examples, max_seq_length):
        q = []

        for t in batch_examples:
            question = t['question_cut'].split()
            # enum = [question, option, background]
            enum = [question]
            output = self.tokenize_parts(parts=enum, max_seq_length=max_seq_length, weights=[1] * len(enum),
                                         generation_ending_token_idx=None, keep=[2], marker_token=Q_marker_token, max_span_length=query_max_span_length)
            q.append(output)

            # return doc_tokens, doc_words, tok_to_orig_seg_map, word_pos0_mask
        # for t in [q, ans, title]:
        #     t[1] = np.array(t[1])
        # return [[torch.tensor(_) for _ in t] for t in [q, ans, title]]
        q = list(zip(*q))
        return [torch.tensor(_) for i, _ in enumerate(q)]

    def tokenize_d_segmented_dict_medqa(self, batch_text, max_seq_length, tqdm_enable=False, to_tensor=True, word=False):
        # num = 4
        d = []
        # batch_text = tqdm(batch_text, disable=len(batch_text) < 1000)
        # all_doc_segments = []
        for t in batch_text:
            # print(t)
            # if 'paragraph_cut' in t:
            #     doc = [t["paragraph_cut"].split()]
            # else:
            #     doc = [t["text_cut"].split()]
            doc = [t.split()]
            output = self.tokenize_parts(parts=doc, max_seq_length=max_seq_length, weights=[1] * len(doc),
                                         generation_ending_token_idx=None, keep=[], marker_token=D_marker_token, output_real=to_tensor,
                                         max_span_length=doc_max_span_length)
            d.append(output)
        # d[1] = np.array(d[1])
        if to_tensor:
            # return [torch.tensor(_) for _ in d]
            d = list(zip(*d))
            # return [(torch.tensor(_) if i != 100 else _) for i, _ in enumerate(d)]
            return [torch.tensor(_) for i, _ in enumerate(d)]
        else:
            return d


def _sort_by_length(ids, mask, bsize):
    if ids.size(0) <= bsize:
        return ids, mask, torch.arange(ids.size(0))

    indices = mask.sum(-1).sort().indices
    reverse_indices = indices.sort().indices

    return ids[indices], mask[indices], reverse_indices


def _split_into_batches(ids, mask, word_mask, bsize):
    batches = []
    for offset in range(0, ids.size(0), bsize):
        batches.append((ids[offset:offset + bsize], mask[offset:offset + bsize], word_mask[offset:offset + bsize]))

    return batches


def _split_into_batches_bundle(bundle, bsize):
    batches = []
    for offset in range(0, bundle[0].size(0), bsize):
        batches.append((_[offset:offset + bsize] for _ in bundle))

    return batches


def join_list(sep, lis):
    return reduce(lambda a, b: a + [sep] + b, lis)


def get_real_inputs(input_ids, word_ids, ignore_word_indices, max_seq_length):
    if len(input_ids) > max_seq_length:
        input_ids = input_ids[:max_seq_length]
        word_ids = word_ids[:max_seq_length]

    max_active_len = len(input_ids)
    input_ids += [0] * (max_seq_length - max_active_len)
    attention_mask = np.zeros((max_seq_length, max_seq_length)).astype(int)
    attention_mask[:max_active_len, :max_active_len] = 1
    active_indices = []

    i = 0
    word_ids = word_ids
    while i < max_active_len:
        # word_idx = inputs.token_to_word(i)
        word_idx = word_ids[i]
        start = i
        end = start + 1
        while end < max_active_len and word_ids[end] == word_ids[start]:
            end += 1
        attention_mask[start:end - 1, :start] = 0
        attention_mask[start:end - 1, end:] = 0
        attention_mask[:start, start:end - 1] = 0
        attention_mask[end:, start:end - 1] = 0
        if word_idx not in ignore_word_indices:
            active_indices.append(end - 1)
        i = end
    active_indices += [0] * (max_seq_length - len(active_indices))
    active_padding = [1] * len(active_indices) + [0] * (max_seq_length - len(active_indices))
    attention_mask = attention_mask.tolist()
    return input_ids, attention_mask, active_indices, active_padding

# def get_tok_avg_inputs(word_ids, ignore_word_indices, max_seq_length):
#     i = 0
#     max_active_len = len([_ for _ in word_ids if _])
#     active_indices = []
#     while i < max_active_len:
#         start = i
#         end = start + 1
#         while end < max_active_len and word_ids[end] == word_ids[start]:
#             end += 1
#         if word_ids[i] not in ignore_word_indices:
#             active_indices.append((start, end))
#         i = end
#
#     active_padding = [1] * len(active_indices) + [0] * (max_seq_length - len(active_indices))
#     # active_indices += [0] * (max_seq_length - len(active_indices))
#     return active_indices, active_padding
