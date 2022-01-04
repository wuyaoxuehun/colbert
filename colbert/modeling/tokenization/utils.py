import torch
from pyserini.eval.evaluate_dpr_retrieval import has_answers, SimpleTokenizer

from colbert import base_config

from transformers import BertTokenizerFast, T5TokenizerFast
from functools import reduce
from tqdm import tqdm
from colbert.base_config import part_weight, puncts, GENERATION_ENDING_TOK
from typing import List, Any
import numpy as np
import string
import nltk
from colbert.utils.func_utils import cache_decorator
from conf import Q_marker_token, D_marker_token, encoder_tokenizer, CLS, SEP, pretrain_choose, answer_SEP, answer_prefix, title_prefix
from nltk.tokenize import sent_tokenize
from colbert.modeling.cpy.hello_world import get_real_inputs2
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
        self.ignore_words = set(list(string.punctuation) + [Q_marker_token, D_marker_token])
        # nlp = spacy.load('en_core_web_sm', disable=['ner', 'tagger', 'parser', 'textcat', 'tok2vec', 'attribute_ruler', 'lemmatizer'])
        # print(nlp.pipe_names)
        # self.word_tokenizer = lambda s: [str(i) for i in nlp(s)]
        # nlp = nltk.ToktokTokenizer()
        # self.word_tokenizer = nlp.tokenize
        # self.word_tokenizer = lambda s: s.split()
        self.word_tokenizer = nltk.word_tokenize

        if pretrain_choose.find("t5") != -1:
            self.ignore_words |= {CLS, SEP, answer_SEP}

        self.tokenize_parts = self.tokenize_multiple_parts_tok if not use_word else self.tokenize_multiple_parts_word

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
                                    marker_token=None, add_special_tokens=True, prefix=None, output_real=True):
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

    def tokenize_q_noopt_segmented_dict(self, batch_examples, max_seq_length, answer_max_seq_length=64, marker_token=None):
        num = 4
        q = [[] for _ in range(num)]
        ans = [[] for _ in range(num)]
        title = [[] for _ in range(num)]

        for t in batch_examples:
            question = t['question']
            if use_word:
                question = nltk.word_tokenize(question)
            enum = [question]
            output = self.tokenize_parts(parts=enum, max_seq_length=max_seq_length, weights=[1] * len(enum),
                                         generation_ending_token_idx=None, keep=[], marker_token=Q_marker_token)
            for i in range(num):
                q[i].append(output[i])

            continue

            titles, sents = self.get_relevant_title_and_sents(t, max_title_num=3, max_sents_num=0)
            # answers = t['answers'][:4] + list(titles)
            answers = t['answers'][:4]
            answers = [nltk.word_tokenize(_) for _ in answers]
            enum = answers
            output = self.tokenize_multiple_parts(parts=enum, max_seq_length=answer_max_seq_length, weights=[1] * len(enum),
                                                  generation_ending_token_idx=None, keep=[], marker_token=Q_marker_token,
                                                  add_special_tokens=False, prefix=None)
            # add_special_tokens=False, prefix=answer_prefix)
            for i in range(num):
                ans[i].append(output[i])

            enum = list(titles)
            enum = [nltk.word_tokenize(_) for _ in enum]
            output = self.tokenize_multiple_parts(parts=enum, max_seq_length=answer_max_seq_length, weights=[1] * len(enum),
                                                  generation_ending_token_idx=None, keep=[], marker_token=Q_marker_token,
                                                  add_special_tokens=False, prefix=None)  # prefix=title_prefix

            for i in range(num):
                title[i].append(output[i])

            # return doc_tokens, doc_words, tok_to_orig_seg_map, word_pos0_mask
        for t in [q, ans, title]:
            t[1] = np.array(t[1])
        return [[torch.tensor(_) for _ in t] for t in [q, ans, title]]

    def tokenize_d_segmented_dict(self, batch_text, max_seq_length, tqdm_enable=False, to_tensor=True, word=False):
        num = 4
        d = [[] for _ in range(num)]

        batch_text = tqdm(batch_text, disable=len(batch_text) < 1000)

        # all_doc_segments = []
        for t in batch_text:
            if t['title'][0] == "\"":
                t['title'] = t['title'][1:-1]
            if use_word:
                doc = [t['title_words'], t['text_words']]
            else:
                doc = [t['title'], t['text']]
            output = self.tokenize_parts(parts=doc, max_seq_length=max_seq_length, weights=[1] * 2,
                                         generation_ending_token_idx=None, keep=[], marker_token=D_marker_token, output_real=to_tensor)
            for i in range(len(output)):
                d[i].append(output[i])
        d[1] = np.array(d[1])
        if to_tensor:
            return [torch.tensor(_) for _ in d]
        else:
            return d[:3]

    def get_relevant_title_and_sents(self, t, max_title_num=2, max_sents_num=0):
        if 'pos_contexts' not in t:
            return {' '}, set()
        answers, pos_contexts = t['answers'], t['pos_contexts']
        titles, sents = set(), set()
        if len(pos_contexts) == 0 and max_sents_num > 0:
            titles.add(" ")
        for ctx in pos_contexts[:max_title_num]:
            titles.add(ctx['title'])
        for ctx in pos_contexts[:max_sents_num]:
            sents = sent_tokenize(text=ctx['text'])
            for sent in sents:
                if has_answers(text=sent, answers=answers, tokenizer=self.dpr_tokenizer, regex=False):
                    sents.add(sent)
        return titles, sents


def tensorize_triples(query_tokenizer, doc_tokenizer, queries, positives, negatives, bsize):
    assert len(queries) == len(positives) == len(negatives)
    # assert bsize is None or len(queries) % bsize == 0

    N = len(queries)
    Q_ids, Q_mask, Q_word_mask = query_tokenizer.tensorize_dict(queries)
    docs = [i + j for i, j in zip(positives, negatives)]
    docs = [j for i in docs for j in i]
    scores = torch.tensor([_['score'] for _ in docs]).view(N, base_config.pn_num, -1)

    D_ids, D_mask, D_word_mask = doc_tokenizer.tensorize_dict(docs)

    D_ids, D_mask, D_word_mask = D_ids.view(N, base_config.pn_num, -1), \
                                 D_mask.view(N, base_config.pn_num, -1), \
                                 D_word_mask.view(N, base_config.pn_num, -1)

    # Compute max among {length of i^th positive, length of i^th negative} for i \in N
    # maxlens = D_mask.sum(-1).max(0).values

    # Sort by maxlens
    # indices = maxlens.sort().indices
    # Q_ids, Q_mask = Q_ids[indices], Q_mask[indices]
    # D_ids, D_mask = D_ids[:, indices], D_mask[:, indices]

    # (positive_ids, negative_ids), (positive_mask, negative_mask) = D_ids, D_mask
    query_batches = _split_into_batches(Q_ids, Q_mask, Q_word_mask, bsize)
    # doc_batches = _split_into_batches(D_ids, D_mask, D_word_mask, bsize)
    doc_batches = _split_into_batches_bundle((D_ids, D_mask, D_word_mask, scores), bsize)
    # negative_batches = _split_into_batches(negative_ids, negative_mask, bsize)
    # print(doc_batches[0][0].size())
    batches = []
    for (q_ids, q_mask, q_word_mask), (d_ids, d_mask, d_word_mask, score) in zip(query_batches, doc_batches):
        Q = (q_ids, q_mask, q_word_mask)
        t_size = d_ids.size(0) * d_ids.size(1)
        D = (d_ids.view(t_size, -1),
             d_mask.view(t_size, -1),
             d_word_mask.view(t_size, -1),
             score.view(t_size))
        batches.append((Q, D))

    return batches


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
