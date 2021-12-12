import torch
from colbert import base_config

from transformers import BertTokenizerFast, T5TokenizerFast
from functools import reduce
from tqdm import tqdm
from colbert.base_config import part_weight, puncts, GENERATION_ENDING_TOK
from typing import List, Any
import numpy as np

from colbert.utils.func_utils import cache_decorator
from conf import Q_marker_token, D_marker_token, encoder_tokenizer, CLS, SEP, pretrain_choose


def cache_function(*args, **kwargs):
    parts = kwargs.get('parts')
    weights = kwargs.get('weights')
    max_seq_length = kwargs.get('max_seq_length')
    if weights is None:
        weights = [1] * len(parts)
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
    def tokenize_multiple_parts(self, parts=None, max_seq_length=None, weights=None, generation_ending_token_idx=None, keep=(), marker_token=None):
        assert len(weights) == len(parts)
        if weights is None:
            weights = [1] * len(parts)

        nparts = len(parts)
        input_sequence = CLS + marker_token + SEP.join(parts) + SEP
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

    def tokenize_q_noopt_segmented_dict(self, batch_examples, max_seq_length, answer_max_seq_length=64, marker_token=None):
        input_ids_all = []
        attention_mask_all = []
        word_pos0_mask_all = []
        all_doc_segments = []

        ans_input_ids_all = []
        ans_attention_mask_all = []
        ans_word_pos0_mask_all = []
        ans_all_doc_segments = []

        for t in batch_examples:
            question = t['question']
            enum = [question]
            input_ids, attention_mask, word_pos0_mask, cur_segments = \
                self.tokenize_multiple_parts(parts=enum, max_seq_length=max_seq_length, weights=[1] * len(enum),
                                             generation_ending_token_idx=None, keep=[], marker_token=Q_marker_token)

            input_ids_all.append(input_ids)
            attention_mask_all.append(attention_mask)
            word_pos0_mask_all.append(word_pos0_mask)
            all_doc_segments.append(cur_segments)

            answers = t['answers'][:4]
            enum = answers
            input_ids, attention_mask, word_pos0_mask, cur_segments = \
                self.tokenize_multiple_parts(parts=enum, max_seq_length=answer_max_seq_length, weights=[1] * len(enum),
                                             generation_ending_token_idx=None, keep=[], marker_token=Q_marker_token)

            ans_input_ids_all.append(input_ids)
            ans_attention_mask_all.append(attention_mask)
            ans_word_pos0_mask_all.append(word_pos0_mask)
            ans_all_doc_segments.append(cur_segments)

            # return doc_tokens, doc_words, tok_to_orig_seg_map, word_pos0_mask
        return (torch.tensor(input_ids_all), torch.tensor(attention_mask_all), torch.tensor(word_pos0_mask_all), all_doc_segments), \
               (torch.tensor(ans_input_ids_all), torch.tensor(ans_attention_mask_all), torch.tensor(ans_word_pos0_mask_all), ans_all_doc_segments)

    def tokenize_d_segmented_dict(self, batch_text, max_seq_length, tqdm_enable=False, to_tensor=True, marker=None):
        input_ids_all = []
        attention_mask_all = []
        word_pos0_mask_all = []

        batch_text = tqdm(batch_text, disable=len(batch_text) < 1000)

        all_doc_segments = []
        for t in batch_text:
            if t['title'][0] == "\"":
                t['title'] = t['title'][1:-1]
            doc = [t['title'], t['text']]
            input_ids, attention_mask, word_pos0_mask, cur_segments = \
                self.tokenize_multiple_parts(parts=doc, max_seq_length=max_seq_length, weights=[1] * 2, generation_ending_token_idx=None, keep=[], marker_token=D_marker_token)

            input_ids_all.append(input_ids)
            attention_mask_all.append(attention_mask)
            word_pos0_mask_all.append(word_pos0_mask)

            # cur_segments = cur_segments[: len(list(filter(lambda x: x != 0, word_pos0_mask)))]
            all_doc_segments.append(cur_segments)
            # return doc_tokens, doc_words, tok_to_orig_seg_map, word_pos0_mask
        if not to_tensor:
            return input_ids_all, attention_mask_all, word_pos0_mask_all, all_doc_segments
        return torch.tensor(input_ids_all), torch.tensor(attention_mask_all), torch.tensor(word_pos0_mask_all), all_doc_segments


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
