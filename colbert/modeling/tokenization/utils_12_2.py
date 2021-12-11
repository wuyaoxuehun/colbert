import torch
from colbert import base_config

from transformers import BertTokenizer
from functools import reduce
from tqdm import tqdm
from colbert.base_config import part_weight, puncts, GENERATION_ENDING_TOK
from typing import List, Any
import numpy as np

from colbert.utils.func_utils import cache_decorator


def cache_function(*args, **kwargs):
    parts = kwargs.get('parts')
    weights = kwargs.get('weights')
    max_seq_length = kwargs.get('max_seq_length')
    key = [str(hash(''.join(parts)))] + [str(_) for _ in weights] + [str(max_seq_length)]
    return '-'.join(key)


class CostomTokenizer(BertTokenizer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.doc_cache = {}
        self.query_cache = {}

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

    def tokenize_q_segmented_dict(self, batch_text, max_seq_length):
        assert False
        input_ids_all = []
        attention_mask_all = []
        word_pos0_mask_all = []
        all_doc_segments = []
        for t in batch_text:
            key = hash(t['question_cut']['tok'] + t['background_cut']['tok'] + t['option_cut']['tok'])
            if key in self.query_cache:
                input_ids, attention_mask, word_pos0_mask, cur_segments = self.query_cache[key]
            else:
                question = t['question_cut']['tok'].strip().split()
                if not t['background_cut'] and not t['option_cut']:
                    enum = [question]
                else:
                    background = t['background_cut']['tok'].strip().split()
                    # background = []
                    answer = t['option_cut']['tok'].strip().split()
                    enum = [background, question, answer]

                # print([''.join(_) for _ in enum])
                # tok_to_orig_seg_map = []
                doc_tokens = [[], [], []]
                word_pos0_mask = [[], [], []]
                # segment_idx = [0, 0, 0]
                cur_segments = [[], [], []]
                for part_id, text in enumerate(enum):
                    segments = text
                    # if part_id < len(enum) - 1:
                    #     segments.append('[SEP]')
                    for word_idx, segment in enumerate(segments):
                        segment = segment.strip()
                        tokens = super().tokenize(segment)
                        if not tokens:
                            print(tokens)
                            continue
                        part_mask = part_weight[part_id] if segment not in puncts else 0
                        word_pos0_mask[part_id].extend([part_mask] + [0] * (len(tokens) - 1))
                        # word_pos0_mask.extend([part_mask] + [1] * (len(tokens) - 1))
                        doc_tokens[part_id].extend(tokens)
                        cur_segments[part_id].append(segment)
                    # print(doc_tokens)
                    # segment_idx[part_id] = len(doc_tokens)
                self.truncate_seq(tokens_list=doc_tokens, max_len=max_seq_length - 5, keep=[1, 2])
                for i in range(3):
                    part_len = len(doc_tokens[i])
                    word_pos0_mask[i] = word_pos0_mask[i][:part_len]
                    cur_segments[i] = cur_segments[i][:len([_ for _ in word_pos0_mask if _ != 0])]

                doc_tokens = ['CLS', '.'] + doc_tokens[0] + ['SEP'] + doc_tokens[1] + ['SEP'] + doc_tokens[2] + ['SEP']
                word_pos0_mask = [1, 0] + word_pos0_mask[0] + [0] + word_pos0_mask[1] + [0] + word_pos0_mask[2] + [0]
                cur_segments = ['[CLS]'] + cur_segments[0] + cur_segments[1] + cur_segments[2]

                # doc_tokens = doc_tokens[:max_seq_length - 2]
                # word_pos0_mask = word_pos0_mask[:max_seq_length - 2]
                padding_length = max_seq_length - len(doc_tokens)
                input_ids = self.convert_tokens_to_ids(doc_tokens) + [0] * padding_length
                attention_mask = [1] * (max_seq_length - padding_length) + [0] * padding_length
                # word_pos0_mask = [1] + word_pos0_mask[:(max_seq_length - padding_length)] + [0] + [0] * padding_length
                word_pos0_mask = word_pos0_mask + [0] * padding_length
                assert len(input_ids) == len(attention_mask) == len(word_pos0_mask) == max_seq_length, \
                    (len(input_ids), len(attention_mask), len(word_pos0_mask))

            input_ids_all.append(input_ids)
            attention_mask_all.append(attention_mask)
            word_pos0_mask_all.append(word_pos0_mask)

            cur_segments = cur_segments[: len(list(filter(lambda x: x != 0, word_pos0_mask)))]
            # assert len(cur_segments) == sum(word_pos0_mask)
            all_doc_segments.append(cur_segments)
            self.query_cache[key] = input_ids, attention_mask, word_pos0_mask, cur_segments

            # return doc_tokens, doc_words, tok_to_orig_seg_map, word_pos0_mask
        return torch.tensor(input_ids_all), torch.tensor(attention_mask_all), torch.tensor(word_pos0_mask_all), all_doc_segments

    def tokenize_q_allopt_segmented_dict(self, batch_examples, max_seq_length):
        input_ids_all = []
        attention_mask_all = []
        word_pos0_mask_all = []
        all_doc_segments = []
        for t in batch_examples:
            for opt in 'ABCD':
                key = hash(t['question_cut']['tok'] + t['background_cut']['tok'] + t[opt + '_cut']['tok'])
                if key in self.query_cache:
                    input_ids, attention_mask, word_pos0_mask, cur_segments = self.query_cache[key]
                else:
                    question = t['question_cut']['tok'].strip().split()
                    if not t['background_cut'] and not t['option_cut']:
                        raise NotImplemented
                        enum = [question]
                    else:
                        background = t['background_cut']['tok'].strip().split()
                        # background = []
                        answer = t[opt + '_cut']['tok'].strip().split()
                        enum = [background, question, answer]

                    # print([''.join(_) for _ in enum])
                    # tok_to_orig_seg_map = []
                    doc_tokens = [[], [], []]
                    word_pos0_mask = [[], [], []]
                    # segment_idx = [0, 0, 0]
                    cur_segments = [[], [], []]
                    for part_id, text in enumerate(enum):
                        segments = text
                        # if part_id < len(enum) - 1:
                        #     segments.append('[SEP]')
                        for word_idx, segment in enumerate(segments):
                            segment = segment.strip()
                            tokens = super().tokenize(segment)
                            if not tokens:
                                print(tokens)
                                continue
                            part_mask = part_weight[part_id] if segment not in puncts else 0
                            word_pos0_mask[part_id].extend([part_mask] + [0] * (len(tokens) - 1))
                            # word_pos0_mask.extend([part_mask] + [1] * (len(tokens) - 1))
                            doc_tokens[part_id].extend(tokens)
                            cur_segments[part_id].append(segment)
                        # print(doc_tokens)
                        # segment_idx[part_id] = len(doc_tokens)
                    self.truncate_seq(tokens_list=doc_tokens, max_len=max_seq_length - 5, keep=[1, 2])
                    for i in range(3):
                        part_len = len(doc_tokens[i])
                        word_pos0_mask[i] = word_pos0_mask[i][:part_len]
                        cur_segments[i] = cur_segments[i][:len([_ for _ in word_pos0_mask if _ != 0])]

                    doc_tokens = ['CLS', '.'] + doc_tokens[0] + ['SEP'] + doc_tokens[1] + ['SEP'] + doc_tokens[2] + ['SEP']
                    word_pos0_mask = [1, 0] + word_pos0_mask[0] + [0] + word_pos0_mask[1] + [0] + word_pos0_mask[2] + [0]
                    cur_segments = ['[CLS]'] + cur_segments[0] + cur_segments[1] + cur_segments[2]

                    # doc_tokens = doc_tokens[:max_seq_length - 2]
                    # word_pos0_mask = word_pos0_mask[:max_seq_length - 2]
                    padding_length = max_seq_length - len(doc_tokens)
                    input_ids = self.convert_tokens_to_ids(doc_tokens) + [0] * padding_length
                    attention_mask = [1] * (max_seq_length - padding_length) + [0] * padding_length
                    # word_pos0_mask = [1] + word_pos0_mask[:(max_seq_length - padding_length)] + [0] + [0] * padding_length
                    word_pos0_mask = word_pos0_mask + [0] * padding_length
                    assert len(input_ids) == len(attention_mask) == len(word_pos0_mask) == max_seq_length, \
                        (len(input_ids), len(attention_mask), len(word_pos0_mask))

                input_ids_all.append(input_ids)
                attention_mask_all.append(attention_mask)
                word_pos0_mask_all.append(word_pos0_mask)

                # cur_segments = cur_segments[: len(list(filter(lambda x: x != 0, word_pos0_mask)))]
                # assert len(cur_segments) == sum(word_pos0_mask)
                all_doc_segments.append(cur_segments)
                self.query_cache[key] = input_ids, attention_mask, word_pos0_mask, cur_segments
            # return doc_tokens, doc_words, tok_to_orig_seg_map, word_pos0_mask
        return torch.tensor(input_ids_all), torch.tensor(attention_mask_all), torch.tensor(word_pos0_mask_all), all_doc_segments

    @cache_decorator(cache_fun=cache_function)
    def tokenize_multiple_parts(self, parts=None, max_seq_length=None, weights=None, generation_ending_token_idx=None, keep=(), ladder_mask=False):
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
                word_pos0_mask[part_id].extend([part_mask] + [0] * (len(tokens) - 1))
                # word_pos0_mask.extend([part_mask] + [1] * (len(tokens) - 1))
                doc_tokens[part_id].extend(tokens)
                cur_segments[part_id].append(segment)
        self.truncate_seq(tokens_list=doc_tokens, max_len=max_seq_length - 2 - len(parts) - int((generation_ending_token_idx is not None)),
                          keep=keep)
        for i in range(nparts):
            part_len = len(doc_tokens[i])
            word_pos0_mask[i] = word_pos0_mask[i][:part_len]
            cur_segments[i] = cur_segments[i][:len([_ for _ in word_pos0_mask[i] if _ != 0])]

        if not ladder_mask:
            doc_tokens = ['[CLS]', '.'] + reduce(lambda a, b: a + ['[SEP]'] + b, doc_tokens) + ['[SEP]'] + \
                         [generation_ending_token_idx] if generation_ending_token_idx is not None else []
            word_pos0_mask = [1, 0] + reduce(lambda a, b: a + [0] + b, word_pos0_mask) + [0] + \
                             [0] if generation_ending_token_idx is not None else []
        else:
            ladders = [2 + len(doc_tokens[0] + doc_tokens[1]) + 2, len(doc_tokens[2]) + 2]
            ladders_presum = [ladders[0], ladders[0] + ladders[1]]
            attention_mask = np.array([[0] * max_seq_length for _ in range(max_seq_length)])
            attention_mask[:ladders_presum[0], :ladders_presum[0]] = 1
            attention_mask[ladders_presum[0]:ladders_presum[1], :ladders_presum[1]] = 1

            doc_tokens = ['[CLS]', '.'] + reduce(lambda a, b: a + ['[SEP]'] + b, doc_tokens) + ['[SEP]'] + ['[CLS]']
            word_pos0_mask = [1, 0] + reduce(lambda a, b: a + [0] + b, word_pos0_mask) + [0] + [0]

        cur_segments = ['[CLS]'] + sum(cur_segments, [])

        padding_length = max_seq_length - len(doc_tokens)
        input_ids = self.convert_tokens_to_ids(doc_tokens) + [0] * padding_length
        if not ladder_mask:
            attention_mask = [1] * (max_seq_length - padding_length) + [0] * padding_length


        # word_pos0_mask = [1] + word_pos0_mask[:(max_seq_length - padding_length)] + [0] + [0] * padding_length
        word_pos0_mask = word_pos0_mask + [0] * padding_length
        assert len(input_ids) == len(attention_mask) == len(word_pos0_mask) == max_seq_length, \
            (len(input_ids), len(attention_mask), len(word_pos0_mask))
        return input_ids, attention_mask, word_pos0_mask, cur_segments

    def tokenize_q_noopt_segmented_dict_(self, batch_examples, max_seq_length, answer_max_seq_lenght=64):
        input_ids_all = []
        attention_mask_all = []
        word_pos0_mask_all = []
        all_doc_segments = []

        answer_input_ids_all = []
        answer_word_pos0_mask_all = []
        answer_all_doc_segments = []
        for t in batch_examples:
            question = t['question_cut']['tok']
            background = t['background_cut']['tok']
            answer = t[t['answer'] + '_cut']['tok']
            enum = [background, question]
            input_ids, attention_mask, word_pos0_mask, cur_segments = \
                self.tokenize_multiple_parts(parts=enum, max_seq_length=max_seq_length, weights=part_weight[:len(enum)], generation_ending_token_idx=GENERATION_ENDING_TOK, keep=[])

            input_ids_all.append(input_ids)
            attention_mask_all.append(attention_mask)
            word_pos0_mask_all.append(word_pos0_mask)
            all_doc_segments.append(cur_segments)

            input_ids, attention_mask, word_pos0_mask, cur_segments = \
                self.tokenize_multiple_parts(parts=[answer], max_seq_length=answer_max_seq_lenght, weights=[1], generation_ending_token_idx=GENERATION_ENDING_TOK, keep=[])

            answer_input_ids_all.append(input_ids)
            answer_word_pos0_mask_all.append(word_pos0_mask)
            answer_all_doc_segments.append(cur_segments)
            # return doc_tokens, doc_words, tok_to_orig_seg_map, word_pos0_mask
        return (torch.tensor(input_ids_all), torch.tensor(attention_mask_all), torch.tensor(word_pos0_mask_all), all_doc_segments), \
               (torch.tensor(answer_input_ids_all), torch.tensor(answer_word_pos0_mask_all), answer_all_doc_segments)

    def tokenize_q_noopt_segmented_dict__(self, batch_examples, max_seq_length, answer_max_seq_lenght=64):
        input_ids_all = []
        attention_mask_all = []
        word_pos0_mask_all = []
        all_doc_segments = []

        answer_input_ids_all = []
        answer_word_pos0_mask_all = []
        answer_all_doc_segments = []
        for t in batch_examples:
            question = t['question_cut']['tok']
            background = t['background_cut']['tok']
            answer = t[t['answer'] + '_cut']['tok']
            enum = [background, question]
            input_ids, attention_mask, word_pos0_mask, cur_segments = \
                self.tokenize_multiple_parts(parts=enum, max_seq_length=max_seq_length, weights=part_weight[:len(enum)], generation_ending_token_idx=GENERATION_ENDING_TOK, keep=[])

            input_ids_all.append(input_ids)
            attention_mask_all.append(attention_mask)
            word_pos0_mask_all.append(word_pos0_mask)
            all_doc_segments.append(cur_segments)

            input_ids, attention_mask, word_pos0_mask, cur_segments = \
                self.tokenize_multiple_parts(parts=[answer], max_seq_length=answer_max_seq_lenght, weights=[1], generation_ending_token_idx=GENERATION_ENDING_TOK, keep=[])

            answer_input_ids_all.append(input_ids)
            answer_word_pos0_mask_all.append(word_pos0_mask)
            answer_all_doc_segments.append(cur_segments)
            # return doc_tokens, doc_words, tok_to_orig_seg_map, word_pos0_mask
        return (torch.tensor(input_ids_all), torch.tensor(attention_mask_all), torch.tensor(word_pos0_mask_all), all_doc_segments), \
               (torch.tensor(answer_input_ids_all), torch.tensor(answer_word_pos0_mask_all), answer_all_doc_segments)

    def tokenize_q_noopt_segmented_dict(self, batch_examples, max_seq_length, answer_max_seq_lenght=64):
        input_ids_all = []
        attention_mask_all = []
        word_pos0_mask_all = []
        all_doc_segments = []

        for t in batch_examples:
            question = t['question_cut']['tok']
            background = t['background_cut']['tok']
            answer = t[t['answer'] + '_cut']['tok']
            enum = [background, question, answer]
            input_ids, attention_mask, word_pos0_mask, cur_segments = \
                self.tokenize_multiple_parts(parts=enum, max_seq_length=max_seq_length, weights=part_weight[:len(enum)],
                                             generation_ending_token_idx=GENERATION_ENDING_TOK, keep=[], ladder_mask=True)

            input_ids_all.append(input_ids)
            attention_mask_all.append(attention_mask)
            word_pos0_mask_all.append(word_pos0_mask)
            all_doc_segments.append(cur_segments)
            # return doc_tokens, doc_words, tok_to_orig_seg_map, word_pos0_mask
        return (torch.tensor(input_ids_all), torch.tensor(attention_mask_all), torch.tensor(word_pos0_mask_all), all_doc_segments), None

    def tokenize_d_segmented_dict(self, batch_text, max_seq_length, tqdm_enable=False):
        input_ids_all = []
        attention_mask_all = []
        word_pos0_mask_all = []

        batch_text = tqdm(batch_text, disable=len(batch_text) < 1000)

        all_doc_segments = []
        for t in batch_text:
            doc = t['paragraph_cut']['tok']
            input_ids, attention_mask, word_pos0_mask, cur_segments = \
                self.tokenize_multiple_parts(parts=[doc], max_seq_length=max_seq_length, weights=[1], generation_ending_token_idx=GENERATION_ENDING_TOK, keep=[])

            input_ids_all.append(input_ids)
            attention_mask_all.append(attention_mask)
            word_pos0_mask_all.append(word_pos0_mask)

            # cur_segments = cur_segments[: len(list(filter(lambda x: x != 0, word_pos0_mask)))]
            all_doc_segments.append(cur_segments)
            # return doc_tokens, doc_words, tok_to_orig_seg_map, word_pos0_mask
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
