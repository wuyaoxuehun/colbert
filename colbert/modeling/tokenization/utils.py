import torch
from colbert import base_config

from transformers import BertTokenizer

from tqdm import tqdm
from colbert.base_config import part_weight, puncts
from typing import List, Any
import numpy as np


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

    # def tokenize(self, batch_text, segmenter, max_seq_length):
    #     input_ids_all = []
    #     attention_mask_all = []
    #     word_pos0_mask_all = []
    #     if len(batch_text) > 100:
    #         batch_text = tqdm(batch_text)
    #     for text in batch_text:
    #         segments = segmenter(text)
    #         # tok_to_orig_seg_map = []
    #         doc_tokens = []
    #         word_pos0_mask = []
    #         for word_idx, segment in enumerate(segments):
    #             segment = segment.strip()
    #             tokens = super().tokenize(segment)
    #             if not tokens:
    #                 continue
    #             # tok_to_orig_seg_map.extend([word_idx] * len(tokens))
    #             word_pos0_mask.extend([1] + [0] * (len(tokens) - 1))
    #             doc_tokens.extend(tokens)
    #
    #         doc_tokens = doc_tokens[:max_seq_length - 2]
    #         word_pos0_mask = word_pos0_mask[:max_seq_length - 2]
    #         padding_length = max_seq_length - len(doc_tokens) - 2
    #         input_ids = self.convert_tokens_to_ids(['[CLS]'] + doc_tokens + ['[SEP]']) + [0] * padding_length
    #         attention_mask = [1] * (max_seq_length - padding_length) + [0] * padding_length
    #         # word_pos0_mask = [1] + word_pos0_mask[:(max_seq_length - padding_length)] + [0] + [0] * padding_length
    #         word_pos0_mask = [1] + word_pos0_mask + [0] + [0] * padding_length
    #         if len(input_ids) != max_seq_length or len(attention_mask) != max_seq_length or len(word_pos0_mask) != max_seq_length:
    #             print(text)
    #             print(input_ids)
    #             print(attention_mask)
    #             print(word_pos0_mask)
    #         assert len(input_ids) == len(attention_mask) == len(word_pos0_mask) == max_seq_length, \
    #             (len(input_ids), len(attention_mask), len(word_pos0_mask))
    #
    #         input_ids_all.append(input_ids)
    #         attention_mask_all.append(attention_mask)
    #         word_pos0_mask_all.append(word_pos0_mask)
    #
    #         # return doc_tokens, doc_words, tok_to_orig_seg_map, word_pos0_mask
    #     return torch.tensor(input_ids_all), torch.tensor(attention_mask_all), torch.tensor(word_pos0_mask_all)
    #
    # def tokenize_q_dict(self, batch_text, segmenter, max_seq_length):
    #     input_ids_all = []
    #     attention_mask_all = []
    #     word_pos0_mask_all = []
    #     for t in batch_text:
    #         background = t['background']
    #         question = t['question']
    #         answer_all = t['answer_all']
    #         # tok_to_orig_seg_map = []
    #         doc_tokens = []
    #         word_pos0_mask = []
    #         # part_weight = part
    #         segment_idx = [0, 0, 0]
    #         for part_id, text in enumerate([background, question, answer_all]):
    #             segments = segmenter(text)
    #
    #             for word_idx, segment in enumerate(segments):
    #                 segment = segment.strip()
    #                 tokens = super().tokenize(segment)
    #                 if not tokens:
    #                     continue
    #                 # tok_to_orig_seg_map.extend([word_idx] * len(tokens))
    #                 print(part_weight)
    #                 input()
    #
    #                 part_mask = part_weight[part_id] if segment not in puncts else 0
    #                 word_pos0_mask.extend([part_mask + 1] + [0] * (len(tokens) - 1))
    #                 doc_tokens.extend(tokens)
    #             segment_idx[part_id] = len(doc_tokens)
    #
    #         # for i in range(2, len(word_pos0_mask)):
    #         #     if word_pos0_mask[i] == 0:
    #         #         continue
    #         #     if 0 <= i <segment_idx[0]:
    #         #         word_pos0_mask[i] *= len(doc_tokens) / segment_idx[0]
    #         #     elif segment_idx[0] <= i < segment_idx[1]:
    #         #         word_pos0_mask[i] *= len(doc_tokens) / segment_idx[1]
    #         #     elif segment_idx[1] <= i < segment_idx[2]:
    #         #         word_pos0_mask[i] *= len(doc_tokens) / segment_idx[2]
    #
    #         doc_tokens.insert(1, '.')  ## spare for special Q token
    #         word_pos0_mask.insert(1, 1)
    #
    #         doc_tokens = doc_tokens[:max_seq_length - 2]
    #         word_pos0_mask = word_pos0_mask[:max_seq_length - 2]
    #         padding_length = max_seq_length - len(doc_tokens) - 2
    #         input_ids = self.convert_tokens_to_ids(['[CLS]'] + doc_tokens + ['[SEP]']) + [0] * padding_length
    #         attention_mask = [1] * (max_seq_length - padding_length) + [0] * padding_length
    #         # word_pos0_mask = [1] + word_pos0_mask[:(max_seq_length - padding_length)] + [0] + [0] * padding_length
    #         word_pos0_mask = [1] + word_pos0_mask + [0] + [0] * padding_length
    #         if len(input_ids) != max_seq_length or len(attention_mask) != max_seq_length or len(word_pos0_mask) != max_seq_length:
    #             print(text)
    #             print(input_ids)
    #             print(attention_mask)
    #             print(word_pos0_mask)
    #         assert len(input_ids) == len(attention_mask) == len(word_pos0_mask) == max_seq_length, \
    #             (len(input_ids), len(attention_mask), len(word_pos0_mask))
    #
    #         input_ids_all.append(input_ids)
    #         attention_mask_all.append(attention_mask)
    #         word_pos0_mask_all.append(word_pos0_mask)
    #
    #         # return doc_tokens, doc_words, tok_to_orig_seg_map, word_pos0_mask
    #     return torch.tensor(input_ids_all), torch.tensor(attention_mask_all), torch.tensor(word_pos0_mask_all)

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

                cur_segments = cur_segments[: len(list(filter(lambda x: x != 0, word_pos0_mask)))]
                # assert len(cur_segments) == sum(word_pos0_mask)
                all_doc_segments.append(cur_segments)
                self.query_cache[key] = input_ids, attention_mask, word_pos0_mask, cur_segments
            # return doc_tokens, doc_words, tok_to_orig_seg_map, word_pos0_mask
        return torch.tensor(input_ids_all), torch.tensor(attention_mask_all), torch.tensor(word_pos0_mask_all), all_doc_segments

    def tokenize_d_segmented_dict(self, batch_text, max_seq_length, tqdm_enable=False):
        input_ids_all = []
        attention_mask_all = []
        word_pos0_mask_all = []

        batch_text = tqdm(batch_text, disable=len(batch_text) < 1000)

        all_doc_segments = []
        for t in batch_text:
            # print(''.join(t['paragraph_cut']['tok'].strip().split()))
            key = hash(t['paragraph_cut']['tok'])
            if key in self.doc_cache:
                input_ids, attention_mask, word_pos0_mask, cur_segments = self.query_cache[key]
            else:
                segments = t['paragraph_cut']['tok'].strip().split()
                doc_tokens = []
                word_pos0_mask = []
                cur_segments = []
                for word_idx, segment in enumerate(segments):
                    segment = segment.strip()
                    tokens = super().tokenize(segment)
                    if not tokens:
                        continue
                    # tok_to_orig_seg_map.extend([word_idx] * len(tokens))
                    part_mask = 1 if segment not in puncts else 0
                    # word_pos0_mask.extend([part_mask + 1] + [0] * (len(tokens) - 1))
                    word_pos0_mask.extend([part_mask] + [0] * (len(tokens) - 1))
                    # word_pos0_mask.extend([part_mask] + [1] * (len(tokens) - 1))
                    doc_tokens.extend(tokens)
                    cur_segments.append(segment)

                doc_tokens.insert(0, '.')  ## spare for special Q token
                word_pos0_mask.insert(0, 0)  # special token Q or D also matters

                doc_tokens = doc_tokens[:max_seq_length - 2]
                word_pos0_mask = word_pos0_mask[:max_seq_length - 2]
                padding_length = max_seq_length - len(doc_tokens) - 2
                input_ids = self.convert_tokens_to_ids(['[CLS]'] + doc_tokens + ['[SEP]']) + [0] * padding_length
                attention_mask = [1] * (max_seq_length - padding_length) + [0] * padding_length
                # word_pos0_mask = [1] + word_pos0_mask[:(max_seq_length - padding_length)] + [0] + [0] * padding_length
                word_pos0_mask = [1] + word_pos0_mask + [0] + [0] * padding_length
                assert len(input_ids) == len(attention_mask) == len(word_pos0_mask) == max_seq_length, \
                    (len(input_ids), len(attention_mask), len(word_pos0_mask))

            input_ids_all.append(input_ids)
            attention_mask_all.append(attention_mask)
            word_pos0_mask_all.append(word_pos0_mask)

            cur_segments = ['[CLS]', '.'] + cur_segments
            cur_segments = cur_segments[: sum(word_pos0_mask)]
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
