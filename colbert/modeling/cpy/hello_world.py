# def get_real_inputs2(int *input_ids, int *word_ids, set ignore_word_indices, max_seq_length):
#     if len(input_ids) > max_seq_length:
#         input_ids = input_ids[:max_seq_length]
#         word_ids = word_ids[:max_seq_length]
#
#     end_ids = [0] * max_seq_length
#     end_sets = set()
#     cur_end = len(word_ids) - 1
#     for idx, word_id in reversed(list(enumerate(word_ids))):
#         if word_id != word_ids[cur_end]:
#             cur_end = idx
#         end_ids[idx] = cur_end
#         end_sets.add(cur_end)
#
#     max_active_len = len(input_ids)
#     input_ids += [0] * (max_seq_length - max_active_len)
#     # attention_mask = np.zeros((max_seq_length, max_seq_length)).astype(int)
#     attention_mask = [[0] * max_seq_length for _ in range(max_seq_length)]
#     # attention_mask[:max_active_len, :max_active_len] = 0
#     # active_indices = []
#
#     start_id = 0
#     # print(end_sets)
#     for i in range(max_active_len):
#         if end_ids[i] != i:
#             for j in range(start_id, end_ids[i] + 1):
#                 attention_mask[i][j] = 1
#         else:
#             t = list(end_sets | set(list(range(start_id, i + 1))))
#             for j in t:
#                 attention_mask[i][j] = 1
#             start_id = i + 1
#     # active_indices = list(end_sets)
#
#     active_indices = list(end_sets - set([end_ids[_] for _ in ignore_word_indices]))
#     active_indices += [0] * (max_seq_length - len(active_indices))
#     active_padding = [1] * len(active_indices) + [0] * (max_seq_length - len(active_indices))
#     # attention_mask = attention_mask.tolist()
#     return input_ids, attention_mask, active_indices, active_padding
import numpy as np
import torch


def get_real_inputs2(input_ids, word_ids, ignore_word_indices, max_seq_length):
    # if len(input_ids) > max_seq_length:
    input_ids = input_ids[:max_seq_length]
    word_ids = word_ids[:max_seq_length]

    max_active_len = len(input_ids)
    input_ids += [0] * (max_seq_length - max_active_len)
    # attention_mask = np.zeros((max_seq_length, max_seq_length)).astype(int)
    # attention_mask[:max_active_len, :max_active_len] = 1
    active_indices = []

    # i = 0
    # word_ids = word_ids
    # while i < max_active_len:
    #     # word_idx = inputs.token_to_word(i)
    #     word_idx = word_ids[i]
    #     start = i
    #     end = start + 1
    #     while end < max_active_len and word_ids[end] == word_ids[start]:
    #         end += 1
    #     attention_mask[start:end - 1, :start] = 0
    #     attention_mask[start:end - 1, end:] = 0
    #     attention_mask[:start, start:end - 1] = 0
    #     attention_mask[end:, start:end - 1] = 0
    #     if word_idx not in ignore_word_indices:
    #         active_indices.append(end - 1)
    #     i = end

    attention_mask = np.zeros((max_seq_length, max_seq_length), dtype=int)

    attention_mask[:max_active_len, :max_active_len] = 1
    i = 0
    while i < max_active_len:
        start = i
        end = start + 1
        while end < max_active_len and word_ids[end] == word_ids[start]:
            end += 1
        # print(start, end)
        attention_mask[start:end - 1, :start] = 0
        attention_mask[start:end - 1, end:] = 0
        attention_mask[:start, start:end - 1] = 0
        attention_mask[end:, start:end - 1] = 0
        if word_ids[i] not in ignore_word_indices:
            active_indices.append(end - 1)
        i = end
        # input(attention_mask)

    # active_indices = [_ for _ in active_indices if word_ids[_] not in ignore_word_indices]

    active_padding = [1] * len(active_indices) + [0] * (max_seq_length - len(active_indices))
    active_indices += [0] * (max_seq_length - len(active_indices))
    # attention_mask = attention_mask.tolist()
    return input_ids, attention_mask, active_indices, active_padding


def get_real_inputs3(input_ids, word_ids, ignore_word_indices, max_seq_length):
    if len(input_ids) > max_seq_length:
        input_ids = input_ids[:max_seq_length]
        word_ids = word_ids[:max_seq_length]

    end_sets_list = []
    # end_idx = []
    # for idx in range(len(word_ids)-1):
    #     if word_ids[idx + 1] != word_ids[idx]:
    #         end_sets_list.append(idx)
    # end_sets_list.append(len(word_ids) - 1)

    max_active_len = len(input_ids)
    input_ids += [0] * (max_seq_length - max_active_len)
    attention_mask = np.zeros((max_seq_length, max_seq_length), dtype=int)
    active_indices = []
    i = 0
    while i < max_active_len:
        # word_idx = inputs.token_to_word(i)
        # word_idx =
        start = i
        end = start + 1
        while end < max_active_len and word_ids[end] == word_ids[start]:
            end += 1
        attention_mask[start: end, start:end] = 1
        # attention_mask[end - 1, end_sets_list] = 1
        # if word_ids[i] not in ignore_word_indices:
        #     active_indices.append(end - 1)

        end_sets_list.append(end - 1)
        i = end
    # print(end_sets_list)
    attention_mask[np.ix_(end_sets_list, end_sets_list)] = 1
    active_indices = [_ for _ in end_sets_list if word_ids[_] not in ignore_word_indices]

    # active_indices = list(end_sets - set([end_ids[_] for _ in ignore_word_indices]))
    active_padding = [1] * len(active_indices) + [0] * (max_seq_length - len(active_indices))
    active_indices += [0] * (max_seq_length - len(active_indices))
    # attention_mask = attention_mask.tolist()
    return input_ids, attention_mask, active_indices, active_padding


def get_tok_avg_inputs(word_ids, ignore_word_indices, max_span_length):
    i = 0
    max_active_len = len([_ for _ in word_ids if _])
    active_indices = []
    while i < max_active_len:
        start = i
        end = start + 1
        while end < max_active_len and word_ids[end] == word_ids[start]:
            end += 1
        if word_ids[i] not in ignore_word_indices:
            active_indices.append((start, end))
        i = end

    active_indices = active_indices[:max_span_length]
    active_padding = [1] * len(active_indices) + [0] * (max_span_length - len(active_indices))
    active_indices = active_indices + [(0, 1)] * (max_span_length - len(active_indices))
    # active_indices += [0] * (max_seq_length - len(active_indices))
    return active_indices, active_padding

# def span_mean(Q, spans):
#     output = torch.zeros_like(Q)
#     for i, t_spans in enumerate(spans):
#         for j, (s, e) in enumerate(t_spans):
#             output[i, j, ...] = Q[i, s:e, ...].mean(0)
#     return output
