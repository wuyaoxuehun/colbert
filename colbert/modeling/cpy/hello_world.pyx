from __future__ import division
import numpy as np
cimport numpy as np
cimport cython
from cpython cimport array
import array

def get_real_inputs2(array.array input_ids, array.array word_ids, ignore_word_indices, int max_seq_length):
    input_ids = input_ids[:max_seq_length]
    word_ids = word_ids[:max_seq_length]

    max_active_len = len(input_ids)
    input_ids += [0] * (max_seq_length - max_active_len)
    active_indices = []

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

    active_padding = [1] * len(active_indices) + [0] * (max_seq_length - len(active_indices))
    active_indices += [0] * (max_seq_length - len(active_indices))
    # attention_mask = attention_mask.tolist()
    return input_ids, attention_mask, active_indices, active_padding
