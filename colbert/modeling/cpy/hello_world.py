def get_tok_avg_inputs(word_ids, ignore_word_indices, max_span_length):
    """
    获得有效词的开始和结束，只用于词级别模型，速度比原生python快
    :param word_ids:
    :param ignore_word_indices:
    :param max_span_length:
    :return:
    """
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
