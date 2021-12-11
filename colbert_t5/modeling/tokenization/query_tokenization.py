import torch

from colbert.modeling.tokenization.utils import _split_into_batches, _split_into_batches_bundle, CostomTokenizer
from colbert import base_config
from conf import answer_max_seqlen


class QueryTokenizer():
    def __init__(self, query_maxlen, segmenter=None):
        self.tok = CostomTokenizer.from_pretrained(base_config.pretrain)
        self.query_maxlen = query_maxlen
        self.segmenter = segmenter
        self.Q_marker_token, self.Q_marker_token_id = '[Q]', self.tok.convert_tokens_to_ids('[unused1]')
        self.cls_token, self.cls_token_id = self.tok.cls_token, self.tok.cls_token_id
        self.sep_token, self.sep_token_id = self.tok.sep_token, self.tok.sep_token_id
        self.mask_token, self.mask_token_id = self.tok.mask_token, self.tok.mask_token_id

        # assert self.Q_marker_token_id == 1 and self.mask_token_id == 103

    def tokenize(self, batch_text, add_special_tokens=False):
        assert type(batch_text) in [list, tuple], (type(batch_text))

        tokens = [self.tok.tokenize(x, add_special_tokens=False) for x in batch_text]

        if not add_special_tokens:
            return tokens

        prefix, suffix = [self.cls_token, self.Q_marker_token], [self.sep_token]
        tokens = [prefix + lst + suffix + [self.mask_token] * (self.query_maxlen - (len(lst) + 3)) for lst in tokens]

        return tokens

    def encode(self, batch_text, add_special_tokens=False):
        assert type(batch_text) in [list, tuple], (type(batch_text))

        ids = self.tok(batch_text, add_special_tokens=False)['input_ids']

        if not add_special_tokens:
            return ids

        prefix, suffix = [self.cls_token_id, self.Q_marker_token_id], [self.sep_token_id]
        ids = [prefix + lst + suffix + [self.mask_token_id] * (self.query_maxlen - (len(lst) + 3)) for lst in ids]

        return ids

    def tensorize(self, batch_text, bsize=None):
        assert type(batch_text) in [list, tuple], (type(batch_text))

        # add placehold for the [Q] marker
        batch_text = ['. ' + x for x in batch_text]

        # obj = self.tok(batch_text, padding='max_length', truncation=True,
        #                return_tensors='pt', max_length=self.query_maxlen)
        obj = self.tok.tokenize(batch_text, self.segmenter, self.query_maxlen)
        # ids, mask = obj['input_ids'], obj['attention_mask']
        ids, mask, word_mask = obj
        # postprocess for the [Q] marker and the [MASK] augmentation
        ids[:, 1] = self.Q_marker_token_id
        ids[ids == 0] = self.mask_token_id

        if bsize:
            batches = _split_into_batches(ids, mask, word_mask, bsize)
            return batches

        return ids, mask, word_mask

    def tensorize_dict(self, batch_text, bsize=None, output_tokens=False):
        assert type(batch_text) in [list, tuple], (type(batch_text))
        # obj = self.tok(batch_text, padding='max_length', truncation=True,
        #                return_tensors='pt', max_length=self.query_maxlen)
        # obj = self.tok.tokenize_q_dict(batch_text, self.segmenter, self.query_maxlen)
        obj = self.tok.tokenize_q_segmented_dict(batch_text, self.query_maxlen)
        # ids, mask = obj['input_ids'], obj['attention_mask']
        ids, mask, word_mask, tokens = obj
        # postprocess for the [Q] marker and the [MASK] augmentation
        ids[:, 1] = self.Q_marker_token_id
        ids[ids == 0] = self.mask_token_id

        if bsize:
            if output_tokens:
                batches = _split_into_batches_bundle((ids, mask, word_mask, tokens), bsize)
            else:
                batches = _split_into_batches(ids, mask, word_mask, bsize)
            return batches
        if output_tokens:
            return ids, mask, word_mask, tokens
        return ids, mask, word_mask

    def tensorize_allopt_dict(self, batch_text, bsize=None, output_tokens=False):
        assert type(batch_text) in [list, tuple], (type(batch_text))
        # obj = self.tok(batch_text, padding='max_length', truncation=True,
        #                return_tensors='pt', max_length=self.query_maxlen)
        # obj = self.tok.tokenize_q_dict(batch_text, self.segmenter, self.query_maxlen)
        obj = self.tok.tokenize_q_allopt_segmented_dict(batch_text, self.query_maxlen)
        # ids, mask = obj['input_ids'], obj['attention_mask']
        ids, mask, word_mask, tokens = obj
        # postprocess for the [Q] marker and the [MASK] augmentation
        ids[:, 1] = self.Q_marker_token_id
        # ids[ids == 0] = self.mask_token_id

        if bsize:
            if output_tokens:
                batches = _split_into_batches_bundle((ids, mask, word_mask, tokens), bsize)
            else:
                batches = _split_into_batches(ids, mask, word_mask, bsize)
            return batches
        if output_tokens:
            return ids, mask, word_mask, tokens
        return ids, mask, word_mask

    def tensorize_noopt_dict(self, batch_text, bsize=None, output_tokens=False):
        assert type(batch_text) in [list, tuple], (type(batch_text))
        # obj = self.tok(batch_text, padding='max_length', truncation=True,
        #                return_tensors='pt', max_length=self.query_maxlen)
        # obj = self.tok.tokenize_q_dict(batch_text, self.segmenter, self.query_maxlen)
        obj = self.tok.tokenize_q_noopt_segmented_dict(batch_text, self.query_maxlen, answer_max_seq_length=answer_max_seqlen)
        # ids, mask = obj['input_ids'], obj['attention_mask']
        ids, mask, word_mask, tokens = obj[0]
        answer_ids, ans_mask, answer_word_mask, answer_tokens = obj[1]

        # postprocess for the [Q] marker and the [MASK] augmentation
        ids[:, 1] = self.Q_marker_token_id
        answer_ids[:, 1] = self.Q_marker_token_id
        # print(self.tok.convert_ids_to_tokens(ids[0]))
        # print(self.tok.convert_ids_to_tokens(answer_ids[0]))
        # input()
        # ids[answer_ids == 0] = -100
        # answer_ids[answer_ids == 0] = -100
        # ids[ids == 0] = self.mask_token_id
        if bsize:
            if output_tokens:
                batches = _split_into_batches_bundle((ids, mask, word_mask, tokens), bsize)
            else:
                batches = _split_into_batches(ids, mask, word_mask, bsize)
            return batches
        if output_tokens:
            return ids, mask, word_mask, tokens
        # return (ids, mask), answer_ids
        return (ids, mask, word_mask), (answer_ids, ans_mask, answer_word_mask)