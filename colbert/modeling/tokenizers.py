import torch
import string
from transformers import BertTokenizerFast
from zhon.hanzi import punctuation


class CostomTokenizer:
    def __init__(self, args):
        # self.dpr_tokenizer = SimpleTokenizer()
        self.args = args
        self.query_maxlen = args.dense_training_args.query_maxlen
        self.doc_maxlen = args.dense_training_args.doc_maxlen
        self.ce_maxlen = args.ce_training_args.max_seq_len
        self.pretrain = args.dense_training_args.pretrain
        self.tokenizer = BertTokenizerFast.from_pretrained(self.pretrain)
        puncts = set(punctuation + string.punctuation)
        self.ignore_words = {"[SEP]"} | puncts
        self.tokenize_q_noopt_segmented_dict = self.tokenize_q
        self.tokenize_d_segmented_dict = self.tokenize_d
        if not args.enable_multiview:
            self.tokenize_parts = self.tokenize_seqs
            self.tokenizer.add_special_tokens({"additional_special_tokens": [f"[unused1]", "[unused2]"]})
            self.q_makers = "[unused1]"
            self.d_makers = "[unused2]"
        else:
            self.q_view, self.d_view = args.dense_multiview_args.q_view, args.dense_multiview_args.d_view
            self.tokenize_parts = self.tokenize_multiview
            self.tokenizer.add_special_tokens({"additional_special_tokens": [f"[unused{i}]" for i in range(1, self.q_view + self.d_view + 1)]})
            self.q_makers = ''.join([f"[unused{i}]" for i in range(1, self.q_view + 1)])
            self.d_makers = ''.join([f"[unused{i}]" for i in range(self.q_view + 1, self.d_view + self.d_view + 1)])

    def tokenize_seqs(self, seqs=None, max_seq_length=None, is_query=False, is_indexing=False):
        # words = CLS + (self.q_makers if is_query else self.d_makers) + parts + SEP
        input_seqs = ["[CLS]" + ((self.q_makers if is_query else self.d_makers) + _ + "[SEP]") for _ in seqs]
        tokens = [self.tokenizer.tokenize(_)[:max_seq_length] for _ in input_seqs]
        active_padding = [[(1 if j not in self.ignore_words else 0) for j in i] + [0] * (max_seq_length - len(i))
                          for i in tokens]
        input_ids = [self.tokenizer.convert_tokens_to_ids(_) + [0] * (max_seq_length - len(_)) for _ in tokens]
        attention_mask = [[1] * len(_) + [0] * (max_seq_length - len(_)) for _ in tokens]
        return input_ids, attention_mask, active_padding

    def tokenize_multiview(self, seqs=None, max_seq_length=None, is_query=False, is_indexing=False):
        words = [((self.q_makers if is_query else self.d_makers) + _ + "[SEP]") for _ in seqs]
        inputs = self.tokenizer.batch_encode_plus(words,
                                                  padding='max_length',
                                                  # padding='longest' if not is_indexing else "max_length",
                                                  max_length=max_seq_length,
                                                  truncation=True,
                                                  add_special_tokens=False)

        input_ids = inputs['input_ids']
        # print(self.batch_decode(input_ids))
        # input()
        attention_mask = inputs['attention_mask']
        view_num = self.q_view if is_query else self.d_view
        # active_padding = [[1] * view_num + [0] * (len(input_ids[0]) - view_num)] * len(input_ids)
        active_padding = [[1] * view_num] * len(input_ids)
        # active_len = sum(attention_mask)
        # active_indices = list(range(active_len)) + [0] * (max_seq_length - active_len)
        # active_padding = [1] * active_len + [0] * (max_seq_length - active_len)
        # word_ids = inputs.word_ids()
        # return input_ids, attention_mask, active_indices, active_padding
        return input_ids, attention_mask, active_padding

    def tokenize_ce(self, qp_seqs):
        words = ["[CLS]" + q + "[SEP]" + p + "[SEP]" for q, p in qp_seqs]
        inputs = self.tokenizer.batch_encode_plus(words,
                                                  padding='max_length',
                                                  # padding='longest' if not is_indexing else "max_length",
                                                  max_length=self.ce_maxlen,
                                                  truncation=True,
                                                  add_special_tokens=False)

        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        return input_ids, attention_mask

    def tokenize_q(self, batch_examples):
        questions = []
        for t in batch_examples:
            question = t['question']
            questions.append(question)
        q = self.tokenize_parts(seqs=questions, max_seq_length=self.query_maxlen, is_query=True)
        return [torch.tensor(_) for _ in q]

    def tokenize_d(self, batch_text, to_tensor=True):
        docs = []
        for t in batch_text:
            doc = t
            docs.append(doc)
        d = self.tokenize_parts(seqs=docs, max_seq_length=self.doc_maxlen, is_query=False, is_indexing=not to_tensor)
        if to_tensor:
            return [torch.tensor(_) for i, _ in enumerate(d)]
        else:
            return d
