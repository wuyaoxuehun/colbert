import torch
import string
from transformers import BertTokenizerFast
from zhon.hanzi import punctuation


class CostomTokenizer:
    def __init__(self, args):
        # self.dpr_tokenizer = SimpleTokenizer()
        self.args = args
        self.tokenizer = BertTokenizerFast.from_pretrained(self.args.pretrain)
        puncts = set(punctuation + string.punctuation)
        self.ignore_words = {"[SEP]"} | puncts
        # self.word_tokenizer = nltk.word_tokenize
        # if pretrain_choose.find("t5") != -1:
        #     self.ignore_words |= {CLS, SEP, answer_SEP}
        self.tokenize_q_noopt_segmented_dict = self.tokenize_q
        self.tokenize_d_segmented_dict = self.tokenize_d
        self.tokenize_parts = self.tokenize_seqs
        self.tokenizer.add_special_tokens({"additional_special_tokens": [f"[unused1]", "[unused2]"]})
        self.q_makers = "[unused1]"
        self.d_makers = "[unused2]"

    def tokenize_seqs(self, seqs=None, max_seq_length=None, is_query=False, is_indexing=False):
        # words = CLS + (self.q_makers if is_query else self.d_makers) + parts + SEP
        input_seqs = ["[CLS]" + ((self.q_makers if is_query else self.d_makers) + _ + "[SEP]") for _ in seqs]
        tokens = [self.tokenizer.tokenize(_)[:max_seq_length] for _ in input_seqs]
        active_padding = [[(1 if j not in self.ignore_words else 0) for j in i] + [0] * (max_seq_length - len(i))
                          for i in tokens]
        input_ids = [self.tokenizer.convert_tokens_to_ids(_) + [0] * (max_seq_length - len(_)) for _ in tokens]
        attention_mask = [[1] * len(_) + [0] * (max_seq_length - len(_)) for _ in tokens]
        return input_ids, attention_mask, active_padding

    def tokenize_q(self, batch_examples):
        questions = []
        for t in batch_examples:
            question = t['question']
            questions.append(question)
        q = self.tokenize_parts(seqs=questions, max_seq_length=self.args.query_maxlen, is_query=True)
        return [torch.tensor(_) for _ in q]

    def tokenize_d(self, batch_text, to_tensor=True):
        docs = []
        for t in batch_text:
            doc = t
            docs.append(doc)
        d = self.tokenize_parts(seqs=docs, max_seq_length=self.args.doc_maxlen, is_query=False, is_indexing=not to_tensor)
        if to_tensor:
            return [torch.tensor(_) for i, _ in enumerate(d)]
        else:
            return d
