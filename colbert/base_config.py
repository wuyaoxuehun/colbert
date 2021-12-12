# pretrain = "../../../pretrain/chinese-bert-wwm-ext"
# pretrain = "../../../pretrain/chinese_roberta_wwm_ext"
# pretrain = "../../../pretrain/rbt3"
# pretrain = "../../../../pretrain/macbert-large/"
# pretrain = "../../../pretrain/bert-base-uncased"
pos_num = 1
neg_num = 1
max_pos_sample = 10
max_neg_sample = 10
pn_num = pos_num + neg_num

from colbert.modeling.colbert_list import ColBERT_List

# ColBert = ColBERT_DPR
# ColBert = ColBERT_List_Weight
ColBert = ColBERT_List
# part_weight = [1, 1, 1]
part_weight = [1, 15, 20]
GENERATION_ENDING_TOK = '[unused3]'

overwrite_mask_cache = False

from zhon.hanzi import punctuation

puncts = ''.join(punctuation.split())

import os

stopwords = set()


def read_stop_words():
    print(os.getcwd())
    file = os.getcwd() + '/data/stopwords.txt'
    with open(file, encoding='utf8') as f:
        for line in f.readlines():
            stopwords.add(line.strip())


# read_stop_words()
# puncts = stopwords
import string

puncts += string.punctuation  # only filter punctuation
# read_stop_words()
# puncts = set(list(puncts)) | set(stopwords)
puncts = set(puncts)
# print(puncts)

pre_batch_num_base = 0
pre_batch_warm_up_rate = 0.5

pos_threshold = 0.5
neg_threshold = 0.3

# print(puncts)
# input()
# part_weight = [1, 5, 10]
# train_part_weight = [1, 1, 1]
# inference_part_weight = [1, 1, 1]
# import hanlp
# HanLP = hanlp.load(hanlp.pretrained.mtl.CLOSE_TOK_POS_NER_SRL_DEP_SDP_CON_ELECTRA_BASE_ZH, device=[-1])
# segmenter = lambda sent:HanLP(sent, tasks='tok')['tok/fine']
segmenter = lambda x: x

from transformers import BertTokenizerFast

# bert_tokenizer = BertTokenizerFast.from_pretrained(pretrain)
