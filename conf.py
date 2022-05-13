import random
import string
from pathlib import Path
import numpy as np
from zhon.hanzi import punctuation

# pretrain_base = Path("/home/1108037/awu/experiments/pretrain/")
# pretrain_base = Path("../../../pretrain/")
# from corpus_cb import load_all_paras_geo, load_all_paras_medqa
from proj_utils.dureader_utils import load_all_paras_dureader

pretrain_base = Path("/home/awu/experiments/pretrain")
from transformers import BertModel, BertConfig, BertTokenizerFast, T5TokenizerFast, T5Config, T5Model, T5EncoderModel, T5ForConditionalGeneration, BertForMaskedLM

from colbert.training.losses import BiEncoderNllLoss, listMLE, lambdaLoss, listMLEWeighted, listnet_loss, BiEncoderNllLossTri

# pretrain_base = Path("/home/awu/experiments/pretrain/")

# pretrain_choose = 'bert-base-en_uncased'
pretrain_choose = 'bert'
# pretrain_choose = 't5_large'
# pretrain_choose = 't5_base'

pretrain_map = {
    'ernie_c3': pretrain_base / "ernie_c3",
    'ernie': pretrain_base / "ernie_pytorch_ch",
    'bert_c3': pretrain_base / "bert_wwm_ext_c3",
    'bert': pretrain_base / "chinese-bert-wwm-ext",
    # 'bert': pretrained_model,
    'roberta-large': "c3pretrain/chinese_roberta_wwm_large_ext",
    'macbert': "../../pretrain/macbert-large",
    'bert-base-en_uncased': pretrain_base / "bert-base-uncased",
    "t5_large": pretrain_base / "t5_large",
    # "t5_base": pretrain_base / "t5_base"
    "t5_base": pretrain_base / "t5_base",
    # "t5_base": pretrain_base / "t5_base_1_1"
}

pretrain = str(pretrain_map[pretrain_choose])

index_type = "cblist"
dim = 64
# faiss_depth, nprobe = 256, 128
# faiss_depth, nprobe = 512, 32
faiss_depth, nprobe = 128, 128
# 256, 128
len_dir_dic = {
    # 'rougelr': lambda ds_type, idx: f"data/bm25/sorted/{ds_type}_{idx}_rougelr_0.8_0.3_sorted.json",
    "colbert0": [256, 320, 256, 256],
    # "medqa_filter_merge": [32, 512, 32, 320],
    "medqa_filter_merge": [32, 512, 32, 320],
}
query_maxlen, doc_maxlen, query_max_span_length, doc_max_span_length = len_dir_dic['medqa_filter_merge']

index_config = {
    # "faiss_index_path": f"index/{index_type}/"
    #                     "0/GKMC.cosine.256_300_20_128/ivfpq.2000.faiss",
    # "index_path": "index/cblist_word_mask_no_ww_shuffle_id_rouge_filter_stopword_bm25_irsort_bg_o_overlap_256_300_cosine_128/0/GKMC.cosine.256_300_30_128/",
    # "index_path": f"index/{index_type}/0/GKMC.cosine.256_300_20_128/",
    "faiss_depth": faiss_depth,  # 16
    'n_probe': nprobe,
    "part_range": None,
    # "depth": 10,
    "dim": dim,
    "query_maxlen": query_maxlen,
    "doc_maxlen": doc_maxlen
}
print(index_config)

colbert_config = {
    "pretrain_path": "",
    # "checkpoint": "example/cblist_word_mask_no_ww_shuffle_id_rouge_filter_stopword_bm25_irsort_bg_o_overlap_256_300_cosine_128/0/"
    #               "MSMARCO-psg/train.py/listcbert/checkpoints/colbert-30.dnn",
    # "checkpoint": f"example/{index_type}/0/"
    #               "exp1/train.py/listcbert/checkpoints/colbert-20.dnn",
    # "checkpoint": str(pretrain_map['bert']),
    'init': False,
    "query_maxlen": query_maxlen,
    "doc_maxlen": doc_maxlen,
    "dim": dim,
    "similarity": "cosine"
}

reader_config = {
    # 'reader_path': "/home/awu/experiments/dcmn/geo/output/"
    #                "bm25_1e-5_bs2_gpu1_gracc2_bert_c3_bertxh_top10/"
    #                "Jun05_11-42-11_simbaLR(1e-05)_DS(['bm25-train-0'])_SEQLEN(256)_PNUM(10)_EPOCH(6.0)_ptm_(bert_c3)_model(bertxh)_(bs_2_1_acc_2)",
    'reader_path': str(pretrain_map['bert_c3']),
    'p_num': 10,
    'p_topk': 10,
    'gnn_layer': 9,
    'n_head': 8,
    "max_seq_length": 256
}

model_config = {
    "ir_topk": 20
}

# task = 'medqa_filter_merge'
task = 'dureader'
# train_task = f"{task}-train-0,rougelr-train-0"
# task = "colbert0"
fold = 0
train_task = f"{task}-train-{fold}"
dev_task = f"{task}-dev-{fold}"
# test_task = f"{task}-test-{fold}"
test_task = f"{task}-test-{fold}"
save_result_file = f"data/bm25/sorted/temp_weight.json"

data_dir_dic = {
    # 'rougelr': lambda ds_type, idx: f"data/bm25/sorted/{ds_type}_{idx}_rougelr_0.8_0.3_sorted.json",
    'rougelr': lambda ds_type, idx: f"data/bm25/sorted/{ds_type}_{idx}_rougelr_0.8_0.3_sorted.json",
    "colbert0": lambda ds_type, idx: f"data/bm25/sorted/{ds_type}_{idx}_colbert.json",
    "generated": lambda ds_type, idx: f"data/collection/question_opt_paras_sample_681.json" if ds_type == "train" else f"data/bm25/sorted/dev_0_rougelr_0.8_0.3_sorted.json",
    'webq': lambda ds_type, idx: f"tests/webqdata/{ds_type}.json",
    "medqa_short": lambda ds_type, idx: f"/home2/awu/testcb/data/MedQA/short/{ds_type}.json",
    "medqa_long": lambda ds_type, idx: f"/home2/awu/testcb/data/MedQA/long/{ds_type}.json",
    "medqa_merge": lambda ds_type, idx: f"/home2/awu/testcb/data/MedQA/merge/{ds_type}.json",
    "medqa_filter_merge": lambda ds_type, idx: f"/home2/awu/testcb/data/MedQA_filter/merge/{ds_type}.json",
    "dureader": lambda dstype, idx: f"/home2/awu/testcb/data/dureader/{dstype}_cut.json"
}

model_map = {
    'bert-base-en_uncased': (BertModel, BertConfig, BertTokenizerFast, '[unused1] ', '[unused2] ', " [CLS] ", " [SEP] ", " [SEP] ", 3e-5),
    'bert': (BertForMaskedLM, BertConfig, BertTokenizerFast, '[unused1] ', '[unused2] ', " [CLS] ", " [SEP] ", " [SEP] ", 3e-5),
    'ernie': (BertModel, BertConfig, BertTokenizerFast, '[unused1] ', '[unused2] ', " [CLS] ", " [SEP] ", " [SEP] ", 3e-5),
    "t5_base": (T5ForConditionalGeneration, T5Config, T5TokenizerFast, "query: ", "document: ", " ", " </s> ", " . ", 1e-4)
    # "t5_base": (T5ForConditionalGeneration, T5Config, T5TokenizerFast, " query: ", " document: ", " ", " , ", " , ", 1e-4)
    # "t5_base": (T5EncoderModel, T5Config, T5TokenizerFast, " <extra_id_0> ", " <extra_id_1> ", " ", " </s> ", 3e-4)
}

encoder_model, encoder_config, encoder_tokenizer, Q_marker_token, D_marker_token, CLS, SEP, answer_SEP, lr = model_map[pretrain_choose]

# pretrained_model = "/home2/awu/testcb//output/geo/colbert_schedule/pytorch.bin"
choose_retriever_model = "colbert_schedule_2e-2_decouple"
pretrained_model = f"/home2/awu/testcb//output/geo/{choose_retriever_model}/pytorch.bin"
index_config['index_path'] = f"/home2/awu/testcb/index/geo/{choose_retriever_model}/"
index_config['faiss_index_path'] = index_config['index_path'] + "ivfpq.2000.faiss"

# load_trained = pretrained_model
load_trained = None

query_word_weight = False
doc_word_weight = False
use_part_weight = False

collection_path = "data/collection/"

answer_prefix = "answer: "
title_prefix = "title: "
sent_prefix = "sent: "

Enable_tqdm = True
ir_topk = 20
retrieve_topk = 40
p_num = 4
pos_thre = 2.5
neg_thre = 2.5
pos_num = p_num // 2
# pos_num = p_num
# neg_num = 0
neg_num = p_num - pos_num
# eval_p_num = 4
eval_p_num = 8
eval_neg_num = 1
max_pos = 20
# max_pos = 10
max_neg = 20
padding_neg_num = 12
padding_neg_num = 12
schedule = True

reader_p_num = 15
num_rerank_keep = 10
n_keep = 2
epoch_thre = 2
reader_max_seq_length = 256
training_rank = False
reader_pretrained = str(pretrain_map["bert_c3"])
retriever_reader_trained = "/home2/awu/testcb//output/geo/colbert_reader_1_rank/pytorch.bin"
# reader_pretrained = str(pretrain_map["bert"])
retriever_bias_temperature = 2e-2
index_rank = None
# DEVICE = f"cuda:{index_rank}"
DEVICE = f"cuda"

stop_epoch = 100

Q_TOPK = 1000
D_TOPK = 1000
# query_aug_topk = 7
query_aug_topk = 16
# query_aug_topk = 4
# query_aug_topk = 0
print_generate = False
# load_trained = "output/webq/webq_colbert_t5_answer_1/pytorch.bin"
# load_trained = "output/webq/webq_colbert_t5_answer_title_pretrain/pytorch.bin"


calc_re_loss = True
save_every_eval = False
teacher_aug = False
calc_dual_negp = True

# load_all_paras = load_all_paras_geo
load_all_paras = load_all_paras_dureader

use_word = True
l2norm = True

use_pre_tensorize = True

use_prf = False
# save_result_file = "./result/result" + ("" if not use_prf else "_prf") + ".json"
# save_result_file = f"data/bm25/sorted/dev_0_colbert0.json"

# train_dev_doc_pre_tensorize_file = "/home2/awu/testcb/tests/webqdata/train_dev_doc_pretensorize.pt"
# train_dev_doc_pre_tensorize_file = f"/home2/awu/testcb/tests/webqdata/train_dev_doc_pretensorize_{pretrain_choose}_{'word' if use_word else 'tok'}.pt"
doc_pre_tensorize_file = f"data/collection/doc_pretensorize_{pretrain_choose}_{'word' if use_word else 'tok'}.pt"
train_dev_query_pre_tensorize_file = "data/collection/train_dev_query_pretensorize.pt"
# corpus_tokenized_prefix = f"/home2/awu/testcb/tests/webqdata/webq_corpus_word_" \
#                           f"{doc_max_span_length if use_word else doc_maxlen}_{pretrain_choose}_tokenized_{'word' if use_word else 'tok'}_"
# corpus_tokenized_prefix = f"/home/awu/experiments/geo/others/testcb/data/collection/all_paragraph_segmented_" \
#                           f"{doc_max_span_length if use_word else doc_maxlen}_{pretrain_choose}_tokenized_{'word' if use_word else 'tok'}"
corpus_tokenized_prefix = f"/home2/awu/testcb/data/dureader/collection//dureader_segmented_" \
                          f"{doc_max_span_length if use_word else doc_maxlen}_{pretrain_choose}_tokenized_{'word' if use_word else 'tok'}"
corpus_index_term_path = f"/home2/awu/testcb/tests/webqdata/webq_corpus_word_" \
                         f"{doc_max_span_length if use_word else doc_maxlen}_{pretrain_choose}_tokenized_{'word' if use_word else 'tok'}_term.pt"

kl_answer_re_loss = False
kl_temperature = 0.1

retriever_criterion = BiEncoderNllLossTri
# retriever_criterion = listMLE
# retriever_criterion = listMLEWeighted
# retriever_criterion = listnet_loss
# retriever_criterion = lambdaLoss

# context_random = random.Random(1234)
context_random = np.random.default_rng(1234)

partweights = [0, 1, 1]
# partweights = [0, 0, 1]
rouge2_weight = 4
# rouge2_weight = 8

opt_num = 1

mix_num = 10

padded_p_num = 0
# Temperature = 1
# SCORE_TEMPERATURE = 0.05 if not query_word_weight else 1e-7
SCORE_TEMPERATURE = 0.05 if not query_word_weight else 5e-5
SCORE_TEMPERATURE = 0.05 if not query_word_weight else 1e-2
SCORE_TEMPERATURE = 0.05 if not query_word_weight else 1e-1
SCORE_TEMPERATURE = 0.05 if not query_word_weight else 1e-4
SCORE_TEMPERATURE = 0.05 if not query_word_weight else 1e-5
SCORE_TEMPERATURE = 0.05 if not query_word_weight else 1e-3
SCORE_TEMPERATURE = 0.05 if not query_word_weight else 5e-3
SCORE_TEMPERATURE = 0.05 if not query_word_weight else 5e-2
SCORE_TEMPERATURE = 0.05 if not query_word_weight else 3e-2
# SCORE_TEMPERATURE = 0.05 if not query_word_weight else 1e-7
# SCORE_TEMPERATURE = 1

print({"temperature:": SCORE_TEMPERATURE})
# puncts = set(punctuation + string.punctuation) - set('()（）%./”“')
puncts = set(punctuation + string.punctuation)
# print(puncts)
