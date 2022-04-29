import json
import math

import torch
from tqdm import tqdm
from conf import *
from colbert.modeling.tokenization import DocTokenizer
from multiprocessing import Pool

# from dense_pretrain.create_pretrain_corpus import load_geo_corpus
from awutils.file_utils import dump_json, load_json

doc_tokenizer = DocTokenizer(doc_maxlen)

import os

os.environ['TOKENIZERS_PARALLELISM'] = 'true'


def pre_tensorize(examples):
    # print(examples)
    # return [_[0] for _ in doc_tokenizer.tensorize_dict([examples], to_tensor=False)]
    # return [_[0] for _ in doc_tokenizer.tensorize_dict([examples], to_tensor=False)]
    res = doc_tokenizer.tensorize_dict([examples], to_tensor=False)[0]
    # print(len(res), len(res[0]))
    # input(0)
    return res
    # return ids[0], mask[0], word_mask[0]


# def query_pre_tensorize():

def pre_tokenize_corpus():
    # collection_path = "tests/webqdata/webq_corpus_word.json"
    # collection_path = "/home2/awu/testcb/tests/webqdata/webq_corpus_word.json"
    # base_dir, file_name = os.path.split(collection_path)
    # output_dir = "/home2/awu/testcb/tests/webqdata/"
    # file_prefix, file_ext = os.path.splitext(file_name)
    # pref = file_prefix + f'_{doc_maxlen}_{pretrain_choose}_tokenized_{"word" if use_word else "tok"}'
    output_dir = os.path.dirname(corpus_tokenized_prefix)
    pref = corpus_tokenized_prefix
    # pre_tok_file = pref + '.pt'
    # pre_tok_file = file_prefix + '_tokenized_no_word_mask.pt'
    # pre_tok_path = os.path.join(base_dir, pre_tok_file)
    # pre_tok_path = os.path.join(output_dir, pre_tok_file)
    # if not os.path.exists(pre_tok_path):
    # collection = open(collection_path, encoding='utf8')
    # collection = json.load(collection)[:]
    # collection = load_geo_corpus()
    collection = load_all_paras()
    print("collection total:", len(collection))
    # self.all_paras = [para['paragraph_cut']['tok'].split() for para in all_paras]
    # collection = pre_tensorize(collection)[:1000]

    # for t in collection:
    #     pre_tensorize(t)
    # return
    # res = list(zip(*res))
    # print(len(res))
    # print("saving corpus")
    # torch.save(res, pre_tok_path)
    # exit()
    # print(f"loading tokenized file from {pre_tok_path}")
    # split_num = 12
    split_num = 12
    print("spliting collection")
    # collection = torch.load(pre_tok_path)
    # collection = res
    # all_d_ids, all_d_mask, all_d_word_mask = collection
    # with Pool(int(16 * 1.5)) as p:

    # get_train_dev_pretensorize(res, collection)
    # exit()
    part_len = math.ceil(len(collection) / split_num)
    for i in tqdm(range(split_num)):
        start, end = i * part_len, (i + 1) * part_len
        sub_collection = collection[start:end]
        with Pool(2) as p:
            res = list(tqdm(p.imap(pre_tensorize, sub_collection), total=len(sub_collection)))
        # for t in tqdm(sub_collection):
        #     pre_tensorize(t)
        # return

        # return
        # res = list(zip(*res))
        # sub_collection = [_[start:end] for _ in collection]

        sub_collection_path = os.path.join(output_dir, f'{pref}_{i}.pt')
        # torch.save(res[start:end], os.path.join(output_dir, sub_collection_path))
        torch.save(res, os.path.join(output_dir, sub_collection_path))
        del res
        # dump_json(res, sub_collection_path, indent=False)
        print(f'saved to {sub_collection_path}')
        # return
    print("collection splitted")


def load_train_dev_test_collection_docs():
    data = load_json(data_dir_dic['rougelr']("dev", 0))
    data += load_json(data_dir_dic['rougelr']("train", 0))
    data += load_json(data_dir_dic['rougelr']("test", 0))
    docs = {para["p_id"]: para for t in data for opt in "abcd" for para in t['paragraph_' + opt]}
    collection = load_geo_corpus()
    collection_ = {para["p_id"]: para for para in collection}
    collection_.update(docs)
    collection = [collection[0]] + list(collection_.values())
    dump_json(collection, 'data/collection/all_paragraph_segmented.json')
    print(len(collection))
    return


def get_train_dev_pretensorize(res, collection):
    train_file = data_dir_dic['rougelr']("train", 0)
    train_data = load_json(train_file) + load_json(data_dir_dic['generated']('train', 0))
    dev_data = load_json(data_dir_dic['rougelr']("dev", 0))

    data = train_data + dev_data
    doc_ids = set([para["p_id"] for t in data for opt in "abcd" for para in t['paragraph_' + opt]])
    doc_pretensorize_dic = {}
    for para, tensorize in zip(collection, res):
        if int(para["p_id"]) in doc_ids:
            doc_pretensorize_dic[int(para["p_id"])] = tensorize
    print(len(doc_ids))
    assert len(doc_ids) == len(doc_pretensorize_dic), (len(doc_ids), len(doc_pretensorize_dic))
    torch.save(doc_pretensorize_dic, doc_pre_tensorize_file)
    print(len(doc_pretensorize_dic))
    print('saved doc pretensorize')


def get_doc_maxlen_corpus_tokenizing():
    pass


def modify_word_pos_mask():
    collection_path = "tests/webqdata/webq_corpus.json"
    base_dir, file_name = os.path.split(collection_path)
    file_prefix, file_ext = os.path.splitext(file_name)
    # pre_tok_file = file_prefix + f'_{doc_maxlen}_tokenized.pt'
    for i in tqdm(range(12)):
        sub_collection_path = file_prefix + f'_{doc_maxlen}_tokenized_{i}.pt'
        sub_collection_path = os.path.join(base_dir, sub_collection_path)
        d_ids, d_mask, d_word_mask = torch.load(sub_collection_path)
        new_d_word_mask = []
        for dmask in d_mask:
            # masklen = sum(dmask)
            masklen = 1
            t = [0] * len(dmask)
            for j in range(masklen):
                t[j] = 1
            new_d_word_mask.append(t)
        torch.save([d_ids, d_mask, new_d_word_mask], sub_collection_path)


import nltk


def sub_word_tokenize(t):
    t['title_words'] = nltk.word_tokenize(t['title'])
    t['text_words'] = nltk.word_tokenize(t['text'])
    return t


def pre_word_tokenize():
    collection_path = "tests/webqdata/webq_corpus.json"
    base_dir, file_name = os.path.split(collection_path)
    output_dir = "/home2/awu/testcb/tests/webqdata/"
    file_prefix, file_ext = os.path.splitext(file_name)
    pref = file_prefix + f'_word'
    # pre_tok_file = pref + '.pt'
    pre_tok_file = pref + '.json'
    pre_tok_path = os.path.join(output_dir, pre_tok_file)
    collection = open(collection_path, encoding='utf8')
    collection = json.load(collection)[:]
    with Pool(2) as p:
        res = list(tqdm(p.imap(sub_word_tokenize, collection), total=len(collection)))
    print('saving to ' + pre_tok_path)
    # dump_json(res, pre_tok_path)
    dump_json(res, pre_tok_path, indent=False)
    # torch.save(res, pre_tok_path)


def json_to_pt():
    file = "/home2/awu/testcb/tests/webqdata/webq_corpus_word.json"
    data = load_json(file)
    torch.save(data, "/home2/awu/testcb/tests/webqdata/webq_corpus_word.pt")


def test_speed():
    for i in range(3):
        file = "/home2/awu/testcb/tests/webqdata/webq_corpus_word.json"

        load_json(file)
        file = "/home2/awu/testcb/tests/webqdata/webq_corpus_word.pt"
        torch.load(file)


if __name__ == '__main__':
    # pre_tokenize_corpus()
    # exit()
    # pre_word_tokenize()
    # json_to_pt()
    # lp = LineProfiler()
    #
    # lp_wrapper = lp(test_speed)
    # lp_wrapper()
    # lp.print_stats(sys.stdout)  # 打印出性能分析结果
    # exit()
    pre_tokenize_corpus()
    # load_train_dev_test_collection_docs()
    # from colbert.modeling.tokenization.utils import CostomTokenizer
    #
    # lp = LineProfiler()
    # lp.add_function(CostomTokenizer.tokenize_multiple_parts)
    # lp.add_function(CostomTokenizer.tokenize_d_segmented_dict)
    # lp_wrapper = lp(pre_tokenize_corpus)
    # lp_wrapper()
    # lp.print_stats(sys.stdout)  # 打印出性能分析结果
    # pre_tokenize_corpus()
    # modify_word_pos_mask()
