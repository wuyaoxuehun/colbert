import json
import math
import os

import torch
from tqdm import tqdm
import os
from conf import *
from colbert.modeling.tokenization import DocTokenizer
import multiprocessing
from multiprocessing import Pool

from file_utils import dump_json, load_json

doc_tokenizer = DocTokenizer(doc_maxlen)

import os

os.environ['TOKENIZERS_PARALLELISM'] = 'true'


def pre_tensorize(examples):
    # print(examples)
    return [_[0] for _ in doc_tokenizer.tensorize_dict([examples], to_tensor=False)]
    # return ids[0], mask[0], word_mask[0]


def pre_tokenize_corpus():
    # collection_path = "tests/webqdata/webq_corpus_word.json"
    collection_path = "/home2/awu/testcb/tests/webqdata/webq_corpus_word.json"
    base_dir, file_name = os.path.split(collection_path)
    output_dir = "/home2/awu/testcb/tests/webqdata/"
    file_prefix, file_ext = os.path.splitext(file_name)
    pref = file_prefix + f'_{doc_maxlen}_{pretrain_choose}_tokenized_{"word" if use_word else "tok"}'
    pre_tok_file = pref + '.pt'
    # pre_tok_file = file_prefix + '_tokenized_no_word_mask.pt'
    # pre_tok_path = os.path.join(base_dir, pre_tok_file)
    pre_tok_path = os.path.join(output_dir, pre_tok_file)
    # if not os.path.exists(pre_tok_path):
    collection = open(collection_path, encoding='utf8')
    collection = json.load(collection)[:]
    # collection = pre_tensorize(collection)[:1000]

    # for t in collection:
    #     pre_tensorize(t)
    # return
    # res = list(zip(*res))
    # print(len(res))
    # print("saving corpus")
    # torch.save(res, pre_tok_path)
    # exit()
    print(f"loading tokenized file from {pre_tok_path}")
    split_num = 12
    print("spliting collection")
    # collection = torch.load(pre_tok_path)
    # collection = res
    # all_d_ids, all_d_mask, all_d_word_mask = collection
    part_len = math.ceil(len(collection) / split_num)
    for i in tqdm(range(split_num)):
        start, end = i * part_len, (i + 1) * part_len
        sub_collection = collection[start:end]
        # for t in tqdm(sub_collection):
        #     pre_tensorize(t)
        # return
        with Pool(4) as p:
            res = list(tqdm(p.imap(pre_tensorize, sub_collection), total=len(sub_collection)))
        # return
        res = list(zip(*res))
        # sub_collection = [_[start:end] for _ in collection]
        sub_collection_path = os.path.join(output_dir, f'{pref}_{i}.pt')
        torch.save(res, os.path.join(output_dir, sub_collection_path))
        # dump_json(res, sub_collection_path, indent=False)
        print(f'saved to {sub_collection_path}')
        # return
    print("collection splitted")


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
    with Pool(8) as p:
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


from line_profiler import LineProfiler
import sys

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
