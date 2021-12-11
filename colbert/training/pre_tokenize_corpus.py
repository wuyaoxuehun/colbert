import json
import math
import os

import torch
from tqdm import tqdm
import os
from conf import doc_maxlen
from colbert.modeling.tokenization import DocTokenizer
import multiprocessing
from multiprocessing import Pool

doc_tokenizer = DocTokenizer(doc_maxlen)

import os

os.environ['TOKENIZERS_PARALLELISM'] = 'true'


def pre_tensorize(examples):
    # print(examples)
    ids, mask, word_mask = doc_tokenizer.tensorize_dict([examples], to_tensor=False)
    return ids[0], mask[0], word_mask[0]


def pre_tokenize_corpus():
    collection_path = "tests/webqdata/webq_corpus.json"
    base_dir, file_name = os.path.split(collection_path)
    file_prefix, file_ext = os.path.splitext(file_name)
    pre_tok_file = file_prefix + f'_{doc_maxlen}_tokenized.pt'
    # pre_tok_file = file_prefix + '_tokenized_no_word_mask.pt'
    pre_tok_path = os.path.join(base_dir, pre_tok_file)
    # if not os.path.exists(pre_tok_path):
    collection = open(collection_path, encoding='utf8')
    collection = json.load(collection)[:]
    # collection = pre_tensorize(collection)[:1000]
    with Pool(os.cpu_count()) as p:
        res = list(tqdm(p.imap(pre_tensorize, collection), total=len(collection)))
    res = list(zip(*res))
    print(len(res))
    torch.save(res, pre_tok_path)
    # exit()
    print(f"loading tokenized file from {pre_tok_path}")
    split_num = 12
    print("spliting collection")
    collection = torch.load(pre_tok_path)
    all_d_ids, all_d_mask, all_d_word_mask = collection
    part_len = math.ceil(len(all_d_ids) / split_num)
    for i in tqdm(range(split_num)):
        start, end = i * part_len, (i + 1) * part_len
        d_ids = all_d_ids[start:end]
        d_mask = all_d_mask[start:end]
        d_word_mask = all_d_word_mask[start:end]
        sub_collection_path = file_prefix + f'_{doc_maxlen}_tokenized_{i}.pt'
        torch.save([d_ids, d_mask, d_word_mask],
                   os.path.join(base_dir, sub_collection_path))
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


if __name__ == '__main__':
    # pre_tokenize_corpus()
    modify_word_pos_mask()
