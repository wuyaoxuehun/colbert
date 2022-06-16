import numpy as np
from tqdm import tqdm

from awutils.file_utils import load_json, dump_json


def csv_reader(input_file, delimiter='\t'):
    def gen():
        with open(input_file, 'r', encoding='utf8') as fd:
            for i, line in enumerate(fd):
                slots = line.rstrip('\n').split(delimiter)
                yield slots

    return gen()


def get_dureader_ori_corpus():
    # medqa_dir = "/home2/awu/testcb/data/dureader/dureader-retrieval-baseline-dataset/passage-collection/"
    medqa_dir = "/home/awu/experiments/geo/others/testcb/data/dureader_dataset/passage-collection/"
    sents = []
    for i in tqdm(range(0, 4)):
        corpus_file = medqa_dir + f"part-0{i}"
        # print(i)
        tsents = [_[2] for _ in csv_reader(corpus_file)]
        for sent in tqdm(tsents):
            sents.append(sent)
    return sents


def gen_ce():
    for ds_type in ['train', 'dev']:
        res_data = load_json(f"data/{ds_type}_1.json")
        for t in tqdm(res_data):
            t['hard_negative_ctxs'] = [_[2] for _ in t['res'][:50] if _[2] not in set(t['positive_ctxs'])]
            del t['res']
        dump_json(res_data, f"data/dureader_dataset/{ds_type}_ce.json")


def gen_dev_for_ce_test():
    res_data = load_json(f"data/dev_11.json")
    for t in tqdm(res_data):
        # t['hard_negative_ctxs'] = [_[2] for _ in t['res'][:20] if _[2] not in set(t['positive_ctxs'])]
        # del t['res']
        t['retrieval_res'] = [_[2] for _ in t['res'][:300]]
        # del t['positive_ctxs']
        del t['hard_negative_ctxs']
        del t['res']
    dump_json(res_data, f"data/dureader_dataset/dev_ce_rerank.json")


def eval_dureader(output_data, topk=10, recall_topk=[50, 100]):
    # topk = 10
    # recall_topk = 50
    max_recall_topk = max(recall_topk)
    res = 0
    # recall_res = 0
    recall_k = {k: [] for k in recall_topk}
    for t in tqdm(output_data):
        for i in range(topk):
            if t['res'][i][2] in t['positive_ctxs']:
                res += 1 / (i + 1)
                break
        true_index = max_recall_topk
        for i in range(max_recall_topk):
            if t['res'][i][2] in t['positive_ctxs']:
                true_index = i
                break
        for k in recall_topk:
            recall_k[k].append(1 if true_index + 1 <= k else 0)

    print(f"mrr@10 = {res / len(output_data)}")
    for k, v in recall_k.items():
        print(f"recall@{k} = {np.mean(v)}")


def gen_iter_colbert_train_dev():
    for ds_type in ['train', 'dev']:
        data = load_json(f"data/{ds_type}_11.json")
        for t in tqdm(data):
            temp_hard_neg = t['hard_negative_ctxs'][:10]
            t['hard_negative_ctxs'] = temp_hard_neg + [_[2] for _ in t['res'][:50] if _[2] not in temp_hard_neg]
            del t['res']
        dump_json(data, f"data/dureader_dataset/{ds_type}_iter.json")


if __name__ == '__main__':
    # gen_ce()
    # gen_dev_for_ce_test()
    gen_iter_colbert_train_dev()