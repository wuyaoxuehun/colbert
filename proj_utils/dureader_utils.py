import json
from collections import defaultdict

import hanlp
import numpy as np
import pandas as pd
from tqdm import tqdm

from awutils import dicutils
from awutils.es_utils import ES

from awutils.file_utils import load_json, dump_json
from awutils.seg_utils import get_segmenter, sents_cut, batch_sents_cut


def get_dureader_segmented(part=4):
    # output_dir = "/home2/awu/testcb/data/dureader/collection/dureader_segmented.json"
    # return load_json(output_dir)
    output_dir = "/home2/awu/testcb/data/dureader/collection/"
    segmented_sents = []
    for i in range(0, part):
        segmented_sents += load_json(output_dir + f"/dureader_segmented_{i}.json")
    return segmented_sents


def load_all_paras_dureader():
    return get_dureader_segmented()


def csv_reader(input_file, delimiter='\t'):
    def gen():
        with open(input_file, 'r', encoding='utf8') as fd:
            for i, line in enumerate(fd):
                slots = line.rstrip('\n').split(delimiter)
                # if len(slots) == 1:
                #     yield slots,
                # else:
                #     yield slots
                yield slots

    return gen()


def get_dureader_ori_corpus():
    medqa_dir = "/home2/awu/testcb/data/dureader/dureader-retrieval-baseline-dataset/passage-collection/"
    sents = []
    for i in range(0, 4):
        idx = (i + 1) * 100
        corpus_file = medqa_dir + f"part-0{i}"
        print(i)
        # tsents = pd.read_csv(corpus_file, delimiter='\t', header=None, error_bad_lines=True, usecols=[2], dtype={'passage_text': "string"}, names=['passage_text'])['passage_text']
        # tsents = pd.read_csv(corpus_file, delimiter='\t', header=None, error_bad_lines=True, names=["1", "2", 'passage_text', "3"])['passage_text']
        tsents = [_[2] for _ in csv_reader(corpus_file)]
        # tsents = tsents.astype("string")
        for sent in tqdm(tsents):
            # print(sent)
            if pd.isna(sent):
                sent = f"nan {idx}"
                idx += 1
                # print(sent)
            sents.append(sent)
    return sents


def index_dureader():
    segmented_sents = get_dureader_segmented()
    es = ES(index_name="dureader_seg")
    es.delete()
    es.build_index()
    # sents = get_dureader_ori_corpus()
    # es.index_corpus(sents)
    es.index_corpus(segmented_sents)


es = ES(index_name="dureader_seg")

topk = 50


def search_one_dureader(query):
    # query = sents_cut([query])[0]
    res = es.search(query=query, topk=topk)
    return [_[0] for _ in res]


# get_segmenter()


def search_for_dureader_test():
    test_data = [json.loads(_) for _ in open("/home2/awu/testcb/data/dureader/dureader-retrieval-test1/test1.json", encoding='utf8')]
    dureader_corpus_dir = "/home2/awu/testcb/data/dureader/dureader-retrieval-baseline-dataset/passage-collection/"
    # passage_id_map = [json.loads(_) for _ in open(dureader_corpus_dir + "passage2id.map.json", encoding='utf8')]
    # res = []
    # for t in tqdm(test_data):
    #     question, question_id = t['question'], t['question_id']
    #     tres = search_one_dureader(query=question)
    #     res.append({
    #         question_id: [passage_id_map[_] for _ in tres]
    #     })
    # test_data = test_data[:100]
    questions = batch_sents_cut([_['question'] for _ in test_data], batch_size=16)
    from multiprocessing import Pool
    with Pool(4) as p:
        res = list(tqdm(p.imap(search_one_dureader, questions), total=len(test_data)))
    passage_id_map = load_json(dureader_corpus_dir + "passage2id.map.json")
    res = {t["question_id"]: [passage_id_map[j] for j in tres] for t, tres in zip(test_data, res)}
    dump_json(res, "/home2/awu/testcb/data/dureader/test_es.json")


@dicutils.name_print_decorater
def load_dureader_seg_ori_map():
    segmented_sents = get_dureader_segmented()
    sents = get_dureader_ori_corpus()
    # return set(sents)
    assert len(sents) == len(segmented_sents), (len(sents), len(segmented_sents))
    return {i.strip(): j.strip() for i, j in zip(sents, segmented_sents)}


def gen_dev():
    dureader_dir = "/home2/awu/testcb/data/dureader/dureader-retrieval-baseline-dataset/"
    dev_data = load_json(dureader_dir + "/dev/dev.json", line=True)
    dev_aux_data = csv_reader(dureader_dir + "auxiliary/dev.retrieval.top50.res.tsv")
    corpus_dict = load_dureader_seg_ori_map()
    dev_aux_dic = defaultdict(list)
    for t in dev_aux_data:
        query, passage = t[0], t[2]
        dev_aux_dic[query].append(passage)

    dev = []
    for t in tqdm(dev_data):
        # assert all([_["paragraph_text"] in dureader_ori_map for _ in t['answer_paragraphs']])
        # assert all([_ in dureader_ori_map for _ in dev_aux_dic[t['question']]])
        # continue
        pos = [_["paragraph_text"] for _ in t['answer_paragraphs']]
        neg = [_ for _ in dev_aux_dic[t['question']] if _ not in set(pos)]
        dev.append({
            "question": t['question'],
            "positive_ctxs": [corpus_dict[_.strip()] for _ in pos],
            "hard_negative_ctxs": [corpus_dict[_.strip()] for _ in neg],
        })
    dump_json(dev, "/home2/awu/testcb/data/dureader/dev.json")


def cut_questions_for_train_dev():
    dureader_dir = "/home2/awu/testcb/data/dureader/"
    from awutils.seg_utils import sents_cut, get_segmenter, batch_sents_cut
    get_segmenter()
    for dstype in ['train', 'dev']:
        data = load_json(dureader_dir + f"{dstype}.json")
        cuts = [_['question'] for _ in data]
        cuts = batch_sents_cut(cuts, batch_size=16)
        for t, cut in tqdm(zip(data, cuts)):
            t['question_cut'] = cut

        dump_json(data, dureader_dir + f"{dstype}_cut.json")


def dureader_word_stat():
    data = load_json("/home2/awu/testcb/data/dureader/dev_cut.json")
    alllen = []
    for t in data:
        paras = t['positive_ctxs'] + t['hard_negative_ctxs']
        alllen += [len(_.split()) for _ in paras]
    print(np.mean(alllen), np.std(alllen), np.percentile(alllen, 95))


if __name__ == '__main__':
    # from torch.multiprocessing import Pool, set_start_method
    #
    # try:
    #     set_start_method('spawn')
    # except RuntimeError:
    #     pass
    # index_dureader()
    search_for_dureader_test()
    # gen_dev()
    # cut_questions_for_train_dev()
    # dureader_word_stat()