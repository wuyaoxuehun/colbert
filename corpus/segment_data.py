import json
import os
import time
from collections import defaultdict

import pandas as pd

from awutils.file_utils import load_json, dump_json
# from remove_nonsense import filterQuestion, filterBackground
import numpy as np
from awutils import dicutils

import hanlp
from tqdm import tqdm





# def get_segmenter():
#     import jieba
#     return jieba.cut

# def sents_cut(sents):




def pre_segment_collection():
    collection_path = "../../../data/GKMC/dpr_eval/all_paragraph.tsv"
    collection = open(collection_path, encoding='utf8')
    next(collection)
    collection = list(collection)
    collection_len = len(collection) + 1
    output_collection = []
    paras = []
    for line_idx, line in tqdm(enumerate(collection), total=len(collection)):
        line_parts = line.strip().split('\t')
        if len(line_parts) < 2:
            print(line_parts)
            print(line_idx)
            if len(line_parts) == 1:
                line_parts = [-1, line_parts[0]]
                print('len line parts == 1')
            else:
                line_parts = [-2, '#']
                print('empty line parts')

        pid, passage, *other = line_parts
        paras.append([pid, passage])

    batch_size = 240
    # segmenter = get_segmenter()
    # paras = paras[:4000]
    for idx in tqdm(range(0, len(paras), batch_size)):
        paras_batch = paras[idx:idx + batch_size]
        paras_batch = list(zip(*paras_batch))
        paras_batch_segments = sents_cut(paras_batch[1], segmenter)
        for pid, p in zip(paras_batch[0], paras_batch_segments):
            output_collection.append({
                'p_id': int(pid),
                'paragraph_cut': p
            })

    output_collection.insert(0, {
        'p_id': -100,
        'paragraph_cut': "占位段落"
    })
    assert len(output_collection) == collection_len
    output_file = "../data/collection/all_paragraph_segmented.json"
    json.dump(output_collection, open(output_file, 'w', encoding='utf8'), indent=2, ensure_ascii=False)


topk = 20
segmenter = None


# get_segmenter()


def segment_example(example):
    out_example = {'id': example['id'],
                   'answer': example['answer']}
    example['background'] = filterBackground(example['background'])
    example['question'] = filterQuestion(example['question'])
    cut_keys = ['background', 'question', ] + list('ABCD')
    sents = [example[_] if example[_] else '#' for _ in cut_keys]
    sents_cuts = sents_cut(sents)
    for idx, key in enumerate(cut_keys):
        out_example[key] = example[key]
        out_example[key + '_cut'] = sents_cuts[idx]

    for opt in list('abcd'):
        out_example['paragraph_' + opt] = []
        topk_paras = example['paragraph_' + opt][:topk]
        paras = [_['paragraph'] for _ in topk_paras]
        paras_cut = sents_cut(paras, segmenter)
        for para, para_cut in zip(topk_paras, paras_cut):
            out_example['paragraph_' + opt].append({
                "p_id": para['p_id'],
                "paragraph": para['paragraph'],
                "paragraph_cut": para_cut,
                "score": para['score'],
                "rank": para['rank'],
            })
    return out_example


def segment_dataset():
    '''
    切割数据集，合并为一个切割好的数据集
    :return:
    '''
    get_segmenter()

    base_dir = "/home2/zxhuang/MultipleChoice/Bert-Choice/data/retrieval_final/"
    data = []
    for fold in range(5):
        file = os.path.join(base_dir, f'test_{fold}.json')
        data.extend(json.load(open(file, encoding='utf8')))
    # output_data = []
    # for example in tqdm(data):
    #     output_data.append(segment_example(example))
    from torch.multiprocessing import Pool
    with Pool(os.cpu_count() // 2) as p:
        output_data = list(tqdm(p.imap(segment_example, data), total=len(data)))

    output_dir = "../data/bm25/segmented"
    output_file = os.path.join(output_dir, 'test_all_segmented.json')
    with open(output_file, 'w', encoding='utf8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    pass


def corpus_static():
    file = '../data/collection/all_paragraph_segmented.json'
    data = json.load(open(file, encoding='utf8'))
    all_len = []
    for example in data[1:]:
        all_len.append(len(''.join(example['paragraph_cut']['tok'].split())))

    from scipy import stats
    print(f'average_tokens={np.mean(all_len)}\n'
          f'tokens_std={np.std(all_len)}\n'
          f'95%={np.percentile(all_len, 95)}\n')
    for seq in range(256, 384, 10):
        print(f'{seq}={stats.percentileofscore(all_len, seq)}\n')


def segment_generated_questions():
    context_questions = load_json("data/collection/sampled_questions.json")
    res = []
    # get_segmenter()
    for p1, p2, question in context_questions:
        question = "".join(question.split())
        res.append(question)
        # input(question)
    segmented_sents = sents_cut(res)
    # input(segmented_sents[0])
    res = [(p1, p2, sent["tok"]) for (p1, p2, _), sent in zip(context_questions, segmented_sents)]
    dump_json(res, "data/collection/generated_questions_segmented.json")


def segment_generated_options():
    context_questions = load_json("data/collection/filtered_options.json")
    res = []
    # get_segmenter()
    for question, options in tqdm(context_questions):
        segmented_options = sents_cut(options)
        res.append((question, [_['tok'] for _ in segmented_options]))
        # input(question)
    dump_json(res, "data/collection/filtered_options_segmented.json")




def segment_medqa():
    medqa_dir = "/home2/zxhuang/python_workstation/DPR/data/MedQA/"
    output_dir = "/home2/awu/testcb/data/MedQA/"
    corpus_file = medqa_dir + "paragraphs.tsv"
    corpus = []
    sents = []
    with open(corpus_file, encoding='utf8') as f:
        for line in tqdm(f.readlines()):
            index, text, title = line.split("\t")
            if index[0].isalpha():
                index = -1
            sents.append([index, text])

    indices, texts = list(zip(*sents))
    segmented_sents = batch_sents_cut(texts, batch_size=24)
    for index, text, t in zip(indices, texts, segmented_sents):
        corpus.append({
            "p_id": int(index),
            "paragraph": text,
            "paragraph_cut": t
        })
    output_file = output_dir + "collection/medqa_segmented.json"
    dump_json(corpus, output_file)


def segment_medqa_dataset():
    global segmenter
    # medqa_dir = "/home2/zxhuang/python_workstation/DPR/data/MedQA/"
    medqa_dir = "/home2/zxhuang/python_workstation/LongQuestionIR/data/MedQA/cn/"
    output_dir = "/home2/awu/testcb/data/MedQA_filter/"
    # corpus_file = medqa_dir + "paragraphs.tsv"
    corpus_dir = "/home2/awu/testcb/data/MedQA/"
    corpus_segmented_file = corpus_dir + "collection/medqa_segmented.json"
    corpus_dict = {}
    for t in load_json(corpus_segmented_file):
        corpus_dict[t['paragraph']] = t
    # length = "merge/"
    # topk_pos, topk_neg = 10, 10
    merge = defaultdict(lambda: defaultdict(list))
    if segmenter is None:
        segmenter = get_segmenter()
    # for length in ["short/", "long/", ]:
    for length in ["merge/"]:
        for ds in ['train', "dev", "test"]:
            data = load_json(medqa_dir + length + ds + ".json")
            # output_data = []
            for t in tqdm(data):
                for p in t['positive_ctxs']:
                    p["p_id"] = corpus_dict[p['text']]["p_id"]
                    p['text_cut'] = corpus_dict[p['text']]["paragraph_cut"]
                for p in t['hard_negative_ctxs']:
                    p["p_id"] = corpus_dict[p['text']]["p_id"]
                    p['text_cut'] = corpus_dict[p['text']]["paragraph_cut"]
                t['question_cut'] = sents_cut(t['question'])
                merge[length][ds] = data
            dump_json(data, output_dir + length + ds + ".json")
    # for ds in ['train', "dev", "test"]:
    #     data = []
    #     for length in ["short/", "long/"]:
    #         data += merge[length][ds]
    #     # merge[length][ds] = data
    #     dump_json(data, output_dir + "merge/" + ds + ".json")


def medqa_statistics():
    # medqa_dir = "/home2/zxhuang/python_workstation/DPR/data/MedQA/"
    medqa_dir = "/home2/awu/testcb/data/MedQA/"
    # corpus_file = medqa_dir + "paragraphs.tsv"
    corpus_segmented_file = medqa_dir + "collection/medqa_segmented.json"
    corpus_dict = {}
    for t in load_json(corpus_segmented_file):
        corpus_dict[t['paragraph']] = t
    questions = []
    for length in ["short/", ]:  # "long/"]:
        for ds in ['train', "dev", "test"]:
            data = load_json(medqa_dir + length + ds + ".json")
            questions += [(len(_['question']), len(_["question_cut"]['tok'].split())) for _ in data]
    question_len, question_cut_len = list(zip(*questions))
    ratio = 95
    print([(np.mean(question_len), np.std(question_len), np.percentile(question_len, ratio))])
    print([(np.mean(question_cut_len), np.std(question_cut_len), np.percentile(question_cut_len, ratio))])
    corpus = []
    for t in corpus_dict.values():
        corpus.append([len(t['paragraph']), len(t['paragraph_cut']['tok'].split())])
    question_len, question_cut_len = list(zip(*corpus))
    print([(np.mean(question_len), np.std(question_len), np.percentile(question_len, ratio))])
    print([(np.mean(question_cut_len), np.std(question_cut_len), np.percentile(question_cut_len, ratio))])








def test_reader():
    medqa_dir = "/home2/awu/testcb/data/dureader/dureader-retrieval-baseline-dataset/passage-collection/"
    sents = []
    for i in range(0, 4):
        corpus_file = medqa_dir + f"part-0{i}"
        for line in tqdm(csv_reader(corpus_file), total=1012084 * 2):
            sent = line[2]
            sents.append(len(sent))
    print(np.mean(sents), np.std(sents), np.percentile(sents, 95), np.percentile(sents, 90), np.percentile(sents, 85))
    # # if "表现为言语流频繁地被与正常流利的人在频率和强度上不同" in str(sent):
    # if '''""''' in str(sent):
    #     print(sent)
    #     input()
    # tsents = [_[2] for _ in csv_reader(corpus_file)]


def segment_dureader():
    global segmenter
    # for i in tqdm(range(60 * 60 * 9), colour="green"):
    #     time.sleep(1)
    index = 4
    # os.environ['CUDA_VISIBLE_DEVICES'] = f"{index}"
    os.environ['CUDA_VISIBLE_DEVICES'] = f"4"
    if segmenter is None:
        segmenter = get_segmenter()
    medqa_dir = "/home2/awu/testcb/data/dureader/dureader-retrieval-baseline-dataset/passage-collection/"
    output_dir = "/home2/awu/testcb/data/dureader/collection"
    # corpus = []
    sents = []
    # idx = index * 100
    # passage2id = load_json(medqa_dir + "passage2id.map.json")
    max_length = 512
    for i in range(index - 1, index):
        # for i in range(0, 4):
        corpus_file = medqa_dir + f"part-0{i}"
        print(i)
        # tsents = pd.read_csv(corpus_file, delimiter='\t', header=None, error_bad_lines=True, usecols=[2], dtype={'passage_text': "string"}, names=['passage_text'])['passage_text']
        # tsents = tsents.astype("string")
        tsents = [_[2][:max_length] for _ in csv_reader(corpus_file)]
        # sents += tsents
        # for sent in tqdm(tsents):
        #     # print(sent)
        #     if pd.isna(sent):
        #         sent = f"nan {idx}"
        #         idx += 1
        #         print(sent)
        #     # if "表现为言语流频繁地被与正常流利的人在频率和强度上不同" in str(sent):
        #     if "频率和强度上不同" in str(sent):
        #         print(sent)
        #         input()
        #     sents.append(sent)

        # with open(corpus_file, encoding='utf8') as f:
        #     for line in tqdm(f.readlines()[:]):
        #         _, _, passage_text, _ = line.split("\t")
        #         # real_id = passage2id[str(idx)]
        #         # sents.append([idx, passage_text, real_id])
        #         sents.append(passage_text)
        #         idx += 1
        # input(passage_text)
        segmented_sents = batch_sents_cut(tsents, batch_size=16)
        dump_json(segmented_sents, output_dir + f"/dureader_segmented_{i}.json")
    print(len(sents))
    return
    indices, texts, real_ids = list(zip(*sents))
    segmented_sents = batch_sents_cut(texts, batch_size=12)
    for index, text, t, real_id in zip(indices, texts, segmented_sents, real_ids):
        corpus.append({
            "p_id": int(index),
            "paragraph": text,
            "paragraph_cut": t,
            "real_id": real_id
        })
    output_file = output_dir + "collection/medqa_segmented.json"
    dump_json(corpus, output_file)


def segment_dureader_merge():
    # medqa_dir = "/home2/awu/testcb/data/dureader/dureader-retrieval-baseline-dataset/passage-collection/"
    output_dir = "/home2/awu/testcb/data/dureader/collection"
    # passage2id = load_json(medqa_dir + "passage2id.map.json")
    segmented_sents = []
    for i in range(0, 4):
        segmented_sents += load_json(output_dir + f"/dureader_segmented_{i}.json")
    print(len(segmented_sents))
    output_file = output_dir + f"/dureader_segmented.txt"
    segmented_sents = segmented_sents[:]
    pd.DataFrame(segmented_sents).astype("string").to_csv(header=False, path_or_buf=output_file, sep="\t")
    # setns = [_['tok'] if type(_) is not str else '0' for _ in segmented_sents]
    # with open(output_file, 'w', encoding='utf8') as f:
    #     print([_ for _ in segmented_sents if type(_) is str])
    #     # input()
    #     all_lines = len()
    #     print(len(all_lines))
    #     exit()
    # f.writelines('\n'.join())
    return
    # indices, texts, real_ids = list(zip(*sents))
    # segmented_sents = batch_sents_cut(texts, batch_size=12)


def dureader_transform():
    sents = get_dureader_segmented()
    output_dir = "/home2/awu/testcb/data/dureader/collection"
    output_file = output_dir + f"/dureader_segmented.json"
    dump_json([' '.join(_) for _ in sents], output_file)


@dicutils.name_print_decorater
def load_dureader_corpus():
    output_dir = "/home2/awu/testcb/data/dureader/collection"
    output_file = output_dir + f"/dureader_segmented.txt"
    idx = 0
    corpus = []
    with open(output_file, 'r', encoding='utf8') as f:
        segmented_sents = f.readlines()
        print(len(segmented_sents))
        for sent in tqdm(segmented_sents):
            sent = sent.strip()
            corpus.append({
                "p_id": int(idx),
                # "paragraph": "".join(sent.split()),
                "paragraph_cut": {'tok': sent},
                # "real_id": real_id
            })
            idx += 1
        # print(idx, len(passage2id))
        # output_file = output_dir + f"/dureader_segmented.json"
        # dump_json(corpus, output_file)
        return corpus





def read_dureader_dataset():
    data_dir = "/home2/awu/testcb/data/dureader/dureader-retrieval-baseline-dataset/"
    data = csv_reader(data_dir + "/train/dual.train.tsv")
    train = defaultdict(lambda: defaultdict(set))
    for query, _, pos, _, neg, _ in data:
        train[query]['pos'].add(pos)
        train[query]['neg'].add(neg)
    return train


def segment_dureader_dataset():
    global segmenter
    output_dir = "/home2/awu/testcb/data/dureader/"
    # corpus_file = medqa_dir + "paragraphs.tsv"
    # data_dir = "/home2/awu/testcb/data/dureader/dureader-retrieval-baseline-dataset/"
    # corpus = load_dureader_corpus()
    # corpus_dict = dicutils.index_by_key(lambda _: "".join(_['paragraph_cut']['tok'].split()), corpus, enable_tqdm=True)
    corpus_dict = load_dureader_seg_ori_map()
    # for i in corpus_dict:
    #     t = "单击要解除阻止的发行商"
    # if len((set(''''单击要解除阻止的发行商''')
    #         & set(i))) / len(set(i)) > 0.9:
    # if t in i:
    #     print(i)
    # exit()
    # import pandas as pd
    # data = pd.read_csv(data_dir + "/train/dual.train.tsv", delimiter='\t', header=None, error_bad_lines=False, usecols=[0, 2, 4], names=['query', 'pos', 'neg'])
    data = read_dureader_dataset()
    train = []
    # for t in tqdm(list(data.groupby('query'))):
    for t in tqdm(data, total=86400):
        query = t
        pos = data[t]['pos']
        neg = data[t]['neg']
        # query = t[0]
        # pos = set(t[1]['pos'])
        # neg = set(t[1]['neg'])
        # assert all([_ in corpus_dict for _ in (pos | neg)]), ([_ for _ in (pos | neg) if _ not in corpus_dict])
        # continue
        try:
            train.append({
                "question": query,
                "positive_ctxs": [corpus_dict[_.strip()] for _ in pos],
                "hard_negative_ctxs": [corpus_dict[_.strip()] for _ in neg],
            })
        except Exception as e:
            print()
            # print(e)
            print(query, [_ for _ in (pos | neg) if _.strip() not in corpus_dict])
            # exit()
            # input()

        # print(train[-1])
        # input()
    # output_data = []
    # for t in tqdm(data):
    #     for p in t['positive_ctxs']:
    #         p["p_id"] = corpus_dict[p['text']]["p_id"]
    #         p['text_cut'] = corpus_dict[p['text']]["paragraph_cut"]
    #     for p in t['hard_negative_ctxs']:
    #         p["p_id"] = corpus_dict[p['text']]["p_id"]
    #         p['text_cut'] = corpus_dict[p['text']]["paragraph_cut"]
    #     t['question_cut'] = sents_cut(t['question'])
    #     merge[length][ds] = data
    # dump_json(data, output_dir + length + ds + ".json")
    dump_json(train, output_dir + "train.json")


if __name__ == '__main__':
    # pre_segment_collection()
    # corpus_static()
    from torch.multiprocessing import Pool, set_start_method

    try:
        set_start_method('spawn')
    except RuntimeError:
        pass
    # segment_dataset()
    # segment_generated_questions()
    # segment_generated_options()
    # segment_medqa()
    # segment_dureader()
    # test_reader()
    index_dureader()
    # segment_medqa_dataset()
    # segment_dureader_merge()
    # test_load_dureader()
    # medqa_statistics()
    # segment_dureader_dataset()
    # dureader_transform()
