import copy
import os
import random
import sys

import ujson

sys.path.append("../")
import json
from collections import defaultdict
from rouge import Rouge as rouge
from multiprocessing import Pool
import multiprocessing as mp
import numpy as np
from tqdm import tqdm
from awutils.file_utils import load_json, dump_json

# all_pos, all_neg = Manager().list(), Manager().list()
# rouge = rouge(metrics=('rouge-1', 'rouge-2', 'rouge-l'))
rouge = rouge(metrics=('rouge-1', 'rouge-2'))
stopwords = set()


def get_meta_data():
    '''
    获取数据集分割的metadata
    :return:
    '''

    def get_fold_split():
        base_dir = "/home2/zxhuang/MultipleChoice/Bert-Choice/../data/retrieval_final/"
        split_data = []
        for fold in range(5):
            fold_map = {}
            for mode in ['train', 'dev', 'test']:
                file = os.path.join(base_dir, f'{mode}_{fold}.json')
                data = json.load(open(file, encoding='utf8'))
                # output_data = []
                fold_ids = []
                for example in data:
                    fold_ids.append(example['id'])
                fold_map[mode] = fold_ids
            split_data.append(fold_map)
        output_file = os.path.join("../data/meta../data/", f'split_data.json')
        print(output_file)

        json.dump(split_data, open(output_file, 'w', encoding='utf8'), indent=2, ensure_ascii=False)

    get_fold_split()


def split_5_fold_for_sorted_data(data, sort_type):
    '''
    将切割好的数据集分为k折
    :return:
    '''
    # all_sorted_data = json.load(open("../data/bm25/sorted/test_all_sorted.json", encoding='utf8'))
    all_sorted_data = data
    sorted_dic = {}
    for example in all_sorted_data:
        sorted_dic[example['id']] = example
    split_data = json.load(open('../data/metadata/split_data.json'))
    output_dir = "../data/bm25/sorted"
    for i in range(1):
        for mode in ['train', 'dev', 'test']:
            file = os.path.join(output_dir, f'{mode}_{i}_{sort_type}.json')
            print(file)
            # json.dump(data[i * 320:(i + 1) * 320], open(file, 'w', encoding='utf8'), indent=2, ensure_ascii=False)
            fold_ids = split_data[i][mode]
            fold_data = []
            for fid in fold_ids:
                # if fid not in sorted_dic:
                #     continue
                fold_data.append(sorted_dic[fid])
            json.dump(fold_data, open(file, 'w', encoding='utf8'), indent=2, ensure_ascii=False)


def filter_stop_words(text_list, join=True):
    res = [_ for _ in text_list if _ not in stopwords]
    if join:
        return ' '.join(res)
    else:
        return res


def read_stop_words():
    file = './../data/stopwords.txt'
    with open(file, encoding='utf8') as f:
        for line in f.readlines():
            stopwords.add(line.strip())


def max_span_rouge_(hyps, refs, output_pos=False):
    if type(hyps) == str:
        hpy_tokens = hyps.split()
    else:
        hpy_tokens = hyps
    hyp_len = len(hpy_tokens)
    # cur_max = {'rouge-l': defaultdict(int), 'rouge-2': defaultdict(int)}
    cur_max = {'rouge-l': defaultdict(int)}
    max_pos = (None, None)
    for i in range(hyp_len):
        for j in range(i + 1, hyp_len):
            try:
                cur_score = rouge.get_scores(hyps=' '.join(hpy_tokens[i:j]), refs=refs, ignore_empty=True)[0]
                for r in ['rouge-l']:
                    for m in list('rf'):
                        cur_max[r][m] = max(cur_max[r][m], cur_score[r][m])
                # cur_score = rouge.get_scores(hyps=' '.join(hpy_tokens[i:j]), refs=refs)[0]['rouge-l']['f']
            except Exception as e:
                print(e)
                print(hpy_tokens)
                print(hpy_tokens[i:j])
                print(refs)
                continue
            # if cur_max < cur_score:
            #     cur_max = max(cur_max, cur_score)
            #     max_pos = (i, j)
    if output_pos:
        return cur_max, max_pos
    else:
        return cur_max


from functools import lru_cache


# @lru_cache(maxsize=None)
# def get_scores(hyps, refs):
#     return rouge.get_scores(hyps=hyps, refs=refs, ignore_empty=True)[0]
#
# rouge.get_scores = get_scores

def max_span_rouge(hyps, refs, output_pos=False):
    if type(hyps) == str:
        hpy_tokens = hyps.split()
    else:
        hpy_tokens = hyps
    # hyp_len = len(hpy_tokens)
    # cur_max = {'rouge-l': defaultdict(int), 'rouge-2': defaultdict(int)}
    cur_max = {'rouge-l': defaultdict(int), 'rouge-1': defaultdict(int), 'rouge-2': defaultdict(int)}
    max_pos = (None, None)
    hyps = " ".join(list("".join(hpy_tokens)))
    refs = " ".join(list("".join(refs.split())))

    try:
        cur_score = rouge.get_scores(hyps=hyps, refs=refs, ignore_empty=True)[0]
        for r in ['rouge-l', 'rouge-1', 'rouge-2']:
            for m in list('rf'):
                cur_max[r][m] = max(cur_max[r][m], cur_score[r][m])
        # cur_score = rouge.get_scores(hyps=' '.join(hpy_tokens[i:j]), refs=refs)[0]['rouge-l']['f']
    except Exception as e:
        # print(e)
        # print(hpy_tokens)
        # print(refs if refs else "No Refs")
        exit()
    # if cur_max < cur_score:
    #     cur_max = max(cur_max, cur_score)
    #     max_pos = (i, j)
    if output_pos:
        return cur_max, max_pos
    else:
        return cur_max


@lru_cache(maxsize=None)
def max_span_rouge_str(hyps, refs, ):
    try:
        cur_score = rouge.get_scores(hyps=hyps, refs=refs, ignore_empty=True)[0]
        rougelr = cur_score['rouge-l']['r']
        # cur_score = rouge.get_scores(hyps=' '.join(hpy_tokens[i:j]), refs=refs)[0]['rouge-l']['f']
    except Exception as e:
        print(e)
        print(hyps)
        print(refs if refs else "No Refs")
    return rougelr


@lru_cache(maxsize=None)
def max_span_rouge12_str(hyps, refs, rouge2_weight):
    try:
        cur_score = rouge.get_scores(hyps=hyps, refs=refs, ignore_empty=True)[0]
        rougelr = cur_score['rouge-1']['r'] + cur_score['rouge-2']['r'] * rouge2_weight
        # print(hyps, refs, rougelr)
        # input()
        # rougelr = cur_score['rouge-1']['f'] + cur_score['rouge-2']['f'] * rouge2_weight
        # rougelr = cur_score['rouge-l']['r']
        # cur_score = rouge.get_scores(hyps=' '.join(hpy_tokens[i:j]), refs=refs)[0]['rouge-l']['f']
    except Exception as e:
        # print(e)
        # print(hyps)
        # print(refs if refs else "No Refs")
        return 0.0
    return rougelr


@lru_cache(maxsize=1000)
def max_span_rouge12f_str(hyps, refs, rouge2_weight):
    try:
        cur_score = rouge.get_scores(hyps=hyps, refs=refs, ignore_empty=True)[0]
        rougelr = cur_score['rouge-1']['f'] + cur_score['rouge-2']['f'] * rouge2_weight
    except Exception as e:
        # print(e)
        # print(hyps)
        # print(refs if refs else "No Refs")
        # return 0.0
        print(e)
        exit()
    return rougelr


def sort_para_by_rougeL(example, topk=20):
    bg_tokens = example['background_cut']['tok'].split()
    q_tokens = example['question_cut']['tok'].split()
    bg_tokens = filter_stop_words(bg_tokens)
    q_tokens = filter_stop_words(q_tokens)
    # pos = neg = 0
    # pos_cut_off = 0.6
    # neg_cut_off = 0.3
    for opt in list('abcd'):
        opt_tokens = example[opt.upper() + '_cut']['tok'].split()
        opt_tokens = filter_stop_words(opt_tokens, join=True)
        if not opt_tokens:
            for t in list('ABCD'):
                print(example[t])
            input()
        choose_paras = []
        for para in example['paragraph_' + opt][:topk]:
            para_tokens = para['paragraph_cut']['tok'].split()
            para_tokens = filter_stop_words(para_tokens, join=False)
            rouge_o = max_span_rouge(hyps=para_tokens, refs=opt_tokens)
            rouge_q = max_span_rouge(hyps=para_tokens, refs=q_tokens)
            rouge_bg = max_span_rouge(hyps=para_tokens, refs=bg_tokens)
            rouge_t = {'o': rouge_o, 'q': rouge_q, 'bg': rouge_bg}
            choose_paras.append({
                "p_id": para['p_id'],
                "paragraph": para["paragraph"],
                'paragraph_cut': para['paragraph_cut'],
                "score": para['score'],
                # 'rouge': rouge.get_scores(hyps=para_tokens, refs=' '.join([bg_tokens, q_tokens]))[0]['rouge-l']['f']
                # 'rouge_o_': max_span_rouge(hyps=para_tokens, refs=opt_tokens)['rouge-l']['r'],
                # 'rouge_q': max_span_rouge(hyps=para_tokens, refs=q_tokens)['rouge-l']['r'],
                # 'rouge_bg': max_span_rouge(hyps=para_tokens, refs=bg_tokens)['rouge-l']['r'],
            })
            # print(rouge_t)
            for i in ['o', 'q', 'bg']:
                for r in ['1', '2', 'l']:
                    choose_paras[-1][f'rouge_{i}_{r}'] = rouge_t[i][f'rouge-{r}']['r']

            # if choose_paras[-1]['rouge'] >= pos_cut_off:
            #     choose_paras[-1]['judge'] = 1
            # elif choose_paras[-1]['rouge'] < neg_cut_off:
            #     choose_paras[-1]['judge'] = -1
            # else:
            #     choose_paras[-1]['judge'] = 0
            # #
            # if choose_paras[-1]['judge'] == 1:
            #     pos += 1
            # elif choose_paras[-1]['judge'] == -1:
            #     neg += 1
        #
        # # choose_paras.sort(key=lambda x: (x['judge'],x['locs_in_bqo'] * 2 + x['score']), reverse=True)
        # choose_paras.sort(key=lambda x: (x['judge'], x['score']), reverse=True)
        # choose_paras.sort(key=lambda x: (x['rouge'], x['score']), reverse=True)
        example['paragraph_' + opt] = choose_paras
    # return pos, neg
    return example


def parse_one(example):
    keep_keys = ['id', 'background', 'question', 'answer'] + sum([[opt, 'paragraph_' + opt.lower()] for opt in list('ABCD')], [])
    example = sort_para_by_rougeL(example)
    t = {}
    for key in keep_keys:
        t[key] = example[key]
        if key in ['background', 'question'] + list('ABCD'):
            t[key + '_cut'] = example[key + '_cut']
    return t


def parse_sort_examples(data, sort_type='rougeL'):
    print(mp.cpu_count())
    with Pool(mp.cpu_count() // 2) as p:
        r = list(tqdm(p.imap(parse_one, data), total=len(data)))
    return r


def transform_to_sort_dataset(sort_type='o'):
    base_dir = "../data/bm25/segmented"
    # base_dir = "/home/zxhuang/MultipleChoice/Bert-Choice/../data/retrieval_final/"
    # data = []
    read_stop_words()
    for fold in range(1):
        file = os.path.join(base_dir, f'test_{fold}_segmented.json')
        data = json.load(open(file, encoding='utf8'))
        output_data = parse_sort_examples(data, sort_type=sort_type)
        output_dir = "../data/bm25/sorted"
        output_file = os.path.join(output_dir, f'test_{fold}_{sort_type}_sorted.json')
        # output_file = os.path.join(output_dir, f'test_{fold}_sorted_bg_o_overlap.json')
        print(output_file)
        with open(output_file, 'w', encoding='utf8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)


def sort_by_rouge_cutoff(data, pos_cut_off, neg_cut_off, metric, topk=20):
    # data = json.load(open('./../data/bm25/sorted/test_0_rouge_sorted.json'))
    pos, neg = [], []
    # base_metric, sub_metric = metric
    pos_exps = []
    for example in data:
        for opt in list('abcd'):
            cur_pos = cur_neg = 0
            choose_paras = example['paragraph_' + opt][:topk]
            pos_exps.append(0)
            for para in choose_paras:
                # del para['paragraph_cut']
                # score = sum([para['rouge_' + _] for _ in metric])
                partweights = [1, 1, 0]
                # rouge2weights = [5, 5, 5]
                rouge2weights = [4] * 3
                score = 0
                for i, t in enumerate(['o', 'q', 'bg']):
                    score += (para[f'rouge_{t}_1'] + para[f'rouge_{t}_2'] * rouge2weights[i]) * partweights[i]
                if score >= pos_cut_off:
                    para['judge'] = 1
                    cur_pos += 1
                    pos_exps[-1] = 1
                elif score < neg_cut_off:
                    para['judge'] = -1
                    cur_neg += 1
                else:
                    para['judge'] = 0
                para['rouge_score'] = score

            pos.append(cur_pos)
            neg.append(cur_neg)
            choose_paras.sort(key=lambda x: (x['rouge_score'],), reverse=True)
            # choose_paras.sort(key=lambda x: (x['judge'], x['score']), reverse=True)
            # choose_paras.sort(key=lambda x: (x['rouge'][base_metric][sub_metric], x['score']), reverse=True)
            example['paragraph_' + opt] = choose_paras
    print(sum(pos_exps), len(pos_exps))
    print(np.mean(pos), np.mean(neg))


def read_all_sorted():
    sorted_dir = "../data/bm25/sorted"
    all_sorted_file = os.path.join(sorted_dir, f'test_all_sorted.json')

    all_sorted_data = json.load(open(all_sorted_file, encoding='utf8'))
    sorted_dic = {}
    for example in all_sorted_data:
        sorted_dic[example['id']] = example
    return sorted_dic


def read_all_segmented():
    base_dir = "../data/bm25/segmented"
    file = os.path.join(base_dir, f'test_all_segmented.json')
    # file = os.path.join(base_dir, f'test_{fold}.json')
    data = json.load(open(file, encoding='utf8'))
    return data


def transform_all_to_sort_dataset():
    """
    transform bm25 topk to sorted dataset, with rouge-l-r
    """
    metric = 'rougelr'
    # base_dir = "/home/zxhuang/MultipleChoice/Bert-Choice/../data/retrieval_final/"
    # data = []
    # read_stop_words()
    all_segmented_data = read_all_segmented()
    # all_segmented_data = read_all_segmented()
    output_data = parse_sort_examples(all_segmented_data, sort_type='o')
    dump_json(output_data, f'../data/bm25/sorted/all_{metric}_test.json')

    # sort_by_rouge_cutoff(output_data, pos_cut_off=0.5, neg_cut_off=0.2, metric=("rouge-l", 'r'))

    # split_5_fold_for_sorted_data(output_data, sort_type=metric)


def calc_metric_for_sort(file=None, sort_type='bq_o'):
    output_dir = "../data/bm25/sorted"
    output_files = [os.path.join(output_dir, f'test_0_{sort_type}.json') for i in range(1)]

    # return ir_result_main_ori(output_files, 1, display=False)
    return ir_result_main_ori(file, 1, display=True)


def test_rouge_sorted():
    # metric = ('rouge-l', 'r')
    data = load_json(f"../data/bm25/sorted/test_0_rougelr.json")
    # pcs, ncs = [], []
    # metrics = []
    best_metric = [[], []]
    best_record = []

    for pc_ in tqdm(range(0, 10, 1)):
        for nc_ in range(0, pc_ + 1, 1):
            pc = pc_ / 10
            nc = nc_ / 10
            # print(f'''
            #     pc : {pc}
            #     nc : {nc}
            #     '''
            #       )
            # pc = nc = 1
            sort_by_rouge_cutoff(data, pos_cut_off=pc, neg_cut_off=nc, metric=('o'), topk=20)
            # output_file = f"../data/bm25/sorted/test_0_rougelr_{pc}_{nc}_sorted.json"
            output_file = f"../data/bm25/sorted/test_0_rougelr_sorted.json"
            dump_json(data, output_file)
            # rouge_cutoff(pc, nc, metric)
            #
            metric = calc_metric_for_sort(sort_type=f'rougelr_sorted')
            # metrics.append(sum([j for i in metric for j in i]))
            t_metric = sum([j for i in metric for j in i])
            if t_metric > sum([j for i in best_metric for j in i]):
                best_metric = metric
                best_record = [pc, nc]
                print(best_metric, best_record)
            # print('*' * 100)

    print(best_metric, best_record)


def get_best_rouge_sorted():
    metric = ('rouge-l', 'r')
    pc, nc = 0.5
    rouge_cutoff(pc, nc, metric)
    #
    calc_metric_for_sort(sort_type=f'rouge_{metric[0]}_{metric[1]}_{pc}_{nc}')
    print('*' * 100)


def test_rouge():
    read_stop_words()

    ref = filter_stop_words("山谷 中 的 城市".split())
    # ref = "1 2 3"
    hypo = filter_stop_words("下列 天气 系统 中 不 利于 雾霾 消散 的 最 可能 是 上 弱 高压".split())
    # hypo = '1 1 1'
    print(ref)
    print(hypo)
    # rouge = Rouge()

    # scores, (i, j) = max_span_rouge(hyps=hypo, refs=ref, output_pos=True)
    scores = rouge.get_scores(hyps=hypo, refs=ref, ignore_empty=True)
    print(scores)

    from rouge_score import rouge_scorer
    scorer = rouge_scorer.RougeScorer(['rougeLsum'])
    print(ref, hypo)
    scores = scorer.score(target=ref, prediction=hypo)
    print(scores)
    # print(''.join(hypo.split()[i: j]))


def test_bm25_metric():
    calc_metric_for_sort(sort_type=f'rougelr')


def test_rouge_sorted_metric():
    # data = load_json(f"../data/bm25/sorted/test_0_rougelr.json")
    data = load_json(f"../data/bm25/sorted/test_0_rouge12l.json")
    pos, neg = 0.6, 0.6
    sort_by_rouge_cutoff(data, pos_cut_off=pos, neg_cut_off=neg, metric=("o"), topk=20)
    # output_file = f"../data/bm25/sorted/test_0_rougelr_{pos}_{neg}_sorted_test.json"
    output_file = f"../data/bm25/sorted/all_rouge_{pos}_{neg}_sorted_test.json"
    # output_file = f"../data/bm25/sorted/test_0_rougelr_sorted.json"
    dump_json(data, output_file)
    # rouge_cutoff(pc, nc, metric)
    #
    # calc_metric_for_sort(sort_type=f'rougelr_sorted')
    calc_metric_for_sort(file=[output_file])


def transform_to_ir_dataset():
    pos, neg = 0.5, 0.3
    for ds in ['train', 'dev', 'test']:
        data = load_json(f"../data/bm25/sorted/{ds}_0_rougelr.json")
        sort_by_rouge_cutoff(data, pos_cut_off=pos, neg_cut_off=neg, metric=("o"), topk=20)
        output_file = f"../data/bm25/sorted/{ds}_0_rougelr_{pos}_{neg}_sorted.json"
        # output_file = f"../data/bm25/sorted/test_0_rougelr_sorted.json"
        print(output_file)
        dump_json(data, output_file)


import pandas as pd
# import plotnine
# from plotnine import *
from sklearn.metrics import precision_recall_curve


def get_best_cut():
    from ir_score_silver import tag_relevant_label
    data = load_json(f"../data/bm25/sorted/test_0_rougelr.json")
    tag_relevant_label(data)
    train_x = []
    train_y = []
    for sample in data:
        for opt in list('abcd'):
            paragraphs = sample['paragraph_' + opt]
            for p in paragraphs:
                label = p.get('label', 0)
                # train_x.append([p['rouge_o'], p['rouge_q']])
                train_x.append(p['rouge_o'])
                train_y.append(label)

    precision, recall, thresholds = precision_recall_curve(y_true=train_y, probas_pred=train_x)
    for re, thre in zip(recall, thresholds):
        if abs(re - 0.9) < 0.01:
            print(re, thre)
    exit()

    # 计算F分数
    fscore = (2 * precision * recall) / (precision + recall)
    # 绘制ROC曲线
    df_recall_precision = pd.DataFrame({'Precision': precision[:-1],
                                        'Recall': recall[:-1],
                                        'Threshold': thresholds})
    # 找到最佳阈值
    index = np.argmax(fscore)
    thresholdOpt = round(thresholds[index], ndigits=4)
    print(thresholdOpt)
    fscoreOpt = round(fscore[index], ndigits=4)
    recallOpt = round(recall[index], ndigits=4)
    precisionOpt = round(precision[index], ndigits=4)
    print('Best Threshold: {} with F-Score: {}'.format(thresholdOpt, fscoreOpt))
    print('Recall: {}, Precision: {}'.format(recallOpt, precisionOpt))

    plotnine.options.figure_size = (8, 4.8)
    (
            ggplot(data=df_recall_precision) +
            geom_point(aes(x='Recall',
                           y='Precision'),
                       size=0.4) +
            # 最佳阈值
            geom_point(aes(x=recallOpt,
                           y=precisionOpt),
                       color='#981220',
                       size=4) +
            geom_line(aes(x='Recall',
                          y='Precision')) +
            # 注释
            geom_text(aes(x=recallOpt,
                          y=precisionOpt),
                      label='Optimal threshold \n for class: {}'.format(thresholdOpt),
                      nudge_x=0.18,
                      nudge_y=0,
                      size=10,
                      fontstyle='italic') +
            labs(title='Recall Precision Curve') +
            xlab('Recall') +
            ylab('Precision') +
            theme_minimal()
    ).save(filename="./test.jpg")
    # from sklearn.linear_model import LogisticRegression
    # lr = LogisticRegression(fit_intercept=True, class_weight="balanced")
    # lr.fit(train_x, train_y)
    # pred = lr.predict(train_x)
    # from sklearn.metrics import classification_report
    # print(classification_report(y_true=train_y, y_pred=pred))
    # print(lr.coef_)
    # print(lr.intercept_)


# from ir_score_silver import ir_result_main_ori, eval_metric_for_data


def test():
    (ir_result_main_ori(["../test_0_ranking.json"], 1, display=True))


def split_ds():
    output_data = load_json(f'../data/bm25/sorted/all_rougelr_test.json')
    split_5_fold_for_sorted_data(output_data, sort_type='rouge12l')


def get_score_rouge12(inst, para, part, rouge2_weight):
    hyps = ' '.join(list(para['paragraph']))
    refs = ' '.join(list(inst[part]))
    return max_span_rouge12_str(hyps=hyps, refs=refs, rouge2_weight=rouge2_weight)


def get_score_for_para(inst, para, opt, rouge2_weight, partweights):
    if partweights[0] == 0:
        para['background_score'] = 0
    else:
        para['background_score'] = get_score_rouge12(inst, para, 'background', rouge2_weight) if inst['background'] else 1
    para['question_score'] = get_score_rouge12(inst, para, 'question', rouge2_weight) if inst['question'] else 1
    para['opt_score'] = get_score_rouge12(inst, para, opt.upper(), rouge2_weight)
    para['score'] = sum([score * weight for score, weight in
                         zip([para['background_score'], para['question_score'], para['opt_score']], partweights)])


def eval_dataset_for_weights(data, rouge2_weight, partweights):
    for t in tqdm(data):
        for opt in "abcd":
            paras = t['paragraph_' + opt]
            for para in paras:
                get_score_for_para(t, para, opt, rouge2_weight, partweights)
            paras.sort(key=lambda x: x['score'], reverse=True)
            t[f'paragraph_{opt}'] = paras

    eval_res = eval_metric_for_data(data)


def calc_dataset_for_weights(data, rouge2_weight, partweights, rank_topk=5):
    for inst in data:
        for opt in "abcd":
            paras = inst['paragraph_' + opt][:rank_topk]
            for para in paras:
                cur_score = 0
                for idx, part in enumerate(['background', "question", opt.upper()]):
                    cur_score += (para.get(part + "_rouge-1-f", 0) + para.get(part + "_rouge-2-f", 0) * rouge2_weight) * partweights[idx]
                para['score'] = cur_score
            paras.sort(key=lambda x: x['score'], reverse=True)
            inst[f'paragraph_{opt}'] = paras + inst['paragraph_' + opt][rank_topk:]
    return eval_metric_for_data(data, verbose=False)


def colbert_0_find():
    data_ = load_json("data/bm25/sorted/train_0_colbert0_rouge.json")
    random.seed(10)
    random.shuffle(data_)
    data_ = data_[:]
    # data = load_json("data/bm25/sorted/train_0_rougelr_0.8_0.3_sorted.json")
    eval_metric_for_data(data_)
    # data = copy.deepcopy(data_)
    rouge2_weight, partweight = 4, [0, 1, 1]
    print(rouge2_weight, partweight, 3)
    print(calc_dataset_for_weights(data_, rouge2_weight, partweight, rank_topk=5))
    # print(calc_dataset_for_weights(data_, 4, [0, 1, 2], rank_topk=3))
    # print(calc_dataset_for_weights(data_, 4, [0, 1, 2], rank_topk=3))
    exit()
    cur_max = 0
    for rouge2_weight in tqdm(range(4, 1, -1)):
        for bg_weight in range(0, 1):
            for question_weight in range(1, 2):
                for opt_weight in range(1, 5):
                    for rank_topk in range(3, 8):
                        data = copy.deepcopy(data_)
                        res = calc_dataset_for_weights(data, rouge2_weight, [bg_weight, question_weight, opt_weight], rank_topk=rank_topk)

                        # input(res['hit'][0])
                        if res['map'][0] > cur_max:
                            cur_max = res['map'][0]
                            print(rouge2_weight, [bg_weight, question_weight, opt_weight], rank_topk)

                            print(res)
                        continue
                        if res['hit'][0] > 0.5:
                            print(rouge2_weight, [bg_weight, question_weight, opt_weight], rank_topk)
                            print(res)
                            input()


def calc_rouge_for_dataset():
    data = load_json("data/bm25/sorted/train_0_colbert0.json")
    from rouge import Rouge as rouge
    rouge = rouge(metrics=('rouge-1', 'rouge-2'))

    for inst in tqdm(data):
        for opt in "abcd":
            paras = inst['paragraph_' + opt]
            for para in paras:
                for part in ['background', "question", opt.upper()]:
                    hyps = ' '.join(list(para['paragraph']))
                    refs = ' '.join(list(inst[part]))
                    try:
                        cur_score = rouge.get_scores(hyps=hyps, refs=refs, ignore_empty=True)[0]
                        for i in ["1", "2"]:
                            for j in ["f", "r", "p"]:
                                para[part + f"_rouge-{i}-{j}"] = cur_score[f'rouge-{i}'][f'{j}']
                    except:
                        for i in ["1", "2"]:
                            for j in ["f", "r", "p"]:
                                para[part + f"_rouge-{i}-{j}"] = 0

            # paras.sort(key=lambda x: x['score'], reverse=True)
            inst[f'paragraph_{opt}'] = paras
    dump_json(data, "data/bm25/sorted/train_0_colbert0_rouge.json")


def split_datasets():
    data = []
    for fold in ['train', 'dev', 'test']:
        data += load_json(f"data/bm25/sorted/{fold}_0_colbert0.json")
    data_dict = {t['id']: t for t in data}
    split_meta = load_json("data/metadata/split_data.json")
    for idx, split in enumerate(split_meta):
        for fold in ['train', 'dev', 'test']:
            fold_data = [data_dict[_] for _ in split[fold]]
            dump_json(fold_data, f"data/bm25/sorted/{fold}_{idx}_colbert.json")


def calc_dataset_statics():
    # data = []
    # for fold in ['train', 'dev', 'test']:
    #     data += load_json(f"data/bm25/sorted/{fold}_0_colbert0.json")
    # bg, q, o = 0, 0, 0
    # for t in data:
    #     bg += len(t['background_cut']['tok'].split())
    #     q += len(t["question_cut"]['tok'].split())
    #     for opt in 'ABCD':
    #         o += len(t[opt + "_cut"]['tok'].split())
    # n = len(data)
    # print(bg / n, q / n, o / n / 4)

    file = 'data/collection/all_paragraph_segmented.json'
    with open(file, encoding='utf8') as f:
        corpus = ujson.load(f)[1:]
    print(len(corpus))
    p = 0
    for t in corpus:
        # p += len(t['paragraph_cut']['tok'].split())
        p += len(''.join(t['paragraph_cut']['tok'].split()))
    print(p / len(corpus))



if __name__ == '__main__':
    pass
    # transform_all_to_sort_dataset()
    # split_ds()
    # transform_to_ir_dataset()
    # test_rouge_sorted_metric()
    # colbert_0_find()
    # split_datasets()
    # calc_dataset_statics()
    heatmap()
    # calc_rouge_for_dataset()
    # test_bm25_metric()
    # test()
    # get_best_cut()
    # test_rouge_sorted()
