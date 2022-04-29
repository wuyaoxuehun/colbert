import json
from collections import defaultdict

from awutils.ir_metric_utils import hit_fun_geo, dcg_fun_geo, idcg_fun_geo, ap_fun_geo


def load_all_paras_geo():
    # file = '../../data/GKMC/dpr_eval/all_paragraph.tsv'
    file = 'data/collection/all_paragraph_segmented.json'
    para_dic = {}
    with open(file, encoding='utf8') as f:
        data = json.load(f)
        for idx, line in enumerate(data[1:]):
            # for idx, line in enumerate(data):
            # p_id, text, *_ = line.strip().split('\t')
            para_dic[idx] = line
    # print(len(para_dic))
    return para_dic


def load_all_paras_medqa(to_dict=False):
    # file = '../../data/GKMC/dpr_eval/all_paragraph.tsv'
    file = '/home2/awu/testcb/data/MedQA/collection/medqa_segmented.json'
    para_dic = {}
    with open(file, encoding='utf8') as f:
        data = json.load(f)
        data = data[1:]
        if not to_dict:
            return data
        for idx, line in enumerate(data):
            # for idx, line in enumerate(data):
            # p_id, text, *_ = line.strip().split('\t')
            para_dic[idx] = line
    # print(len(para_dic))
    return para_dic


def load_all_paras_geo_dict():
    file = 'data/collection/all_paragraph_segmented.json'
    para_dic = {}
    with open(file, encoding='utf8') as f:
        data = json.load(f)
        for idx, line in enumerate(data[1:]):
            para_dic[int(line["p_id"])] = line
    # print(len(para_dic))
    return para_dic


def load_all_paras_webq(path='tests/webqdata/webq_corpus.json'):
    # file = '../../data/GKMC/dpr_eval/all_paragraph.tsv'
    file = path
    para_dic = dict()
    with open(file, encoding='utf8') as f:
        data = json.load(f)
        for idx, line in enumerate(data[:]):
            # p_id, text, *_ = line.strip().split('\t')
            para_dic[idx] = line
    # print(len(para_dic))
    return para_dic


def read_ground_truth(path=None):
    if not path:
        path = 'data/metadata/ground_truth.json'
    f = open(path, encoding='utf8')
    json_List = json.load(f)
    ground_truth_map = {}
    for question_map in json_List:
        q_id = question_map['id']
        for i, opt in enumerate("abcd"):
            pids = [int(paragraph['p_id']) for paragraph in question_map['paragraph_' + opt]]
            if len(pids) > 0:
                key = str(q_id) + "-" + str(i)
                ground_truth_map[key] = pids
    return ground_truth_map


def eval_metric_for_data(data, verbose=True):
    ground_truth = read_ground_truth()
    res_dic = defaultdict(list)
    for para_num in [2, 10]:
        count, hit_count, map, ndcg = 0, 0, 0, 0
        for sample in data:
            for opt in list('abcd'):
                paragraphs = sample['paragraph_' + opt]
                q_id = sample['id']
                key = str(q_id) + '-' + str(ord(opt) - ord('a'))
                if key in ground_truth:
                    p_id_gt = ground_truth[key]
                    if len(p_id_gt) > 0:
                        count += 1
                        hit_count += hit_fun_geo(paragraphs, p_id_gt, para_num)
                        map += ap_fun_geo(paragraphs, p_id_gt, para_num)
                        ndcg += dcg_fun_geo(paragraphs, p_id_gt, para_num) / idcg_fun_geo(p_id_gt, para_num)
        hit = round(hit_count / count, 4)
        map = round(map / count, 4)
        ndcg = round(ndcg / count, 4)
        if verbose:
            print('hit@', para_num, ":", hit, 'map@', para_num, ":", map, 'ndcg@', para_num, ":",
                  para_num, ":", ndcg)
        res_dic['hit'].append(hit)
        res_dic['map'].append(map)
        res_dic['ndcg'].append(ndcg)
    return res_dic


# coding=utf-8
from collections import defaultdict

import math
import json


def read_ground_truth_allopt(path=None):
    if not path:
        path = 'data/metadata/ground_truth.json'
    f = open(path, encoding='utf8')
    json_List = json.load(f)
    ground_truth_map = {}
    for question_map in json_List:
        paragraph_a = question_map['paragraph_a']
        paragraph_b = question_map['paragraph_b']
        paragraph_c = question_map['paragraph_c']
        paragraph_d = question_map['paragraph_d']
        q_id = question_map['id']
        p_ids = []
        for i, paragraphs in enumerate([paragraph_a, paragraph_b, paragraph_c, paragraph_d]):
            for paragraph in paragraphs:
                p_ids.append(int(paragraph['p_id']))
        if len(p_ids) > 0:
            key = str(q_id)
            ground_truth_map[key] = p_ids

    # print(len(ground_truth_map))
    return ground_truth_map


def get_metric_for_data(data, para_num, ground_truth=None):
    if ground_truth == None:
        path_gt = 'data/metadata/ground_truth.json'
        ground_truth = read_ground_truth(path_gt)
    count, hit_count, map, ndcg = 0, 0, 0, 0
    for sample in data:
        paragraphs = sample['ctxs']
        key = str(sample['id']) + '-' + str(ord(sample['opt']) - ord('A'))
        if key in ground_truth:
            p_id_gt = ground_truth[key]
            if len(p_id_gt) > 0:
                count += 1
                hit_count += hit_fun_geo(paragraphs, p_id_gt, para_num)
                map += ap_fun_geo(paragraphs, p_id_gt, para_num)
                ndcg += dcg_fun_geo(paragraphs, p_id_gt, para_num) / idcg_fun_geo(p_id_gt, para_num)

    hit = round(hit_count / count, 4)
    map = round(map / count, 4)
    ndcg = round(ndcg / count, 4)
    print(count)
    # print('hit@', para_num, ":", hit, 'map@', para_num, ":", map, 'ndcg_err@', para_num, ":", ndcg_err, 'ndcg',
    #       para_num, ":", ndcg)
    return hit, map, ndcg


def get_metric_for_data_allopt(data, para_num, ground_truth=None):
    if ground_truth == None:
        path_gt = 'data/metadata/ground_truth.json'
        ground_truth = read_ground_truth_allopt(path_gt)
    count, hit_count, map, ndcg = 0, 0, 0, 0
    for sample in data:
        paragraphs = sample['ctxs']
        key = str(sample['id'])
        if key in ground_truth:
            p_id_gt = ground_truth[key]
            if len(p_id_gt) > 0:
                count += 1
                hit_count += hit_fun_geo(paragraphs, p_id_gt, para_num)
                map += ap_fun_geo(paragraphs, p_id_gt, para_num)
                ndcg += dcg_fun_geo(paragraphs, p_id_gt, para_num) / idcg_fun_geo(p_id_gt, para_num)

    hit = round(hit_count / count, 4)
    map = round(map / count, 4)
    ndcg = round(ndcg / count, 4)
    print(count)
    # print('hit@', para_num, ":", hit, 'map@', para_num, ":", map, 'ndcg_err@', para_num, ":", ndcg_err, 'ndcg',
    #       para_num, ":", ndcg)
    return hit, map, ndcg


def readFile_ir(path, para_num, ground_truth):
    f = open(path, encoding='utf8')
    json_List = json.load(f)

    # print('hit@', para_num, ":", hit, 'map@', para_num, ":", map, 'ndcg_err@', para_num, ":", ndcg_err, 'ndcg',
    #       para_num, ":", ndcg)
    return get_metric_for_data(json_List, para_num, ground_truth)


def read_file_ir_ori(path, para_num, ground_truth):
    f = open(path, encoding='utf8')
    json_List = json.load(f)
    count, hit_count, map, ndcg = 0, 0, 0, 0
    for sample in json_List:
        for opt in list('abcd'):
            paragraphs = sample['paragraph_' + opt]
            q_id = sample['id']
            key = str(q_id) + '-' + str(ord(opt) - ord('a'))
            if key in ground_truth:
                p_id_gt = ground_truth[key]
                if len(p_id_gt) > 0:
                    count += 1
                    hit_count += hit_fun_geo(paragraphs, p_id_gt, para_num)
                    map += ap_fun_geo(paragraphs, p_id_gt, para_num)
                    ndcg += dcg_fun_geo(paragraphs, p_id_gt, para_num) / idcg_fun_geo(p_id_gt, para_num)

    # print(count)
    hit = round(hit_count / count, 4)
    map = round(map / count, 4)
    ndcg = round(ndcg / count, 4)
    # print('hit@', para_num, ":", hit, 'map@', para_num, ":", map, 'ndcg_err@', para_num, ":", ndcg_err, 'ndcg',
    #       para_num, ":", ndcg)

    return hit, map, ndcg


def eval_metric_for_data_noopt(data):
    ground_truth = read_ground_truth_allopt()
    for para_num in [2, 10]:
        count, hit_count, map, ndcg = 0, 0, 0, 0
        for sample in data:
            paragraphs = sample['paragraph_abcd']
            q_id = sample['id']
            key = str(q_id)
            if key in ground_truth:
                p_id_gt = ground_truth[key]
                if len(p_id_gt) > 0:
                    # print(p_id_gt, [_['p_id'] for _ in paragraphs])
                    # input()
                    count += 1
                    hit_count += hit_fun_geo(paragraphs, p_id_gt, para_num)
                    map += ap_fun_geo(paragraphs, p_id_gt, para_num)
                    ndcg += dcg_fun_geo(paragraphs, p_id_gt, para_num) / idcg_fun_geo(p_id_gt, para_num)

        print(count)
        hit = round(hit_count / count, 4)
        map = round(map / count, 4)
        ndcg = round(ndcg / count, 4)
        print('hit@', para_num, ":", hit, 'map@', para_num, ":", map, 'ndcg@', para_num, ":",
              para_num, ":", ndcg)

    return hit, map, ndcg


def tag_relevant_label(data):
    path_gt = '../data/metadata/ground_truth.json'
    ground_truth = read_ground_truth(path_gt)
    for sample in data:
        for opt in list('abcd'):
            paragraphs = sample['paragraph_' + opt]
            q_id = sample['id']
            key = str(q_id) + '-' + str(ord(opt) - ord('a'))
            if key in ground_truth:
                p_id_gt = ground_truth[key]
                for p in paragraphs:
                    p['label'] = int(int(p.get('id', p.get('p_id'))) in p_id_gt)


def ir_result_main(path, fold_num):
    path_gt = 'data/metadata/ground_truth.json'
    ground_truth = read_ground_truth(path_gt)
    para_nums = [2, 10]
    for para_num in para_nums:
        hit_all, map_all, ndcg_all = 0, 0, 0
        for i in range(fold_num):
            # path = f"rankresult/loose1_{i}/rank_output.json"
            # hit_avg, map_avg, ndcg_avg_err, ndcg_avg = readFile_ir(path, para_num, ground_truth)
            hit_avg, map_avg, ndcg_avg = readFile_ir(path, para_num, ground_truth)
            # print(hit_avg, map_avg, ndcg_avg)
            hit_all += hit_avg
            map_all += map_avg
            ndcg_all += ndcg_avg
        print(f'-----------{fold_num}折平均-----------')
        print('hit@', para_num, ":", round(hit_all / fold_num, 4), 'map@', para_num, round(map_all / fold_num, 4),
              'ndcg@', para_num, round(ndcg_all / fold_num, 4))


def ir_result_main_ori(path, fold_num=1, display=True):
    # path_gt = '../../data/GKMC/dpr_eval/ground_truth.json'
    # path_gt = '../../data/GKMC/dpr_eval/ground_truth.json'
    # path_gt = 'data/metadata/ground_truth.json'
    path_gt = '../data/metadata/ground_truth.json'
    # path_gt = '/home2/zxhuang/python_workstation/HMRS_history/data/ground_truth.json'
    ground_truth = read_ground_truth(path_gt)
    para_nums = [2, 10]
    if type(path) == list:
        fold_num = len(path)
    res = []
    for para_num in para_nums:
        hit_all, map_all, ndcg_all = 0, 0, 0
        for i in range(fold_num):
            # path = f"rankresult/loose1_{i}/rank_output.json"
            # hit_avg, map_avg, ndcg_avg_err, ndcg_avg = readFile_ir(path, para_num, ground_truth)
            # print(path[i])
            hit_avg, map_avg, ndcg_avg = read_file_ir_ori(path[i], para_num, ground_truth)
            # print(hit_avg, map_avg, ndcg_avg)
            hit_all += hit_avg
            map_all += map_avg
            ndcg_all += ndcg_avg
        hit = round(hit_all / fold_num, 4)
        map_ = round(map_all / fold_num, 4)
        ndcg = round(ndcg_all / fold_num, 4)
        if display:
            print(f'-----------{fold_num}折平均-----------')

            print('hit@', para_num, ":", hit, 'map@', para_num, map_,
                  'ndcg@', para_num, ndcg)
        res.append([hit, map_, ndcg])
    return res


if __name__ == '__main__':
    import sys

    print(sys.argv)
    ir_result_main(sys.argv[1], int(sys.argv[2]) + 1)
