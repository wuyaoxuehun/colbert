from mip import Model, xsum, maximize, BINARY
import mip
from tqdm import tqdm
import logging
import numpy as np
import sys


sys.path.append("../../")
from colbert.training.training_utils import moving_average
from ir_score_silver import eval_metric_for_data
from file_utils import load_json, dump_json

mip.logger.setLevel(logging.WARN)


def test_mip():
    p = [10, 13, 18, 31, 7, 15]
    w = [11, 15, 20, 35, 10, 33]
    c, I = 47, range(len(w))

    m = Model("knapsack")
    m.verbose = 0
    x = [m.add_var(var_type=BINARY) for i in I]

    m.objective = maximize(xsum(p[i] * x[i] for i in I))

    m += xsum(w[i] * x[i] for i in I) <= c

    m.optimize()

    selected = [i for i in I if x[i].x >= 0.99]
    # print("selected items: {}".format(selected))

def ilp_solve(w, occ, l):
    m = Model("ilp")
    m.verbose = 0
    word_num, sent_num = len(w), len(occ[0])
    weight_sum = sum(w)
    c = [m.add_var(var_type=BINARY) for _ in range(word_num)]
    s = [m.add_var(var_type=BINARY) for _ in range(sent_num)]
    sent_word_num = [sum([occ[i][j] for i in range(word_num)]) for j in range(sent_num)]
    # print(sent_word_num)
    m.objective = maximize(xsum(w[i] * c[i] for i in range(word_num)) +
                           - 0.0001 * xsum(l[i] * s[i] for i in range(sent_num)) +
                           0.0000001 * xsum(sent_word_num[i] * s[i] for i in range(sent_num)))
    # m.objective = maximize(xsum(w[i] * c[i] for i in range(word_num)))
    m += xsum(s) <= 5
    for i in range(word_num):
        for j in range(sent_num):
            m += s[j] * occ[i][j] <= c[i]

    for j in range(sent_num):
        m += xsum(occ[i][j] * w[i] * s[j] for i in range(word_num)) >= ((1 / 10) * weight_sum) * s[j]
    for i in range(word_num):
        m += xsum(s[j] * occ[i][j] for j in range(sent_num)) >= c[i]
    m.optimize()
    # print(c, s)
    # print(m.objective_value)
    selected_c = [i for i in range(word_num) if c[i].x >= 0.99]
    # print("selected wrods: {}".format(selected))
    selected_s = [i for i in range(sent_num) if s[i].x >= 0.99]
    return selected_c, selected_s


def ilp_solve_one(background_cut, question_cut, option_cut, paras):
    part_weights = [1, 5, 20]
    # part_x_weights = []
    word_weights = {}
    word_idx = 0
    words_set = []
    for idx, t in enumerate([background_cut, question_cut, option_cut]):
        words = t['tok'].split()
        # print(words)
        for word in words:
            if word not in word_weights:
                word_weights[word] = [part_weights[idx], word_idx]
                word_idx += 1
                words_set.append(word)
            else:
                word_weights[word][0] = part_weights[idx]

    # assert len(words_set) == len(word_weights)
    occ = [[0] * len(paras) for _ in range(len(word_weights))]
    l = [0] * len(paras)
    for idx, para in enumerate(paras):
        words = para['paragraph_cut']['tok'].split()
        for word in words:
            if word in word_weights:
                occ[word_weights[word][1]][idx] = 1
        l[idx] = len(words)

    w = [word_weights[word][0] for word in words_set]
    # print(w)
    # print(occ)
    # print(l)
    return ilp_solve(w, occ, l)

def test():
    data = load_json("data/bm25/sorted/test_0_rougelr_0.8_0.3_sorted.json")
    a = moving_average()
    res = 0
    # eval_metric_for_data(data)
    for example in tqdm(data):
        # example['paragraph_a']
        top_k = 20
        for opt in 'abcd':

            example['paragraph_' + opt].sort(key=lambda x: x['score'], reverse=True)
            # continue
            selected_c, selected_s = ilp_solve_one(background_cut=example['background_cut'],
                                                   question_cut=example['question_cut'],
                                                   option_cut=example[f'{opt.upper()}_cut'],
                                                   paras=example['paragraph_a'][:top_k])
            # example['selected_paras'] = [example['paragraph_' + opt][i]['paragraph'] for i in selected_s]
            # example['paragraph_a'] = [example['paragraph_a'][i]['paragraph'] for i in range(len(example['paragraph_a']))]
            selected = []
            unselected = []
            for idx in range(20):
                cur = example['paragraph_' + opt][idx]
                if idx in selected_s:
                    selected.append(cur)
                else:
                    unselected.append(cur)
            example['paragraph_' + opt] = selected + unselected
            res = a.send(len(selected_s))

        # print(res)
        # print(selected_s)
        # dump_json(example, "../../data/sample.json")
        # input()
    print(res)
    eval_metric_for_data(data)


def test_solve_ilp():
    wn, sn = 30, 20
    w = np.random.randint(1, 10, wn).tolist()
    occ = np.random.randint(0, 2, size=(wn, sn)).tolist()
    l = np.random.randint(100, 300, size=sn).tolist()
    ilp_solve(w, occ, l)


if __name__ == '__main__':
    # for i in tqdm(range(100)):
    #     test_mip()

    # test_solve_ilp()
    test()
