from collections import defaultdict
import json

def load_all_paras():
    # file = '../../data/GKMC/dpr_eval/all_paragraph.tsv'
    file = 'data/collection/all_paragraph_segmented.json'
    para_dic = defaultdict(dict)
    with open(file, encoding='utf8') as f:
        data = json.load(f)
        for idx, line in enumerate(data[1:]):
            # p_id, text, *_ = line.strip().split('\t')
            para_dic[idx] = line
    # print(len(para_dic))
    return para_dic