import math


def hit_fun(pred_pids, gt_pids, topk):
    for index, pid in enumerate(pred_pids):
        if index >= topk:
            break
        if pid in gt_pids:
            return 1
    return 0


def hit_fun_geo(paragraphs, p_id_gt, para_num):
    for index, paragraph in enumerate(paragraphs):
        if index >= para_num:
            break
        p_id = int(paragraph.get('id', paragraph.get('p_id')))
        if p_id in p_id_gt:
            return 1
    return 0


def ap_fun(pred_pids, gt_pids, topk):
    ap_right_num = 0
    ap = 0
    for index, pid in enumerate(pred_pids):
        if index >= topk:
            break
        if pid in gt_pids:
            ap_right_num += 1
            ap += ap_right_num / (index + 1)
    if ap_right_num > 0:
        ap = ap / topk
    return ap


def ap_fun_geo(paragraphs, p_id_gt, para_num):
    ap_right_num = 0
    ap = 0
    for index, paragraph in enumerate(paragraphs):
        if index >= para_num:
            break
        p_id = int(paragraph.get('id', paragraph.get('p_id')))
        if p_id in p_id_gt:
            ap_right_num += 1
            ap += ap_right_num / (index + 1)
    if ap_right_num > 0:
        ap = ap / ap_right_num
    return ap


def dcg_fun(pred_pids, gt_pids, topk):
    dcg = 0
    for index, pid in enumerate(pred_pids):
        if index >= topk:
            break
        if pid in gt_pids:
            dcg += 1 / math.log2(index + 2)
    return dcg


def dcg_fun_geo(paragraphs, p_id_gt, para_num):
    dcg = 0
    for index, paragraph in enumerate(paragraphs):
        if index >= para_num:
            break
        p_id = int(paragraph.get('id', paragraph.get('p_id')))
        if p_id in p_id_gt:
            dcg += 1 / math.log2(index + 2)
    return dcg

def idcg_fun(gt_pids, topk):
    idcg = 0
    for i in range(topk):
        if i < len(gt_pids):
            idcg += 1 / math.log2(i + 2)
    return idcg

# p_id_gt 所有相关段落的 list
# para_num  @N的大小
def idcg_fun_geo(p_id_gt, para_num):
    idcg = 0
    for i in range(para_num):
        if i < len(p_id_gt):
            idcg += 1 / math.log2(i + 2)
    return idcg
