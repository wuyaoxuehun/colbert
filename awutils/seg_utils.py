import hanlp
import os
from tqdm import tqdm

segmenter = None


# os.environ['CUDA_VISIBLE_DEVICES'] = f"2"


def get_segmenter():
    global segmenter
    segmenter = hanlp.load(hanlp.pretrained.mtl.CLOSE_TOK_POS_NER_SRL_DEP_SDP_CON_ELECTRA_BASE_ZH)
    # segmenter = hanlp.load(hanlp.pretrained.tok.COARSE_ELECTRA_SMALL_ZH)
    # print(segmenter(['商品和服务。', '阿婆主来到北京立方庭参观自然语义科技公司。']))
    return segmenter


def sents_cut(sents):
    global segmenter
    # if type(sents) == str:
    #     sents = [sents]
    # ts = segmenter(sents, tasks=['tok/fine', 'pos/pku', 'ner/msra'])
    ts = segmenter(sents, tasks=['tok/coarse'])
    # ts = segmenter(sents, )
    return [" ".join(_) for _ in ts['tok/coarse']]
    rets = []
    t = range(len(sents))
    if len(sents) > 100:
        t = tqdm(t)
    for idx in t:
        # tok = ' '.join(ts['tok/fine'][idx])
        tok = ' '.join(ts[idx])
        # pos = ' '.join(ts['pos/pku'][idx])
        # ner = [list(_) for _ in ts['ner/msra'][idx]]
        rets.append({
            'tok': tok,
            # 'pos': pos,
            # 'ner': ner
        })
    if len(sents) == 1:
        return rets[0]
    return rets


def batch_sents_cut(sents, batch_size=96):
    n = len(sents)
    segmented_sents = []
    for i in tqdm(range(0, n, batch_size)):
        t_sents = sents[i:i + batch_size]
        segmented_sents += sents_cut(t_sents)
    return segmented_sents
