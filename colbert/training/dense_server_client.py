import os
import sys
from multiprocessing.connection import Listener, Client

import numpy as np
import torch
from tqdm import tqdm

from awutils.file_utils import load_json, dump_json
from colbert.indexing.faiss_indexers import ColbertRetriever, DPRRetriever
from colbert.modeling.colbert_model import ColbertModel
from colbert.training.training_utils import qd_mask_to_realinput
# QueryData = namedtuple("QueryData", ["question", "topk", "faiss_depth", "nprobe"])
from colbert.utils.dense_conf import load_dense_conf, data_dir_dic
from proj_utils.dureader_utils import get_dureader_ori_corpus, eval_dureader


# os.environ["CUDA_VISIBLE_DEVICES"] = "2,3,4"


class DenseRetrieverServer:
    def __init__(self):
        args = load_dense_conf()
        self.index_path = args.dense_index_args.index_path
        self.dim = args.dense_training_args.dim
        faiss_type = args.faiss_index_args.faiss_type
        self.colbert = ColbertModel(args)
        self.colbert.load(args.dense_index_args.checkpoint + "/pytorch_model.bin")
        self.colbert = self.colbert.cuda()
        self.retriever = ColbertRetriever(index_path=self.index_path, dim=self.dim, model=self.colbert) if faiss_type == "colbert" \
            else DPRRetriever(index_path=self.index_path, dim=self.dim)
        self.retriever.load_index()
        self.all_paras = get_dureader_ori_corpus()

    @torch.no_grad()
    def retrieve(self, questions, topk, faiss_depth, nprobe, bs=144):
        batch_pid_score_paras = []
        self.retriever.set_faiss_nprobe(nprobe)
        for i in tqdm(range(0, len(questions), bs)):
            batch_questions = questions[i:i + bs]
            q = self.colbert.tokenize_for_retriever(batch=[{"question": _} for _ in batch_questions])
            q_ids, q_attention_mask, q_active_padding, *_ = [_.cuda() for _ in q]
            Q = self.colbert.query(q_ids, q_attention_mask)
            for _, (q, q_mask) in enumerate(zip(Q, q_active_padding)):
                q, q_mask = qd_mask_to_realinput(t=q, t_mask=q_mask, keep_dim=False)
                pid_scores = self.retriever.search(query=q, topk_doc=topk, faiss_depth=faiss_depth)
                paras = [self.all_paras[pid] for pid in pid_scores[0]]
                batch_pid_score_paras.append(list(zip(pid_scores[0], pid_scores[1], paras)))
        return batch_pid_score_paras

    def server(self):
        # torch.multiprocessing.set_sharing_strategy("file_system")
        port_start = 9090
        address = ('localhost', port_start)  # family is deduced to be 'AF_INET'
        listener = Listener(address, authkey=b'1')
        while True:
            print("receiving connection --- --- ")
            conn = listener.accept()
            try:
                print('connection accepted from', listener.last_accepted)
                params = conn.recv()
                pids = self.retrieve(*params)
                conn.send(pids)
            except:
                print('retrieval error')


class DenseRetrieverClient:
    def __init__(self):
        self.address = ('localhost', 9090)
        # self.conn = Client(self.address, authkey=b'1')
        print('connected to server')

    def retrieve(self, questions, topk, faiss_depth, nprobe):
        self.conn = Client(self.address, authkey=b'1')
        self.conn.send((questions, topk, faiss_depth, nprobe))
        data = self.conn.recv()
        return data


nprobe, faiss_depth = 128, 512


# nprobe, faiss_depth = 128, 256


def dureader_evaluate():
    dr_client = DenseRetrieverClient()

    def test_to_submit():
        dureader_corpus_dir = "/home/awu/experiments/geo/others/testcb/data/dureader_dataset/passage-collection/"
        passage_id_map = load_json(dureader_corpus_dir + "passage2id.map.json")
        # test_res = load_json("data/bm25/sorted/temp_test_res.json")
        test_ori = load_json("/home/awu/experiments/geo/others/testcb/data/dureader_dataset/test1.json", line=True)
        output = {}
        test_res = eval_for_ds(test_ori)
        for t, t_ori in tqdm(zip(test_res, test_ori)):
            output[t_ori['question_id']] = [
                # passage_id_map[str(min(len(passage_id_map), int(seg_dict[_['paragraph_cut']] + 1)))]
                passage_id_map[str(_[0])]
                for _ in t['res'][:50]
            ]
        dump_json(output, "data/test_res.json")

    def eval_for_ds(data):
        questions = [_['question'] for _ in data]
        bs = 1024
        res = []
        for i in tqdm(range(0, len(data), bs)):
            sub_questions = questions[i:i + bs]
            sub_res = dr_client.retrieve(questions=sub_questions, topk=100, faiss_depth=faiss_depth, nprobe=nprobe)
            res.extend(sub_res)
        print(len(res), len(res[0]))
        # input()
        print(len(questions), len(res))
        for t_ori, t in zip(data, res):
            t_ori['res'] = t
        return data

    def eval_for_dataset(ds_type='dev'):
        dev_data = load_json(data_dir_dic['dureader'](ds_type, 0))
        dev_data = eval_for_ds(dev_data)
        eval_dureader(output_data=dev_data)
        # dump_json(dev_data, f"data/{ds_type}_11.json")

    def eval_for_test():
        pass

    eval_for_dataset('dev')
    # test_to_submit()


def test_res_to_test_rerank():
    data = load_json("data/test_res.json")
    all_paras = get_dureader_ori_corpus()
    dureader_corpus_dir = "/home/awu/experiments/geo/others/testcb/data/dureader_dataset/passage-collection/"
    passage_id_map = load_json(dureader_corpus_dir + "passage2id.map.json")
    id_passage_map = {idx: passage for passage, idx in passage_id_map.items()}
    # test_res = load_json("data/bm25/sorted/temp_test_res.json")
    test_ori = load_json("/home/awu/experiments/geo/others/testcb/data/dureader_dataset/test1.json", line=True)
    for t_ori, t_res in tqdm(zip(test_ori, data.items())):
        assert t_res[0] == t_ori['question_id']
        t_ori['retrieval_res'] = [all_paras[int(id_passage_map[_])] for _ in t_res[1]]
        t_ori['ids'] = t_res[1]
    dump_json(test_ori, "data/dureader_dataset/test_ce_rerank.json")


def test_rerank_to_submit():
    data = load_json("data/output_res.json")
    output = {}
    for t in tqdm(data):
        res_ids = []
        for _, _, p in t['res']:
            tid = t['retrieval_res'].index(p)
            res_ids.append(t['ids'][1][tid])
        output[t['question_id']] = res_ids
        assert len(res_ids) == 50
    dump_json(output, "data/test_rerank_res.json")


def OBQAEvaluate():
    # todo
    def eval_obqa(data):
        topk = [10, 20, 30, 100, 200, 500]
        accuracy = {k: [] for k in topk}
        max_k = max(topk)
        for t in tqdm(data):
            answer = t['answers'][0]
            contexts = t['res']
            has_ans_idx = max_k  # first index in contexts that has answers
            for idx, ctx in enumerate(contexts):
                if idx >= max_k:
                    break
                if answer in ctx['paragraph']:
                    has_ans_idx = idx
                    break

            for k in topk:
                accuracy[k].append(0 if has_ans_idx >= k else 1)
                t['hit@' + str(k)] = accuracy[k][-1]

        for k in topk:
            print(f'Top{k}\taccuracy: {np.mean(accuracy[k])}')
        return accuracy


def test_dense_retriever():
    dr = DenseRetrieverServer()
    print(dr.retrieve(questions=["中国的首都"], topk=16, faiss_depth=1024, nprobe=32))


def server_start():
    dr = DenseRetrieverServer()
    dr.server()


def test_client():
    dr_client = DenseRetrieverClient()
    print(dr_client.retrieve(questions=["北京奥运会"], topk=16, faiss_depth=256, nprobe=128))


if __name__ == '__main__':
    # test_client()
    # server_start()
    # test_res_to_test_rerank()
    # test_rerank_to_submit()
    # exit()
    comm = sys.argv[1]
    if comm == "server":
        server_start()
    else:
        dureader_evaluate()
