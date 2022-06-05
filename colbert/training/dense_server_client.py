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
from proj_utils.dureader_utils import get_dureader_ori_corpus

os.environ["CUDA_VISIBLE_DEVICES"] = "2,3,4"


class DenseRetrieverServer:
    def __init__(self):
        args = load_dense_conf()
        self.index_path = args.dense_index_args.index_path
        self.dim = args.dense_training_args.dim
        faiss_type = args.faiss_index_args.faiss_type
        self.colbert = ColbertModel(args.dense_training_args)
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
            print('connection accepted from', listener.last_accepted)
            params = conn.recv()
            pids = self.retrieve(*params)
            conn.send(pids)


class DenseRetrieverClient:
    def __init__(self):
        address = ('localhost', 9090)
        self.conn = Client(address, authkey=b'1')
        print('connected to server')

    def retrieve(self, questions, topk, faiss_depth, nprobe):
        self.conn.send((questions, topk, faiss_depth, nprobe))
        data = self.conn.recv()
        return data


def dureader_evaluate():
    dr_client = DenseRetrieverClient()

    def eval_dureader(output_data):
        topk = 10
        recall_topk = 50
        res = 0
        recall_res = 0
        for t in output_data:
            for i in range(topk):
                if t['res'][i][2] in t['positive_ctxs']:
                    res += 1 / (i + 1)
                    break
            for i in range(recall_topk):
                if t['res'][i][2] in t['positive_ctxs']:
                    recall_res += 1
                    break

        print(f"mrr@10 = {res / len(output_data)}")
        print(f"recall@50 = {recall_res / len(output_data)}")

    def test_to_submit():
        dureader_corpus_dir = "/home2/awu/testcb/data/dureader/dureader-retrieval-baseline-dataset/passage-collection/"
        passage_id_map = load_json(dureader_corpus_dir + "passage2id.map.json")
        # test_res = load_json("data/bm25/sorted/temp_test_res.json")
        test_ori = load_json("/home2/awu/testcb/data/dureader/dureader-retrieval-test1/test1.json", line=True)
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
        res = dr_client.retrieve(questions=questions, topk=50, faiss_depth=256, nprobe=128)
        print(len(res), len(res[0]))
        input()
        print(len(questions), len(res))
        for t_ori, t in zip(data, res):
            t_ori['res'] = t
        return data

    def eval_for_dev():
        dev_data = load_json(data_dir_dic['dureader']('dev', 0))
        dev_data = eval_for_ds(dev_data)
        eval_dureader(output_data=dev_data)

    def eval_for_test():
        pass

    # eval_for_dev()
    test_to_submit()


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
    print(dr.retrieve(questions=["中国的首都"], topk=16, faiss_depth=256, nprobe=128))


def server_start():
    dr = DenseRetrieverServer()
    dr.server()


def test_client():
    dr_client = DenseRetrieverClient()
    print(dr_client.retrieve(questions=["北京奥运会"], topk=16, faiss_depth=256, nprobe=128))


if __name__ == '__main__':
    # test_client()
    # server_start()
    comm = sys.argv[1]
    if comm == "server":
        server_start()
    else:
        dureader_evaluate()
