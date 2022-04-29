import sys

import torch
from line_profiler import LineProfiler

from colbert.modeling.tokenization import QueryTokenizer
from conf import index_config
# from corpus_cb import load_all_paras
from colbert.indexing.faiss_indexers import ColbertRetriever
# from colbert.training.training_utils import *
import argparse
import multiprocessing as mp


class ModelHelperServer:
    def __init__(self, index_config, rank=0, model=None):
        self.query_tokenizer = QueryTokenizer(index_config["query_maxlen"], None)
        self.index_config = index_config
        self.faiss_index = None
        self.retrieve = None
        self.model = model
        self.rank = rank
        # self.depth = index_config['depth']
        self.dim = index_config['dim']
        self.index = None
        # self.all_paras = load_all_paras()
        self.init()

    # def query_tokenize(self, queries):
    #     Q = self.query_tokenizer.tensorize_allopt_dict(queries)
    #     return Q

    def init(self):
        self.retriever = ColbertRetriever(index_path=self.index_config['index_path'], rank=self.rank, index_config=self.index_config, model=self.model)
        self.retriever.load_index()
        # self.retriever = DPRRetriever(index_path=self.index_config['index_path'], dim=dim, rank=self.rank)
        # self.retriever.load_index()
        self.retrieve = self.retriever.search

    def retrieve_for_encoded_queries(self, batches, retrieve_topk=10, faiss_depth=None, nprobe=None, **kwargs):
        # batches = torch.clone(batches).cpu().detach().to(dtype=torch.float16)
        batch_pids = []
        if faiss_depth is not None and nprobe is not None:
            self.retriever.set_faiss_depth_nprobe(faiss_depth, nprobe)
        with torch.no_grad():
            if False:
                Q = np.concatenate([np.array(_[0]) for _ in batches], axis=0).astype(np.float32)
                batch_pids += self.retrieve(query=Q, topk_doc=retrieve_topk)
            else:
                for i, (q, q_word_mask) in enumerate(batches):
                    # print(q.size(), q_word_mask.size())
                    # input()
                    pids = self.retrieve(query=q, topk_doc=retrieve_topk, **kwargs)
                    batch_pids.append(pids)

        return batch_pids

    def direct_retrieve(self, msg):
        batches, retrieve_topk, *others = msg
        # if len(others) > 0:
        #     faiss_depth, nprobe = others[:2]
        #     self.retriever.set_faiss_depth_nprobe(faiss_depth, nprobe)

        # batch_pids = self.retrieve_for_encoded_queries(batches, q_word_mask, retrieve_topk, **(others[2]))[-1]
        batch_pids = self.retrieve_for_encoded_queries(batches, retrieve_topk, **(others[2]))
        return batch_pids


global helper


def serve_one(conn, lock):
    try:
        while True:
            msg = conn.recv()
            if msg == 'close':
                conn.close()
                break
            batches, retrieve_topk, *others = msg
            # if len(others) > 0:
            #     faiss_depth, nprobe = others[:2]
            #     helper.retriever.set_faiss_depth_nprobe(faiss_depth, nprobe)
            # batch_pids = helper.retrieve_for_encoded_queries(batches, q_word_mask, retrieve_topk, **(others[2]))[-1]
            # lock.acquire()
            batch_pids = helper.retrieve_for_encoded_queries(batches, retrieve_topk, **(others[2]))
            conn.send(batch_pids)
            # lock.release()
        # listener.close()
    except Exception as e:
        print(e)
        print('error')
        # listener.close()
        pass


def start_server(index_config):
    # torch.multiprocessing.set_start_method('spawn')
    from multiprocessing.connection import Listener
    from multiprocessing import Manager
    from threading import Thread
    manager = Manager()

    torch.multiprocessing.set_sharing_strategy("file_system")
    port_start = 9090
    address = ('localhost', port_start)  # family is deduced to be 'AF_INET'
    listener = Listener(address, authkey=b'1')
    lock = manager.Lock()
    while True:
        print("receiving connection --- --- ")
        conn = listener.accept()
        print('connection accepted from', listener.last_accepted)
        p1 = Thread(target=serve_one, args=(conn, lock))
        p1.start()


if __name__ == '__main__':
    # mp.set_start_method('spawn')
    parser = argparse.ArgumentParser()
    parser.add_argument("--index_path", dest='index_path', required=True, type=str)
    args = parser.parse_args()
    global index_config
    index_config['index_path'] = args.index_path
    index_config['faiss_index_path'] = args.index_path + "ivfpq.2000.faiss"
    helper = ModelHelperServer(index_config)

    start_server(index_config)
    exit()
    lp = LineProfiler()  # 把函数传递到性能分析器
    for f in [ColbertRetriever.search]:
        lp.add_function(f)

    lp_wrapper = lp(start_server)
    lp_wrapper(index_config)
    # profile.disable()  # 停止分析
    lp.print_stats(sys.stdout)  # 打印出性能分析结果
