import torch

from colbert.modeling.tokenization import QueryTokenizer
from conf import index_config
from corpus_cb import load_all_paras
from colbert.indexing.faiss_indexers import ColbertRetriever, DPRRetriever
from colbert.training.training_utils import qd_mask_to_realinput
import argparse


class ModelHelper:
    def __init__(self, index_config, rank=0):
        self.query_tokenizer = QueryTokenizer(index_config["query_maxlen"], None)
        self.index_config = index_config
        self.faiss_index = None
        self.retrieve = None
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
        self.retriever = ColbertRetriever(index_path=self.index_config['index_path'], rank=self.rank, index_config=self.index_config)
        self.retriever.load_index()
        # self.retriever = DPRRetriever(index_path=self.index_config['index_path'], dim=dim, rank=self.rank)
        # self.retriever.load_index()
        self.retrieve = self.retriever.search

    def retrieve_for_encoded_queries(self, batches, q_word_mask=None, retrieve_topk=10):
        # batches = torch.clone(batches).cpu().detach().to(dtype=torch.float16)
        batches = batches.cpu().detach().to(dtype=torch.float16)
        q_word_mask = q_word_mask.cpu().detach()
        batches = [qd_mask_to_realinput(Q=q, q_word_mask=qw_mask, keep_dim=False) for q, qw_mask in zip(batches, q_word_mask)]

        batch_pids = []
        with torch.no_grad():
            if False:
                Q = np.concatenate([np.array(_[0]) for _ in batches], axis=0).astype(np.float32)
                batch_pids += self.retrieve(query=Q, topk_doc=retrieve_topk)
            else:
                for i, (q, q_word_mask) in enumerate(batches):
                    # print(q.size(), q_word_mask.size())
                    # input()
                    pids = self.retrieve(query=(q, q_word_mask), topk_doc=retrieve_topk)
                    batch_pids.append(pids)
                # print([''.join(self.all_paras[pid]['paragraph_cut']['tok'].split()) for pid in pids[:2]])
            # batch_D, batch_D_mask = self.index.ranker.get_doc_embeddings_by_pids(sum(batch_pids, []))

            # batch_D = batch_D.view(len(batches), retrieve_topk, -1, self.dim)
            # batch_D_mask = batch_D_mask.view(len(batches), retrieve_topk, -1)

        # batch_paras = [[self.all_paras[pid] for pid in pids] for pids in batch_pids]
        # batch_paras = [[self.all_paras[np.random.randint(0, len(self.all_paras))] for pid in pids] for pids in batch_pids]
        # return batch_D, batch_D_mask, batch_paras
        # return None, None, batch_paras
        return None, None, batch_pids


def start_server(index_config):
    from multiprocessing.connection import Listener
    helper = ModelHelper(index_config)
    port_start = 6001
    while True:
        address = ('localhost', port_start)  # family is deduced to be 'AF_INET'
        listener = Listener(address, authkey=b'secret password')
        print("receiving connection --- --- ")
        conn = listener.accept()
        print('connection accepted from', listener.last_accepted)
        # try:
        while True:
            msg = conn.recv()
            if msg == 'close':
                conn.close()
                break
            batches, q_word_mask, retrieve_topk, *others = msg
            if len(others) > 0:
                faiss_depth, nprobe = others
                helper.retriever.set_faiss_depth_nprobe(faiss_depth, nprobe)

            batch_pids = helper.retrieve_for_encoded_queries(batches, q_word_mask, retrieve_topk)[-1]
            conn.send(batch_pids)
        # except Exception as e:
        #     print(e)

        listener.close()
        # port_start += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--index_path", dest='index_path', required=True, type=str)
    args = parser.parse_args()
    global index_config
    index_config['index_path'] = args.index_path
    index_config['faiss_index_path'] = args.index_path + "ivfpq.2000.faiss"
    start_server(index_config)
