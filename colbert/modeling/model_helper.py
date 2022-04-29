class ModelHelper:
    def __init__(self, index_config, rank=0):
        self.conn = None
        self.all_paras = None
        self.all_paras = None
        # index_config['index_path'] = args.index_path
        # index_config['faiss_index_path'] = args.index_path + "ivfpq.2000.faiss"
        self.helper = None
        # if self.all_paras is None:
        #     self.all_paras = load_all_paras()
        #     from colbert.training.model_helper_server import ModelHelperServer
        #     self.helper = ModelHelperServer(index_config, rank=get_rank())

    def load_paras(self):
        self.all_paras = load_all_paras()

    def to_real_batch(self, batches, q_word_mask, retrieve_topk):
        batches = batches.cpu().detach().to(dtype=torch.float16)
        q_word_mask = q_word_mask.cpu().detach()
        batches = [qd_mask_to_realinput(Q=q, q_word_mask=qw_mask, keep_dim=False) for q, qw_mask in
                   zip(batches, q_word_mask)]
        kwargs = {"expand_size": 5, "expand_center_size": 24, "expand_per_emb": 16, "expand_topk_emb": 3, "expand_weight": 0.000000000001}
        res = batches, retrieve_topk, faiss_depth, nprobe, kwargs
        return res

    def retrieve_for_encoded_queries(self, batches, q_word_mask=None, retrieve_topk=10):
        if self.conn is None:
            address = ('localhost', 9090)
            self.conn = Client(address, authkey=b'1')
            print('connected to server')
            self.all_paras = load_all_paras()
        retrieve_input = self.to_real_batch(batches, q_word_mask, retrieve_topk)
        self.conn.send(retrieve_input)
        data = self.conn.recv()
        # batch_pids, *extra = list(zip(*data))
        #
        # batch_paras = [[self.all_paras[pid] for pid in pids] for pids in batch_pids]
        # # batch_paras = [[self.all_paras[np.random.randint(0, len(self.all_paras))] for pid in pids] for pids in batch_pids]
        # # return batch_D, batch_D_mask, batch_paras
        # return None, None, batch_paras, extra
        batch_pids = list(zip(*data))
        batch_pids, batch_scores = batch_pids
        batch_paras = [[self.all_paras[pid] for pid in pids] for pids in batch_pids]
        return batch_paras, batch_scores

    @torch.no_grad()
    def retrieve_for_encoded_queries_(self, batches, q_word_mask=None, retrieve_topk=10):
        # if self.all_paras is None:
        #     self.all_paras = load_all_paras()
        #     from colbert.training.model_helper_server import ModelHelperServer
        #
        #     self.helper = ModelHelperServer(index_config, rank=get_rank())
        retrieve_input = self.to_real_batch(batches, q_word_mask, retrieve_topk)
        data = self.helper.direct_retrieve(retrieve_input)
        # print(len(data))
        # batch_pids, *extra = list(zip(*data))
        batch_pids = list(zip(*data))
        batch_pids, batch_scores = batch_pids
        batch_paras = [[self.all_paras[pid] for pid in pids] for pids in batch_pids]
        # input(batch_scores)

        # batch_paras = [[self.all_paras[np.random.randint(0, len(self.all_paras))] for pid in pids] for pids in batch_pids]
        # return batch_D, batch_D_mask, batch_paras
        return batch_paras, batch_scores

    def close(self):
        if self.conn is not None:
            self.conn.send("close")
            print("closed")
            return