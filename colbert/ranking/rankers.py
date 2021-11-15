import torch

from functools import partial

from colbert.ranking.index_part import IndexPart
from colbert.ranking.faiss_index import FaissIndex
import myfaiss

class Ranker():
    def __init__(self, args, inference, faiss_depth=1024):
        self.inference = inference
        self.faiss_depth = faiss_depth
        # print(args.index_path, args.faiss_index_path, args.part_range)
        # input()
        if faiss_depth is not None:
            self.faiss_index = FaissIndex(args.index_path, args.faiss_index_path, args.nprobe, part_range=args.part_range)
            self.retrieve = partial(self.faiss_index.retrieve, self.faiss_depth)

        self.index = IndexPart(args.index_path, dim=inference.colbert.dim, part_range=args.part_range, verbose=True)

    def encode(self, queries, bsize=64):
        assert type(queries) in [list, tuple], type(queries)
        # Q = self.inference.query_tokenizer.tensorize(queries)
        Q = self.inference.query_tokenizer.tensorize_dict(queries)
        Q = self.inference.queryFromTensorize(Q, keep_dims=False, bsize=bsize if len(queries) > 512 else None)

        return Q

    def rank(self, Q, pids=None):
        pids = self.retrieve(Q, verbose=False)[0] if pids is None else pids

        assert type(pids) in [list, tuple], type(pids)
        assert Q.size(0) == 1, (len(pids), Q.size())
        assert all(type(pid) is int for pid in pids)
        scores = []
        if len(pids) > 1:
            if len(Q[0]) > 0:
                Q = Q.permute(0, 2, 1)
                scores = self.index.rank(Q, pids)
                scores_sorter = torch.tensor(scores).sort(descending=True)
                pids, scores = torch.tensor(pids)[scores_sorter.indices].tolist(), scores_sorter.values.tolist()
            else:
                pids = pids
                scores = [0] * len(pids)
        return pids, scores
