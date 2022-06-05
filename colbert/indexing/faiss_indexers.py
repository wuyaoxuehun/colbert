import logging
import os
import pickle
from typing import List, Tuple
import faiss
import math
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from colbert.indexing.faiss_index import FaissIndexing
from colbert.indexing.index_manager import load_index_part
from colbert.indexing.loaders import get_parts, load_doclens
from colbert.utils.utils import print_message

logger = logging.getLogger(__name__)


class DenseIndexer(object):
    def __init__(self, buffer_size: int = 50000):
        self.buffer_size = buffer_size
        self.index_id_to_db_id = []
        self.index = None

    def init_index(self, vector_sz: int):
        raise NotImplementedError

    def index_data(self, data: List[Tuple[object, np.array]]):
        raise NotImplementedError

    def get_index_name(self):
        raise NotImplementedError

    def search_knn(self, query_vectors: np.array, top_docs: int) -> List[Tuple[List[object], List[float]]]:
        raise NotImplementedError

    def serialize(self, file: str):
        logger.info("Serializing index to %s", file)

        if os.path.isdir(file):
            index_file = os.path.join(file, "index.dpr")
            meta_file = os.path.join(file, "index_meta.dpr")
        else:
            index_file = file + ".index.dpr"
            meta_file = file + ".index_meta.dpr"

        faiss.write_index(self.index, index_file)
        with open(meta_file, mode="wb") as f:
            pickle.dump(self.index_id_to_db_id, f)

    def get_files(self, path: str):
        if os.path.isdir(path):
            index_file = os.path.join(path, "index.dpr")
            meta_file = os.path.join(path, "index_meta.dpr")
        else:
            index_file = path + ".{}.dpr".format(self.get_index_name())
            meta_file = path + ".{}_meta.dpr".format(self.get_index_name())
        return index_file, meta_file

    def index_exists(self, path: str):
        index_file, meta_file = self.get_files(path)
        return os.path.isfile(index_file) and os.path.isfile(meta_file)

    def deserialize(self, path: str):
        logger.info("Loading index from %s", path)
        index_file, meta_file = self.get_files(path)

        self.index = faiss.read_index(index_file)
        logger.info("Loaded index of type %s and size %d", type(self.index), self.index.ntotal)

        with open(meta_file, "rb") as reader:
            self.index_id_to_db_id = pickle.load(reader)
        assert (
                len(self.index_id_to_db_id) == self.index.ntotal
        ), "Deserialized index_id_to_db_id should match faiss index size"

    def _update_id_mapping(self, db_ids: List) -> int:
        self.index_id_to_db_id.extend(db_ids)
        return len(self.index_id_to_db_id)


class DenseFlatIndexer(DenseIndexer):
    def __init__(self, buffer_size: int = 50000):
        super(DenseFlatIndexer, self).__init__(buffer_size=buffer_size)

    def init_index(self, vector_sz: int):
        # self.index = faiss.IndexFlatIP(vector_sz)
        self.index = faiss.IndexFlatL2(vector_sz)

    def to_gpu(self, rank):
        if rank is not None:
            print('index to gpu!')
            res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(res, rank, self.index)

    def index_data(self, data: List[Tuple[object, np.array]]):
        n = len(data)
        # indexing in batches is beneficial for many faiss index types
        for i in tqdm(range(0, n, self.buffer_size)):
            db_ids = [t[0] for t in data[i: i + self.buffer_size]]
            vectors = [np.reshape(t[1], (1, -1)) for t in data[i: i + self.buffer_size]]
            vectors = np.concatenate(vectors, axis=0)
            total_data = self._update_id_mapping(db_ids)
            self.index.add(vectors)
            # logger.info("data indexed %d", total_data)

        indexed_cnt = len(self.index_id_to_db_id)
        logger.info("Total data indexed %d", indexed_cnt)
        self.index = faiss.index_gpu_to_cpu(self.index)

    # def search_knn(self, query_vectors: np.array, top_docs: int) -> List[Tuple[List[object], List[float]]]:
    def search_knn(self, query_vectors: np.array, top_docs: int):
        scores, indexes = self.index.search(query_vectors, top_docs)

        # convert to external ids
        db_ids = [[self.index_id_to_db_id[i] for i in query_top_idxs] for query_top_idxs in indexes]
        # result = [(db_ids[i], scores[i]) for i in range(len(db_ids))]
        # return result
        return db_ids, scores

    def get_index_name(self):
        return "flat_index"


class DenseFaissRetriever:
    def __init__(self, index_path):
        self.index_path = index_path
        self.num_embeddings = sum(load_doclens(index_path))
        print("#> num_embeddings =", self.num_embeddings)
        # self.dim = dim
        # self.rank = rank

    def load_index(self):
        raise NotImplementedError

    def encode_corpus(self, encoded_corpus_path):
        raise NotImplementedError

    def search(self, query, topk_doc=None):
        raise NotImplementedError

    def get_corpus_vectors(self, encoded_corpus_path, l2norm=True):
        print_message("#> Starting..")
        parts, parts_paths, samples_paths = get_parts(encoded_corpus_path)
        slice_parts_paths = parts_paths
        for filename in tqdm(slice_parts_paths):
            if filename is None:
                continue
            sub_collection = load_index_part(filename)
            sub_collection = sub_collection.cpu().float().numpy()
            print_message("#> Processing a sub_collection with shape", sub_collection.shape)
            yield sub_collection
            del sub_collection


from colbert.ranking.colbert_ranker import ColbertIndex, ColbertRanker


class ColbertRetriever(DenseFaissRetriever):
    def __init__(self, index_path=None, faiss_index_path=None, faiss_depth=None, nprobe=None, rank=None, partitions=None, model=None, dim=None, **kwargs):
        super().__init__(index_path)
        self.faiss_index = None
        # self.retrieve = None
        self.index = None
        self.dim = dim
        self.partitions = partitions if partitions is not None else get_best_partitons(self.num_embeddings)
        self.rank = rank
        self.model = model
        self.faiss_depth = faiss_depth if faiss_depth is not None else 256
        self.nprobe = nprobe if nprobe is not None else 64
        self.faiss_index_path = faiss_index_path if faiss_index_path is not None else (os.path.join(self.index_path, "ivfpq.2000.faiss"))
        # self.faiss_index_path = os.path.join(self.index_path, "index.dpr")

    def load_index(self):
        logger.info('loading index from ' + self.index_path)
        self.faiss_index = ColbertIndex(index_path=self.index_path, faiss_index_path=self.faiss_index_path, nprobe=self.nprobe, rank=self.rank)
        # self.retrieve = partial(self.faiss_index.retrieve, self.index_config['faiss_depth'])
        # self.index = IndexPart(index_config['index_path'], dim=index_config['dim'], part_range=index_config['part_range'], verbose=True, model=self.model)
        self.ranker = ColbertRanker(index_path=self.index_path, model=self.model, dim=self.dim)

    # def set_faiss_depth_nprobe(self, faiss_depth=None, nprobe=None):
    # if faiss_depth is not None:
    #     self.retrieve = partial(self.faiss_index.retrieve, faiss_depth)
    # if nprobe is not None:
    #     self.faiss_index.faiss_index.nprobe = nprobe

    def set_faiss_nprobe(self, nprobe=None):
        if nprobe is not None:
            self.faiss_index.faiss_index.nprobe = nprobe

    def get_trained_colbert_index(self, collection):
        training_sample = collection
        dim = training_sample.shape[-1]
        index = FaissIndexing(dim, self.partitions)
        print_message("#> Training with the vectors...")
        index.train(training_sample)
        print_message("Done training!\n")
        return index

    def get_sample_corpus(self, encoded_corpus_path):
        # print_message("#> Starting..")
        parts, parts_paths, samples_paths = get_parts(encoded_corpus_path)
        slice_parts_paths = parts_paths
        sub_collection = load_index_part(slice_parts_paths[0])  # use the first part as collection for training
        sub_collection = sub_collection.cpu().float().numpy()
        # print_message("#> Processing a sub_collection with shape", sub_collection.shape)
        return sub_collection

    def encode_corpus(self, encoded_corpus_path):
        sample_collection = self.get_sample_corpus(encoded_corpus_path)
        colbert_faiss_index = self.get_trained_colbert_index(sample_collection)
        for sub_collection in self.get_corpus_vectors(encoded_corpus_path):
            colbert_faiss_index.add(sub_collection)
        print_message("Done indexing!")
        output_path = os.path.join(self.index_path, "ivfpq.2000.faiss")
        print_message(f"#> writing to {output_path}.")
        colbert_faiss_index.save(output_path)

    def search(self, query: torch.Tensor, topk_doc=None, faiss_depth=None, **kwargs):
        Q = query
        assert len(Q.shape) == 2
        # pids = self.retrieve(Q.unsqueeze(0), verbose=False)[0]
        self.faiss_index: ColbertIndex
        faiss_depth = self.faiss_depth if faiss_depth is None else faiss_depth
        pids = self.faiss_index.retrieve(faiss_depth, Q=query.unsqueeze(0))
        pids = pids[0]
        Q = Q.unsqueeze(0)
        Q = Q.permute(0, 2, 1)
        pid_scores = self.ranker.rank_forward(Q, pids, depth=topk_doc)
        return pid_scores


class DPRRetriever(DenseFaissRetriever):
    def __init__(self, index_path, dim):
        super().__init__(index_path)
        self.index = DenseFlatIndexer()

        # self.index = DenseHNSWFlatIndexer()
        # self.index = DenseHNSWSQIndexer()
        self.index.init_index(vector_sz=dim)
        # self.rank = rank

    def load_index(self, rank=None):
        self.index.deserialize(path=self.index_path)
        if rank is not None:
            self.index.to_gpu(rank=rank)
        # self.index.to_gpu(0)
        print(f"loaded index from {self.index_path}")

    def set_faiss_nprobe(self, nprobe=None):
        if nprobe is not None:
            self.index.nprobe = nprobe

    def encode_corpus(self, encoded_corpus_path):
        idx = 0
        collection = self.get_corpus_vectors(encoded_corpus_path)
        for sub_collection in collection:
            data = [(i + idx, vec) for i, vec in enumerate(sub_collection)]
            # print("indexing data")
            self.index.index_data(data)
            idx += len(data)
        self.index.serialize(file=self.index_path)
        print(f"index completed, saved to {self.index_path}")

    def search(self, query, topk_doc=None, **kwargs):
        # Q, q_word_mask = query
        # assert len(Q.size()) == 2
        query_vectors = np.array(query.cpu()).astype(np.float32)
        # query_vectors = query
        pids = self.index.search_knn(query_vectors=query_vectors, top_docs=topk_doc)[0]
        return pids


def get_best_partitons(num_embeddings):
    # partitions = 1 << math.ceil(math.log2(8 * math.sqrt(num_embeddings)))
    partitions = 1 << round(math.log2(8 * math.sqrt(num_embeddings)))
    print('\n\n')
    print("You did not specify --partitions!")
    print(f'''Default computation chooses {partitions} partitions (for {num_embeddings} embeddings)''')
    print('\n\n')
    return partitions


def test_colbert_indexer():
    index_path = "/home/awu/testcb/index/colbert"
    # from line_profiler import LineProfiler
    # lp = LineProfiler()
    # lp_wrapper = lp(ColbertRetriever)
    # retriever = ColbertRetriever(index_path=index_config['index_path'], rank=0, index_config=index_config)
    retriever = ColbertRetriever(index_path=index_path, dim=128)
    retriever.load_index()
    query = torch.randn(2, 128)
    query = F.normalize(query, p=2, dim=-1)
    res = retriever.search(query, topk_doc=4)
    print(res)


def test_indexer():
    flat_indexer = DenseFlatIndexer()
    vec_num = 1000000
    dim = 64
    flat_indexer.init_index(vector_sz=dim)
    vecs = np.random.random((vec_num, dim)).astype(np.float32)
    ids = list(range(vec_num))
    flat_indexer.index_data(list(zip(ids, vecs)))
    for i in tqdm(range(10)):
        tvecs = vecs[i * 10:(i + 1) * 10].copy()
        print(tvecs.shape)
        res = flat_indexer.search_knn(query_vectors=tvecs, top_docs=10)
        print(res)


if __name__ == '__main__':
    # test_indexer()
    test_colbert_indexer()
    # test_term_manager()
