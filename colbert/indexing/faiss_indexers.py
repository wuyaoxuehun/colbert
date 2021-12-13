#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
 FAISS-based index components for dense retriever
"""
from functools import partial

import faiss
import logging
import numpy as np
import os
import pickle

from typing import List, Tuple

import torch
from tqdm import tqdm

from colbert.utils.utils import print_message
from colbert.indexing.loaders import get_parts
from colbert.indexing.index_manager import load_index_part
from conf import dim
from colbert.ranking.faiss_index import FaissIndex
from colbert.ranking.index_part import IndexPart
from colbert.indexing.myfaiss import prepare_faiss_index

logger = logging.getLogger()


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
        self.index = faiss.IndexFlatIP(vector_sz)

    def to_gpu(self, rank):
        if rank is not None:
            print('index to gpu!')
            res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(res, rank, self.index)

    def index_data(self, data: List[Tuple[object, np.array]]):
        n = len(data)
        # indexing in batches is beneficial for many faiss index types
        for i in range(0, n, self.buffer_size):
            db_ids = [t[0] for t in data[i: i + self.buffer_size]]
            vectors = [np.reshape(t[1], (1, -1)) for t in data[i: i + self.buffer_size]]
            vectors = np.concatenate(vectors, axis=0)
            total_data = self._update_id_mapping(db_ids)
            self.index.add(vectors)
            logger.info("data indexed %d", total_data)

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
    def __init__(self, index_path, dim=None, rank=None):
        self.index_path = index_path
        self.dim = dim
        self.rank = rank

    def load_index(self):
        raise NotImplementedError

    def encode_corpus(self, encoded_corpus_path):
        raise NotImplementedError

    def search(self, query, topk_doc=None):
        raise NotImplementedError

    def get_corpus_vectors(self, encoded_corpus_path):
        print_message("#> Starting..")
        parts, parts_paths, samples_paths = get_parts(encoded_corpus_path)
        slice_parts_paths = parts_paths
        sub_collection = [load_index_part(filename) for filename in slice_parts_paths if filename is not None]
        sub_collection = torch.cat(sub_collection)
        sub_collection = sub_collection.float().cpu().numpy()
        print_message("#> Processing a sub_collection with shape", sub_collection.shape)
        return sub_collection


class ColbertRetriever(DenseFaissRetriever):
    def __init__(self, index_path=None, dim=None, rank=None, index_config=None, partitions=None, sample=None):
        super().__init__(index_path, dim, rank)
        self.index_config = index_config
        self.faiss_index = None
        self.retrieve = None
        self.index = None
        self.partitions = partitions
        self.rank = rank
        self.sample = sample

    def load_index(self):
        index_config = self.index_config
        logger.info('loading index from ' + index_config['index_path'])
        logger.info(index_config)
        self.faiss_index = FaissIndex(index_config['index_path'], index_config['faiss_index_path'],
                                      index_config['n_probe'], part_range=index_config['part_range'], rank=self.rank)
        self.retrieve = partial(self.faiss_index.retrieve, self.index_config['faiss_depth'])
        self.index = IndexPart(index_config['index_path'], dim=index_config['dim'], part_range=index_config['part_range'], verbose=True)

    def get_colbert_index(self):
        parts, parts_paths, samples_paths = get_parts(self.index_path)
        return prepare_faiss_index(parts_paths, self.partitions, self.sample)

    def encode_corpus(self, encoded_corpus_path):
        collection = self.get_corpus_vectors(encoded_corpus_path)
        colbert_faiss_index = self.get_colbert_index()
        colbert_faiss_index.add(collection)
        print_message("Done indexing!")
        output_path = os.path.join(self.index_path, "ivfpq.2000.faiss")
        print_message(f"#> writing to {output_path}.")
        colbert_faiss_index.save(output_path)

    def search(self, query, topk_doc=None):
        Q, q_word_mask = query
        assert len(Q.size()) == 2
        pids = self.retrieve(Q.unsqueeze(0), verbose=False)[0]
        weighted_q = Q * q_word_mask[:, None]
        Q = weighted_q.unsqueeze(0)
        Q = Q.permute(0, 2, 1)
        pids = self.index.ranker.rank_forward(Q, pids, depth=topk_doc)
        return pids


class DPRRetriever(DenseFaissRetriever):
    def __init__(self, index_path, dim=None, rank=None):
        super().__init__(index_path, dim, rank)
        self.index = DenseFlatIndexer()

        # self.index = DenseHNSWFlatIndexer()
        # self.index = DenseHNSWSQIndexer()
        self.index.init_index(vector_sz=dim)
        self.rank = rank

    def load_index(self):
        self.index.deserialize(path=self.index_path)
        if self.rank is not None:
            self.index.to_gpu(rank=self.rank)

        print(f"loaded index from {self.index_path}")

    def encode_corpus(self, encoded_corpus_path):
        collection = self.get_corpus_vectors(encoded_corpus_path)
        data = [(i, vec) for i, vec in enumerate(collection)]
        print("indexing data")
        self.index.index_data(data)
        self.index.serialize(file=self.index_path)
        print(f"index completed, saved to {self.index_path}")

    def search(self, query, topk_doc=None):
        # Q, q_word_mask = query
        # assert len(Q.size()) == 2
        # query_vectors = np.array(Q).astype(np.float32)
        query_vectors = query
        pids = self.index.search_knn(query_vectors=query_vectors, top_docs=topk_doc)[0]
        return pids


class DenseHNSWFlatIndexer(DenseIndexer):
    """
    Efficient index for retrieval. Note: default settings are for hugh accuracy but also high RAM usage
    """

    def __init__(
            self,
            buffer_size: int = 1e9,
            store_n: int = 512,
            ef_search: int = 128,
            ef_construction: int = 200,
    ):
        super(DenseHNSWFlatIndexer, self).__init__(buffer_size=buffer_size)
        self.store_n = store_n
        self.ef_search = ef_search
        self.ef_construction = ef_construction
        self.phi = 0

    def init_index(self, vector_sz: int):
        # IndexHNSWFlat supports L2 similarity only
        # so we have to apply DOT -> L2 similairy space conversion with the help of an extra dimension
        index = faiss.IndexHNSWFlat(vector_sz + 1, self.store_n)
        index.hnsw.efSearch = self.ef_search
        index.hnsw.efConstruction = self.ef_construction
        self.index = index

    def index_data(self, data: List[Tuple[object, np.array]]):
        n = len(data)

        # max norm is required before putting all vectors in the index to convert inner product similarity to L2
        if self.phi > 0:
            raise RuntimeError(
                "DPR HNSWF index needs to index all data at once," "results will be unpredictable otherwise."
            )
        phi = 1
        # for i, item in enumerate(data):
        #     id, doc_vector = item[0:2]
        #     norms = (doc_vector ** 2).sum()
        #     phi = max(phi, norms)
        # logger.info("HNSWF DotProduct -> L2 space phi={}".format(phi))
        self.phi = phi

        # indexing in batches is beneficial for many faiss index types
        bs = int(self.buffer_size)
        for i in range(0, n, bs):
            db_ids = [t[0] for t in data[i: i + bs]]
            vectors = [np.reshape(np.array(t[1]), (1, -1)) for t in data[i: i + bs]]

            # norms = [(doc_vector ** 2).sum() for doc_vector in vectors]
            # aux_dims = [np.sqrt(phi - norm) for norm in norms]
            aux_dims = [np.sqrt(phi - 1) for vec in vectors]
            hnsw_vectors = [np.hstack((doc_vector, aux_dims[i].reshape(-1, 1))) for i, doc_vector in enumerate(vectors)]
            hnsw_vectors = np.concatenate(hnsw_vectors, axis=0)
            hnsw_vectors = hnsw_vectors.astype(np.float32)
            self.train(hnsw_vectors)

            self._update_id_mapping(db_ids)
            self.index.add(hnsw_vectors)
            logger.info("data indexed %d", len(self.index_id_to_db_id))
        indexed_cnt = len(self.index_id_to_db_id)
        logger.info("Total data indexed %d", indexed_cnt)

    def train(self, vectors: np.array):
        pass

    def search_knn(self, query_vectors: np.array, top_docs: int) -> List[Tuple[List[object], List[float]]]:

        aux_dim = np.zeros(len(query_vectors), dtype="float32")
        query_nhsw_vectors = np.hstack((query_vectors, aux_dim.reshape(-1, 1)))
        logger.info("query_hnsw_vectors %s", query_nhsw_vectors.shape)
        scores, indexes = self.index.search(query_nhsw_vectors, top_docs)
        # convert to external ids
        db_ids = [[self.index_id_to_db_id[i] for i in query_top_idxs] for query_top_idxs in indexes]
        result = [(db_ids[i], scores[i]) for i in range(len(db_ids))]
        return result

    def deserialize(self, file: str):
        super(DenseHNSWFlatIndexer, self).deserialize(file)
        # to trigger exception on subsequent indexing
        self.phi = 1

    def get_index_name(self):
        return "hnsw_index"


class DenseHNSWSQIndexer(DenseHNSWFlatIndexer):
    """
    Efficient index for retrieval. Note: default settings are for hugh accuracy but also high RAM usage
    """

    def __init__(
            self,
            buffer_size: int = 1e10,
            store_n: int = 128,
            ef_search: int = 128,
            ef_construction: int = 200,
    ):
        super(DenseHNSWSQIndexer, self).__init__(
            buffer_size=buffer_size,
            store_n=store_n,
            ef_search=ef_search,
            ef_construction=ef_construction,
        )

    def init_index(self, vector_sz: int):
        # IndexHNSWFlat supports L2 similarity only
        # so we have to apply DOT -> L2 similairy space conversion with the help of an extra dimension
        index = faiss.IndexHNSWSQ(vector_sz + 1, faiss.ScalarQuantizer.QT_8bit, self.store_n)
        index.hnsw.efSearch = self.ef_search
        index.hnsw.efConstruction = self.ef_construction
        self.index = index

    def train(self, vectors: np.array):
        self.index.train(vectors)

    def get_index_name(self):
        return "hnswsq_index"


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
    test_indexer()
