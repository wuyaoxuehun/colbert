#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
 FAISS-based index components for dense retriever
"""
import time
from collections import defaultdict
from functools import partial, lru_cache
import torch.nn.functional as F
import faiss
import logging
import numpy as np
import os
import pickle

from typing import List, Tuple

import torch
from tqdm import tqdm

# from colbert.utils.utils import print_message
# from colbert.indexing.loaders import get_parts
# from colbert.indexing.index_manager import load_index_part
from conf import dim, index_config, corpus_tokenized_prefix, pretrain, corpus_index_term_path, use_prf
from colbert.ranking.index_part import IndexPart
from colbert.indexing.myfaiss import *
from colbert.ranking.faiss_index import FaissIndex as RankingFaissIndex
from colbert.modeling.tokenization import CostomTokenizer
from colbert.modeling.model_utils import batch_index_select

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

    def get_corpus_vectors(self, encoded_corpus_path, l2norm=True):
        print_message("#> Starting..")
        parts, parts_paths, samples_paths = get_parts(encoded_corpus_path)
        slice_parts_paths = parts_paths
        for filename in tqdm(slice_parts_paths):
            if filename is None:
                continue
            sub_collection = load_index_part(filename)
            # if l2norm:
            #     sub_collection = F.normalize(sub_collection, p=2, dim=-1)
            # sub_collection = sub_collection.cpu().numpy().astype(np.float32)
            sub_collection = sub_collection.cpu().float().numpy()
            print_message("#> Processing a sub_collection with shape", sub_collection.shape)
            yield sub_collection
            del sub_collection


from colbert.indexing.kmeans import kmeans


class ColbertRetriever(DenseFaissRetriever):
    def __init__(self, index_path=None, dim=None, rank=None, index_config=None, partitions=None, sample=None, model=None):
        super().__init__(index_path, dim, rank)
        self.index_config = index_config
        self.faiss_index = None
        self.retrieve = None
        self.index = None
        self.partitions = partitions
        self.rank = rank
        self.model = model
        if index_config is not None:
            self.faiss_depth = self.index_config['faiss_depth']
        self.sample = sample
        # self.index_term_manager = IndexTermManager()

    def load_index(self):
        index_config = self.index_config
        logger.info('loading index from ' + index_config['index_path'])
        logger.info(index_config)
        self.faiss_index = RankingFaissIndex(index_config['index_path'], index_config['faiss_index_path'],
                                             index_config['n_probe'], part_range=index_config['part_range'], rank=self.rank)
        self.retrieve = partial(self.faiss_index.retrieve, self.index_config['faiss_depth'])
        self.index = IndexPart(index_config['index_path'], dim=index_config['dim'], part_range=index_config['part_range'], verbose=True, model=self.model)

    def set_faiss_depth_nprobe(self, faiss_depth=None, nprobe=None):
        if faiss_depth is not None:
            self.retrieve = partial(self.faiss_index.retrieve, faiss_depth)
        if nprobe is not None:
            self.faiss_index.faiss_index.nprobe = nprobe

    def get_colbert_index(self, collection):
        # training_sample = np.array([collection[i] for i in np.random.choice(list(range(len(collection))), size=int(len(collection) * self.sample))])
        # training_sample = training_sample.squeeze(1)
        training_sample = collection
        dim = training_sample.shape[-1]
        index = FaissIndex(dim, self.partitions)
        print_message("#> Training with the vectors...")
        index.train(training_sample)
        print_message("Done training!\n")
        return index

    def get_sample_corpus(self, encoded_corpus_path):
        print_message("#> Starting..")
        parts, parts_paths, samples_paths = get_parts(encoded_corpus_path)
        slice_parts_paths = parts_paths
        sub_collection = load_index_part(slice_parts_paths[0])
        sub_collection = sub_collection.cpu().float().numpy()
        print_message("#> Processing a sub_collection with shape", sub_collection.shape)
        return sub_collection

    def encode_corpus(self, encoded_corpus_path):
        sample_collection = self.get_sample_corpus(encoded_corpus_path)
        colbert_faiss_index = self.get_colbert_index(sample_collection)
        for sub_collection in self.get_corpus_vectors(encoded_corpus_path):
            colbert_faiss_index.add(sub_collection)
        print_message("Done indexing!")
        output_path = os.path.join(self.index_path, "ivfpq.2000.faiss")
        print_message(f"#> writing to {output_path}.")
        colbert_faiss_index.save(output_path)

    @staticmethod
    def get_doclens():
        collection_dir = "/home2/awu/testcb/data/dureader/collection/"
        for i in tqdm(range(12)):
            file = f"{collection_dir}dureader_segmented_320_bert_tokenized_word_{i}.pt"
            data = torch.load(file)
            *_, active_padding = zip(*data)
            doclens = torch.tensor([sum(_) for _ in tqdm(active_padding)])
            torch.save(doclens, f"/home2/awu/testcb/index/geo/colbert_medqa_2e-2_weight/doclens.{i}.json")

    def search(self, query, topk_doc=None, filter_topk=None, **kwargs):
        Q = query
        assert len(Q.size()) == 2

        # softmax_word_weight = Q.norm(p=2, dim=-1)
        # print(softmax_word_weight)
        # input()
        # t = softmax_word_weight.sort(-1, descending=True)
        # sorted_ww = t.indices
        # ww = t.values
        # total = q_active_padding.sum(-1)
        # Q_new = Q[sorted_ww[:filter_topk]]
        t1 = time.time()
        search_Q = F.normalize(Q.to(torch.float32), p=2, dim=-1)
        # print(search_Q.norm(p=2, dim=-1))
        # input()
        # search_Q = F.normalize(Q_new.to(torch.float32), p=2, dim=-1)
        pids = self.retrieve(search_Q.unsqueeze(0), verbose=False)[0]
        t2 = time.time()
        # print(len(pids), Q.size())
        # input()
        # input(len(pids))
        # weighted_q = Q * q_word_mask[:, None]
        # Q = weighted_q.unsqueeze(0)
        Q = Q.unsqueeze(0)
        Q = Q.permute(0, 2, 1)
        pids = self.index.ranker.rank_forward(Q, pids, depth=topk_doc)
        t3 = time.time()
        # print(t3-t2, t2-t1)
        # input()
        return pids

    def search_(self, query, topk_doc=None, expand_size=10, expand_center_size=24, expand_per_emb=16, expand_topk_emb=3, expand_weight=0.5):
        Q, q_word_mask = query
        assert len(Q.size()) == 2
        search_Q = F.normalize(Q.to(torch.float32), p=2, dim=-1)
        pids, embedding_ids, scores = self.retrieve(search_Q.unsqueeze(0), verbose=False, output_embedding_ids=True)
        pids = pids[0]
        # input(scores)
        # input(scores.size())
        # print(len(pids), embedding_ids.size())
        # embedding_ids = embedding_ids.view(Q.size(0), self.faiss_depth)
        # scores = scores.view(Q.size(0), self.faiss_depth)
        # embeddings = self.index.tensor[embedding_ids]
        # # print(embeddings.size())
        # embeddings = embeddings[:, :expand_topk_emb, ...].cuda().view(-1, Q.size(-1)).contiguous()
        # scores = scores[:, :expand_topk_emb, ...].cuda().view(-1).contiguous()
        # # input(scores)
        # selected_embs = scores > 0.8
        # if int(sum(selected_embs)) > expand_size and False:
        #     embeddings = embeddings[selected_embs]
        #     # print(embeddings.size())
        #     # from kmeans_pytorch import kmeans
        #     from colbert.indexing.kmeans import kmeans
        #     cluster_ids_x, cluster_centers = kmeans(
        #         X=embeddings, num_clusters=expand_size, distance='euclidean', device="cuda"
        #     )
        #     # print(cluster_centers.size())
        #     cluster_centers = cluster_centers.to(Q.device)
        #     Q = torch.cat([Q, cluster_centers], dim=0)

        # input(len(pids))
        # weighted_q = Q * q_word_mask[:, None]
        # weighted_q = Q
        # Q = weighted_q.unsqueeze(0)
        # Q = Q.permute(0, 2, 1)
        pids, output_D, output_D_mask = self.index.ranker.rank_forward(Q.unsqueeze(0).permute(0, 2, 1), pids, depth=topk_doc, output_D_embedding=True)
        if not use_prf:
            return pids, [], []

        output_D, output_D_mask = output_D[:expand_topk_emb, ...], output_D_mask[:expand_topk_emb, ...]
        output_D, output_D_mask = output_D.view(-1, output_D.size(-1)), output_D_mask.view(-1)

        embeddings = output_D[output_D_mask.bool()]
        # from kmeans_pytorch import kmeans
        cluster_ids_x, cluster_centers = kmeans(
            X=embeddings, num_clusters=expand_center_size, device="cuda"
        )
        # from sklearn.cluster import KMeans
        # kmeans = KMeans(n_clusters=expand_center_size, random_state=0).fit(embeddings.cpu().numpy())
        # cluster_centers = torch.from_numpy(kmeans.cluster_centers_)

        # _, embedding_ids, _ = self.retrieve(cluster_centers.unsqueeze(0), verbose=False, output_embedding_ids=True)
        cluster_centers = F.normalize(cluster_centers, p=2, dim=-1)

        _, embedding_ids, _ = self.faiss_index.retrieve(expand_per_emb, cluster_centers.unsqueeze(0), verbose=False, output_embedding_ids=True)
        sigmas = []
        stemid_chooses = []
        # input(embedding_ids.shape)
        for embids in embedding_ids:
            try:
                stemids = [self.index_term_manager.embid2stemid[embid] for embid in embids]
                u, indices = np.unique(stemids, return_inverse=True)
                stemid_choose = u[np.argmax(np.bincount(indices))]
            except Exception as e:
                print(e)
                print(embids, embedding_ids.shape, cluster_centers.shape, embeddings.shape)
                exit()
            df = self.index_term_manager.stemid2df[stemid_choose]
            stemid_chooses.append(stemid_choose)
            sigma = np.log((self.index_term_manager.doc_num + 1) / (df + 1))
            sigmas.append(sigma)
            # print([self.index_term_manager.id2stem[stemid] for stemid in stemids])
            # print(self.index_term_manager.id2stem[stemid_choose])
            # print(df, sigma)
            # input()

        sigmas = torch.tensor(sigmas)
        sigmas_choose_idx = sigmas.argsort(descending=True)[:expand_size]
        choose_stems = sorted([(self.index_term_manager.id2stem[stemid], float(sigmas[idx]))
                               for idx, stemid in enumerate(stemid_chooses)], key=lambda x: x[1], reverse=True)
        expand_stems = [self.index_term_manager.id2stem[stemid_choose]
                        for idx, stemid_choose in enumerate(stemid_chooses) if idx in sigmas_choose_idx]
        # print("choose stems: ", choose_stems)
        # print("expand stems: ", expand_stems)
        sigmas = sigmas[sigmas_choose_idx]
        cluster_centers = cluster_centers[sigmas_choose_idx]

        # print(cluster_centers.size())
        # cluster_centers = cluster_centers.to(Q.device) * sigmas.to(Q.device)[:, None] * expand_weight
        cluster_centers = cluster_centers.to(Q.device) * expand_weight

        Q = torch.cat([Q, cluster_centers], dim=0)

        pids, output_D, output_D_mask = self.index.ranker.rank_forward(Q.unsqueeze(0).permute(0, 2, 1), pids, depth=topk_doc, output_D_embedding=True)
        # print(output_D.size())
        return pids, choose_stems, expand_stems


def load_pretensorize(start, end):
    collection = []
    for i in tqdm(range(start, end)):
        collection_path = corpus_tokenized_prefix + f"_{i}.pt"
        if not os.path.exists(collection_path):
            break
        sub_collection = torch.load(collection_path)
        collection += sub_collection
    return collection


class IndexTermManager:
    def __init__(self):

        # self.span2stemid = lambda x: '#'.join([str(_) for _ in x])
        self.stem2id = {}
        self.id2stem = {}
        self.stemid2df = defaultdict(int)
        self.stemid2cf = defaultdict(int)
        self.embid2stemid = []
        # self.stem_cache = {}
        self.span_cache = {}
        self.tokenzier = CostomTokenizer.from_pretrained(pretrain)

        from nltk.stem import PorterStemmer
        self.stemmer = PorterStemmer()
        if not os.path.exists(corpus_index_term_path) or False:
            self.parse()
            torch.save([self.stem2id, self.id2stem, self.stemid2df, self.stemid2cf, self.embid2stemid], corpus_index_term_path)
        else:
            self.stem2id, self.id2stem, self.stemid2df, self.stemid2cf, self.embid2stemid = torch.load(corpus_index_term_path)

    @property
    def doc_num(self):
        return 756674

    def embid2df(self, embid):
        return self.stemid2df[self.embid2stemid[embid]]

    def embid2cf(self, embid):
        return self.stemid2cf[self.embid2stemid[embid]]

    def embid2term(self, embid):
        return self.id2stem[self.embid2stemid[embid]]

    @lru_cache(maxsize=None)
    def get_stem(self, word):
        return self.stemmer.stem(word, to_lowercase=True)

    # @lru_cache(maxsize=None)
    def get_stem_by_span(self, span):
        key = '#'.join([str(_) for _ in span])
        if key not in self.span_cache:
            self.span_cache[key] = self.get_stem(self.tokenzier.decode(span))
        return self.span_cache[key]

    def parse(self):
        corpus = load_pretensorize(0, 12)

        input_ids, attention_mask, active_indices, active_padding = list(zip(*corpus))
        cur_stem_id = 0

        for ids, spans, pading in tqdm(zip(input_ids, active_indices, active_padding), total=len(input_ids)):
            doc_stem_ids = []
            for i in range(sum(pading)):
                start, end = spans[i]
                # word = tokenzier.decode(ids[start:end])
                # input(word)
                # stem = self.get_stem(word)
                stem = self.get_stem_by_span(ids[start:end])
                if stem not in self.stem2id:
                    self.stem2id[stem] = cur_stem_id
                    self.id2stem[cur_stem_id] = stem
                    cur_stem_id += 1
                stem_id = self.stem2id[stem]
                doc_stem_ids.append(stem_id)
                self.stemid2cf[stem_id] += 1

            for stem_id in set(doc_stem_ids):
                self.stemid2df[stem_id] += 1
            self.embid2stemid += doc_stem_ids


def test_colbert_indexer():
    index_path = "/home/awu/experiments/geo/others/testcb/index/webq/webq_colbert_t5_seqlen_emb/"
    index_config['index_path'] = index_path
    index_config['faiss_index_path'] = index_path + "ivfpq.2000.faiss"
    # from line_profiler import LineProfiler
    # lp = LineProfiler()
    # lp_wrapper = lp(ColbertRetriever)
    retriever = ColbertRetriever(index_path=index_config['index_path'], rank=0, index_config=index_config)
    retriever.load_index()
    query = [torch.randn(2, 128).cuda(), torch.ones(2).cuda()]
    query[0] = F.normalize(query[0], p=2, dim=-1)
    retriever.search(query, topk_doc=4)


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


def test_term_manager():
    index_term_manager = IndexTermManager()
    # index_term_manager.parse()
    for i in range(200):
        print(index_term_manager.embid2term(i), index_term_manager.embid2cf(i), index_term_manager.embid2df(i))


if __name__ == '__main__':
    # test_indexer()
    # test_colbert_indexer()
    # test_term_manager()
    ColbertRetriever.get_doclens()