# from colbert.base_config import segmenter, overwrite_mask_cache

import gc
from multiprocessing import Pool

import math
import queue

import threading
import torch
import ujson
import os

from tqdm import tqdm

from proj_utils.dureader_utils import get_dureader_ori_corpus

os.environ['TOKENIZERS_PARALLELISM'] = 'true'
from colbert.modeling.tokenization.doc_tokenization import DocTokenizer
from awutils.file_utils import load_json, dump_json
from colbert.evaluation.loaders import load_colbert
from colbert.indexing.index_manager import IndexManager
from colbert.modeling.inference import ModelInference
from colbert.utils.distributed import barrier
# from colbert.modeling.tokenization.utils import get_real_inputs
from colbert.utils.utils import print_message
from conf import corpus_tokenized_prefix, doc_maxlen, load_all_paras

# BUF_SIZE = 256
# q = queue.Queue(BUF_SIZE)
doc_tokenizer = DocTokenizer(doc_maxlen)


def pre_tensorize(examples):
    res = [_[0] for _ in doc_tokenizer.tensorize_dict([examples], to_tensor=False)]
    return res


class BatchGenerator:
    collection = get_dureader_ori_corpus()

    def __init__(self, part, rank, nranks):
        # BUF_SIZE = 64
        BUF_SIZE = 1024
        self.part = part
        self.rank = rank
        self.nranks = nranks
        self.q = queue.Queue(BUF_SIZE)
        collection = self.collection
        split_num = 12
        self.bs = 2048
        print("collection total:", len(collection))
        part_len = math.ceil(len(collection) / split_num)
        # for part in range(12):
        start, end = part * part_len, (part + 1) * part_len
        collection = collection[start:end]
        print("sub_collection total:", len(collection))
        # sub_collection = collection
        sub_collection_num = math.ceil(len(collection) / self.nranks)
        collection = collection[sub_collection_num * self.rank:sub_collection_num * (self.rank + 1)]
        self.sub_collection = collection
        self.rank_collection_size = len(self.sub_collection)
        print("rank_collection total:", len(self.sub_collection))
        threading.Thread(target=self.run).start()

    def run(self):
        # part = self.part
        q = self.q
        bs = self.bs
        sub_collection = self.sub_collection
        # print("rank_collection total:", len(sub_collection))
        from contextlib import closing
        for i in range(0, len(sub_collection), bs):
            batch_data = sub_collection[i:i + bs]
            with closing(Pool(4)) as p:
                res = list(p.imap(pre_tensorize, batch_data))
            # while q.full():
            #     pass
            q.put(res)
        q.put(None)

    def get_generator(self):
        return self.q, self.rank_collection_size // self.bs


class CollectionEncoder:
    def __init__(self, args, process_idx, num_processes, model=None):
        self.args = args
        self.collection = args.collection
        self.process_idx = process_idx
        self.num_processes = num_processes

        assert 0.5 <= args.chunksize <= 128.0
        max_bytes_per_file = args.chunksize * (1024 * 1024 * 1024)

        max_bytes_per_doc = (self.args.doc_maxlen * self.args.dim * 2.0)

        # Determine subset sizes for output
        minimum_subset_size = 10_000
        maximum_subset_size = max_bytes_per_file / max_bytes_per_doc
        maximum_subset_size = max(minimum_subset_size, maximum_subset_size)
        self.possible_subset_sizes = [int(maximum_subset_size)]

        self.print_main("#> Local args.bsize =", args.bsize)
        # self.print_main("#> args.index_root =", args.index_root)
        self.print_main(f"#> self.possible_subset_sizes = {self.possible_subset_sizes}")

        self._load_model(model)
        self.indexmgr = IndexManager(args.dim)
        self.iterator = None
        # self.iterator = self._initialize_iterator()

    def pre_tensorize(self, collection):
        collection = list(collection)
        print(len(collection))
        ids, mask, word_mask = self.inference.doc_tokenizer.tensorize_dict(collection)
        # assert len(ids[0]) == len(collection)
        assert len(ids) == len(collection)
        return ids.cpu().tolist(), mask.cpu().tolist(), word_mask.tolist()

    def _initialize_iterator(self, part):
        collection_path = corpus_tokenized_prefix + f"_{part}.pt"
        print_message(f"loading sub collection {collection_path} for rank {self.args.rank}\n")
        collection = torch.load(collection_path)
        collection = collection[:len(collection) // 1]
        sub_collection_num = math.ceil(len(collection) / self.args.nranks)
        sub_collection = collection[sub_collection_num * self.args.rank:sub_collection_num * (self.args.rank + 1)]
        print_message('collection total %d' % (len(sub_collection),))
        # self.part_len = len(collection)
        return sub_collection

    def _load_model(self, model=None):
        if model is None:
            self.colbert, self.checkpoint = load_colbert(self.args, do_print=(self.process_idx == 0))
        else:
            self.colbert = model
        self.colbert = self.colbert.cuda()
        if not self.args.distributed and torch.cuda.device_count() > 1:
            print_message("having multiple gpus, colbert parralel on %d gpus" % (torch.cuda.device_count(),))
            self.colbert.bert = torch.nn.DataParallel(self.colbert.bert)  # , device_ids=list(range(torch.cuda.device_count()))

        self.colbert.eval()

        self.inference = ModelInference(self.colbert, amp=self.args.amp, segmenter=None, query_maxlen=self.args.query_maxlen, doc_maxlen=self.args.doc_maxlen)

    def encode_simple(self):
        for i in range(0, 12):
            barrier(self.args.rank)
            # self.iterator = self._initialize_iterator(part=i)
            batch_generator = BatchGenerator(part=i, rank=self.args.rank, nranks=self.args.nranks)
            q, batch_num = batch_generator.get_generator()
            embs, doclens = [], []
            with tqdm(total=batch_num, disable=self.args.rank > 0) as pbar:
                for batch in iter(q.get, None):
                    batch_embs = self._encode_batch(batch)
                    embs += batch_embs
                    # doclens += [sum(_[-1]) for _ in batch]
                    doclens += [d.size(0) for d in batch_embs]
                    del batch
                    pbar.update(1)
            embs = torch.cat(embs)
            base_dir = "/home2/awu/testcb/"
            # embs = distributed_concat(embs, concat=True, num_total_examples=self.part_len)
            torch.save(embs, f"/home2/awu/testcb/data/dureader/temp/{self.args.rank}.pt")
            # torch.save(embs, f"/home2/awu/testcb/data/dureader/temp/{self.args.rank}_doclens.pt")
            dump_json(doclens, f"/home2/awu/testcb/data/dureader/temp/{self.args.rank}_doclens.json")
            del embs
            del doclens
            gc.collect()
            barrier(self.args.rank)
            if self.args.rank == 0:
                all_embs = []
                doclens = []
                for j in range(self.args.nranks):
                    all_embs.append(torch.load(f"/home2/awu/testcb/data/dureader/temp/{j}.pt"))
                    # all_embs += torch.load(f"/home2/awu/testcb/data/dureader/temp/{j}.pt")
                    # all_doclens.append(load_json(f"/home2/awu/testcb/data/dureader/temp/{j}_doclens.json"))
                    doclens += load_json(f"/home2/awu/testcb/data/dureader/temp/{j}_doclens.json")
                embs = torch.cat(all_embs, dim=0)
                # doclens = [_ for doclen in all_doclens for _ in doclen]
                if not os.path.exists(self.args.index_path):
                    os.makedirs(self.args.index_path)
                # encode_idx = self.args.rank
                encode_idx = i
                output_path = os.path.join(self.args.index_path, "{}.pt".format(encode_idx))
                # output_sample_path = os.path.join(self.args.index_path, "{}.sample".format(encode_idx))
                doclens_path = os.path.join(self.args.index_path, 'doclens.{}.json'.format(encode_idx))

                # Save the embeddings.
                self.indexmgr.save(embs, output_path)
                # self.indexmgr.save(embs[torch.randint(0, high=embs.size(0), size=(embs.size(0) // 20,))], output_sample_path)
                print_message(f"saved {output_path}")
                # Save the doclens.
                with open(doclens_path, 'w') as output_doclens:
                    ujson.dump(doclens, output_doclens)
            # del self.iterator
                del embs
                del doclens
                gc.collect()
            barrier(self.args.rank)

        metadata_path = os.path.join(self.args.index_path, 'metadata.json')
        print_message("Saving (the following) metadata to", metadata_path, "..")
        print(self.args.input_arguments)

        with open(metadata_path, 'w') as output_metadata:
            ujson.dump(self.args.input_arguments.__dict__, output_metadata)

    def _encode_batch_(self, batch):
        with torch.no_grad():
            embs = self.inference.docFromTensorize(batch, bsize=self.args.bsize, keep_dims=False, to_cpu=True, args=self.args)
            assert type(embs) is list
            # assert len(embs) == len(d_ids)
            print(f'local embs num = {len(embs)}')
            local_doclens = [d.size(0) for d in embs]
            embs = torch.cat(embs)

        return embs, local_doclens

    def _encode_batch(self, batch):
        with torch.no_grad():
            embs = self.inference.docFromTensorize(batch, bsize=self.args.bsize, keep_dims=False, to_cpu=True, args=self.args)
            return embs
            # assert type(embs) is list
            # # assert len(embs) == len(d_ids)
            # print(f'local embs num = {len(embs)}')
            # local_doclens = [d.size(0) for d in embs]
            # embs = torch.cat(embs)

    def print(self, *args):
        print_message("[" + str(self.process_idx) + "]", "\t\t", *args)

    def print_main(self, *args):
        if self.process_idx == 0:
            self.print(*args)


def compute_throughput(size, t0, t1):
    throughput = size / (t1 - t0) * 60

    if throughput > 1000 * 1000:
        throughput = throughput / (1000 * 1000)
        throughput = round(throughput, 1)
        return '{}M'.format(throughput)

    throughput = throughput / (1000)
    throughput = round(throughput, 1)
    return '{}k'.format(throughput)
