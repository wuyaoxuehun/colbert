import itertools
# from colbert.base_config import segmenter, overwrite_mask_cache
import os
import queue
import threading
import time

import math
import numpy as np
import torch
import ujson
import gc
from colbert.evaluation.loaders import load_colbert
from colbert.indexing.index_manager import IndexManager
from colbert.modeling.inference import ModelInference
# from colbert.modeling.tokenization.utils import get_real_inputs
from colbert.training.training_utils import distributed_concat
from colbert.utils.utils import print_message
from conf import corpus_tokenized_prefix
from colbert.utils.distributed import barrier


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

    def _initialize_iterator_(self, overwrite=False):
        # return open(self.collection)
        # collection = iter(list(open(self.collection))[50000:])
        # base_dir, file_name = os.path.split(self.collection)
        # file_prefix, file_ext = os.path.splitext(file_name)
        # pref = file_prefix + f'_{doc_maxlen}_{pretrain_choose}_tokenized'
        # pre_tok_file = f'{pref}_tokenized.pt'
        # pre_tok_file = file_prefix + '_tokenized_no_word_mask.pt'
        # pre_tok_path = os.path.join(base_dir, pre_tok_file)
        split_num = 12
        # collection = [[], [], []]
        collection = []
        part_len = math.floor(split_num / self.args.nranks)
        start, end = self.args.rank * part_len, (self.args.rank + 1) * part_len
        if (split_num - end) < part_len:
            end += 10
        print_message(f"rank{self.args.rank}=[{start}, {end}]")

        # start, end = 0, 1
        for i in range(start, end):
            # sub_collection_path = f'{pref}_{i}.pt'
            # input(sub_collection_path)
            collection_path = corpus_tokenized_prefix + f"_{i}.pt"
            # input(collection_path)
            # input(collection_path)
            if not os.path.exists(collection_path):
                break

            print_message(f"loading sub collection {collection_path} for rank {self.args.rank}")
            sub_collection = torch.load(collection_path)
            # for j in range(3):
            #     collection[j].extend(sub_collection[j])
            collection += sub_collection

        # collection = torch.load(pre_tok_path)
        # collection = open(self.collection, encoding='utf8')
        # next(collection)
        # collection = list(zip(*collection))
        print_message('collection total %d' % (len(collection),))
        return collection

        # return iter(list(zip(*collection))[:1000])

    def _initialize_iterator(self, part):
        collection_path = corpus_tokenized_prefix + f"_{part}.pt"
        print_message(f"loading sub collection {collection_path} for rank {self.args.rank}")
        collection = torch.load(collection_path)
        collection = collection[:len(collection) // 1]
        sub_collection_num = math.ceil(len(collection) // self.args.nranks)
        sub_collection = collection[sub_collection_num * self.args.rank:sub_collection_num * (self.args.rank + 1)]
        print_message('collection total %d' % (len(sub_collection),))
        # self.part_len = len(collection)
        return sub_collection

    def _saver_thread(self):
        for args in iter(self.saver_queue.get, None):
            self._save_batch(*args)

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
        for i in range(2, 10):
            barrier(self.args.rank)
            self.iterator = self._initialize_iterator(part=i)
            embs, doclens = self._encode_batch(self.iterator)
            # embs = distributed_concat(embs, concat=True, num_total_examples=self.part_len)
            torch.save(embs, f"/home2/awu/testcb/data/dureader/temp/{self.args.rank}.pt")
            barrier(self.args.rank)
            if self.args.rank == 0:
                all_embs = []
                for j in range(self.args.nranks):
                    all_embs.append(torch.load(f"/home2/awu/testcb/data/dureader/temp/{j}.pt"))
                embs = torch.cat(all_embs, dim=0)
                if not os.path.exists(self.args.index_path):
                    os.makedirs(self.args.index_path)
                # encode_idx = self.args.rank
                encode_idx = i
                output_path = os.path.join(self.args.index_path, "{}.pt".format(encode_idx))
                output_sample_path = os.path.join(self.args.index_path, "{}.sample".format(encode_idx))
                doclens_path = os.path.join(self.args.index_path, 'doclens.{}.json'.format(encode_idx))

                # Save the embeddings.
                self.indexmgr.save(embs, output_path)
                self.indexmgr.save(embs[torch.randint(0, high=embs.size(0), size=(embs.size(0) // 20,))], output_sample_path)
                print_message(f"saved {output_path}")
                # Save the doclens.
                with open(doclens_path, 'w') as output_doclens:
                    ujson.dump(doclens, output_doclens)
            del self.iterator
            del embs
            del doclens
            gc.collect()
            barrier(self.args.rank)

        metadata_path = os.path.join(self.args.index_path, 'metadata.json')
        print_message("Saving (the following) metadata to", metadata_path, "..")
        print(self.args.input_arguments)

        with open(metadata_path, 'w') as output_metadata:
            ujson.dump(self.args.input_arguments.__dict__, output_metadata)

    def _batch_passages(self, fi):
        """
        Must use the same seed across processes!
        """
        np.random.seed(0)

        offset = 0
        for owner in itertools.cycle(range(self.num_processes)):
            print(self.possible_subset_sizes)
            batch_size = np.random.choice(self.possible_subset_sizes)

            L = [line for _, line in zip(range(batch_size), fi)]

            if len(L) == 0:
                break  # EOF

            yield (offset, L, owner)
            offset += len(L)

            if len(L) < batch_size:
                break  # EOF

        self.print("[NOTE] Done with local share.")

        return

    def _preprocess_batch(self, offset, lines):
        endpos = offset + len(lines)

        batch = []

        for line_idx, line in zip(range(offset, endpos), lines):
            line_parts = line.strip().split('\t')
            if len(line_parts) < 2:
                print(line_parts)
                print(line_idx)
                if len(line_parts) == 1:
                    line_parts = [-1, line_parts[0]]
                else:
                    line_parts = ['@', '@']
            pid, passage, *other = line_parts
            batch.append(passage)
        return batch

    def _encode_batch(self, batch):
        with torch.no_grad():
            # import math
            # rank_data_len = math.ceil(len(batch) / self.args.nranks)
            # start, end = self.args.rank * rank_data_len, (self.args.rank + 1) * rank_data_len
            # print(f'local embs num = {len(d_ids)}')
            # d_word_mask[:, 1:] = 0
            # from colbert.modeling.tokenization.utils import get_real_inputs
            # batch = list(zip(*batch))
            # real_inputs = []
            # print('getting real inputs')
            # for t in tqdm(batch):
            #     real_inputs.append(get_real_inputs(*t, max_seq_length=doc_maxlen))

            embs = self.inference.docFromTensorize(batch, bsize=self.args.bsize, keep_dims=False, to_cpu=True, args=self.args)
            assert type(embs) is list
            # assert len(embs) == len(d_ids)
            print(f'local embs num = {len(embs)}')
            local_doclens = [d.size(0) for d in embs]
            embs = torch.cat(embs)

        return embs, local_doclens

    def _save_batch(self, batch_idx, embs, offset, doclens):
        start_time = time.time()

        output_path = os.path.join(self.args.index_path, "{}.pt".format(batch_idx))
        output_sample_path = os.path.join(self.args.index_path, "{}.sample".format(batch_idx))
        doclens_path = os.path.join(self.args.index_path, 'doclens.{}.json'.format(batch_idx))

        # Save the embeddings.
        self.indexmgr.save(embs, output_path)
        self.indexmgr.save(embs[torch.randint(0, high=embs.size(0), size=(embs.size(0) // 20,))], output_sample_path)

        # Save the doclens.
        with open(doclens_path, 'w') as output_doclens:
            ujson.dump(doclens, output_doclens)

        throughput = compute_throughput(len(doclens), start_time, time.time())
        self.print_main("#> Saved batch #{} to {} \t\t".format(batch_idx, output_path),
                        "Saving Throughput =", throughput, "passages per minute.\n")

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
