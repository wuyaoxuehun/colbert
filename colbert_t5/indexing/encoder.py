import math
import os
import time
import torch
import ujson
import numpy as np

import itertools
import threading
import queue

from colbert.modeling.inference import ModelInference
from colbert.evaluation.loaders import load_colbert
from colbert.utils.utils import print_message

from colbert.indexing.index_manager import IndexManager
from colbert.base_config import segmenter, overwrite_mask_cache
import os
import json
from colbert.utils.distributed import barrier
from tqdm import tqdm


class CollectionEncoder():
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
        self.iterator = self._initialize_iterator()

    def pre_tensorize(self, collection):
        collection = list(collection)
        print(len(collection))
        ids, mask, word_mask = self.inference.doc_tokenizer.tensorize_dict(collection)
        # assert len(ids[0]) == len(collection)
        assert len(ids) == len(collection)
        return ids.cpu().tolist(), mask.cpu().tolist(), word_mask.tolist()

    def _initialize_iterator(self, overwrite=False):
        # return open(self.collection)
        # collection = iter(list(open(self.collection))[50000:])
        base_dir, file_name = os.path.split(self.collection)
        file_prefix, file_ext = os.path.splitext(file_name)
        pre_tok_file = file_prefix + f'_{self.args.doc_maxlen}_tokenized.pt'
        # pre_tok_file = file_prefix + '_tokenized_no_word_mask.pt'
        pre_tok_path = os.path.join(base_dir, pre_tok_file)
        if not os.path.exists(pre_tok_path) or overwrite:
            if self.args.rank == 0:
                collection = open(self.collection, encoding='utf8')
                collection = json.load(collection)
                # collection = collection[1:]
                # collection = collection[:]
                collection = self.pre_tensorize(collection)
                torch.save(collection, pre_tok_path)
        barrier(self.args.rank)

        print_message(f"loading tokenized file from {pre_tok_path}")
        split_num = 12
        if self.args.rank == 0 and overwrite:
            print_message("spliting collection")
            collection = torch.load(pre_tok_path)
            all_d_ids, all_d_mask, all_d_word_mask = collection
            part_len = math.ceil(len(all_d_ids) / split_num)
            for i in tqdm(range(split_num)):
                start, end = i * part_len, (i + 1) * part_len
                d_ids = all_d_ids[start:end]
                d_mask = all_d_mask[start:end]
                d_word_mask = all_d_word_mask[start:end]
                sub_collection_path = file_prefix + f'_{self.args.doc_maxlen}_tokenized_{i}.pt'
                torch.save([d_ids, d_mask, d_word_mask],
                           os.path.join(base_dir, sub_collection_path))
            print_message("collection splitted")
            exit()

        barrier(self.args.rank)
        collection = [[], [], []]
        part_len = math.floor(split_num / self.args.nranks)
        start, end = self.args.rank * part_len, (self.args.rank + 1) * part_len
        if (split_num - end) < part_len:
            end += 10
        print_message(f"rank{self.args.rank}=[{start}, {end}]")
        for i in range(start, end):
            sub_collection_path = file_prefix + f'_{self.args.doc_maxlen}_tokenized_{i}.pt'
            collection_path = os.path.join(base_dir, sub_collection_path)
            if not os.path.exists(collection_path):
                break
            print_message(f"loading sub collection {collection_path} for rank {self.args.rank}")
            sub_collection = torch.load(os.path.join(base_dir, sub_collection_path))
            for j in range(3):
                collection[j].extend(sub_collection[j])

        # collection = torch.load(pre_tok_path)
        # collection = open(self.collection, encoding='utf8')
        # next(collection)
        print_message('collection total %d' % (len(collection[0]),))
        return zip(*collection)

        # return iter(list(zip(*collection))[:1000])

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

        self.inference = ModelInference(self.colbert, amp=self.args.amp, segmenter=segmenter, query_maxlen=self.args.query_maxlen, doc_maxlen=self.args.doc_maxlen)

    def encode(self):
        self.saver_queue = queue.Queue(maxsize=3)
        thread = threading.Thread(target=self._saver_thread)
        thread.start()

        t0 = time.time()
        local_docs_processed = 0

        for batch_idx, (offset, lines, owner) in enumerate(self._batch_passages(self.iterator)):
            if owner != self.process_idx:
                continue

            t1 = time.time()
            # batch = self._preprocess_batch(offset, lines)

            embs, doclens = self._encode_batch(batch_idx, lines)

            t2 = time.time()
            self.saver_queue.put((batch_idx, embs, offset, doclens))

            t3 = time.time()
            local_docs_processed += len(lines)
            overall_throughput = compute_throughput(local_docs_processed, t0, t3)
            this_encoding_throughput = compute_throughput(len(lines), t1, t2)
            this_saving_throughput = compute_throughput(len(lines), t2, t3)

            self.print(f'#> Completed batch #{batch_idx} (starting at passage #{offset}) \t\t'
                       f'Passages/min: {overall_throughput} (overall), ',
                       f'{this_encoding_throughput} (this encoding), ',
                       f'{this_saving_throughput} (this saving)')
        self.saver_queue.put(None)

        self.print("#> Joining saver thread.")
        thread.join()

    def encode_simple(self):
        embs, doclens = self._encode_batch(0, list(self.iterator))
        if not os.path.exists(self.args.index_path):
            os.makedirs(self.args.index_path)
        encode_idx = self.args.rank
        output_path = os.path.join(self.args.index_path, "{}.pt".format(encode_idx))
        output_sample_path = os.path.join(self.args.index_path, "{}.sample".format(encode_idx))
        doclens_path = os.path.join(self.args.index_path, 'doclens.{}.json'.format(encode_idx))

        # Save the embeddings.
        self.indexmgr.save(embs, output_path)
        self.indexmgr.save(embs[torch.randint(0, high=embs.size(0), size=(embs.size(0) // 20,))], output_sample_path)

        # Save the doclens.
        with open(doclens_path, 'w') as output_doclens:
            ujson.dump(doclens, output_doclens)

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

    def _encode_batch(self, batch_idx, batch):
        with torch.no_grad():
            import math
            # rank_data_len = math.ceil(len(batch) / self.args.nranks)
            # start, end = self.args.rank * rank_data_len, (self.args.rank + 1) * rank_data_len
            d_ids = [_[0] for _ in batch]
            d_mask = [_[1] for _ in batch]
            d_word_mask = [_[2] for _ in batch]
            # print(f'local embs num = {len(d_ids)}')
            # d_word_mask[:, 1:] = 0
            embs = self.inference.docFromTensorize((d_ids, d_mask, d_word_mask), bsize=self.args.bsize, keep_dims=False, to_cpu=True, args=self.args)
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
