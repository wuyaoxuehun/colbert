import gc
import os
import queue
import threading
from multiprocessing import Pool
from torch.nn.parallel import DistributedDataParallel as DDP

import logging
import math
import torch
import ujson
from tqdm import tqdm

from colbert.utils.amp import MixedPrecisionManager
from colbert.modeling.model_utils import to_real_input_all
from colbert.modeling.colbert_model import ColbertModel
from proj_utils.dureader_utils import get_dureader_ori_corpus
from colbert.modeling.tokenizers import CostomTokenizer
from colbert.training.training_utils import qd_mask_to_realinput, keep_nonzero

logger = logging.getLogger("__main__")
os.environ['TOKENIZERS_PARALLELISM'] = 'true'
from awutils.file_utils import load_json, dump_json
from colbert.indexing.index_manager import IndexManager
from colbert.utils.distributed import barrier

doc_tokenizer: CostomTokenizer


def init_doc_tokenizer(args):
    global doc_tokenizer
    doc_tokenizer = CostomTokenizer(args)


def pre_tensorize(examples):
    global doc_tokenizer
    res = [_[0] for _ in doc_tokenizer.tokenize_d([examples], to_tensor=False)]
    return res


split_num = 12


class BatchGenerator:
    # 导入语料库
    collection = get_dureader_ori_corpus()[:]

    def __init__(self, part, rank, nranks):
        # BUF_SIZE = 64
        BUF_SIZE = 1024
        self.part = part
        self.rank = rank
        self.nranks = nranks
        self.q = queue.Queue(BUF_SIZE)
        collection = self.collection
        self.bs = 2048 * 2
        logger.info(f"collection total: {len(collection)}")
        part_len = math.ceil(len(collection) / split_num)
        # for part in range(12):
        start, end = part * part_len, (part + 1) * part_len
        collection = collection[start:end]
        logger.info(f"sub_collection total: {len(collection)}")
        # sub_collection = collection
        sub_collection_num = math.ceil(len(collection) / self.nranks)
        collection = collection[sub_collection_num * self.rank:sub_collection_num * (self.rank + 1)]
        self.sub_collection = collection
        self.rank_collection_size = len(self.sub_collection)
        logger.info(f"rank_collection total: {len(self.sub_collection)}")
        threading.Thread(target=self.run).start()

    def run(self):
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
    def __init__(self, args):
        self.args = args
        self.collection = args.dense_index_args.collection
        self.bsize = args.dense_index_args.bsize
        self.index_path = self.args.dense_index_args.index_path
        self.colbert = ColbertModel(args)
        self.colbert.load(args.dense_index_args.checkpoint + "/pytorch_model.bin")
        self.colbert = self.colbert.cuda()
        if args.distributed:
            self.colbert = DDP(self.colbert, device_ids=[args.rank], find_unused_parameters=True).module
        self.colbert.eval()
        # self._load_model(model)
        self.indexmgr = IndexManager()
        # self.iterator = None
        self.amp_manager = MixedPrecisionManager(True)
        init_doc_tokenizer(args)

    def encode_simple(self):
        for i in range(0, split_num):
            barrier(self.args.rank)
            # self.iterator = self._initialize_iterator(part=i)
            batch_generator = BatchGenerator(part=i, rank=self.args.rank, nranks=self.args.nranks)
            q, batch_num = batch_generator.get_generator()
            embs, doclens = [], []
            with tqdm(total=batch_num, disable=self.args.rank > 0) as pbar:
                for batch in iter(q.get, None):
                    batch_embs = self._encode_batch(batch)
                    embs += batch_embs
                    doclens += [d.size(0) for d in batch_embs]
                    del batch
                    pbar.update(1)
            embs = torch.cat(embs)
            tmp_dir = "./tmp/"
            torch.save(embs, f"{tmp_dir}/{self.args.rank}.pt")
            dump_json(doclens, f"{tmp_dir}/{self.args.rank}_doclens.json")
            del embs
            del doclens
            gc.collect()
            barrier(self.args.rank)
            if self.args.rank == 0:
                all_embs = []
                doclens = []
                for j in range(self.args.nranks):
                    all_embs.append(torch.load(f"{tmp_dir}/{j}.pt"))
                    doclens += load_json(f"{tmp_dir}/{j}_doclens.json")
                embs = torch.cat(all_embs, dim=0)
                if not os.path.exists(self.index_path):
                    os.makedirs(self.index_path)
                encode_idx = i
                output_path = os.path.join(self.index_path, "{}.pt".format(encode_idx))
                doclens_path = os.path.join(self.index_path, 'doclens.{}.json'.format(encode_idx))

                # Save the embeddings.
                self.indexmgr.save(embs, output_path)
                logger.info(f"saved {output_path}")
                # Save the doclens.
                with open(doclens_path, 'w') as output_doclens:
                    ujson.dump(doclens, output_doclens)
                    logger.info(f"saved {doclens_path}")
                # del self.iterator
                del embs, doclens
                torch.cuda.empty_cache()
                gc.collect()
            barrier(self.args.rank)

        metadata_path = os.path.join(self.index_path, 'metadata.json')
        logger.info("Saving (the following) metadata to" + metadata_path + "..")
        # logger.info(self.args.input_arguments)
        #
        # with open(metadata_path, 'w') as output_metadata:
        #     ujson.dump(self.args.input_arguments.__dict__, output_metadata)

    @torch.no_grad()
    def _encode_batch(self, tensorizes):
        embs = []
        self.colbert: ColbertModel
        iterator = range(0, len(tensorizes), self.bsize)
        for offset in iterator:
            t = tensorizes[offset:offset + self.bsize]
            input_ids, attention_mask, active_padding = to_real_input_all(t)
            max_len_batch = attention_mask.sum(-1).max(-1).values
            input_ids, attention_mask, active_padding = input_ids[:, :max_len_batch], attention_mask[:, :max_len_batch], active_padding[:, :max_len_batch]
            with self.amp_manager.context():
                batches = self.colbert.doc(input_ids.to("cuda"), attention_mask.to("cuda"), active_padding.to("cuda"))
            batches = batches.cpu().to(dtype=torch.float16)
            batches = [qd_mask_to_realinput(t=d, t_mask=dw_mask, keep_dim=False)[0] for d, dw_mask in zip(batches, active_padding)]
            # batches = [keep_nonzero(d, d_mask)[0] for d, d_mask in zip(batches, active_padding)]
            embs.extend(batches)
        return embs
