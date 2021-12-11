import logging

from torch.nn.parallel import DistributedDataParallel as DDP
from transformers.utils.logging import set_verbosity_error

from colbert.indexing.encoder import CollectionEncoder
# from colbert.modeling.colbert_list_qa import ColBERT_List_qa
from colbert.modeling.colbert_list_qa_gen import ColBERT_List_qa
from colbert.utils.parser import Arguments
from conf import *
import os

set_verbosity_error()
logger = logging.getLogger("__main__")


def index_by_model(args):
    colbert_config['init'] = True
    print(colbert_config)
    colbert_qa = ColBERT_List_qa(config=model_config, colbert_config=colbert_config, reader_config=reader_config, load_old=False)
    colbert_qa.load(args.checkpoint + "/pytorch.bin")
    args.query_maxlen, args.doc_maxlen = colbert_config['query_maxlen'], colbert_config['doc_maxlen']
    args.dim = colbert_config['dim']
    colbert_qa.colbert.to("cuda")
    if args.distributed:
        colbert_qa.colbert = DDP(colbert_qa.colbert, device_ids=[args.rank], find_unused_parameters=True).module

    print(args.rank, args.nranks)
    encoder = CollectionEncoder(args, process_idx=args.rank, num_processes=args.nranks, model=colbert_qa.colbert)
    encoder.encode_simple()
    print('encoded all ')
    if args.rank == 0:
        for file in os.listdir(args.index_path):
            if file.endswith(".pt") and int(file[0]) >= args.nranks:
                os.remove(os.path.join(args.index_path, file))


if __name__ == '__main__':
    parser = Arguments(description='Faiss indexing for end-to-end retrieval with ColBERT.')
    parser.add_index_model_input()
    args = parser.parse()
    index_by_model(args)
