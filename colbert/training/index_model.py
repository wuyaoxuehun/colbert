from colbert.utils.parser import Arguments
from colbert.indexing.loaders import load_doclens
from colbert.indexing.encoder import CollectionEncoder
import math
import logging
from colbert.modeling.colbert_list_qa import ColBERT_List_qa
from colbert.indexing.myfaiss import index_faiss_simple
from conf import *
from transformers.utils.logging import set_verbosity_error
set_verbosity_error()
logger = logging.getLogger("__main__")

def get_best_partitons(num_embeddings):
    partitions = 1 << math.ceil(math.log2(8 * math.sqrt(num_embeddings)))
    print('\n\n')
    logger.info("You did not specify --partitions!")
    logger.info(f'''Default computation chooses {partitions} partitions (for {num_embeddings} embeddings)''')
    print('\n\n')
    return partitions

def index_by_model(args):
    colbert_config['init'] = True
    colbert_qa = ColBERT_List_qa(config=model_config, colbert_config=colbert_config, reader_config=reader_config, load_old=False)
    colbert_qa.load(args.checkpoint + "/pytorch.bin")
    encoder = CollectionEncoder(args, process_idx=0, num_processes=1, model=colbert_qa.colbert)
    encoder.encode_simple()
    print('encoded all ')
    num_embeddings = sum(load_doclens(args.index_path))
    print("#> num_embeddings =", num_embeddings)
    if args.partitions is None:
        args.partitions = get_best_partitons(num_embeddings)
    index_faiss_simple(args)

if __name__ == '__main__':
    parser = Arguments(description='Faiss indexing for end-to-end retrieval with ColBERT.')
    parser.add_index_model_input()
    index_by_model(parser.parse())
