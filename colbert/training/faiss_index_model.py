import logging
import math

from transformers.utils.logging import set_verbosity_error

from colbert.indexing.loaders import load_doclens
# from colbert.modeling.colbert_list_qa import ColBERT_List_qa
from colbert.indexing.myfaiss import index_faiss_simple
from colbert.utils.parser import Arguments
from conf import dim

set_verbosity_error()
logger = logging.getLogger("__main__")


def get_best_partitons(num_embeddings):
    partitions = 1 << math.ceil(math.log2(8 * math.sqrt(num_embeddings)))
    print('\n\n')
    logger.info("You did not specify --partitions!")
    logger.info(f'''Default computation chooses {partitions} partitions (for {num_embeddings} embeddings)''')
    print('\n\n')
    return partitions


def faiss_index_by_encoded(args):
    num_embeddings = sum(load_doclens(args.index_path))
    print("#> num_embeddings =", num_embeddings)
    # if args.partitions is None:
    args.partitions = get_best_partitons(num_embeddings)
    print('best partitions = ', args.partitions)
    if False:
        from colbert.indexing.faiss_indexers import DPRRetriever
        dpr_retriever = DPRRetriever(index_path=args.index_path, dim=dim)
        dpr_retriever.encode_corpus(encoded_corpus_path=args.index_path)
    else:
        # index_faiss_simple(args)
        from colbert.indexing.faiss_indexers import ColbertRetriever
        retriever = ColbertRetriever(index_path=args.index_path, dim=dim, rank=None, index_config=None, partitions=args.partitions, sample=args.sample)
        retriever.encode_corpus(encoded_corpus_path=args.index_path)

if __name__ == '__main__':
    parser = Arguments(description='Faiss indexing for end-to-end retrieval with ColBERT.')
    parser.add_index_model_input()
    args = parser.parse()
    faiss_index_by_encoded(args)
