import logging
import os
from colbert.utils.dense_conf import load_dense_conf
logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s", datefmt="%m/%d/%Y %H:%M:%S",
                    level=logging.INFO if int(os.environ.get('LOCAL_RANK', 0)) in [-1, 0] else logging.WARN)
logger = logging.getLogger(__name__)

def faiss_index_by_encoded(args):
    index_path = args.dense_index_args.index_path
    faiss_type = args.faiss_index_args.faiss_type
    if faiss_type == "dpr":
        from colbert.indexing.faiss_indexers import DPRRetriever
        dpr_retriever = DPRRetriever(index_path=index_path, dim=args.dense_training_args.dim)
        dpr_retriever.encode_corpus(encoded_corpus_path=index_path)
    else:
        # index_faiss_simple(args)
        from colbert.indexing.faiss_indexers import ColbertRetriever
        retriever = ColbertRetriever(index_path=index_path)
        retriever.encode_corpus(encoded_corpus_path=index_path)


if __name__ == '__main__':
    args = load_dense_conf()
    faiss_index_by_encoded(args)
