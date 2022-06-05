import logging
import os
from colbert.indexing.encoder import CollectionEncoder
from colbert.utils.dense_conf import load_dense_conf
from colbert.utils.distributed import init_dist

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s", datefmt="%m/%d/%Y %H:%M:%S",
                    level=logging.INFO if int(os.environ.get('LOCAL_RANK', 0)) in [-1, 0] else logging.WARN)
logger = logging.getLogger(__name__)


def index_by_model(args):
    args.rank, args.nranks, args.distributed = init_dist()
    print(args.rank, args.nranks)
    encoder = CollectionEncoder(args)
    encoder.encode_simple()
    print('encoded all ')


if __name__ == '__main__':
    args = load_dense_conf()
    index_by_model(args)
