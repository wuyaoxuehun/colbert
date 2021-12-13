from transformers import BertTokenizerFast

from conf import pretrain
from colbert.base_config import ColBert
from colbert.parameters import DEVICE
from colbert.utils.utils import print_message, load_checkpoint
from colbert.modeling.tokenization.utils import CostomTokenizer


def load_model(args, do_print=True):
    # tokenizer = BertTokenizerFast.from_pretrained(pretrain)
    colbert = ColBert.from_pretrained(pretrain,
                                      query_maxlen=args.query_maxlen,
                                      doc_maxlen=args.doc_maxlen,
                                      dim=args.dim,
                                      similarity_metric=args.similarity)
    colbert = colbert.to(DEVICE)
    print_message("#> Loading model checkpoint.", condition=do_print)

    checkpoint = load_checkpoint(args.checkpoint, colbert, do_print=do_print)

    colbert.eval()

    return colbert, checkpoint
