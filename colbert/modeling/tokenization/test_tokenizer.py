from colbert.modeling.tokenization.utils import CostomTokenizer
from colbert import base_config


def testtok():
    tokenizer = CostomTokenizer.from_pretrained(base_config.pretrain)
    tokens = tokenizer.convert_tokens_to_ids(['[unused1]', '[unused2]', '[CLS]'])
    print(tokens)

if __name__ == '__main__':
    testtok()
