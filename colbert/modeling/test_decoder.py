import copy

import torch
import yaml
from torch import nn
from transformers import BertTokenizer, BertModel

from transformer_decoder import TransformerDecoder, Generator, TransformerDecoderState, set_parameter_tf


def test():
    config = yaml.safe_load(open("./decoder_config.yml"))
    pretrain = "../../../../../pretrain/bert-base-uncased/"
    tokenizer = BertTokenizer.from_pretrained(pretrain)
    model = BertModel.from_pretrained(pretrain).cuda()
    tgt_embeddings = nn.Embedding(len(tokenizer.vocab), config['decoder_hidden'], padding_idx=0)
    decoder = TransformerDecoder(
        config['decoder_layers'],
        config['decoder_hidden'],
        heads=6,
        d_ff=config['decoder_ff'],
        dropout=config['decoder_dropout'],
        embeddings=tgt_embeddings
    ).cuda()
    generator = Generator(len(tokenizer.vocab), config['decoder_hidden'], tokenizer.vocab['[PAD]']).cuda()
    generator.linear.weight = decoder.embeddings.weight

    for p in decoder.modules():
        set_parameter_tf(p)
    for p in generator.parameters():
        set_parameter_tf(p)

    if "share_emb" in config:
        if 'bert' in config['mode']:
            tgt_embeddings = nn.Embedding(len(tokenizer.vocab), config['decoder_hidden'], padding_idx=0)
            tgt_embeddings.weight = copy.deepcopy(model.bert.embeddings.word_embeddings.weight)
        decoder.embeddings = tgt_embeddings
        generator.linear.weight = decoder.embeddings.weight

    reconstruction_criterion = nn.NLLLoss(ignore_index=tokenizer.vocab['[PAD]'], reduction='mean')
    bert_tokens_masked = torch.randint(low=0, high=tokenizer.vocab_size - 1, size=(2, 128)).cuda()
    # hidden_state = torch.randn(2, config['decoder_hidden']).cuda()
    hidden_state = model(input_ids=bert_tokens_masked, return_dict=True, output_hidden_states=True).last_hidden_state[:, 0]
    decode_context = decoder(bert_tokens_masked[:, :-1], hidden_state,
                             TransformerDecoderState(bert_tokens_masked))
    reconstruction_text = generator(decode_context.view(-1, decode_context.size(2)))

    reconstruction_loss = reconstruction_criterion(reconstruction_text,
                                                   # bert_tokens[:, 1:].reshape(-1))
                                                   bert_tokens_masked[:, 1:].reshape(-1))


if __name__ == '__main__':
    test()
