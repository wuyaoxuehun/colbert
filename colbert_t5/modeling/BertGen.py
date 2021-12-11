from transformers import BertGenerationEncoder, BertGenerationDecoder, EncoderDecoderModel, BertTokenizer
import torch

from base_config import pretrain
from conf import pretrain_map

path = pretrain_map['bert']
tokenizer = BertTokenizer.from_pretrained(path)
from transformers.utils.logging import set_verbosity_error, set_verbosity_debug

set_verbosity_error()


def test_encoder_decoder():
    # leverage checkpoints for Bert2Bert model...
    # use BERT's cls token as BOS token and sep token as EOS token
    path = pretrain_map['bert']
    encoder = BertGenerationEncoder.from_pretrained(path, bos_token_id=101, eos_token_id=102)
    # add cross attention layers and use BERT's cls token as BOS token and sep token as EOS token
    decoder = BertGenerationDecoder.from_pretrained(path, add_cross_attention=True, is_decoder=True, bos_token_id=101, eos_token_id=102)
    bert2bert = EncoderDecoderModel(encoder=encoder, decoder=decoder)

    # create tokenizer...
    tokenizer = BertTokenizer.from_pretrained(path)

    input_ids = tokenizer('This is a long article to summarize', add_special_tokens=False, return_tensors="pt").input_ids
    labels = tokenizer('This is a short summary', return_tensors="pt").input_ids

    # train...
    model_output = bert2bert(input_ids=input_ids, decoder_input_ids=labels, labels=labels)

    # loss.backward()


def test_mask():
    attention_mask = torch.tensor([[1, 1, 1, 1, 1, 1]] * 2)
    input_shape = attention_mask.size()[0], 4
    batch_size, seq_length = input_shape
    seq_ids = torch.arange(seq_length)
    causal_mask = seq_ids[None, None, :].repeat(batch_size, seq_length, 1) <= seq_ids[None, :, None]
    # in case past_key_values are used we need to add a prefix ones mask to the causal mask
    # causal and attention masks must have same type with pytorch version < 1.3
    causal_mask = causal_mask.to(attention_mask.dtype)

    if True or causal_mask.shape[1] < attention_mask.shape[1]:
        prefix_seq_len = attention_mask.shape[1] - causal_mask.shape[1]
        causal_mask = torch.cat(
            [
                torch.ones(
                    (batch_size, seq_length, prefix_seq_len), dtype=causal_mask.dtype
                ),
                causal_mask,
            ],
            axis=-1,
        )

    extended_attention_mask = causal_mask[:, None, :, :] * attention_mask[:, None, None, :]
    print(extended_attention_mask)


def test_generate():
    encoder = BertGenerationEncoder.from_pretrained(path, bos_token_id=101, eos_token_id=102)
    # add cross attention layers and use BERT's cls token as BOS token and sep token as EOS token
    decoder = BertGenerationDecoder.from_pretrained(path, add_cross_attention=True, is_decoder=True, bos_token_id=101, eos_token_id=102)
    model = EncoderDecoderModel(encoder=encoder, decoder=decoder)
    input_ids = tokenizer("Hello, my dog is cute and ", return_tensors="pt")['input_ids']
    # generation_output = model.generate(**inputs, return_dict_in_generate=True, output_scores=True)

    beam_output = model.generate(input_ids,
                                 max_length=50,
                                 num_beams=5,
                                 early_stopping=True)
    output = tokenizer.decode(beam_output[0], skip_special_tokens=True)
    print(output)


def test_conditional_generation():
    torch.manual_seed(0)
    encoder_hidden_states = torch.randn(2, 16, 768)
    # input(encoder_hidden_states[0, 0, :10])
    encoder_hidden_states.requires_grad_(True)
    input_ids = torch.tensor([tokenizer.encode_plus("我的名字是")['input_ids'] for _ in range(2)])
    # input(input_ids)
    decoder = BertGenerationDecoder.from_pretrained(path, add_cross_attention=True, is_decoder=True, bos_token_id=101, eos_token_id=102)
    model_output = decoder(input_ids=input_ids, encoder_hidden_states=encoder_hidden_states, labels=input_ids)
    # print(model_output.logits.size())
    print(model_output.loss, model_output.logits.size())


def train_generation():
    from transformers import AdamW
    from tqdm import tqdm
    torch.manual_seed(0)
    encoder_hidden_states = torch.randn(2, 16, 768)
    # input(encoder_hidden_states[0, 0, :10])
    encoder_hidden_states.requires_grad_(True)
    input_ids = torch.tensor([tokenizer.encode_plus("我的名字是")['input_ids'] + [0] * 10 for _ in range(2)])
    input(input_ids)
    decoder = BertGenerationDecoder.from_pretrained(path, add_cross_attention=True, is_decoder=True, bos_token_id=101, eos_token_id=102).cuda()
    input_ids = input_ids.cuda()
    encoder_hidden_states = encoder_hidden_states.cuda()
    optimizer = AdamW(params=decoder.parameters(), lr=3e-5)
    labels = input_ids.clone()
    labels[input_ids == 0] = -100
    print(labels)
    for i in tqdm(range(50)):
        decoder.train()
        model_output = decoder(input_ids=input_ids, encoder_hidden_states=encoder_hidden_states, labels=labels)
        model_output.loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    decoder.eval()
    beam_output = decoder.generate(input_ids,
                                   max_length=50,
                                   num_beams=5,
                                   early_stopping=True)
    input_ids = torch.tensor([tokenizer.encode_plus("我的")['input_ids'] for _ in range(2)]).cuda()
    output = tokenizer.decode(beam_output[0], skip_special_tokens=True)
    print(output)


def test_t5():
    from transformers import T5ForConditionalGeneration, T5Config
    model = T5ForConditionalGeneration.from_pretrained(pretrain)
    config =

if __name__ == '__main__':
    # test_encoder_decoder()
    # test_mask()
    # test_generate()
    # test_conditional_generation()
    train_generation()
