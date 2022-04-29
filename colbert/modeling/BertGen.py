import os
import string
from time import time

import numpy as np
# import cupy as np
import torch
from tqdm import tqdm
from transformers import BertGenerationEncoder, BertGenerationDecoder, EncoderDecoderModel, BertTokenizer, T5ForConditionalGeneration, T5TokenizerFast, T5EncoderModel, \
    AutoConfig
import sys

from conf import pretrain_map, encoder_tokenizer, pretrain

path = pretrain_map['bert']
tokenizer = BertTokenizer.from_pretrained(path)
from transformers.utils.logging import set_verbosity_error

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


def test_t5_():
    # AutoModel, AutoTokenizer, AutoConfig
    path = "google/t5-v1_1-base"
    # path = pretrain_map['t5_base']
    config = AutoConfig.from_pretrained(path, cache_dir=pretrain_map['t5_base_1_1'])
    print(config)
    tokenizer = T5TokenizerFast.from_pretrained(path, cache_dir=pretrain_map['t5_base_1_1'])
    input_ids = tokenizer('translate English to German: The house is wonderful.', return_tensors='pt').input_ids
    labels = tokenizer('Das Haus ist wunderbar.', return_tensors='pt').input_ids
    # the forward function automatically creates the correct decoder_input_ids
    # output = model(input_ids=input_ids, labels=labels)
    decoder_input_ids = tokenizer(tokenizer.pad_token, return_tensors='pt').input_ids

    print(config.d_model)

    print(decoder_input_ids, tokenizer.pad_token)
    print(config.sep_token_id, config.bos_token_id, tokenizer.eos_token)
    max_source_length = 16
    tokens = tokenizer.tokenize("<extra_id_0>" + "</s>" + " beijing ")
    ids = tokenizer.convert_tokens_to_ids(tokens)
    print(ids)
    input_ids = tokenizer("",
                          padding='longest',
                          max_length=max_source_length,
                          truncation=True,
                          return_tensors='pt')
    print(input_ids)
    # exit()
    print(config.num_decoder_layers)
    model = T5EncoderModel.from_pretrained(path, cache_dir=pretrain_map['t5_base_1_1'])
    # model = T5Model.from_pretrained(path)

    # output = model(input_ids=input_ids, decoder_input_ids=decoder_input_ids, return_dict=True)
    output = model(input_ids=input_ids['input_ids'], return_dict=True)
    print(type(output))

    outputs = model.generate(input_ids=input_ids, output_hidden_states=True, return_dict_in_generate=True)
    print(type(outputs))

    print(tokenizer.decode(outputs.sequences[0], skip_special_tokens=True))
    # print(loss)


def test_t5():
    # AutoModel, AutoTokenizer, AutoConfig
    path = "google/t5-v1_1-base"
    # path = pretrain_map['t5_base']
    path = pretrain_map["t5_base"]
    config = AutoConfig.from_pretrained(path)
    print(config)
    tokenizer = T5TokenizerFast.from_pretrained(path)
    print(tokenizer.pad_token)
    exit()
    input_ids = tokenizer(['''<extra_id_0> who has been married to julia roberts?</s>''']
                          , return_tensors='pt', padding=True, truncation=True).input_ids
    print(tokenizer.tokenize(
        '''can be cold and commanding when she needs to be, or warm and loving as Padmé.\"\" In February 2018, Portman reprised the role for a interview-Rap sketch on \"\"Saturday Night Live\"\". In \"\"The Phantom Menace\"\", Padmé Amidala, in her capacity as queen, is addressed as \"\"Your Majesty\"\", \"\"Your Royal Highness\"\" and \"\"Your Highness\"\". Contrary to usage in real monarchies, where the style is fixed and tied to the person's rank, in Lucas's \"\"Star Wars\"\" universe they are apparently interchangeable. After her tenure as monarch ended and she became a member of the Senate, Padmé Amidala was addressed as \"\"Senator Amidala\"\"'''))
    exit()
    labels = tokenizer('Das Haus ist wunderbar.', return_tensors='pt').input_ids
    model = T5ForConditionalGeneration.from_pretrained(path)

    # output = model(input_ids=input_ids, decoder_input_ids=decoder_input_ids, return_dict=True)
    # output = model(input_ids=input_ids['input_ids'],  return_dict=True)
    # print(type(output))

    outputs = model.generate(input_ids=input_ids, output_hidden_states=True, return_dict_in_generate=True, min_length=10)
    print(type(outputs))
    print(len(outputs.decoder_hidden_states))
    print([torch.cat(_, dim=1).size() for _ in outputs.decoder_hidden_states])
    print(torch.cat([_[-1] for _ in outputs.decoder_hidden_states], dim=1).size())

    for i in range(len(outputs.sequences)):
        print(len(tokenizer.decode(outputs.sequences[i], skip_special_tokens=False)))
        print((tokenizer.decode(outputs.sequences[i], skip_special_tokens=False)))
    # print(loss)


def test_tok():
    tokenizer = encoder_tokenizer.from_pretrained(pretrain)
    tokenizer.add_special_tokens({"additional_special_tokens": ["[unused1]"]})
    encoder = tokenizer.batch_encode_plus(["[unused1]"])
    print(encoder)


def train():
    path = pretrain_map["t5_base"]
    from transformers import T5Tokenizer, T5ForConditionalGeneration
    import torch

    tokenizer = T5Tokenizer.from_pretrained(path)
    model = T5ForConditionalGeneration.from_pretrained(path)

    # the following 2 hyperparameters are task-specific
    max_source_length = 512
    max_target_length = 128

    # Suppose we have the following 2 training examples:
    input_sequence_1 = "Welcome to NYC"
    output_sequence_1 = "Bienvenue à NYC"

    input_sequence_2 = "HuggingFace is a company"
    output_sequence_2 = "HuggingFace est une entreprise"

    # encode the inputs
    task_prefix = "translate English to French: "
    input_sequences = [input_sequence_1, input_sequence_2]
    encoding = tokenizer([task_prefix + sequence for sequence in input_sequences],
                         padding='longest',
                         max_length=max_source_length,
                         truncation=True,
                         return_tensors="pt")
    input_ids, attention_mask = encoding.input_ids, encoding.attention_mask

    # encode the targets
    target_encoding = tokenizer([output_sequence_1, output_sequence_2],
                                padding='longest',
                                max_length=max_target_length,
                                truncation=True)
    labels = target_encoding.input_ids

    # replace padding token id's of the labels by -100
    labels = [
        [(label if label != tokenizer.pad_token_id else -100) for label in labels_example] for labels_example in labels
    ]
    labels = torch.tensor(labels)

    # forward pass
    from colbert.training.training_utils import get_t5_optimizer
    optimizer = get_t5_optimizer(model)
    model.cuda()
    for i in tqdm(range(100)):
        loss = model(input_ids=input_ids.cuda(), attention_mask=attention_mask.cuda(), labels=labels.cuda()).loss
        loss.backward()
        optimizer.step()
    torch.save(model.state_dict(), "output/testsave/pytorch_model.bin")


def anoterh():
    path = pretrain_map["t5_base"]
    tokenizer = T5TokenizerFast.from_pretrained(path)
    # print(tokenizer.bos_token, tokenizer.eos_token)
    # print(tokenizer("</s>"))
    # exit()
    # exit()
    # tokenizer.padding_side = "left"
    # tokenizer.pad_token = tokenizer.eos_token  # to avoid an error

    # task_prefix = "translate English to French: "
    task_prefix = ""
    sentences = ["<extra_id_0> what language they speak in taiwan?</s><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad>",
                 "HuggingFace est une entreprise"]  # use different length sentences to test batching
    inputs = tokenizer([task_prefix + sentence for sentence in sentences], return_tensors="pt", padding='max_length', max_length=64, truncation=True)
    print(tokenizer.batch_decode(inputs['input_ids']))
    print(inputs)
    model = T5ForConditionalGeneration.from_pretrained(path)
    # model.load_state_dict(torch.load("output/testsave/pytorch_model.bin"))
    output_sequences = model.generate(
        input_ids=inputs['input_ids'],
        attention_mask=inputs['attention_mask'],
        do_sample=False,  # disable sampling to test if batching affects output,
        num_beams=5
    )

    print(tokenizer.batch_decode(output_sequences, skip_special_tokens=True))


def train_decoder_prefix():
    path = pretrain_map["t5_base"]
    from transformers import T5Tokenizer, T5ForConditionalGeneration
    import torch

    tokenizer = T5Tokenizer.from_pretrained(path)
    model = T5ForConditionalGeneration.from_pretrained(path)

    # the following 2 hyperparameters are task-specific
    max_source_length = 512
    max_target_length = 128

    # Suppose we have the following 2 training examples:
    input_sequence_1 = "Welcome to NYC"
    output_sequence_1 = "title: HuggingFace is a company"

    input_sequence_2 = "Welcome to NYC"
    output_sequence_2 = "sentence: HuggingFace est une entreprise"

    # encode the inputs
    task_prefix = "query to enrich: "
    input_sequences = [input_sequence_1, input_sequence_2]
    encoding = tokenizer([task_prefix + sequence for sequence in input_sequences],
                         padding='longest',
                         max_length=max_source_length,
                         truncation=True,
                         return_tensors="pt")
    input_ids, attention_mask = encoding.input_ids, encoding.attention_mask

    # encode the targets
    target_encoding = tokenizer([output_sequence_1, output_sequence_2],
                                padding='longest',
                                max_length=max_target_length,
                                truncation=True)
    labels = target_encoding.input_ids

    # replace padding token id's of the labels by -100
    labels = [
        [(label if label != tokenizer.pad_token_id else -100) for label in labels_example] for labels_example in labels
    ]
    labels = torch.tensor(labels)

    # forward pass
    from colbert.training.training_utils import get_t5_optimizer
    optimizer = get_t5_optimizer(model)
    model.cuda()
    for i in tqdm(range(100)):
        loss = model(input_ids=input_ids.cuda(), attention_mask=attention_mask.cuda(), labels=labels.cuda()).loss
        loss.backward()
        optimizer.step()
    torch.save(model.state_dict(), "output/testsave/pytorch_model.bin")


path = pretrain_map["t5_base"]
# path = pretrain_map["bert-base-en_uncased"]
tokenizer = T5TokenizerFast.from_pretrained(path)


# tokenizer = BertTokenizerFast.from_pretrained(path)


def prefix_allowed_tokens_fn(idx, prev_input_ids):
    types = ["<pad>sentence: ", "<pad>title: "]
    ids = tokenizer(types[1 - idx], return_tensors="pt", padding='longest', max_length=64, truncation=True, add_special_tokens=False)['input_ids'][0]
    # input(ids)
    if len(prev_input_ids) < len(ids):
        # print([ids[len(prev_input_ids)]])
        return [ids[len(prev_input_ids)]]
    return None


def test_decoder_prefix():
    # path = "output/testsave/pytorch_model.bin"
    task_prefix = "query to enrich: "
    sentences = ["Welcome to NYC",
                 "Welcome to NYC"]  # use different length sentences to test batching
    inputs = tokenizer([task_prefix + sentence for sentence in sentences], return_tensors="pt", padding='max_length', max_length=64, truncation=True)
    print(tokenizer.batch_decode(inputs['input_ids']))
    print(inputs)
    model = T5ForConditionalGeneration.from_pretrained(path)

    model.load_state_dict(torch.load("output/testsave/pytorch_model.bin"))

    output_sequences = model.generate(
        input_ids=inputs['input_ids'],
        attention_mask=inputs['attention_mask'],
        do_sample=True,  # disable sampling to test if batching affects output,
        prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
        num_beams=5
    )

    print(tokenizer.batch_decode(output_sequences, skip_special_tokens=True))


from colbert.modeling.cpy.hello_world import get_real_inputs2, span_mean


def test_tok_word_map_():
    s = "welcome to new york for youra temasd visit abcdff" * 6
    # s = "did sa cham?" * 1
    import nltk
    puncts = string.punctuation
    ignore_words = list(puncts) + ['query: ']
    print(tokenizer.tokenize(list(puncts), is_split_into_words=True))
    print(tokenizer.tokenize(s))
    words = nltk.word_tokenize(s)
    words = ["  ", "query: "] + words + ['</s>']
    ignore_word_indices = [i for i, w in enumerate(words) if w in ignore_words]
    max_seq_length = 180
    # max_seq_length = 16
    print(tokenizer.tokenize(words, is_split_into_words=True))
    inputs = tokenizer.encode_plus(words,
                                   # return_offsets_mapping=True,
                                   padding='max_length', max_length=max_seq_length, truncation=True,
                                   add_special_tokens=False, is_split_into_words=True)
    print(inputs)
    # words = inputs.word_ids()
    # words = s.split()
    print(words)
    word_ids = inputs.word_ids()
    max_active_len = sum(inputs['attention_mask'])
    attention_mask = torch.zeros(max_seq_length, max_seq_length, dtype=torch.long)
    attention_mask[:max_active_len, :max_active_len] = 1
    # arange = torch.arange(0, max_seq_length)
    # column = arange.expand(max_seq_length, max_seq_length)
    # row = arange[:, None].expand(max_seq_length, max_seq_length)
    # attention_mask[(row < max_active_len) & (column < max_active_len)] = 1
    active_indices = []

    # for i in range(max_active_len):
    #     word_idx = inputs.token_to_word(i)
    #     if word_idx in ignore_word_indices:
    #         continue
    #     start, end = inputs.word_to_tokens(word_idx)
    #     # for j in range(start, end):
    #     if i + 1 < end:
    #         attention_mask[(row == i) & ((column < start) | (column >= end))] = 0
    #         attention_mask[(column == i) & ((row < start) | (row >= end))] = 0
    #     else:
    #         active_indices.append(i)

    # i = 0
    # while i < max_active_len:
    #     word_idx = inputs.token_to_word(i)
    #     start, end = inputs.word_to_tokens(word_idx)
    #     for j in range(start, end - 1):
    #         row_eqi = row == j
    #         col_start = column < start
    #         col_end = column >= end
    #         row_selector = row_eqi & (col_start | col_end)
    #         col_selector = row_selector.T
    #         attention_mask[row_selector] = 0
    #         attention_mask[col_selector] = 0
    #     if word_idx not in ignore_word_indices:
    #         active_indices.append(end - 1)
    #     i = end

    i = 0
    while i < max_active_len:
        word_idx = inputs.token_to_word(i)
        # word_idx = word_ids[i]
        start, end = inputs.word_to_tokens(word_idx)
        attention_mask[start:end - 1, :start] = 0
        attention_mask[start:end - 1, end:] = 0
        attention_mask[:start, start:end - 1] = 0
        attention_mask[end:, start:end - 1] = 0
        if word_idx not in ignore_word_indices:
            active_indices.append(end - 1)
        i = end

    attention_mask1 = torch.zeros(max_seq_length, max_seq_length, dtype=torch.long)
    attention_mask1[:max_active_len, :max_active_len] = 1
    i = 0
    while i < max_active_len:
        # word_idx = inputs.token_to_word(i)
        word_idx = word_ids[i]
        start = i
        end = start + 1
        while end < max_active_len and word_ids[end] == word_ids[start]:
            end += 1
        attention_mask1[start:end - 1, :start] = 0
        attention_mask1[start:end - 1, end:] = 0
        attention_mask1[:start, start:end - 1] = 0
        attention_mask1[end:, start:end - 1] = 0
        if word_idx not in ignore_word_indices:
            active_indices.append(end - 1)
        i = end
    print(torch.allclose(attention_mask1, attention_mask))
    print(max_active_len)
    print(attention_mask)
    for i in tqdm(range(1920)):
        attention_mask = torch.tensor(get_real_inputs2(inputs['input_ids'][:max_active_len], word_ids[:max_active_len], ignore_word_indices, max_seq_length)[1])
        # attention_mask2 = torch.tensor(get_real_inputs3(inputs['input_ids'][:max_active_len], word_ids[:max_active_len], ignore_word_indices, max_seq_length)[1])
        # print(attention_mask2)
        # print(attention_mask)
        # assert (torch.allclose(attention_mask2, attention_mask))

    return
    # i = 0
    # while i < max_active_len:
    #     word_idx = inputs.token_to_word(i)
    #     # if word_idx in ignore_word_indices:
    #     #     i += 1
    #     #     continue
    #     start, end = inputs.word_to_tokens(word_idx)
    #     # for j in range(start, end):
    #     # if i + 1 < end:
    #     # selector = ((row >= start) & (row < end - 1) & ((column < start) | (column >= end))) | \
    #     #            ((column >= start) & (column < end - 1) & ((row < start) | (row >= end)))
    #     for j in range(start, end-1):
    #         for k in range(0, start):
    #             attention_mask[j, k] = 0
    #             attention_mask[k, j] = 0
    #         for k in range(end, max_active_len):
    #             attention_mask[j, k] = 0
    #             attention_mask[k, j] = 0
    #
    #     # attention_mask[selector] = 0
    #     # attention_mask[(start <= row) & (row <= (end - 1)) & ((column >= start) & (column < end-1))] = 1
    #     # attention_mask[((column >= start) & (column < end-1))] = 1
    #     # attention_mask[(column == end - 1)] = 1
    #     # attention_mask[(column == i) & ((row < start) | (row >= end))] = 0
    #     # else:
    #     if word_idx not in ignore_word_indices:
    #         active_indices.append(end - 1)
    #     # i += 1
    #     i = end

    # for i in range(max_active_len):
    #     word_idx = inputs.token_to_word(i)
    #     if word_idx in ignore_word_indices:
    #         continue
    #     start, end = inputs.word_to_tokens(word_idx)
    #     # for j in range(start, end):
    #     if i + 1 < end:
    #         # attention_mask[(row == i) & ((column < start) | (column >= end))] = 0
    #         # attention_mask[(column == i) & ((row < start) | (row >= end))] = 0
    #         row_eqi = row == i
    #         col_start = column < start
    #         col_end = column >= end
    #         row_selector = row_eqi & (col_start | col_end)
    #         col_selector = row_selector.T
    #         attention_mask[row_selector] = 0
    #         attention_mask[col_selector] = 0
    #     else:
    #         active_indices.append(i)
    # attention_mask[(row >= max_active_len) | (column >= max_active_len)] = 0

    active_padding = [1] * len(active_indices) + [0] * (max_seq_length - len(active_indices))

    print(attention_mask)

    for i in range(100):
        idx = inputs.token_to_word(i)
        print(i, idx, words[idx], i in active_indices)
        # print()


def test_tok_word_map():
    s = "did sa cham?"  # , "welcome to new york for your visit"]
    import nltk
    puncts = string.punctuation
    ignore_words = list(puncts) + ['query: ']
    print(tokenizer.tokenize(list(puncts), is_split_into_words=True))
    print(tokenizer.tokenize(s))
    words = nltk.word_tokenize(s)
    words = ["  ", "query: "] + words + ['</s>']
    ignore_word_indices = [i for i, w in enumerate(words) if w in ignore_words]
    max_seq_length = 16
    print(tokenizer.tokenize(words, is_split_into_words=True))
    token_to_word, word_to_tokens = [], {}
    input_ids = []

    for word_id, word in enumerate(words):
        tokens = tokenizer.encode(word, add_special_tokens=False)
        token_to_word += [word_id] * len(tokens)
        word_to_tokens[word_id] = (len(input_ids), len(input_ids) + len(tokens))
        input_ids += tokens

    max_active_len = len(input_ids)
    input_ids += [0] * (max_seq_length - len(input_ids))

    # words = inputs.word_ids()
    # words = s.split()
    print(words)
    attention_mask = torch.zeros(max_seq_length, max_seq_length, dtype=torch.long)
    arange = torch.arange(0, max_seq_length)
    column = arange.expand(max_seq_length, max_seq_length)
    row = arange[:, None].expand(max_seq_length, max_seq_length)
    attention_mask[(row < max_active_len) & (column < max_active_len)] = 1
    active_indices = []

    i = 0
    while i < max_active_len:
        word_idx = token_to_word[i]
        start, end = word_to_tokens[word_idx]
        attention_mask[start:end - 1, :start] = 0
        attention_mask[start:end - 1, end:] = 0
        attention_mask[:start, start:end - 1] = 0
        attention_mask[end:, start:end - 1] = 0
        if word_idx not in ignore_word_indices:
            active_indices.append(end - 1)
        i = end

    active_padding = [1] * len(active_indices) + [0] * (max_seq_length - len(active_indices))

    print(attention_mask)

    for i in range(100):
        idx = token_to_word[i]
        print(i, idx, words[idx], i in active_indices)
        # print()


def get_real_inputs(input_ids, word_ids, ignore_word_indices, max_seq_length):
    if len(input_ids) > max_seq_length:
        input_ids = input_ids[:max_seq_length]
        word_ids = word_ids[:max_seq_length]

    max_active_len = len(input_ids)
    input_ids += [0] * (max_seq_length - max_active_len)
    attention_mask = np.zeros((max_seq_length, max_seq_length)).astype(int)
    attention_mask[:max_active_len, :max_active_len] = 1
    active_indices = []

    i = 0
    word_ids = word_ids
    while i < max_active_len:
        # word_idx = inputs.token_to_word(i)
        word_idx = word_ids[i]
        start = i
        end = start + 1
        while end < max_active_len and word_ids[end] == word_ids[start]:
            end += 1
        attention_mask[start:end - 1, :start] = 0
        attention_mask[start:end - 1, end:] = 0
        attention_mask[:start, start:end - 1] = 0
        attention_mask[end:, start:end - 1] = 0
        if word_idx not in ignore_word_indices:
            active_indices.append(end - 1)
        i = end
    active_indices += [0] * (max_seq_length - len(active_indices))
    active_padding = [1] * len(active_indices) + [0] * (max_seq_length - len(active_indices))
    # attention_mask = attention_mask.tolist()
    return input_ids, attention_mask, active_indices, active_padding


def get_real_inputs2_(input_ids, word_ids, ignore_word_indices, max_seq_length):
    if len(input_ids) > max_seq_length:
        input_ids = input_ids[:max_seq_length]
        word_ids = word_ids[:max_seq_length]

    end_ids = [0] * max_seq_length
    end_sets = set()
    cur_end = len(word_ids) - 1
    for idx, word_id in reversed(list(enumerate(word_ids))):
        if word_id != word_ids[cur_end]:
            cur_end = idx
        end_ids[idx] = cur_end
        end_sets.add(cur_end)

    max_active_len = len(input_ids)
    input_ids += [0] * (max_seq_length - max_active_len)
    # attention_mask = np.zeros((max_seq_length, max_seq_length)).astype(int)
    attention_mask = [[0] * max_seq_length for _ in range(max_seq_length)]
    # attention_mask[:max_active_len, :max_active_len] = 0
    # active_indices = []

    start_id = 0
    # print(end_sets)
    for i in range(max_active_len):
        if end_ids[i] != i:
            for j in range(start_id, end_ids[i] + 1):
                attention_mask[i][j] = 1
        else:
            t = list(end_sets | set(list(range(start_id, i + 1))))
            for j in t:
                attention_mask[i][j] = 1
            start_id = i + 1
    # active_indices = list(end_sets)

    active_indices = list(end_sets - set([end_ids[_] for _ in ignore_word_indices]))
    active_indices += [0] * (max_seq_length - len(active_indices))
    active_padding = [1] * len(active_indices) + [0] * (max_seq_length - len(active_indices))
    # attention_mask = attention_mask.tolist()
    return input_ids, attention_mask, active_indices, active_padding


def profile_attention_mask():
    from line_profiler import LineProfiler
    lp = LineProfiler()
    # lp.add_function(get_real_inputs2)
    # lp.add_function(get_real_inputs)
    # lp.add_function(get_real_inputs3)
    lp_wrapper = lp(test_tok_word_map_)

    lp_wrapper()
    lp.print_stats(sys.stdout)  # 打印出性能分析结果


def test_span():
    os.environ["CUDA_VISIBLE_DEVICES"] = "4"
    Q = torch.randn((2, 4, 3))
    spans = torch.tensor([[[0, 1], [2, 4]]] * 2)

    def test_span1(Q, spans):
        output = torch.zeros([Q.size(0), spans.size(1), Q.size(2)], device=Q.device)
        for i, t_spans in enumerate(spans):
            for j, (s, e) in enumerate(t_spans):
                output[i, j, ...] = Q[i, s:e, ...].mean(0)
        return output

    from colbert.modeling.model_utils import batch_index_select

    # print(Q)
    # print(spans)
    # print(output)
    def span_mean1(Q, spans):
        presum_q = Q.cumsum(1)
        presum_q = torch.cat([torch.zeros([Q.size(0), 1, Q.size(2)], device=Q.device), presum_q], dim=1)
        start_idx, end_idx = spans[:, :, 0], spans[:, :, 1]
        # print(presum_q.size(), start_idx.size(), end_idx.size())
        start_q, end_q = [batch_index_select(presum_q, 1, _) for _ in [start_idx, end_idx]]
        res_q = end_q - start_q
        span_len = end_idx = start_idx
        span_len[span_len == 0] = 1
        res_q = res_q / (end_idx - start_idx)[:, :, None]
        return res_q

    # for method in [span_mean1, test_span1]:
    #     t1 = time()
    #     for i in tqdm(range(1)):
    #
    #         method(Q, spans)
    #         # t2 = span_mean1(Q, spans)
    #     print(time() - t1)

    bs = 88
    span_num = 80
    Q = torch.randn((bs, 180, 768)).cuda()
    # spans = torch.tensor([[[0, 1], [2, 4]]] * 2).cuda()
    start = torch.randint(0, 180, (bs, span_num, 1))
    end = torch.randint(0, 6, (bs, span_num, 1))
    spans = torch.cat([start, (start + end).clamp(0, 180)], dim=-1).cuda()
    for i in range(2):
        for method in [span_mean1, span_mean]:
            t1 = time()
            for i in tqdm(range(2300)):
                # Q = torch.randn((88, 180, 768)).cuda()
                # spans = torch.tensor([[[0, 1], [2, 4]]] * 2).cuda()
                method(Q, spans)
                # t2 = span_mean1(Q, spans)
            print(time() - t1)
        # print(Q)
        # print(spans)
        # print(t1)
        # print(t2)
        # print(t2 - t1)
        # print(torch.allclose(t1, t2))


if __name__ == '__main__':
    # test_encoder_decoder()
    # test_mask()
    # test_generate()
    # test_conditional_generation()
    # train_generation()
    # test_t5()
    # anoterh()
    # train()
    # test_tok()
    # train_decoder_prefix()
    # test_decoder_prefix()
    # test_tok_word_map_()
    # profile_attention_mask()
    test_span()
