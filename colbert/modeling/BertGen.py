from tqdm import tqdm
from transformers import BertGenerationEncoder, BertGenerationDecoder, EncoderDecoderModel, BertTokenizer, T5Config, T5ForConditionalGeneration, T5Tokenizer, T5TokenizerFast, T5Model, T5EncoderModel, \
    AutoModel, AutoTokenizer, AutoConfig
import torch
from conf import pretrain_map, encoder_tokenizer, pretrain

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
tokenizer = T5TokenizerFast.from_pretrained(path)


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


def test_tok_word_map():
    s = "welcome to new york bilibili"  # , "welcome to new york for your visit"]
    print(tokenizer.tokenize(s))
    inputs = tokenizer.encode_plus(s,
                                   return_offsets_mapping=True, padding='max_length', max_length=10, truncation=True,
                                   add_special_tokens=False)
    print(inputs)
    words = s.split()
    for i in range(8):
        idx = inputs.token_to_word(i)
        print(idx, words[idx])
        # print()


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
    test_tok_word_map()
