'''
this is adaptation of original, for every choice with its own context
'''

import json
import logging

import dgl

logger = logging.getLogger(__name__)
from torch.utils.data import Dataset
from dgl import DGLGraph

import torch


def encode_sequence_fever(context, background, question, option, max_seq_length, tokenizer):
    context_tokens = tokenizer.tokenize(context)
    background_tokens = tokenizer.tokenize(background)
    start_ending_tokens = tokenizer.tokenize(question)
    ending_tokens = tokenizer.tokenize(option)

    _truncate_seq_quadruple(context_tokens, background_tokens, start_ending_tokens, ending_tokens, max_seq_length - 5)

    padding_length = max_seq_length - len(context_tokens) - len(background_tokens) - \
                     len(start_ending_tokens) - len(ending_tokens) - 5
    input_ids = tokenizer.convert_tokens_to_ids(['[CLS]'] +
                                                context_tokens + ['[SEP]'] +
                                                background_tokens + ['[SEP]'] +
                                                start_ending_tokens + ['[SEP]'] +
                                                ending_tokens + ['[SEP]']) + [0] * padding_length
    input_mask = [1] * (max_seq_length - padding_length) + [0] * padding_length
    segment_ids = [0] * (len(context_tokens) + 1) + \
                  [1] * (len(background_tokens) + 1) + \
                  [0] * (len(start_ending_tokens) + 1) + \
                  [1] * (len(ending_tokens) + 2) + \
                  [0] * padding_length

    context_len = len(context_tokens)
    background_len = len(background_tokens)
    question_len = len(start_ending_tokens)
    opt_len = len(ending_tokens)
    return torch.LongTensor(input_ids), torch.LongTensor(input_mask), torch.LongTensor(segment_ids), \
           torch.LongTensor([context_len]), torch.LongTensor([background_len]), torch.LongTensor([question_len]), torch.LongTensor([opt_len])


from colbert.base_config import bert_tokenizer


def batch_transform_bert_fever(insts, bert_max_len, p_num=2, device="cuda"):
    all_graphs = []
    all_labels = []
    for inst in insts:
        # print(inst['question'])
        # print(inst['A'])
        # print(inst['paragraph_a'][0])
        # input()
        opt_graph = []
        for opt in list('abcd'):
            g = DGLGraph()
            g.add_nodes(p_num)
            for i in range(p_num):
                for j in range(p_num):
                    if i == j:
                        continue
                    g.add_edge(i, j)

            for i, node in enumerate(inst['paragraph_' + opt][:p_num]):
                # g.nodes[i].data['score'] = torch.tensor(float(node['score'])).unsqueeze(0).type(torch.float)
                context = node['paragraph']
                encoding_inputs, encoding_masks, encoding_ids, \
                context_len, background_len, question_len, opt_len = encode_sequence_fever(context,
                                                                                           inst.get('background', inst.get('scenario')),
                                                                                           inst.get('question'),
                                                                                           inst.get(opt.upper()),
                                                                                           bert_max_len,
                                                                                           bert_tokenizer)
                g.nodes[i].data['encoding'] = encoding_inputs.unsqueeze(0).to(device)
                g.nodes[i].data['encoding_mask'] = encoding_masks.unsqueeze(0).to(device)
                g.nodes[i].data['segment_id'] = encoding_ids.unsqueeze(0).to(device)
                g.nodes[i].data['context_len'] = context_len.unsqueeze(0).to(device)
                g.nodes[i].data['background_len'] = background_len.unsqueeze(0).to(device)
                g.nodes[i].data['question_len'] = question_len.unsqueeze(0).to(device)
                g.nodes[i].data['opt_len'] = opt_len.unsqueeze(0).to(device)

            opt_graph.append(g)

        all_graphs.append(opt_graph)
        all_labels.append(ord(inst['answer'] if inst['answer'] in list('ABCD') else 'A') - ord('A'))

    return all_graphs, torch.tensor(all_labels)


def _truncate_seq_quadruple(tokens_a, tokens_b, tokens_c, tokens_d, max_length):
    while True:
        total_length = len(tokens_a) + len(tokens_b) + len(tokens_c) + len(tokens_d)

        if total_length <= max_length:
            break
        if len(tokens_a) >= max(len(tokens_b), len(tokens_c), len(tokens_d)):
            tokens_a.pop()
        elif len(tokens_b) >= max(len(tokens_c), len(tokens_d)):
            # only truncate the beginning of backgroud+question
            tokens_b.pop()
        elif len(tokens_c) >= len(tokens_d):
            tokens_c.pop()
        else:
            tokens_d.pop()
