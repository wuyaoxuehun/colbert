# Generates problem and choices from enrich text

import argparse
from collections import defaultdict
import copy
from datetime import datetime
import logging
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import trange
from transformers import MT5ForConditionalGeneration, set_seed

from tokenizer import T5PegasusTokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ts = defaultdict(int)
def log_interval(tag, x, intv):
    global ts

    ct = datetime.now().timestamp()
    if ct - ts[tag] > intv:
        ts[tag] = ct
        logger.info(f'[{tag}] {x}')

config: argparse.Namespace = None
model: MT5ForConditionalGeneration = None
tokenizer: Optional[T5PegasusTokenizer] = None

def load_tokenizer() -> T5PegasusTokenizer:
    return T5PegasusTokenizer.from_pretrained(config.pretrain_model)

def load_model() -> MT5ForConditionalGeneration:
    return MT5ForConditionalGeneration.from_pretrained(config.pretrain_model).to(config.device)

def load_dataset(path: str, is_train: bool):
    dataset = json.load(open(path))
    return dataset

def to_tensor(x, dtype):
    if isinstance(x, list):
        return torch.tensor(x, dtype=dtype, device=config.device)
    else:
        return x.to(config.device)

def tokenize_clip(clip: List[Dict[str, Any]]):

    def truncate_one(x: str) -> str:
        l = len(x)
        return x[:l//2] + x[l//2+1:]

    def truncate_multipart_join(parts: List[str], length: int, separator: str) -> str:
        while sum(map(len, parts)) > length:
            mlen = max(map(len, parts))
            for i, v in enumerate(parts):
                if len(v) == mlen:
                    parts[i] = truncate_one(parts[i])
                    break
        return separator.join(parts)

    def preprocess_tokenize_statement(triples, scenario, question, choice):
        input_data = truncate_multipart_join([triples, scenario, question], config.max_input_length, '|')
        # input_data = truncate_multipart_join([triples], config.max_input_length, '|')
        log_interval('input', input_data, 10)
        output_data = truncate_multipart_join([choice], config.max_output_length, '|')
        log_interval('output', output_data, 10)
        return input_data, output_data

    def shift_left(l):
        for sl in l:
            n = len(sl)
            for i in range(n - 1):
                sl[i] = sl[i + 1]
                if sl[i] == tokenizer.pad_token_id:
                    sl[i] = -100
            sl.pop()

    def gen_entity_map(x):
        es: Set[str] = set()

        for y in x:
            if isinstance(x[y], dict) and 'ner' in x[y]:
                for eid in (e['entity'] for e in x[y]['ner']):
                    es.add(eid)
        
        return {y: f'[unused{x+1}]' for (x, y) in enumerate(es)}

    def process_extracted(em: Dict[str, str], x: List[List[str]]) -> str:
        cx = x[:]
        random.shuffle(cx)
        return ''.join([f'<{em[l[0]]},{l[1]},{l[2]}>' for l in cx])

    def process_text(em: Dict[str, str], x: Dict[str, Any]) -> str:
        n = x['text']
        msk = [None for _ in n]
        for ne in x['ner']:
            for i in range(ne['start'], ne['end']):
                msk[i] = 'del'
            msk[ne['start']] = ne['entity']
        ans = []
        for x, y in zip(n, msk):
            if y == 'del':
                continue
            elif y is None:
                ans.append(x)
            else:
                ans.append(em[y])
        return ''.join(ans)

    entity_maps = [gen_entity_map(x) for x in clip]

    text_pairs = [[
        process_extracted(em, x['extracted']),
        process_text(em, x['scenario']),
        process_text(em, x['question']),
        process_text(em, x['choice'])
    ] for x, em in zip(clip, entity_maps)]
    
    text_pairs = [preprocess_tokenize_statement(*x) for x in text_pairs]

    input_data_tokenized = tokenizer([x[0] for x in text_pairs],
        padding='longest',
        return_token_type_ids=False,
        return_attention_mask=True)

    output_data_tokenized = tokenizer([x[1] for x in text_pairs],
        padding='longest',
        return_token_type_ids=False,
        return_attention_mask=False)

    shift_left(output_data_tokenized['input_ids'])

    return input_data_tokenized, output_data_tokenized

def from_clip(clip: List[Dict[str, Any]], is_train: bool, is_generate: bool):
    clip = copy.deepcopy(clip)
    inputs, outputs = tokenize_clip(clip)
    i_masks = to_tensor(inputs['attention_mask'], torch.long)
    i_encoding = to_tensor(inputs['input_ids'], torch.long)
    o_encoding = to_tensor(outputs['input_ids'], torch.long)

    if is_generate:
        return {
            "input_ids": i_encoding,
            "attention_mask": i_masks,
        }
    else:
        return {
            "input_ids": i_encoding,
            "attention_mask": i_masks,
            "labels": o_encoding,
        }

def process_generated(prob: Dict[str, str], generated: str) -> Dict[str, str]:
    return {**prob, 'generated': generated}

def main():
    global config, model, tokenizer

    parser = argparse.ArgumentParser()

    # paths
    parser.add_argument('--pretrain_model', default='/data1/PTLM/t5_pegasus_base/')
    parser.add_argument('--load_path')
    parser.add_argument('--save_path')
    parser.add_argument('--gendata_save_dir')
    parser.add_argument('--train', required=True)
    parser.add_argument('--dev', required=True)
    parser.add_argument('--test', required=True)

    # hyper parameters
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--eval_batch_size', type=int, default=24)
    parser.add_argument('--max_input_length', type=int, default=512)
    parser.add_argument('--max_output_length', type=int, default=512)
    parser.add_argument('--max_generate_length', type=int, default=512)
    parser.add_argument('--pre_epoch', type=int, default=10)
    parser.add_argument('--epoch', type=int, default=10)

    # misc
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--generate_repetition', type=int, default=10)

    config = parser.parse_args()

    set_seed(config.seed)
    logger.info('config %s', config)

    model = load_model()
    tokenizer = load_tokenizer()

    train_set = load_dataset(config.train, is_train=True)
    dev_set = load_dataset(config.dev, is_train=False)
    test_set = load_dataset(config.test, is_train=False)

    stage_0_train_set = [x for x in train_set if x['extracted'] == []]
    stage_1_train_set = [x for x in train_set if x['extracted'] != []]
    dev_set = [x for x in dev_set if x['extracted'] != []]
    test_set = [x for x in test_set if x['extracted'] != []]

    unseen_set = dev_set + test_set
    logger.info("train %s", len(train_set))
    logger.info("dev %s", len(dev_set))
    logger.info("test %s", len(test_set))
    logger.info("unseen %s", len(unseen_set))

    if config.load_path is not None:
        model.load_state_dict(torch.load(config.load_path), strict=True)
    trainer = optim.Adam(model.parameters(), lr=config.lr)

    best_dev_loss = 1e100

    for epoch in trange(config.pre_epoch, dynamic_ncols=True):
        logger.info("pre epoch %s", epoch)

        # train init
        model.train()
        indexes = list(range(len(stage_0_train_set)))
        random.shuffle(indexes)

        # train
        all_loss, all_iter = 0, 0
        for i in trange(0, len(indexes), config.batch_size):
            clip = [stage_0_train_set[j] for j in indexes[i:i+config.batch_size]]
            outputs = model.forward(**from_clip(clip, is_train=True, is_generate=False))
            all_loss += outputs.loss.cpu().item()
            all_iter += 1
            trainer.zero_grad()
            outputs.loss.backward()
            trainer.step()
        logger.info('train loss %s', all_loss / all_iter)

    for epoch in trange(config.epoch, dynamic_ncols=True):
        logger.info("epoch %s", epoch)

        # train init
        model.train()
        indexes = list(range(len(stage_1_train_set)))
        random.shuffle(indexes)

        # train
        all_loss, all_iter = 0, 0
        for i in trange(0, len(indexes), config.batch_size):
            clip = [stage_1_train_set[j] for j in indexes[i:i+config.batch_size]]
            outputs = model.forward(**from_clip(clip, is_train=True, is_generate=False))
            all_loss += outputs.loss.cpu().item()
            all_iter += 1
            trainer.zero_grad()
            outputs.loss.backward()
            trainer.step()
        logger.info('train loss %s', all_loss / all_iter)

        # eval init
        model.eval()
        with torch.no_grad():
            # eval dev+test set
            all_loss, all_iter = 0, 0
            for i in trange(0, len(unseen_set), config.eval_batch_size):
                clip = unseen_set[i:i+config.eval_batch_size]
                outputs = model.forward(**from_clip(clip, is_train=False, is_generate=False))
                all_loss += outputs.loss.cpu().item()
                all_iter += 1
            logger.info('dev loss %s', all_loss / all_iter)
            cur_dev_loss = all_loss / all_iter
            if cur_dev_loss < best_dev_loss:
                logger.info('saving best model...')
                if config.save_path is not None:
                    torch.save(model.state_dict(), config.save_path)
                best_dev_loss = cur_dev_loss

            # Currently based on dev+test
            output_segments = []
            probs = []
            for _ in trange(config.generate_repetition):
                for i in trange(0, len(unseen_set), config.eval_batch_size):
                    clip = unseen_set[i:i+config.eval_batch_size]
                    output = model.generate(do_sample=True,
                        decoder_start_token_id=tokenizer.cls_token_id,
                        eos_token_id=tokenizer.sep_token_id,
                        max_length=config.max_generate_length,
                        **from_clip(clip, is_train=False, is_generate=True)
                    )
                    output = output.cpu().numpy()
                    for line in output:
                        output_segments.append(''.join(tokenizer.decode(line[1:])).replace(' ', '').replace('[SEP]', '').replace('[PAD]', ''))
                probs += [{
                    'scenario': x['scenario'],
                    'question': x['question'],
                    'original_choice': x['choice'],
                    'extracted': x['extracted']
                } for x in unseen_set]
                logger.info(f'output_segments {len(output_segments)}')
                logger.info(f'probs {len(probs)}')
            
            generated_sample_list = []
            for prob, generated in zip(probs, output_segments):
                generated_sample = process_generated(prob, generated)
                if generated_sample is not None:
                    generated_sample_list.append(generated_sample)
                else:
                    print(prob, generated)

            with open(Path(config.gendata_save_dir) / f'epoch_{epoch}_gen.json', 'w') as f:
                json.dump(generated_sample_list, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()