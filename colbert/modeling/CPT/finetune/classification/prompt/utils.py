# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
import json
import pickle
import random
import string
from collections import defaultdict
from typing import Dict, List, Optional, Union

import numpy as np
import torch
from torch.nn import functional as F
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, GPT2Tokenizer



class LogitsList:
    """A list of logits obtained from a finetuned PET model"""

    def __init__(self, score: float, logits: List[List[float]]):
        """
        Create a new LogitsList.

        :param score: the corresponding PET model's score on the training set
        :param logits: the list of logits, where ``logits[i][j]`` is the score for label ``j`` at example ``i``
        """
        self.score = score
        self.logits = logits

    def __repr__(self):
        return 'LogitsList(score={}, logits[:2]={})'.format(self.score, self.logits[:2])

    def save(self, path: str) -> None:
        """Save this list to a file."""
        with open(path, 'w') as fh:
            fh.write(str(self.score) + '\n')
            for example_logits in self.logits:
                fh.write(' '.join(str(logit) for logit in example_logits) + '\n')

    @staticmethod
    def load(path: str, with_score: bool = True) -> 'LogitsList':
        """Load a list from a file"""
        score = -1
        logits = []
        with open(path, 'r') as fh:
            for line_idx, line in enumerate(fh.readlines()):
                line = line.rstrip('\n')
                if line_idx == 0 and with_score:
                    score = float(line)
                else:
                    logits.append([float(x) for x in line.split()])
        return LogitsList(score=score, logits=logits)


class InputExample(object):
    """A raw input example consisting of one or two segments of text and a label"""

    def __init__(self, guid, text_a, text_b=None, label=None, idx=0):
        """
        Create a new InputExample.

        :param guid: a unique textual identifier
        :param text_a: the sequence of text
        :param text_b: an optional, second sequence of text
        :param label: an optional label
        :param logits: an optional list of per-class logits
        :param meta: an optional dictionary to store arbitrary meta information
        :param idx: an optional numeric index
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label
        self.idx = idx

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serialize this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serialize this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

    @staticmethod
    def load_examples(path: str) -> List['InputExample']:
        """Load a set of input examples from a file"""
        with open(path, 'rb') as fh:
            return pickle.load(fh)

    @staticmethod
    def save_examples(examples: List['InputExample'], path: str) -> None:
        """Save a set of input examples to a file"""
        with open(path, 'wb') as fh:
            pickle.dump(examples, fh)


class InputFeatures(object):
    """A set of numeric features obtained from an :class:`InputExample`"""

    def __init__(self, input_ids, attention_mask=None, token_type_ids=None, label=None, lm_label=None, mlm_positions=None, idx=0, prompt_idx=0):
        """
        Create new InputFeatures.

        :param input_ids: the input ids corresponding to the original text or text sequence
        :param attention_mask: an attention mask, with 0 = no attention, 1 = attention
        :param token_type_ids: segment ids as used by BERT
        :param label: the label
        :param idx: an optional numeric index
        """
        self.input_ids = input_ids
        self.gen_prompt = None
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.label = label
        self.lm_label = lm_label
        self.mlm_positions = mlm_positions
        self.idx = idx
        self.prompt_idx = prompt_idx

    def __repr__(self):
        return str(self.to_json_string())

    def pretty_print(self, tokenizer):
        return f'input_ids         = {tokenizer.convert_ids_to_tokens(self.input_ids)}\n' + \
               f'attention_mask    = {self.attention_mask}\n' + \
               f'token_type_ids    = {self.token_type_ids}\n' + \
               f'mlm_positions     = {self.mlm_positions}\n' + \
               f'idx               = {self.idx}\n' + \
               f'label             = {self.label}\n' + \
               f'prompt_idx        = {self.prompt_idx}'    

    def to_dict(self):
        """Serialize this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serialize this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"
    
    @classmethod
    def from_dict(cls, data):
        return InputFeatures(
            data['input_ids'], data['attention_mask'], data.get('token_type_ids', None),
        )

class ExampleDataset:
    def __init__(self, exmaples: List[InputExample], tokenizer, converter, pid):
        self.examples = exmaples
        self.tokenizer = tokenizer
        self.converter = converter
        self.pid = pid
        for i, example in enumerate(self.examples):
            example.idx = i

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        example = self.examples[i]
        feature = self.converter.get_input_feature(example, self.pid)
        return feature

    def collate_fn(self, batch_features: List[InputFeatures]) -> Dict:
        batch = {
            'input_ids': [f.input_ids for f in batch_features],
            'attention_mask': [f.attention_mask for f in batch_features],
        }
        if batch_features[0].token_type_ids is not None:
            batch['token_type_ids'] = [f.token_type_ids for f in batch_features]
        
        batch_inputs = self.tokenizer.pad(batch, padding=True)
        batch_inputs = {k: torch.LongTensor(v) for k,v in batch_inputs.items()}
        batch_inputs['labels'] = torch.LongTensor([f.label for f in batch_features])
        batch_inputs['lm_labels'] = torch.LongTensor([f.lm_label for f in batch_features])
        batch_inputs['mlm_positions'] = torch.LongTensor([f.mlm_positions for f in batch_features])
        batch_inputs['idx'] = torch.LongTensor([f.idx for f in batch_features])
        batch_inputs['prompt_idx'] = torch.LongTensor([f.prompt_idx for f in batch_features])

        if batch_features[0].gen_prompt is not None:
            gen_prompt = torch.LongTensor([f.gen_prompt for f in batch_features])
        else:
            gen_prompt = batch_inputs['input_ids']
        
        decoder_input_ids = torch.zeros_like(gen_prompt)
        decoder_input_ids[:, 1:] = gen_prompt[:, :-1]
        decoder_input_ids[:, 0] = self.tokenizer.sep_token_id
        for i in range(decoder_input_ids.size(0)):
            decoder_input_ids[i, batch_inputs['mlm_positions'][i] + 1] = batch_inputs['lm_labels'][i]
        batch_inputs['decoder_input_ids'] = decoder_input_ids
        return batch_inputs


class PLMInputFeatures(InputFeatures):
    """A set of numeric input features for a model pretrained with a permuted language modeling objective."""

    def __init__(self, *_, perm_mask, target_mapping, **kwargs):
        super().__init__(**kwargs)
        self.perm_mask = perm_mask
        self.target_mapping = target_mapping

    def pretty_print(self, tokenizer):
        return super().pretty_print(tokenizer) + '\n' + \
               f'perm_mask         = {self.perm_mask}\n' + \
               f'target_mapping    = {self.target_mapping}'


class DictDataset(Dataset):
    """A dataset of tensors that uses a dictionary for key-value mappings"""

    def __init__(self, **tensors):
        tensors.values()

        assert all(next(iter(tensors.values())).size(0) == tensor.size(0) for tensor in tensors.values())
        self.tensors = tensors

    def __getitem__(self, index):
        return {key: tensor[index] for key, tensor in self.tensors.items()}

    def __len__(self):
        return next(iter(self.tensors.values())).size(0)


def set_seed(seed: int):
    """ Set RNG seeds for python's `random` module, numpy and torch"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def eq_div(N, i):
    """ Equally divide N examples among i buckets. For example, `eq_div(12,3) = [4,4,4]`. """
    return [] if i <= 0 else [N // i + 1] * (N % i) + [N // i] * (i - N % i)


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def remove_final_punc(s: str):
    """Remove the last character from a string if it is some form of punctuation"""
    return s.rstrip(string.punctuation)


def lowercase_first(s: str):
    """Lowercase the first letter of a string"""
    return s[0].lower() + s[1:]


def save_logits(path: str, logits: np.ndarray):
    """Save an array of logits to a file"""
    with open(path, 'w') as fh:
        for example_logits in logits:
            fh.write(' '.join(str(logit) for logit in example_logits) + '\n')
    pass


def softmax(x, temperature=1.0, axis=None):
    """Custom softmax implementation"""
    y = np.atleast_2d(x)

    if axis is None:
        axis = next(j[0] for j in enumerate(y.shape) if j[1] > 1)

    y = y * float(temperature)
    y = y - np.expand_dims(np.max(y, axis=axis), axis)
    y = np.exp(y)

    ax_sum = np.expand_dims(np.sum(y, axis=axis), axis)
    p = y / ax_sum

    if len(x.shape) == 1:
        p = p.flatten()
    return p


def get_verbalization_ids(word: str, tokenizer: PreTrainedTokenizer, force_single_token: bool) -> Union[int, List[int]]:
    """
    Get the token ids corresponding to a verbalization

    :param word: the verbalization
    :param tokenizer: the tokenizer to use
    :param force_single_token: whether it should be enforced that the verbalization corresponds to a single token.
           If set to true, this method returns a single int instead of a list and throws an error if the word
           corresponds to multiple tokens.
    :return: either the list of token ids or the single token id corresponding to this word
    """
    kwargs = {'add_prefix_space': True} if isinstance(tokenizer, GPT2Tokenizer) else {}
    ids = tokenizer.encode(word, add_special_tokens=False, **kwargs)
    if not force_single_token:
        return ids
    assert len(ids) == 1, \
        f'Verbalization "{word}" does not correspond to a single token, got {tokenizer.convert_ids_to_tokens(ids)}'
    verbalization_id = ids[0]
    assert verbalization_id not in tokenizer.all_special_ids, \
        f'Verbalization {word} is mapped to a special token {tokenizer.convert_ids_to_tokens(verbalization_id)}'
    return verbalization_id
    

def trim_input_ids(input_ids: torch.tensor, pad_token_id, mask_token_id, num_masks: int):
    """
    Trim a sequence of input ids by removing all padding tokens and keeping at most a specific number of mask tokens.

    :param input_ids: the sequence of input token ids
    :param pad_token_id: the id of the pad token
    :param mask_token_id: the id of the mask tokens
    :param num_masks: the number of masks to keeps
    :return: the trimmed sequence of input ids
    """
    assert input_ids.shape[0] == 1
    input_ids_without_pad = [x for x in input_ids[0] if x != pad_token_id]

    trimmed_input_ids = []
    mask_count = 0
    for input_id in input_ids_without_pad:
        if input_id == mask_token_id:
            if mask_count >= num_masks:
                continue
            mask_count += 1
        trimmed_input_ids.append(input_id)

    return torch.tensor([trimmed_input_ids], dtype=torch.long, device=input_ids.device)


def exact_match(predictions: np.ndarray, actuals: np.ndarray, question_ids: np.ndarray):
    """Compute the exact match (EM) for a sequence of predictions and actual labels"""
    unique_questions = set(question_ids)

    q_actuals = list(zip(question_ids, actuals))
    q_predictions = list(zip(question_ids, predictions))

    actuals_per_question = defaultdict(list)
    predictions_per_question = defaultdict(list)

    for qid, val in q_actuals:
        actuals_per_question[qid].append(val)
    for qid, val in q_predictions:
        predictions_per_question[qid].append(val)

    em = 0
    for qid in unique_questions:
        if actuals_per_question[qid] == predictions_per_question[qid]:
            em += 1
    em /= len(unique_questions)

    return em


def distillation_loss(predictions, targets, temperature):
    """Compute the distillation loss (KL divergence between predictions and targets) as described in the PET paper"""
    p = F.log_softmax(predictions / temperature, dim=1)
    q = F.softmax(targets / temperature, dim=1)
    return F.kl_div(p, q, reduction='sum') * (temperature ** 2) / predictions.shape[0]


def batch_index_select(tensor, index):
    """
    tensor [B, T1, H]
    index [B, T2]
    return [B, T2, H]
    """
    errors = (index >= tensor.size(1)).sum().item()
    assert errors == 0, errors
    batch_idx = torch.arange(0, tensor.size(0), device=tensor.device)
    if index.dim() > 1:
        batch_idx = batch_idx.view(-1, 1)
    return tensor[batch_idx, index]

def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, power=1, last_epoch=-1):
    """
    Create a schedule with a learning rate that decreases linearly from the initial lr set in the optimizer to 0,
    after a warmup period during which it increases linearly from 0 to the initial lr set in the optimizer.

    Args:
        optimizer (:class:`~torch.optim.Optimizer`):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (:obj:`int`):
            The number of steps for the warmup phase.
        num_training_steps (:obj:`int`):
            The totale number of training steps.
        last_epoch (:obj:`int`, `optional`, defaults to -1):
            The index of the last epoch when resuming training.

    Return:
        :obj:`torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
        ) ** power

    return LambdaLR(optimizer, lr_lambda, last_epoch)
