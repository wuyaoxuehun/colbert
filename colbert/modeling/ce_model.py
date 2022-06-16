# import faiss_indexers
import logging
from typing import Dict, List

import numpy as np
import torch.distributed
from torch import nn
from transformers import BertPreTrainedModel, BertModel, BertConfig

from colbert.modeling.BaseModel import BaseModel
from colbert.training.losses import BiEncoderNllLoss
from colbert.utils.dense_conf import context_random

torch.multiprocessing.set_sharing_strategy("file_system")
# torch.multiprocessing.set_start_method('spawn', force=True)
from colbert.training.training_utils import collection_qd_masks
from colbert.modeling.tokenizers import CostomTokenizer

logger = logging.getLogger("__main__")


def get_mrr_(scores, pos_num=1, neg_num=7):
    # pos_num, neg_num = 1, 4
    # labels = torch.tensor([list(range(i * (pos_num + neg_num), i * (pos_num + neg_num) + pos_num)) for i in range(scores.size(0))])
    labels = torch.tensor([[]])
    sorted_idx = scores.sort(dim=-1, descending=True)[1]
    positive_idxes = torch.zeros_like(scores)
    positive_idxes.scatter_(dim=-1, index=labels.to(scores.device), value=1)
    res = positive_idxes.gather(1, sorted_idx)
    return res.nonzero()[:, 1].float().mean()


def get_mrr(scores, positive_idx_per_question):
    sorted_idx = scores.sort(dim=-1, descending=True)[1]
    positive_idxes = torch.zeros_like(scores)
    positive_idxes.scatter_(dim=-1, index=positive_idx_per_question[:, None].to(scores.device), value=1)
    res = positive_idxes.gather(1, sorted_idx)
    return res.argmax(-1).sum().float()


class CEModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        pretrain = args.dense_training_args.pretrain
        self.score_temperature = args.ce_training_args.score_temperature
        self.neg_num = args.ce_training_args.neg_num
        self.eval_topk = args.ce_test_args.eval_topk
        self.config: BertConfig = BertConfig.from_pretrained(pretrain)
        self.model = BertModel.from_pretrained(pretrain)
        # if not self.args.enable_multiview:
        # self.linear = nn.Linear(self.config.hidden_size, 1, bias=False)
        self.linear = nn.Linear(self.config.hidden_size, 1)
        self.tokenizer = CostomTokenizer(args)

    def tokenize_for_ce(self, batch, mode='train'):
        qp_seq = []
        for t in batch:
            question = t['question']
            if mode == 'test':
                qp_seq.extend([(question, _) for _ in t['retrieval_res'][:self.eval_topk]])
                continue

            # assert len(t['positive_ctxs']) == 1
            if mode == 'dev':
                pos = t['positive_ctxs'][:1]
                neg_num = self.neg_num * 2
            else:
                pos = list(context_random.choice(t['positive_ctxs'][:], 1))
                # pos = t['positive_ctxs'][:1]
                neg_num = self.neg_num
            while len(t['hard_negative_ctxs']) < neg_num:
                t['hard_negative_ctxs'].append(t['hard_negative_ctxs'][-1])
            if mode == 'train':
                # neg_random = np.random.default_rng(hash(t['question']) % 123457)
                neg = list(context_random.choice(t['hard_negative_ctxs'][5:50], neg_num, replace=False))
                # neg = list(neg_random.choice(t['hard_negative_ctxs'][5:50], neg_num, replace=False))
                # neg = t['hard_negative_ctxs'][:neg_num]
            else:
                # neg = t['hard_negative_ctxs'][5:5 + neg_num]
                neg = t['hard_negative_ctxs'][:neg_num]
            qp_seq.extend([(question, pos[0])] + [(question, _) for _ in neg])
        return self.tokenizer.tokenize_ce(qp_seqs=qp_seq)

    def forward(self, batch, mode='train'):
        ids, attention_mask = [torch.tensor(_).cuda() for _ in self.tokenize_for_ce(batch, mode=mode)]
        pooled_output = self.model(ids, attention_mask=attention_mask, return_dict=True, output_hidden_states=True).hidden_states[-1][:, 0, ...]
        logits = self.linear(pooled_output)
        pred_scores = logits.view(len(batch), -1)
        # labels = torch.tensor([1 if _ % len(batch) == 0 else 0 for _ in range(ids.size(0))], device=ids.device)
        # loss = nn.CrossEntropyLoss()(logits, labels)
        # return {"loss": loss}

        positive_idx_per_question = torch.tensor([0 for _ in range(len(batch))], device=ids.device)
        if mode != 'train':
            # return get_mrr(pred_scores),
            return {"loss": get_mrr(pred_scores, positive_idx_per_question=positive_idx_per_question),
                    'logits': pred_scores}
        score_temperature = self.score_temperature
        cur_retriever_loss = BiEncoderNllLoss(scores=pred_scores / score_temperature, positive_idx_per_question=positive_idx_per_question)
        return {"loss": cur_retriever_loss}

    def save(self: BertPreTrainedModel, save_dir: str):
        to_save = self.state_dict()
        if isinstance(self, nn.DataParallel):
            to_save = self.module
        import os
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
        logger.info('*' * 20 + "saving checkpoint to " + save_dir + '*' * 20)

        torch.save(to_save, os.path.join(save_dir, "pytorch.bin"))

    def load(self: BertPreTrainedModel, checkpoint: str):
        # logger.info('*' * 20 + "loading checkpoint from " + checkpoint + '*' * 20)
        print('*' * 20 + "loading checkpoint from " + checkpoint + '*' * 20)
        model = torch.load(checkpoint, map_location=lambda storage, loc: storage)
        load = {k: v for k, v in model.items() if (k in self.state_dict() and 'reader.para_choice_linear' not in k)}
        return self.load_state_dict(load, strict=False)


if __name__ == '__main__':
    pass
