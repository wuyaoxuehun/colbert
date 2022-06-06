# import faiss_indexers
import logging
from typing import Dict, List

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


def get_mrr(scores):
    pos_num, neg_num = 2, 8
    labels = torch.tensor([list(range(i * (pos_num + neg_num), i * (pos_num + neg_num) + pos_num)) for i in range(scores.size(0))])
    sorted_idx = scores.sort(dim=-1, descending=True)[1]
    positive_idxes = torch.zeros_like(scores)
    positive_idxes.scatter_(dim=-1, index=labels.to(scores.device), value=1)
    res = positive_idxes.gather(1, sorted_idx)
    return res.nonzero()[:, 1].float().mean()


def get_mrr_(scores, positive_idx_per_question):
    sorted_idx = scores.sort(dim=-1, descending=True)[1]
    positive_idxes = torch.zeros_like(scores)
    positive_idxes.scatter_(dim=-1, index=positive_idx_per_question[:, None].to(scores.device), value=1)
    res = positive_idxes.gather(1, sorted_idx)
    return res.argmax(-1).sum()


class ColbertModel(BaseModel):
    def __init__(self, args):
        super().__init__()
        self.args = args
        pretrain = args.dense_training_args.pretrain
        dim = args.dense_training_args.dim
        self.score_temperature = args.dense_training_args.score_temperature
        self.config: BertConfig = BertConfig.from_pretrained(pretrain)
        self.model = BertModel.from_pretrained(pretrain)
        # if not self.args.enable_multiview:
        self.linear = nn.Linear(self.config.hidden_size, dim, bias=False)
        self.tokenizer = CostomTokenizer(args)

    def tokenize_for_retriever(self, batch):
        obj = self.tokenizer.tokenize_q(batch)
        return obj

    def tokenize_for_train_retriever(self, batch: List[Dict], is_evaluating=False):
        docs = []
        for t in batch:
            if not is_evaluating:
                # if len(t['positive_ctxs']) == 1:
                #     t['positive_ctxs'].append(t['positive_ctxs'][-1])
                cur_pos_docs = list(context_random.choice(t['positive_ctxs'][:], 1))
                cur_neg_docs = list(context_random.choice(t['hard_negative_ctxs'][:], 1))
            else:
                if len(t['positive_ctxs']) < 2:
                    t['positive_ctxs'].append(t['positive_ctxs'][0])
                cur_pos_docs = t['positive_ctxs'][:2]
                if len(cur_pos_docs) < 2:
                    cur_pos_docs.append(cur_pos_docs[-1])
                assert len(t['hard_negative_ctxs']) >= 18
                cur_neg_docs = list(t['hard_negative_ctxs'][10:18])
            cur_docs = cur_pos_docs + cur_neg_docs
            docs += cur_docs
        D = self.tokenizer.tokenize_d(docs)
        return D

    def forward(self, batch, is_evaluating=False, is_testing_retrieval=False):
        q = self.tokenize_for_retriever(batch)
        q_ids, q_attention_mask, q_active_padding, *_ = [_.cuda() for _ in q]
        Q = self.query(q_ids, q_attention_mask)
        d = self.tokenize_for_train_retriever(batch, is_evaluating=is_evaluating)
        d_ids, d_attention_mask, d_active_padding, *_ = [_.cuda() for _ in d]
        D = self.doc(d_ids, d_attention_mask)
        # input((Q.size(), D.size(), q_active_padding.size(), d_active_padding.size()))
        Q, q_active_padding, D, d_active_padding = collection_qd_masks([Q, q_active_padding, D, d_active_padding])

        positive_idx_per_question = torch.tensor([_ * 2 for _ in range(Q.size(0))], device=Q.device)
        pred_scores = self.score(Q, D, q_active_padding, d_active_padding)
        if is_evaluating:
            # return get_mrr(pred_scores),
            return {"loss": get_mrr(pred_scores)}
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
