import json

import torch
import torch.nn.functional as F
from torch import nn, einsum
from transformers import BertPreTrainedModel


class BaseModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = None
        self.linear = None
        self.args = None
        self.get_representation = self.get_representation

    @property
    def encoder(self):
        return self.model

    def get_representation(self, t, is_query):
        if self.args.enable_multiview:
            view_num = self.args.dense_multiview_args.q_view if is_query else self.args.dense_multiview_args.d_view
            t = t[:, :view_num, ...]
        t = self.linear(t)
        t = torch.nn.functional.normalize(t, p=2, dim=2)
        return t

    def query(self, input_ids, attention_mask, active_indices=None, active_padding=None, **kwargs):
        output = self.encoder(input_ids, attention_mask=attention_mask, return_dict=True, output_hidden_states=True).hidden_states[-1]
        Q = self.get_representation(output, is_query=True)
        return Q

    def doc(self, input_ids, attention_mask, active_indices=None, active_padding=None, **kwargs):
        output = self.encoder(input_ids, attention_mask=attention_mask, return_dict=True, output_hidden_states=True).hidden_states[-1]
        D = self.get_representation(output, is_query=False)
        return D

    @staticmethod
    def score(Q, D, q_mask, d_mask, *args, **kwargs):
        D = D * d_mask[..., None]
        Q = Q * q_mask[..., None]
        simmat = einsum("qmh,dnh->qdmn", Q, D)
        scores_match, indices = simmat.max(-1)
        scores = scores_match.sum(-1)
        return scores

    def save(self: BertPreTrainedModel, save_dir: str):
        save_info = self.save_info
        if save_info is None:
            save_info = {}
        to_save = self.state_dict()
        if isinstance(self, nn.DataParallel):
            to_save = self.module
        import os
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
        print('*' * 20 + "saving checkpoint to " + save_dir + '*' * 20)

        torch.save(to_save, os.path.join(save_dir, "pytorch.bin"))
        args = save_info
        json.dump(args, open(os.path.join(save_dir, "training_args.bin"), 'w', encoding='utf8'), ensure_ascii=False, indent=4)

    def load(self: BertPreTrainedModel, checkpoint: str):
        # logger.info('*' * 20 + "loading checkpoint from " + checkpoint + '*' * 20)
        print('*' * 20 + "loading checkpoint from " + checkpoint + '*' * 20)
        return self.load_state_dict(torch.load(checkpoint, map_location=lambda storage, loc: storage), strict=True)


def test_score():
    Q = torch.tensor([[[1, 5, 4], [2, 8, 1]]]).float()
    D = torch.tensor([[[0, 0, 0], [1, 1, 1]], [[3, 2, 1], [1, 1, 3]]]).float()
    q_mask, d_mask = torch.ones(Q.size()[:2]), torch.ones(D.size()[:2])
    score = BaseModel.score(Q, D, q_mask, d_mask)
    print(score)


if __name__ == '__main__':
    test_score()
