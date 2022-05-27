import json
import time
from multiprocessing.connection import Client

import torch.distributed
from torch import nn
from torch.distributed import get_rank
from transformers import BertPreTrainedModel

from colbert.training.CBQADataset_gen_medqa import CBQADataset
from colbert.modeling.BaseModel import BaseModel
# import faiss_indexers
from colbert.utils.utils import load_checkpoint
from conf import *
import logging
import torch.nn.functional as F
from colbert.training.losses import kl_loss, BiEncoderNllLossMS
from proj_utils.dureader_utils import get_dureader_ori_corpus

torch.multiprocessing.set_sharing_strategy("file_system")
# torch.multiprocessing.set_start_method('spawn', force=True)
from colbert.training.training_utils import create_geo_retriever_score, qd_mask_to_realinput, collection_qd_masks
from colbert.modeling.reader_models import BertForMC

logger = logging.getLogger("__main__")


# import pydevd_pycharm
# pydevd_pycharm.settrace('114.212.84.202', port=8899, stdoutToServer=True, stderrToServer=True)

class ModelHelper:
    def __init__(self, index_config, rank=0):
        self.conn = None
        self.all_paras = None
        # self.all_paras = None
        # index_config['index_path'] = args.index_path
        # index_config['faiss_index_path'] = args.index_path + "ivfpq.2000.faiss"
        self.helper = None
        # if self.all_paras is None:
        #     self.all_paras = load_all_paras()
        #     from colbert.training.model_helper_server import ModelHelperServer
        #     self.helper = ModelHelperServer(index_config, rank=get_rank())

    def load_paras(self):
        self.all_paras = load_all_paras()

    def to_real_batch(self, batches, q_word_mask, retrieve_topk):
        batches = batches.cpu().detach().to(dtype=torch.float16)
        q_word_mask = q_word_mask.cpu().detach()
        batches = [qd_mask_to_realinput(Q=q, q_word_mask=qw_mask, keep_dim=False) for q, qw_mask in
                   zip(batches, q_word_mask)]
        kwargs = {"expand_size": 5, "expand_center_size": 24, "expand_per_emb": 16, "expand_topk_emb": 3, "expand_weight": 0.000000000001}
        res = batches, retrieve_topk, faiss_depth, nprobe, kwargs
        return res

    def retrieve_for_encoded_queries_(self, batches, q_word_mask=None, retrieve_topk=10):
        if self.conn is None:
            address = ('localhost', 9090)
            self.conn = Client(address, authkey=b'1')
            print('connected to server')
            # self.all_paras = load_all_paras()
            self.all_paras = get_dureader_ori_corpus()
        retrieve_input = self.to_real_batch(batches, q_word_mask, retrieve_topk)
        self.conn.send(retrieve_input)
        data = self.conn.recv()
        # batch_pids, *extra = list(zip(*data))
        #
        # batch_paras = [[self.all_paras[pid] for pid in pids] for pids in batch_pids]
        # # batch_paras = [[self.all_paras[np.random.randint(0, len(self.all_paras))] for pid in pids] for pids in batch_pids]
        # # return batch_D, batch_D_mask, batch_paras
        # return None, None, batch_paras, extra
        batch_pids = list(zip(*data))
        batch_pids, batch_scores = batch_pids
        batch_paras = [[self.all_paras[pid] for pid in pids] for pids in batch_pids]
        return batch_paras, batch_scores

    @torch.no_grad()
    def retrieve_for_encoded_queries(self, batches, q_word_mask=None, retrieve_topk=10, model=None):
        if self.all_paras is None:
            # self.all_paras = list(load_all_paras(to_dict=True).values())
            # self.all_paras = list(load_all_paras(to_dict=True))
            self.all_paras = get_dureader_ori_corpus()
            # self.all_paras = self.all_paras[1:] + self.all_paras[:1]
            from colbert.training.model_helper_server import ModelHelperServer

            # self.helper = ModelHelperServer(index_config, rank=get_rank())
            self.helper = ModelHelperServer(index_config, rank=0, model=model)
        retrieve_topk = 200
        retrieve_input = self.to_real_batch(batches, q_word_mask, retrieve_topk)
        data = self.helper.direct_retrieve(retrieve_input)
        # print(len(data))
        # batch_pids, *extra = list(zip(*data))
        batch_pids = list(zip(*data))
        batch_pids, batch_scores = batch_pids
        batch_paras = [[self.all_paras[pid] for pid in pids] for pids in batch_pids]
        # input(batch_scores)

        # batch_paras = [[self.all_paras[np.random.randint(0, len(self.all_paras))] for pid in pids] for pids in batch_pids]
        # return batch_D, batch_D_mask, batch_paras
        return batch_paras, batch_scores, batch_pids

    def close(self):
        if self.conn is not None:
            self.conn.send("close")
            print("closed")
            return


model_helper: ModelHelper = None


def load_model_helper(rank=0):
    global model_helper
    if model_helper is None:
        model_helper = ModelHelper(index_config, rank=rank)
    return model_helper


class ColBERT_List_qa(BaseModel):
    def __init__(self, load_old=False):
        super().__init__()
        self.config = encoder_config.from_pretrained(pretrain)

        # self.model = encoder_model(self.config)
        self.model = encoder_model.from_pretrained(pretrain)

        # self.linear = nn.Linear(self.config.hidden_size, dim, bias=False)
        # self.q_word_weight_linear = nn.Sequential(
        #     nn.Linear(self.config.hidden_size, self.config.hidden_size),
        #     nn.ReLU(),
        #     nn.Linear(self.config.hidden_size, 1, bias=False),
        #     # nn.Linear(self.config.hidden_size, 1)
        # )
        # self.d_word_weight_linear = nn.Sequential(
        #     nn.Linear(self.config.hidden_size, self.config.hidden_size),
        #     nn.ReLU(),
        #     nn.Linear(self.config.hidden_size, 1),
        #     # nn.ReLU6()
        # )

        # if load_old:
        #     self.old_colbert = load_model(colbert_config)
        #     self.doc_colbert_fixed = load_model(colbert_config)
        # self.reader = BertForMC.from_pretrained(reader_pretrained)
        self.tokenizer = encoder_tokenizer.from_pretrained(pretrain)
        # self.dummy_labels = tokenizer
        # for module in [self.linear, self.q_word_weight_linear, self.esim_linear_42, self.esim_linear_80]:
        # for module in [self.linear, self.q_word_weight_linear,self.comatch]:
        #     module.apply(self.model._init_weights)
        # self.load_retriever()
        # self.load(checkpoint=retriever_reader_trained)

    def get_mrr_(self, scores, positive_idx_per_question):
        sorted_idx = scores.sort(dim=-1, descending=True)[1]
        positive_idxes = torch.zeros_like(scores)
        positive_idxes.scatter_(dim=-1, index=positive_idx_per_question[:, None].to(scores.device), value=1)
        res = positive_idxes.gather(1, sorted_idx)
        return res.argmax(-1).sum()

    def get_mrr(self, scores):
        pos_num, neg_num = 2, 8
        labels = torch.tensor([list(range(i * (pos_num + neg_num), i * (pos_num + neg_num) + pos_num)) for i in range(scores.size(0))])
        sorted_idx = scores.sort(dim=-1, descending=True)[1]
        positive_idxes = torch.zeros_like(scores)
        positive_idxes.scatter_(dim=-1, index=labels.to(scores.device), value=1)
        res = positive_idxes.gather(1, sorted_idx)
        return res.nonzero()[:, 1].float().mean()

    # def forward(self, batch, labels):
    def forward(self, batch, train_dataset, is_evaluating=False, merge=False, doc_enc_training=True,
                eval_p_num=None, is_testing_retrieval=False, pad_p_num=None, neg_weight_mask=None, args=None):

        # global pos_num, neg_num
        q = train_dataset.tokenize_for_retriever(batch)

        # q_ids, q_attention_mask, q_active_spans, q_active_padding, *word_weight_parts = [_.cuda() for _ in q]
        # q_ids, q_attention_mask, q_active_spans, q_active_padding, *_ = [_.cuda() for _ in q]
        q_ids, q_attention_mask, *_ = [_.cuda() for _ in q]
        # Q, Q_output = self.query(q_ids, q_attention_mask, q_active_spans, q_active_padding, with_word_weight=query_word_weight, dpr=True, output_ori=True)
        t1 = time.time()
        Q = self.query(q_ids, q_attention_mask)
        # Q = self.query(q_ids, q_attention_mask, q_active_spans, q_active_padding, with_word_weight=query_word_weight)
        if is_testing_retrieval:
            # Q = Q * q_active_padding[..., None]
            # softmax_word_weight = Q.norm(p=2, dim=-1)
            # torch.set_printoptions(precision=4)
            # print(softmax_word_weight[0][q_active_padding[0].bool()])
            # opt_idx = 0
            # sum_all = 0.
            # d, scores = train_dataset.tokenize_for_train_retriever(batch, eval_p_num=eval_p_num, is_evaluating=is_evaluating)
            # d_ids, d_attention_mask, d_active_indices, d_active_padding, *_ = [_.cuda() for _ in d]
            # D = self.doc(d_ids, d_attention_mask, d_active_indices, with_word_weight=doc_word_weight, dpr=False, output_ori=False)
            # pred_scores, pred_match_scores = self.score(Q, D, q_mask=q_active_padding, d_mask=d_active_padding, output_match_weight=True)
            # for (s, e), weight, match_weight in \
            #         zip(q_active_spans[opt_idx][:sum(q_active_padding[opt_idx].bool())],
            #             softmax_word_weight[opt_idx],
            #             pred_match_scores[0][0]
            #             ):
            #     print(self.tokenizer.decode(q_ids[opt_idx][s:e]).replace(" ", ""), '\t',
            #           round(float(weight), 3), '\t',
            #           round(float(match_weight), 3))
            #     sum_all += weight
            # input(sum_all)

            # Q, q_active_padding = self.filter_query(Q, q_active_padding)
            # Q = F.normalize(Q, p=2, dim=-1)
            t2 = time.time()
            # d_paras = model_helper.retrieve_for_encoded_queries(Q, q_word_mask=q_active_padding, retrieve_topk=eval_p_num, model=self)
            d_paras = model_helper.retrieve_for_encoded_queries(Q, q_word_mask=torch.ones(Q.size(0), QView), retrieve_topk=eval_p_num, model=self)
            train_dataset.merge_to_reader_input(batch, d_paras)
            t3 = time.time()
            # print(t2-t1, t3-t2)
            # input()
            # d, scores = train_dataset.tokenize_for_train_retriever(batch, [], eval_p_num=eval_p_num, is_evaluating=True)
            # assert len(batch)
            return
        # if merge:
        #     with torch.no_grad():
        #         Q = self.old_colbert.query(ids, q_word_mask)
        #     d_paras = self.retriever_forward(Q, q_word_mask=q_word_mask, labels=None)
        #     model_helper.merge_to_reader_input(batch, d_paras)

        d, scores = train_dataset.tokenize_for_train_retriever(batch, eval_p_num=eval_p_num, is_evaluating=is_evaluating)

        # d_ids, d_attention_mask, d_active_indices, d_active_padding, *_ = [_.cuda() for _ in d]
        d_ids, d_attention_mask, *_ = [_.cuda() for _ in d]
        # D = self.doc(d_ids, d_attention_mask, d_active_indices, with_word_weight=doc_word_weight, dpr=False, output_ori=False)
        D = self.doc(d_ids, d_attention_mask)

        # Q, D = Q.to(DEVICE), D.to(DEVICE)
        # Q, q_word_mask, D, d_word_mask = [_.to(DEVICE).contiguous()
        #                                   for _ in qd_mask_to_realinput(Q=Q, D=D, q_word_mask=q_active_padding, d_word_mask=d_active_padding)]
        #
        # # Q, q_word_mask, D, d_word_mask, scores = collection_qd_masks([Q, q_word_mask, D, d_word_mask, scores])
        # Q, q_word_mask, D, d_word_mask = collection_qd_masks([Q, q_word_mask, D, d_word_mask])

        # Q, Q_ori, D, D_ori = Q.to(DEVICE), Q_ori.to(DEVICE), D.to(DEVICE), D_ori.to(DEVICE)
        # Q, D = Q.to(DEVICE), D.to(DEVICE)
        # Q, q_word_mask, D, d_word_mask = [_.to(DEVICE).contiguous()
        #                                   for _ in qd_mask_to_realinput(Q=Q, D=D, q_word_mask=q_active_padding, d_word_mask=d_active_padding)]
        # q_word_mask, d_word_mask = q_attention_mask, d_attention_mask
        # Q_ori, _, D_ori, _ = [_.to(DEVICE).contiguous()
        #                       for _ in qd_mask_to_realinput(Q=Q_ori, D=D_ori, q_word_mask=q_active_padding, d_word_mask=d_active_padding)]

        # Q, q_word_mask, D, d_word_mask, scores = collection_qd_masks([Q, q_word_mask, D, d_word_mask, scores])
        # Q, q_word_mask, D, d_word_mask = collection_qd_masks([Q, q_word_mask, D, d_word_mask])
        Q, D = collection_qd_masks([Q, D])
        # Q, Q_ori, q_word_mask, D, D_ori, d_word_mask = collection_qd_masks([Q, Q_ori, q_word_mask, D, D_ori, d_word_mask])

        if not is_evaluating:
            positive_idx_per_question = torch.tensor([_ * 2 for _ in range(Q.size(0))])
        else:
            positive_idx_per_question = torch.tensor([_ * 6 for _ in range(Q.size(0))])
        pred_scores = self.score(Q, D, lce=False)
        # pred_esim_scores = self.score_esim(Q, D, q_mask=q_word_mask, d_mask=d_word_mask)
        # pred_esim_scores = self.score_comatch(Q, D, q_mask=q_word_mask, d_mask=d_word_mask)
        # pred_esim_scores = self.score_comatch(Q_ori, D_ori, q_mask=q_word_mask, d_mask=d_word_mask)
        # pred_esim_scores = self.score_comatch(Q_ori, D_ori, q_mask=q_word_mask, d_mask=d_word_mask)
        # scores = scores.cuda()
        # all_scores = pred_esim_scores, pred_scores
        if is_evaluating:
            # pred_scores = self.score(Q, D, q_mask=q_word_mask, d_mask=d_word_mask)
            # return self.get_mrr(all_scores[0], positive_idx_per_question), self.get_mrr(all_scores[1], positive_idx_per_question),
            # return self.get_mrr(pred_scores, positive_idx_per_question),
            return self.get_mrr(pred_scores),

        if args is not None and hasattr(args, "scoretemperature"):
            score_temperature = args.scoretemperature
        else:
            score_temperature = SCORE_TEMPERATURE
        # if torch.distributed.get_rank() == 0:
        #     torch.set_printoptions(threshold=np.inf)
        #     print(pred_scores)
        #     print(pred_scores / score_temperature)
        #     print(torch.nn.functional.softmax(pred_scores / score_temperature, dim=-1))
        # input()
        # cur_retriever_loss = retriever_criterion(y_pred=pred_scores / score_temperature, y_true=scores, neg_weight_mask=neg_weight_mask)
        score_temperature = 2e-2
        # if is_evaluating:
        #     score_temperature = 1
        # score_temperature = 1
        cur_retriever_loss = retriever_criterion(scores=pred_scores / score_temperature, positive_idx_per_question=positive_idx_per_question)
        # cur_retriever_esim_loss = retriever_criterion(scores=pred_esim_scores / 2e-2, positive_idx_per_question=positive_idx_per_question)
        # cur_loss = BiEncoderNllLossMS(Q, D, q_word_mask, d_word_mask, positive_idx_per_question)

        # kd_loss = torch.tensor(0.).to(cur_retriever_esim_loss.device)
        # if args.epoch <
        # kd_loss = kl_loss(y_pred=pred_scores / score_temperature * 5, y_true=pred_esim_scores / score_temperature * 5)
        # return cur_retriever_loss, 0.01 * lce_loss,
        return cur_retriever_loss,
        # return cur_retriever_loss * 0, cur_retriever_esim_loss * 1
        # return cur_loss,
        # return cur_retriever_loss * 1 / 3, cur_retriever_esim_loss * 1 / 3, kd_loss * 1 / 3
        # return cur_retriever_esim_loss, kd_loss

    def save(self: BertPreTrainedModel, save_dir: str):
        to_save = self.state_dict()
        if isinstance(self, nn.DataParallel):
            to_save = self.module
        import os
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
        logger.info('*' * 20 + "saving checkpoint to " + save_dir + '*' * 20)

        torch.save(to_save, os.path.join(save_dir, "pytorch.bin"))
        # args = {
        #     'colbert_config': self.colbert_config,
        #     'reader_config': self.reader_config
        # }
        # json.dump(args, open(os.path.join(save_dir, "training_args.bin"), 'w', encoding='utf8'), ensure_ascii=False, indent=4)

    def load(self: BertPreTrainedModel, checkpoint: str):
        # logger.info('*' * 20 + "loading checkpoint from " + checkpoint + '*' * 20)
        print('*' * 20 + "loading checkpoint from " + checkpoint + '*' * 20)
        model = torch.load(checkpoint, map_location=lambda storage, loc: storage)
        load = {k: v for k, v in model.items() if (k in self.state_dict() and 'reader.para_choice_linear' not in k)}
        return self.load_state_dict(load, strict=False)

    def load_retriever(self):
        checkpoint = load_trained
        print('*' * 20 + "loading checkpoint from " + checkpoint + '*' * 20)
        return self.load_state_dict(torch.load(checkpoint, map_location=lambda storage, loc: storage), strict=False)


def test_one():
    base_dir = os.getcwd()
    file = base_dir + "/data/bm25cb/test_0_bm25_irsort.json"
    data = json.load(open(file, encoding='utf8'))
    from conf import reader_config

    colbert_qa = ColBERT_List_qa(colbert_config=colbert_config, reader_config=reader_config)
    colbert_qa.to(DEVICE)
    data = data[8:12]
    for i in data:
        del i['positive_ctxs']
        del i['hard_negative_ctxs']
    q_ids, q_mask, q_word_mask = model_helper.query_tokenize(data)

    # print(data)
    colbert_qa.retrieval_forward((q_ids, q_mask), q_word_mask)


if __name__ == '__main__':
    pass
