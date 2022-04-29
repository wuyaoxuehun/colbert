import json
from multiprocessing.connection import Client

import torch.distributed
from torch import nn
from transformers import BertPreTrainedModel

from colbert.modeling.BaseModel import BaseModel
# import faiss_indexers
from colbert.utils.utils import load_checkpoint
from conf import *
import logging

from colbert.modeling.model_utils import batch_index_select

torch.multiprocessing.set_sharing_strategy("file_system")

from colbert.training.training_utils import create_geo_retriever_score, qd_mask_to_realinput

logger = logging.getLogger("__main__")


# import pydevd_pycharm
# pydevd_pycharm.settrace('114.212.84.202', port=8899, stdoutToServer=True, stderrToServer=True)
def load_model(colbert_config, do_print=True):
    init = colbert_config.get('init', True)
    # colbert = ColBert.from_pretrained(pretrain,
    colbert = ColBert.from_pretrained(pretrain,
                                      query_maxlen=colbert_config['query_maxlen'],
                                      doc_maxlen=colbert_config['doc_maxlen'],
                                      dim=colbert_config['dim'],
                                      similarity_metric=colbert_config['similarity'])
    if not init:
        print_message("#> Loading model checkpoint.", condition=do_print)
        load_checkpoint(colbert_config['checkpoint'], colbert, do_print=do_print)
    colbert.eval()
    return colbert


class ModelHelper__:
    def __init__(self, index_config, rank=0):
        self.conn = None
        self.all_paras = None
        self.all_paras = None
        # index_config['index_path'] = args.index_path
        # index_config['faiss_index_path'] = args.index_path + "ivfpq.2000.faiss"
        self.helper = None

    def load_paras(self):
        self.all_paras = load_all_paras()

    def retrieve_for_encoded_queries_(self, batches, q_word_mask=None, retrieve_topk=10):
        if self.conn is None:
            address = ('localhost', 9090)
            self.conn = Client(address, authkey=b'secret password')
            print('connected to server')
            self.all_paras = load_all_paras()

        kwargs = {"expand_size": 5, "expand_center_size": 24, "expand_per_emb": 16, "expand_topk_emb": 3, "expand_weight": 0.000000000001}
        self.conn.send((batches, q_word_mask, retrieve_topk, faiss_depth, nprobe, kwargs))
        data = self.conn.recv()
        batch_pids, *extra = list(zip(*data))

        batch_paras = [[self.all_paras[pid] for pid in pids] for pids in batch_pids]
        # batch_paras = [[self.all_paras[np.random.randint(0, len(self.all_paras))] for pid in pids] for pids in batch_pids]
        # return batch_D, batch_D_mask, batch_paras
        return None, None, batch_paras, extra

    def retrieve_for_encoded_queries(self, batches, q_word_mask=None, retrieve_topk=10):
        if self.all_paras is None:
            self.all_paras = load_all_paras()
            from colbert.training.model_helper_server import ModelHelperServer

            self.helper = ModelHelperServer(index_config)

        kwargs = {"expand_size": 5, "expand_center_size": 24, "expand_per_emb": 16, "expand_topk_emb": 3, "expand_weight": 0.000000000001}
        data = self.helper.direct_retrieve((batches, q_word_mask, retrieve_topk, faiss_depth, nprobe, kwargs))
        # print(len(data))
        # batch_pids, *extra = list(zip(*data))
        batch_pids = list(zip(*data))
        batch_pids, batch_scores = batch_pids
        batch_paras = [[self.all_paras[pid] for pid in pids] for pids in batch_pids]
        # input(batch_scores)

        # batch_paras = [[self.all_paras[np.random.randint(0, len(self.all_paras))] for pid in pids] for pids in batch_pids]
        # return batch_D, batch_D_mask, batch_paras
        return batch_paras

    def close(self):
        if self.conn is not None:
            self.conn.send("close")
            print("closed")
            return


class ModelHelper:
    def __init__(self, index_config, rank=0):
        self.conn = None
        self.all_paras = None
        self.all_paras = None
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
        kwargs = {"expand_size": 5, "expand_center_size": 24, "expand_per_emb": 16, "expand_topk_emb": 3, "expand_weight": 0.000000000001, "filter_topk": 10000}
        res = batches, retrieve_topk, faiss_depth, nprobe, kwargs
        return res

    def retrieve_for_encoded_queries(self, batches, q_word_mask=None, retrieve_topk=10):
        if self.conn is None:
            address = ('localhost', 9090)
            self.conn = Client(address, authkey=b'1')
            print('connected to server')
            self.all_paras = load_all_paras()
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
    def retrieve_for_encoded_queries_(self, batches, q_word_mask=None, retrieve_topk=10):
        # if self.all_paras is None:
        #     self.all_paras = load_all_paras()
        #     from colbert.training.model_helper_server import ModelHelperServer
        #
        #     self.helper = ModelHelperServer(index_config, rank=get_rank())
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
        return batch_paras, batch_scores

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
    # def __init__(self, config, colbert_config, reader_config, load_old=True):
    def __init__(self, load_old=False):
        super().__init__()
        # self.colbert_config = colbert_config
        # self.reader_config = reader_config
        self.config = encoder_config.from_pretrained(pretrain)

        # self.model = encoder_model(self.config)
        self.model = encoder_model.from_pretrained(pretrain)
        self.linear = nn.Linear(self.config.hidden_size, dim, bias=False)
        self.q_word_weight_linear = nn.Sequential(
            nn.Linear(self.config.hidden_size, self.config.hidden_size),
            nn.ReLU(),
            nn.Linear(self.config.hidden_size, 1),
        )
        self.d_word_weight_linear = nn.Sequential(
            nn.Linear(self.config.hidden_size, self.config.hidden_size),
            nn.ReLU(),
            nn.Linear(self.config.hidden_size, 1),
            # nn.ReLU6()
        )

        # self.part_linear = nn.Sequential(
        #     nn.Linear(self.config.hidden_size, self.config.hidden_size),
        #     nn.ReLU(),
        #     nn.Linear(self.config.hidden_size, 1, bias=False),
        # )
        if load_old:
            self.old_colbert = load_model(colbert_config)
            self.doc_colbert_fixed = load_model(colbert_config)
        self.reader = nn.Linear(2, 2)
        self.ir_topk = ir_topk
        self.ir_linear = nn.Linear(20, 1)
        self.tokenizer = encoder_tokenizer.from_pretrained(pretrain)
        # self.dummy_labels = tokenizer
        for module in [self.linear, self.q_word_weight_linear, self.d_word_weight_linear, self.part_linear]:
            module.apply(self.model._init_weights)

    # def forward(self, batch, labels):

    def filter_query(self, Q, q_active_padding):
        Q = Q * q_active_padding[..., None]
        softmax_word_weight = Q.norm(p=2, dim=-1)
        t = softmax_word_weight.sort(-1, descending=True)
        sorted_ww = t.indices
        ww = t.values
        # input(ww)

        # total = q_active_padding.sum(-1)
        Q_new = batch_index_select(Q, dim=1, inds=sorted_ww[:, :20])
        padding_new = q_active_padding.gather(1, sorted_ww[:, :20])
        return Q_new, padding_new

    def forward(self, batch, train_dataset, is_evaluating=False, merge=False, doc_enc_training=True,
                eval_p_num=None, is_testing_retrieval=False, pad_p_num=None, neg_weight_mask=None, args=None):
        # global pos_num, neg_num
        q = train_dataset.tokenize_for_retriever(batch)

        # q_ids, q_attention_mask, q_active_spans, q_active_padding, *word_weight_parts = [_.cuda() for _ in q]
        q_ids, q_attention_mask, q_active_spans, q_active_padding, *_ = [_.cuda() for _ in q]
        Q = self.query(q_ids, q_attention_mask, q_active_spans, q_active_padding, with_word_weight=query_word_weight)
        # Q = self.query(q_ids, q_attention_mask, q_active_spans, q_active_padding, with_word_weight=query_word_weight)
        if is_testing_retrieval:
            # Q = Q * q_active_padding[..., None]
            # softmax_word_weight = Q.norm(p=2, dim=-1)
            # print(softmax_word_weight[0][q_active_padding[0].bool()])
            # print(len(batch))
            # print([batch[0][_] for _ in ['question_cut', 'B_cut', "background_cut"]])
            # opt_idx = 0
            # for (s, e), weight in zip(q_active_spans[opt_idx][:sum(q_active_padding[opt_idx].bool())], softmax_word_weight[opt_idx]):
            #     print(self.tokenizer.decode(q_ids[opt_idx][s:e]).replace(" ", ""), float(weight))
            # input()
            # Q, q_active_padding = self.filter_query(Q, q_active_padding)

            d_paras = model_helper.retrieve_for_encoded_queries(Q, q_word_mask=q_active_padding, retrieve_topk=eval_p_num)
            train_dataset.merge_to_reader_input(batch, d_paras)
            # d, scores = train_dataset.tokenize_for_train_retriever(batch, [], eval_p_num=eval_p_num, is_evaluating=True)
            # assert len(batch)
            return
        if merge:
            with torch.no_grad():
                Q = self.old_colbert.query(ids, q_word_mask)
            d_paras = self.retriever_forward(Q, q_word_mask=q_word_mask, labels=None)
            model_helper.merge_to_reader_input(batch, d_paras)

        d, scores = train_dataset.tokenize_for_train_retriever(batch, eval_p_num=eval_p_num, is_evaluating=is_evaluating)

        d_ids, d_attention_mask, d_active_indices, d_active_padding, *_ = [_.cuda() for _ in d]
        scores = scores.cuda()

        if not doc_enc_training:
            with torch.no_grad():
                D = self.doc_colbert_fixed.doc(d_ids, d_attention_mask)
                D = D.requires_grad_(requires_grad=True)
        else:
            D = self.doc(d_ids, d_attention_mask, d_active_indices, with_word_weight=doc_word_weight)

        Q, D = Q.to(DEVICE), D.to(DEVICE)

        Q, q_word_mask, D, d_word_mask = [_.to(DEVICE).contiguous()
                                          for _ in qd_mask_to_realinput(Q=Q, D=D, q_word_mask=q_active_padding, d_word_mask=d_active_padding)]

        # Q, q_word_mask, D, d_word_mask, scores = collection_qd_masks([Q, q_word_mask, D, d_word_mask, scores])

        # if not is_evaluating:  # 评估的时候不考虑in batch negative
        scores = create_geo_retriever_score(scores)
        pred_scores = self.score(Q, D, q_mask=q_word_mask, d_mask=d_word_mask)

        # if torch.distributed.get_rank() == 0:
        #     torch.set_printoptions(threshold=np.inf)
        #     print(pred_scores)
        # input()
        scores = scores.cuda()
        if args is not None and hasattr(args, "scoretemperature"):
            score_temperature = args.scoretemperature
        else:
            score_temperature = SCORE_TEMPERATURE
        cur_retriever_loss = retriever_criterion(y_pred=pred_scores / score_temperature, y_true=scores, neg_weight_mask=neg_weight_mask)
        return cur_retriever_loss,

    def forward_(self, batch, train_dataset, is_evaluating=False, merge=False, doc_enc_training=True,
                 eval_p_num=None, is_testing_retrieval=False, pad_p_num=None, neg_weight_mask=None):
        # global pos_num, neg_num
        q = train_dataset.tokenize_for_retriever(batch)

        q_ids, q_attention_mask, q_active_spans, q_active_padding, *word_weight_parts = [_.cuda() for _ in q]
        # q_ids, q_attention_mask, q_active_spans, q_active_padding = [_.cuda() for _ in q]
        Q = self.query(q_ids, q_attention_mask, q_active_spans, with_word_weight=query_word_weight)
        # if query_word_weight:
        #     Q, q_word_weight = Q
        #     # q_word_weight[...] = 10000
        #     # Q = Q * q_word_weight[..., None]
        #     q_word_weight[q_active_padding == 0] = -1e4
        #     softmax_q_word_weight = F.softmax(q_word_weight, dim=-1)
        #     # print(softmax_q_word_weight[1][q_active_padding[1].bool()])
        #     # print(len(batch))
        #     # print([batch[0][_] for _ in ['question_cut', 'B_cut', "background_cut"]])
        #     # input()
        #     Q = Q * softmax_q_word_weight[..., None]
        # Q, q_word_weight = self.query(q_ids, q_attention_mask, q_active_spans, with_word_weight=True)
        # Q = Q * q_word_weight[..., None]

        # Q = Q * word_weight_parts[..., None]
        if is_testing_retrieval:
            # tokens = self.tokenizer.convert_ids_to_tokens(q_ids[0])
            # weights = (Q.norm(p=2, dim=-1)[0][:q_active_padding[0].bool().sum()])
            # for (s,t), weight in zip(q_active_spans[0], weights):
            #     print(self.tokenizer.decode(q_ids[0][s:t], skip_special_tokens=False).replace(" ", ""), float(weight))
            # input()
            Q = Q * q_active_padding[..., None]
            *_, d_paras, extra = model_helper.retrieve_for_encoded_queries(Q, q_word_mask=q_active_padding, retrieve_topk=self.ir_topk)
            train_dataset.merge_to_reader_input(batch, d_paras, extra)
            # d, scores = train_dataset.tokenize_for_train_retriever(batch, [], eval_p_num=eval_p_num, is_evaluating=True)
            # assert len(batch[0]["paragraph_a"]) == 10, len(batch[0]["paragraph_a"])
            return
        if merge:
            with torch.no_grad():
                Q = self.old_colbert.query(ids, q_word_mask)
            retrieval_scores, d_paras = self.retriever_forward(Q, q_word_mask=q_word_mask, labels=None)
            model_helper.merge_to_reader_input(batch, d_paras)
        padded_negs = []

        # print(len(batch[0]['paragraph_a']))
        d, scores = train_dataset.tokenize_for_train_retriever(batch, padded_negs, eval_p_num=eval_p_num, is_evaluating=is_evaluating)
        # print(len(batch[0]['paragraph_a']))

        d_ids, d_attention_mask, d_active_indices, d_active_padding, *_ = [_.cuda() for _ in d]
        scores = scores.cuda()

        if not doc_enc_training:
            with torch.no_grad():
                D = self.doc_colbert_fixed.doc(d_ids, d_attention_mask)
                D = D.requires_grad_(requires_grad=True)
        else:
            D = self.doc(d_ids, d_attention_mask, d_active_indices, with_word_weight=doc_word_weight)

        Q, D = Q.to(DEVICE), D.to(DEVICE)

        Q, q_word_mask, D, d_word_mask = [_.to(DEVICE).contiguous()
                                          for _ in qd_mask_to_realinput(Q=Q, D=D, q_word_mask=q_active_padding, d_word_mask=d_active_padding)]

        # Q, q_word_mask, D, d_word_mask, scores = collection_qd_masks([Q, q_word_mask, D, d_word_mask, scores])
        # positive_idxes = torch.tensor([_ * p_num for _ in range(Q.size(0))])

        # if not is_evaluating:  # 评估的时候不考虑in batch negative
        scores = create_geo_retriever_score(scores)
        pred_scores = self.score(Q, D, q_mask=q_word_mask, d_mask=d_word_mask)

        # if torch.distributed.get_rank() == 0:
        #     torch.set_printoptions(threshold=np.inf)
        #     print(pred_scores / SCORE_TEMPERATURE)
        # input()
        scores = scores.cuda()
        # print(pred_scores[0] / SCORE_TEMPERATURE)
        # input()
        # input()
        # cur_retriever_loss = retriever_criterion(scores=scores / SCORE_TEMPERATURE,
        #                                          positive_idx_per_question=positive_idxes,
        #                                          hard_negative_idx_per_question=None, dual=not is_evaluating)
        cur_retriever_loss = retriever_criterion(y_pred=pred_scores / SCORE_TEMPERATURE, y_true=scores, neg_weight_mask=neg_weight_mask)

        # dual_retriever_loss = retriever_criterion(y_pred=pred_scores.T / SCORE_TEMPERATURE, y_true=scores.T)
        # cur_q_answer_retriever_loss, q_answer_kl_loss = torch.tensor(0.0).cuda(), torch.tensor(0.0).cuda()
        # return cur_retriever_loss, cur_q_answer_retriever_loss, q_answer_kl_loss
        return cur_retriever_loss,
        # return cur_retriever_loss,
        # return scores
        # return scores, D_scores

    def retriever_forward(self, Q, q_word_mask=None, labels=None):
        D, d_word_mask, d_paras = model_helper.retrieve_for_encoded_queries(Q, q_word_mask=q_word_mask, retrieve_topk=self.ir_topk)
        return None, d_paras

    def reader_forward(self, graphs, labels):
        loss, preds = self.reader((graphs, labels), DEVICE)
        return loss, preds

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
