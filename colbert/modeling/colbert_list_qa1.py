import json
import os
from functools import partial

import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import BertPreTrainedModel

# from colbert.modeling.colbert_list import ColBERT_List
from colbert.base_config import ColBert, pretrain
from colbert.modeling.model_xh import ModelXH
from colbert.modeling.tokenization.query_tokenization import QueryTokenizer
# os.environ['CUDA_VISIBLE_DEVICES'] = "0"
from colbert.parameters import DEVICE
from colbert.ranking.faiss_index import FaissIndex
from colbert.ranking.index_part import IndexPart
from colbert.utils.utils import print_message, load_checkpoint
from corpus_cb import load_all_paras


# import pydevd_pycharm
# pydevd_pycharm.settrace('114.212.84.202', port=8899, stdoutToServer=True, stderrToServer=True)
def load_model(colbert_config, do_print=True):
    colbert = ColBert.from_pretrained(pretrain,
                                      query_maxlen=colbert_config['query_maxlen'],
                                      doc_maxlen=colbert_config['doc_maxlen'],
                                      dim=colbert_config['dim'],
                                      similarity_metric=colbert_config['similarity'])
    print_message("#> Loading model checkpoint.", condition=do_print)
    load_checkpoint(colbert_config['checkpoint'], colbert, do_print=do_print)
    colbert.eval()
    return colbert


class ModelHelper:
    def __init__(self, index_config):
        self.query_tokenizer = QueryTokenizer(index_config["query_maxlen"], None)

        self.faiss_index = FaissIndex(index_config['index_path'], index_config['faiss_index_path'], index_config['n_probe'], part_range=index_config['part_range'])
        self.retrieve = partial(self.faiss_index.retrieve, index_config['faiss_depth'])
        self.depth = index_config['depth']
        self.dim = index_config['dim']
        self.index = IndexPart(index_config['index_path'], dim=self.dim, part_range=index_config['part_range'], verbose=True)
        self.all_paras = load_all_paras()

    def query_tokenize(self, queries):
        Q = self.query_tokenizer.tensorize_allopt_dict(queries)
        return Q

    def merge_to_reader_input(self, batch_examples, batch_paras):
        para_idx = 0
        for example in batch_examples:
            for opt in list('abcd'):
                example['paragraph_' + opt] = [{"p_id": _['p_id'], "paragraph": ''.join(_['paragraph_cut']['tok'].split())} for _ in batch_paras[para_idx]]
                para_idx += 1

    @torch.no_grad()
    def retrieve_for_encoded_queries(self, batches, q_word_mask, retrieve_topk=10):
        batches = torch.clone(batches)
        q_word_mask = torch.clone(q_word_mask)
        batches, q_word_mask_bool = batches.cpu().to(dtype=torch.float16), q_word_mask.cpu().bool().squeeze(-1)
        batches = [q[q_word_mask_bool[idx]] for idx, q in enumerate(batches)]

        batch_pids = []
        for i, q in enumerate(batches):
            only_pos_q_word_mask = q_word_mask[i][q_word_mask_bool[i]]
            weighted_q = q * (only_pos_q_word_mask.unsqueeze(-1))
            Q = weighted_q.unsqueeze(0)
            pids = self.retrieve(Q, verbose=False)[0]
            Q = Q.permute(0, 2, 1)
            pids = self.index.ranker.rank_forward(Q, pids, depth=retrieve_topk)
            batch_pids.append(pids)
            # print([''.join(self.all_paras[pid]['paragraph_cut']['tok'].split()) for pid in pids[:2]])

        batch_D, batch_D_mask = self.index.ranker.get_doc_embeddings_by_pids(sum(batch_pids, []))
        batch_D = batch_D.view(len(batches), retrieve_topk, -1, self.dim)
        batch_D_mask = batch_D_mask.view(len(batches), retrieve_topk, -1)

        # print(batch_D.size(), batch_D_mask.size())
        # print()
        batch_paras = [[self.all_paras[pid] for pid in pids] for pids in batch_pids]
        return batch_D, batch_D_mask, batch_paras


from conf import index_config, colbert_config
from colbert.modeling.utils_xh import batch_transform_bert_fever

# colbert = load_model(colbert_config=colbert_config)

# model_helper = ModelHelper(index_config)
model_helper: ModelHelper = None


def load_model_helper():
    global model_helper
    if model_helper is None:
        model_helper = ModelHelper(index_config)
    return model_helper


class ColBERT_List_qa(nn.Module):
    def __init__(self, config, colbert_config, reader_config):
        super().__init__()
        self.colbert_config = colbert_config
        self.reader_config = reader_config
        # self.colbert = ColBert.from_pretrained(pretrain,
        #                                        query_maxlen=colbert_config['query_maxlen'],
        #                                        doc_maxlen=colbert_config['doc_maxlen'],
        #                                        dim=colbert_config['dim'],
        # similarity_metric = colbert_config['similarity'])

        self.colbert = load_model(colbert_config)
        self.reader = ModelXH.from_pretrained(reader_config['reader_path'],
                                              p_num=reader_config['p_num'],
                                              p_topk=reader_config['p_topk'],
                                              gnn_layer=reader_config['gnn_layer'],
                                              n_head=reader_config['n_head'])
        self.config = config
        self.ir_topk = config['ir_topk']
        self.ww_linear = nn.Sequential(
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        # self.ir_linear = nn.Sequential(
        #     nn.Linear(config['ir_topk'], 1),
        #     nn.Tanh(),
        #     nn.Linear(128, 1),
        # )
        self.ir_linear = nn.Linear(config['ir_topk'], 1)

    # def forward(self, batch, labels):
    def forward(self, batch, labels):
        q_ids, q_mask, q_word_mask = model_helper.query_tokenize(batch)
        ir_preds, ir_loss, d_paras = self.retriever_forward((q_ids, q_mask), q_word_mask, labels=labels)
        # model_helper.merge_to_reader_input(batch, d_paras)
        # graphs, labels = batch_transform_bert_fever(batch, self.reader_config['max_seq_length'], self.reader_config['p_num'], device=DEVICE)
        # reader_loss, reader_preds = self.reader_forward(graphs, labels)
        # return ir_loss, reader_loss, ir_preds, reader_preds
        return ir_loss, ir_preds

    def retriever_forward(self, Q, q_word_mask=None, labels=None):
        # with torch.no_grad():
        Q = self.colbert.query(*Q)
        # Q_ww = self.ww_linear(Q).squeeze(-1)
        # Q = (Q.permute(2, 0, 1) * Q_ww).permute(1, 2, 0)

        D, d_word_mask, d_paras = model_helper.retrieve_for_encoded_queries(Q, q_word_mask=q_word_mask, retrieve_topk=self.ir_topk)
        scores = self.query_wise_score(Q, D, q_word_mask=q_word_mask, d_word_mask=d_word_mask)
        scores = self.ir_linear(scores)

        scores = scores.view(-1, 4)
        loss_fct = CrossEntropyLoss()
        if labels is not None:
            labels = labels.to(DEVICE)
            ir_loss = loss_fct(scores, labels)
            return scores, ir_loss, d_paras
        return scores, d_paras

    def query_wise_score(self, Q, D, q_word_mask=None, d_word_mask=None):
        D = D.to(DEVICE)
        if d_word_mask is not None:
            d_word_mask = d_word_mask.to(DEVICE)
            D = (D.permute(3, 0, 1, 2) * d_word_mask).permute(1, 2, 3, 0)
        Q = Q.unsqueeze(1)
        scores = (Q @ (D.permute(0, 1, 3, 2))).max(-1).values
        if q_word_mask is not None:
            q_word_mask = q_word_mask.to(DEVICE)
            scores = scores.permute(1, 0, 2)
            scores = (scores * q_word_mask).sum(-1).T

        return scores  # Q_bs * topk

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
        print('*' * 20 + "saving checkpoint to " + save_dir + '*' * 20)

        torch.save(to_save, os.path.join(save_dir, "pytorch.bin"))
        args = {
            'colbert_config': self.colbert_config,
            'reader_config': self.reader_config
        }
        json.dump(args, open(os.path.join(save_dir, "training_args.bin"), 'w', encoding='utf8'), ensure_ascii=False, indent=4)

    def load(self: BertPreTrainedModel, checkpoint: str):
        print('*' * 20 + "loading checkpoint from " + checkpoint + '*' * 20)
        return self.load_state_dict(torch.load(checkpoint, map_location=lambda storage, loc: storage))


if __name__ == '__main__':
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
