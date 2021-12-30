import copy
import json
import logging
import os
from functools import partial
from multiprocessing.connection import Client

import torch
import yaml
from torch import nn, einsum
from torch.nn import CrossEntropyLoss
from transformers import BertPreTrainedModel, BertGenerationDecoder, T5ForConditionalGeneration, T5Config, T5Tokenizer, T5TokenizerFast, T5EncoderModel
import numpy as np
# from colbert.modeling.colbert_list import ColBERT_List
# import faiss_indexers
from transformers.modeling_outputs import BaseModelOutput

from colbert.base_config import ColBert, GENERATION_ENDING_TOK
from colbert.modeling.tokenization.query_tokenization import QueryTokenizer
# os.environ['CUDA_VISIBLE_DEVICES'] = "0"
from colbert.parameters import DEVICE
from colbert.ranking.faiss_index import FaissIndex
from colbert.ranking.index_part import *
from colbert.utils.utils import print_message, load_checkpoint
from corpus_cb import load_all_paras
from colbert.modeling.reader_models import BertForDHC
from conf import *
from conf import teacher_aug
from colbert.training.losses import kl_loss
from colbert.training.training_utils import *
# from colbert.modeling.transformer_decoder import TransformerDecoder, Generator, set_parameter_tf, TransformerDecoderState
from colbert.indexing.faiss_indexers import *
from colbert.modeling.hyper.PoincareDistance import PoincareDistance, exp_map, hyperdist
import torch.nn.functional as F
from colbert.modeling.model_utils import batch_index_select

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


class ModelHelper_:
    def __init__(self, index_config, rank=0):
        self.query_tokenizer = QueryTokenizer(index_config["query_maxlen"], None)
        self.index_config = index_config
        self.faiss_index = None
        self.retrieve = None
        self.rank = rank
        # self.depth = index_config['depth']
        self.dim = index_config['dim']
        self.index = None
        self.all_paras = load_all_paras()

    # def query_tokenize(self, queries):
    #     Q = self.query_tokenizer.tensorize_allopt_dict(queries)
    #     return Q

    def retrieve_for_encoded_queries(self, batches, q_word_mask=None, retrieve_topk=10):
        if self.retrieve is None:
            # logger.info('loading index from ' + index_config['index_path'])
            # logger.info(index_config)
            # self.faiss_index = FaissIndex(self.index_config['index_path'], self.index_config['faiss_index_path'],
            #                               self.index_config['n_probe'], part_range=self.index_config['part_range'], rank=self.rank)
            # self.retrieve = partial(self.faiss_index.retrieve, self.index_config['faiss_depth'])
            # self.index = IndexPart(self.index_config['index_path'], dim=self.dim, part_range=self.index_config['part_range'], verbose=True)
            self.retriever = ColbertRetriever(index_path=self.index_config['index_path'], rank=self.rank, index_config=index_config)
            self.retriever.load_index()
            # self.retriever = DPRRetriever(index_path=self.index_config['index_path'], dim=dim, rank=self.rank)
            # self.retriever.load_index()
            self.retrieve = self.retriever.search

        batches = torch.clone(batches).cpu().detach().to(dtype=torch.float16)
        q_word_mask = torch.clone(q_word_mask).cpu().detach()
        # q_word_mask = torch.clone(q_word_mask)
        # batches, q_word_mask_bool = batches.cpu().to(dtype=torch.float16), q_word_mask.cpu().bool().squeeze(-1)
        # batches = [q[q_word_mask_bool[idx]][:topk_token] for idx, q in enumerate(batches)]
        # only_pos_q_word_mask = [q_word_mask[idx][:topk_token][q_word_mask_bool[idx][:topk_token]] for idx, q in enumerate(batches)]
        # only_pos_q_word_mask = [q_word_mask[idx][:topk_token][q_word_mask_bool[idx][:topk_token]] for idx, q in enumerate(batches)]
        batches = [qd_mask_to_realinput(Q=q, q_word_mask=qw_mask, keep_dim=False) for q, qw_mask in zip(batches, q_word_mask)]

        batch_pids = []
        with torch.no_grad():
            if False:
                Q = np.concatenate([np.array(_[0]) for _ in batches], axis=0).astype(np.float32)
                batch_pids += self.retrieve(query=Q, topk_doc=retrieve_topk)
            else:
                for i, (q, q_word_mask) in enumerate(batches):
                    # print(q.size(), q_word_mask.size())
                    # input()
                    pids = self.retrieve(query=(q, q_word_mask), topk_doc=retrieve_topk)
                    batch_pids.append(pids)
                # print([''.join(self.all_paras[pid]['paragraph_cut']['tok'].split()) for pid in pids[:2]])
            # batch_D, batch_D_mask = self.index.ranker.get_doc_embeddings_by_pids(sum(batch_pids, []))

            # batch_D = batch_D.view(len(batches), retrieve_topk, -1, self.dim)
            # batch_D_mask = batch_D_mask.view(len(batches), retrieve_topk, -1)

        batch_paras = [[self.all_paras[pid] for pid in pids] for pids in batch_pids]
        # batch_paras = [[self.all_paras[np.random.randint(0, len(self.all_paras))] for pid in pids] for pids in batch_pids]
        # return batch_D, batch_D_mask, batch_paras
        return None, None, batch_paras


class ModelHelper:
    def __init__(self, index_config, rank=0):
        self.conn = None
        self.all_paras = None
        # conn.send('close')
        # can also send arbitrary objects:
        # conn.send(['a', 2.5, None, int, sum])
        # conn.close()

    # def query_tokenize(self, queries):
    #     Q = self.query_tokenizer.tensorize_allopt_dict(queries)
    #     return Q

    def load_paras(self):
        self.all_paras = load_all_paras()

    def retrieve_for_encoded_queries(self, batches, q_word_mask=None, retrieve_topk=10):
        if self.conn is None:
            address = ('localhost', 6001)
            self.conn = Client(address, authkey=b'secret password')
            print('connected to server')
            self.all_paras = load_all_paras()

        self.conn.send((batches, q_word_mask, retrieve_topk, faiss_depth, nprobe))
        batch_pids = self.conn.recv()
        batch_paras = [[self.all_paras[pid] for pid in pids] for pids in batch_pids]
        # batch_paras = [[self.all_paras[np.random.randint(0, len(self.all_paras))] for pid in pids] for pids in batch_pids]
        # return batch_D, batch_D_mask, batch_paras
        return None, None, batch_paras

    def close(self):
        self.conn.send("close")
        print("closed")
        return


from conf import index_config, colbert_config, p_num

# colbert = load_model(colbert_config=colbert_config)

# model_helper = ModelHelper(index_config)
model_helper: ModelHelper = None


def load_model_helper(rank=0):
    global model_helper
    if model_helper is None:
        model_helper = ModelHelper(index_config, rank=rank)
    return model_helper


class ColBERT_List_qa(nn.Module):
    def __init__(self, config, colbert_config, reader_config, load_old=True):
        super().__init__()
        self.colbert_config = colbert_config
        self.reader_config = reader_config
        # self.colbert = ColBert.from_pretrained(pretrain,
        #                                        query_maxlen=colbert_config['query_maxlen'],
        #                                        doc_maxlen=colbert_config['doc_maxlen'],
        #                                        dim=colbert_config['dim'],
        # similarity_metric = colbert_config['similarity'])

        self.model = encoder_model.from_pretrained(pretrain)
        self.config = encoder_config.from_pretrained(pretrain)
        # self.linear = nn.Linear(self.config.d_model, dim, bias=False)
        self.linear = nn.Linear(self.config.hidden_size, dim, bias=False)
        # self.decoder = BertGenerationDecoder.from_pretrained(pretrain, add_cross_attention=True, is_decoder=True, bos_token_id=101, eos_token_id=GENERATION_ENDING_TOK).cuda()
        # self.embedding_size, self.hidden_size = self.colbert.bert.embeddings.word_embeddings.weight.size()
        # self.colbert = self
        if load_old:
            self.old_colbert = load_model(colbert_config)
            self.doc_colbert_fixed = load_model(colbert_config)
        # self.reader = BertForDHC.from_pretrained(reader_config['reader_path'])
        self.reader = nn.Linear(2, 2)
        # self.config = config
        self.ir_topk = ir_topk
        self.ir_linear = nn.Linear(20, 1)
        self.tokenizer = encoder_tokenizer.from_pretrained(pretrain)
        # self.dummy_labels = tokenizer

    @property
    def encoder(self):
        return self.model.encoder if str(pretrain).find("t5") != -1 else self.model

    # def get_encoder(self):

    def get_dummy_labels(self, n):
        # return self.tokenizer(["" for _ in range(n)],
        return self.tokenizer([self.tokenizer.pad_token for _ in range(n)],
                              padding='max_length',
                              max_length=2,
                              truncation=True,
                              return_tensors="pt")['input_ids'].to(DEVICE)

    def query_(self, input_ids, attention_mask, decoder_labels=None, **kwargs):
        input_ids, attention_mask = input_ids.to(DEVICE), attention_mask.to(DEVICE)
        ar_losses = None
        Q = None
        if decoder_labels is not None:
            if type(decoder_labels) != list:
                decoder_labels = [decoder_labels]
            ar_losses = []
            for idx, decoder_label in enumerate(decoder_labels):
                decoder_labels[decoder_labels == 0] = -100
                if idx == 0:
                    output = self.model(input_ids, attention_mask=attention_mask, return_dict=True, labels=decoder_label)
                    encoder_outputs = BaseModelOutput(
                        last_hidden_state=output.encoder_last_hidden_state,
                        hidden_states=output.encoder_hidden_states,
                        attentions=output.encoder_attentions
                    )
                    Q = output.encoder_last_hidden_state
                else:
                    output = self.model(encoder_outputs=encoder_outputs, attention_mask=attention_mask, return_dict=True, labels=decoder_label)
                ar_loss = output.loss
                ar_losses.append(ar_loss)

            if print_generate:
                decoder_labels[decoder_labels == -100] = 0
                print(self.tokenizer.decode(decoder_labels[0], skip_special_tokens=True))
        else:
            Q = self.encoder(input_ids, attention_mask=attention_mask, return_dict=True).last_hidden_state
        # Q = Q.to(torch.float32)
        # Q = self.model(input_ids, attention_mask=attention_mask, return_dict=True).last_hidden_state
        # Q = self.linear_q(Q)
        Q = self.linear(Q)
        Q = torch.nn.functional.normalize(Q, p=2, dim=2)
        if decoder_labels is not None:
            return Q, ar_losses
        return Q

    def query(self, input_ids, attention_mask, active_indices=None, decoder_labels=None, **kwargs):
        input_ids, attention_mask = input_ids.to(DEVICE), attention_mask.to(DEVICE)
        ar_losses = None
        Q = None
        teacher_aug_Q = None
        encoder_outputs = None
        # print(print_generate, decoder_labels is not None)
        if decoder_labels is not None:
            if type(decoder_labels) != list:
                decoder_labels = [decoder_labels]
            ar_losses = []
            for idx, decoder_label in enumerate(decoder_labels):
                decoder_label[decoder_label == 0] = -100
                if idx == 0:
                    output = self.model(input_ids, attention_mask=attention_mask,
                                        output_hidden_states=True, return_dict=True, labels=decoder_label)
                    encoder_outputs = BaseModelOutput(
                        last_hidden_state=output.encoder_last_hidden_state,
                        hidden_states=output.encoder_hidden_states,
                        attentions=output.encoder_attentions
                    )
                    Q = output.encoder_last_hidden_state
                    if teacher_aug:
                        teacher_aug_Q = output.decoder_hidden_states[-1][:, :query_aug_topk, ...]
                        teacher_aug_Q_mask = torch.zeros((input_ids.size(0), query_aug_topk), dtype=torch.long).to(input_ids.device)
                        aug_labels = decoder_label[:, :query_aug_topk]
                        teacher_aug_Q_mask[(aug_labels != 0) & (aug_labels != -100)] = 1
                        # Q = torch.cat([Q, teacher_aug_Q], dim=1)
                        # q_word_mask = torch.cat([q_word_mask, teacher_aug_Q_mask], dim=1)
                else:
                    output = self.model(encoder_outputs=encoder_outputs, attention_mask=attention_mask, return_dict=True, labels=decoder_label)
                ar_loss = output.loss
                ar_losses.append(ar_loss)

        else:
            Q = self.encoder(input_ids, attention_mask=attention_mask, return_dict=True).last_hidden_state
        # Q = Q.to(torch.float32)
        # Q = self.model(input_ids, attention_mask=attention_mask, return_dict=True).last_hidden_state
        # Q = self.linear_q(Q)
        Q = batch_index_select(Q, dim=1, inds=active_indices)
        Q = self.linear(Q)
        Q = torch.nn.functional.normalize(Q, p=2, dim=2)
        if decoder_labels is not None:
            if teacher_aug:
                teacher_aug_Q = torch.nn.functional.normalize(self.linear(teacher_aug_Q), p=2, dim=2)
                return Q, ar_losses, teacher_aug_Q, teacher_aug_Q_mask
            return Q, ar_losses
        return Q

    def doc(self, input_ids, attention_mask, active_indices=None, **kwargs):
        # input_ids, attention_mask = input_ids.to(DEVICE), attention_mask.to(DEVICE)
        # D = self.t5(input_ids, attention_mask=attention_mask, return_dict=True, labels=self.get_dummy_labels(n=input_ids.size(0))).encoder_last_hidden_state
        # print("inputIds", input_ids[-3, ...], attention_mask[-3, ...])
        D = self.encoder(input_ids, attention_mask=attention_mask, return_dict=True).last_hidden_state
        # D = D.to(torch.float32)
        D = batch_index_select(D, dim=1, inds=active_indices)
        # t = attention_mask[0]
        # print(t.sum(-1).max(0)[0], active_indices[0])
        D = self.linear(D)
        # print("123", D[-3, 0, ...])
        D = torch.nn.functional.normalize(D, p=2, dim=2)
        return D

    @staticmethod
    def score(Q, D, q_mask=None, d_mask=None):
        # print(Q.size(), D.size(), q_mask.size(), d_mask.size())
        # input()
        # print(D[-3, 0, ...])
        if d_mask is not None and q_mask is not None:
            D = D * d_mask[..., None]
            Q = Q * q_mask[..., None]

        scores = einsum("qmh,dnh->qdmn", Q, D).max(-1)[0].sum(-1)
        scores = scores / (q_mask.bool().sum(-1)[:, None])
        # scores = F.relu(einsum("qmh,dnh->qdmn", Q, D)).max(-1)[0].sum(-1)
        # if not torch.isfinite(scores[0]):
        # print(scores[0][-4:], scores[1][-4:])
        # input()
        return scores

    @staticmethod
    def score___(Q, D, q_mask=None, d_mask=None):
        # if d_mask is not None and q_mask is not None:
        #     D = D * d_mask[..., None]
        #     Q = Q * q_mask[..., None]
        # 保存encode之前的向量，保留在欧式空间，从而可以使用ANN
        qbs, qn, dim = Q.size()
        dbs, dn = D.size()[:2]
        Q_ = Q.view(-1, dim).repeat_interleave(dbs * dn, dim=0)
        D_ = D.view(-1, dim).repeat((qbs * qn, 1))
        QQ, DD = exp_map(Q_), exp_map(D_)
        dist = PoincareDistance.apply(QQ, DD, 1e-5)
        # dist = torch.cosh(dist) ** 2
        # dist = - (dist ** 2) / 2
        # dist = - dist
        # print(dist)
        d_mask[d_mask != 1] = 1e4
        dist = dist.view(qbs, qn, dbs, dn)
        # mask = q_mask[..., None, None] * d_mask[None, None, ...]
        # mask[mask.bool()]
        dist = (dist * d_mask[None, None, ...]).min(-1)[0]
        dist = (dist * q_mask[..., None]).sum(1)
        scores = - dist
        scores = scores / (q_mask.bool().sum(-1)[:, None])
        return scores

    def score__(self, Q, D, q_mask=None, d_mask=None):
        # if d_mask is not None and q_mask is not None:
        #     D = D * d_mask[..., None]
        #     Q = Q * q_mask[..., None]
        qbs, qn, dim = Q.size()
        dbs, dn = D.size()[:2]
        dist = torch.zeros(size=(qbs, dbs, qn, dn)).to(Q.device)
        for i in range(qbs):
            for j in range(dbs):
                ti = Q[i, ...].repeat_interleave(dn, dim=0)
                tj = D[j, ...].repeat((qn, 1))
                ti, tj = exp_map(ti), exp_map(tj)
                tdist = PoincareDistance.apply(ti, tj, 1e-5)
                # tdist = hyperdist(ti, tj)
                dist[i, j, :, :] = tdist.view(qn, dn).contiguous()
        # input(dist)
        dist = dist.permute(0, 2, 1, 3)
        # dist = torch.cosh(dist) ** 2
        # dist = - (dist ** 2) / 2
        # dist = - dist
        # print(dist)
        d_mask1 = d_mask.clone()
        d_mask1[d_mask != 1] = 1e4
        # mask = q_mask[..., None, None] * d_mask[None, None, ...]
        # mask[mask.bool()]
        dist = (dist * d_mask1[None, None, ...]).min(-1)[0]
        dist = (dist * q_mask[..., None]).sum(1)
        scores = - dist
        # input(scores)
        scores = scores / (q_mask.bool().sum(-1)[:, None])
        return scores

    def generate_aug_Q(self, input_ids):
        with torch.no_grad():
            # D = self.model.generate(input_ids=input_ids, output_hidden_states=True, return_dict_in_generate=True, min_length=10, num_beams=5, num_return_sequences=3)
            # outputs = self.model.generate(input_ids=input_ids, do_sample=False, output_hidden_states=True, return_dict_in_generate=True, min_length=query_aug_topk + 1)
            outputs = self.model.generate(input_ids=input_ids, do_sample=False, output_hidden_states=True, return_dict_in_generate=True)  # , no_repeat_ngram_size=4
            Q = outputs.decoder_hidden_states

            if print_generate:
                print(self.tokenizer.decode(input_ids[0], skip_special_tokens=True))
                print(len(self.tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)))
                print((self.tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)))
                input()
            Q = torch.cat([_[-1] for _ in Q[0:query_aug_topk]], dim=1)
            Q = self.linear(Q)
            Q = torch.nn.functional.normalize(Q, p=2, dim=2)
            aug_Q = Q
            if Q.size(1) < query_aug_topk:
                aug_Q = torch.zeros((Q.size(0), query_aug_topk, Q.size(2)), dtype=Q.dtype).to(Q.device)
                aug_Q[:, :Q.size(1), ...] = Q
            aug_mask = torch.zeros((Q.size(0), Q.size(1) - 1), dtype=torch.long).to(input_ids.device)
            aug_sequences = outputs.sequences[:, 1:Q.size(1)]  # shift left (for position i output contain information of i+1 token)
            aug_mask[:, :Q.size(1) - 1][(aug_sequences != 0) & (aug_sequences != 1)] = 1

            return aug_Q, aug_mask, aug_sequences

    def augment_query(self, input_ids, attention_mask, q_word_mask, decoder_labels=None):
        if decoder_labels is None:
            decoder_labels = self.get_dummy_labels(n=input_ids.size(0))

        Q, ar_loss = self.query(input_ids=input_ids, attention_mask=attention_mask, decoder_labels=decoder_labels)
        aug_Q, aug_mask, _ = self.generate_aug_Q(input_ids)

        if not print_generate:
            Q = torch.cat([Q, aug_Q], dim=1)

        q_word_mask = torch.cat([q_word_mask, aug_mask], dim=1)
        return Q, q_word_mask, ar_loss

    def aug_query_and_reencode(self, input_ids, attention_mask, q_word_mask):
        """
        使用generate 扩充query，得到aug id， 然后结合原来的query重新输入encoder
        :return:
        """
        aug_Q, aug_mask, aug_sequences = self.generate_aug_Q(input_ids)
        input_ids, attention_mask = torch.cat([input_ids, aug_sequences], dim=1), torch.cat([attention_mask, aug_mask], dim=1)
        q_word_mask = torch.cat([q_word_mask, aug_mask], dim=1)
        Q = self.query(input_ids=input_ids, attention_mask=attention_mask)
        # print(self.tokenizer.decode(input_ids[0], skip_special_tokens=True))
        # print((self.tokenizer.decode(aug_sequences[0], skip_special_tokens=True)))
        # print(attention_mask[0])
        # input()
        return Q, q_word_mask

    # def reconstruct_forward(self, input_ids, hidden_states):
    #     # encoder_hidden_states = Q[:, 0:1, ...]
    #     # labels = ids.clone()
    #     # labels[labels == 0] = ignore_index
    #     # model_output = self.decoder(input_ids=ids, encoder_hidden_states=encoder_hidden_states, labels=labels)
    #     # query_reconstruction_loss = model_output.loss
    #     reconstruction_criterion = nn.NLLLoss(ignore_index=0, reduction='mean')
    #     decode_context = self.decoder(input_ids[:, :-1], hidden_states, TransformerDecoderState(input_ids))
    #     reconstruction_text = self.generator(decode_context.view(-1, decode_context.size(2)))
    #     answer_reconstruction_loss = reconstruction_criterion(reconstruction_text, input_ids[:, 1:].reshape(-1))
    #     return answer_reconstruction_loss

    # def forward(self, batch, labels):
    def forward(self, batch, train_dataset, is_evaluating=False, merge=False, doc_enc_training=False, eval_p_num=None, is_testing_retrieval=False, pad_p_num=None):
        obj = train_dataset.tokenize_for_retriever(batch)
        q_ids, q_attention_mask, q_active_indices, q_active_padding = [_.to(DEVICE) for _ in obj[0]]
        # input_ids, attention_mask, active_indices, active_padding = [_.to(DEVICE) for _ in obj[1]]
        # input_ids, attention_mask, active_indices, active_padding = [_.to(DEVICE) for _ in obj[2]]
        # Q = model.colbert.query(ids, mask)
        # print(batch[0]['question'], '\n', batch[0]['A'], '\n', d_paras[0][:2])
        # input()
        Q = self.query(q_ids, q_attention_mask, q_active_indices)
        # Q, ar_losses = self.query(input_ids=ids, attention_mask=mask, decoder_labels=[answer_ids, title_ids])
        # Q, ar_losses = self.query(input_ids=ids, attention_mask=mask, decoder_labels=title_ids)
        # q_answer_ids, q_answer_mask, q_answer_word_mask = \
        #     torch.cat([ids, answer_ids, title_ids], dim=1), \
        #     torch.cat([mask, answer_mask, title_mask], dim=1), \
        #     torch.cat([q_word_mask, answer_word_mask, title_word_mask], dim=1)
        # Q_Answer = self.query(q_answer_ids, q_answer_mask)

        # Q, ar_losses, teacher_aug_Q, teacher_aug_Q_mask = self.query(input_ids=ids, attention_mask=mask, decoder_labels=answer_ids)

        # ar_loss = torch.tensor(0.0).requires_grad_(True).to(ids.device)
        # Q, q_word_mask, ar_loss = self.augment_query(ids, mask, q_word_mask, answer_ids)
        # Q, q_word_mask, ar_loss = self.augment_query(ids, mask, q_word_mask)

        global teacher_aug

        # Q, q_word_mask = torch.cat([Q, teacher_aug_Q], dim=1), torch.cat([q_word_mask, teacher_aug_Q_mask], dim=1)

        if is_testing_retrieval:
            # Q = self.colbert.query(ids, q_word_mask)
            if print_generate:
                print(self.tokenizer.decode(answer_ids[0], skip_special_tokens=True))
            # teacher_aug = False
            # Q, q_word_mask, ar_loss = self.augment_query(ids, mask, q_word_mask, answer_ids)
            # Q, q_word_mask = self.aug_query_and_reencode(ids, mask, q_word_mask)
            if print_generate:
                return
            # Q, q_word_mask = torch.cat([Q, teacher_aug_Q], dim=1), torch.cat([q_word_mask, teacher_aug_Q_mask], dim=1)
            # retrieval_scores, d_paras = self.retriever_forward(Q, q_word_mask=q_word_mask, labels=None)
            retrieval_scores, d_paras = self.retriever_forward(Q, q_word_mask=q_active_padding, labels=None)
            train_dataset.merge_to_reader_input(batch, d_paras)
            return
        # return Q.contiguous(), q_word_mask.contiguous(), None, None, ar_loss
        if merge:
            with torch.no_grad():
                Q = self.old_colbert.query(ids, q_word_mask)
            retrieval_scores, d_paras = self.retriever_forward(Q, q_word_mask=q_word_mask, labels=None)
            model_helper.merge_to_reader_input(batch, d_paras)

        # assert CrossEntropyLoss().ignore_index == -100
        # ignore_index = CrossEntropyLoss().ignore_index
        # Q = self.query(ids, mask)
        # if False:
        #     answer_reconstruction_loss = self.reconstruct_forward(answer_ids, Q[:, 1, ...])
        #     # reconstruction_criterion(reconstruction_text, answer_ids[:, 1:].reshape(-1))
        # else:
        #     answer_reconstruction_loss = torch.tensor(0).to(DEVICE)
        #
        # if False:
        #     query_reconstruction_loss = self.reconstruct_forward(ids, Q[:, 0, ...])
        # else:
        #     query_reconstruction_loss = torch.tensor(0).to(DEVICE)

        # padded_negs = [model_helper.all_paras[_] for _ in np.random.randint(1, len(model_helper.all_paras), padded_p_num)] if not is_evaluating else []
        # padded_negs = [model_helper.all_paras[_] for _ in np.random.randint(1, len(model_helper.all_paras), pad_p_num if pad_p_num is not None else padded_p_num)]
        padded_negs = []

        d_ids, d_attention_mask, d_active_indices, d_active_padding = [_.to(DEVICE) for _ in
                                                                       train_dataset.tokenize_for_train_retriever(batch, padded_negs, eval_p_num=eval_p_num, is_evaluating=is_evaluating)]

        # D = model.colbert.doc(D_ids, D_mask)
        if not doc_enc_training:
            with torch.no_grad():
                D = self.doc_colbert_fixed.doc(d_ids, d_attention_mask)
                D = D.requires_grad_(requires_grad=True)
        else:
            D = self.doc(d_ids, d_attention_mask, d_active_indices)

        Q, D = Q.to(DEVICE), D.to(DEVICE)
        # if False:
        #     doc_reconstruction_loss = self.reconstruct_forward(D_ids, D[:, 0, ...])
        # else:
        #     doc_reconstruction_loss = torch.tensor(0).to(DEVICE)

        Q, q_word_mask, D, d_word_mask = [_.to(DEVICE).contiguous()
                                          for _ in qd_mask_to_realinput(Q=Q, D=D, q_word_mask=q_active_padding, d_word_mask=d_active_padding)]
        # scores = self.colbert.score(Q1, D, q_mask=q_word_mask, d_mask=d_word_mask)
        # scores = (scores / q_word_mask.bool().sum(1)[:, None])

        # input(scores)
        # return scores, D_scores
        # return scores, D_scores, answer_reconstruction_loss, query_reconstruction_loss, doc_reconstruction_loss
        # return scores, D_scores, scores_ans, ans_sim_loss
        # return scores, D_scores, answer_reconstruction_loss, query_reconstruction_loss
        # return scores, D_scores, answer_reconstruction_loss
        # return scores, answer_reconstruction_loss
        # return scores, query_reconstruction_loss, doc_reconstruction_loss

        # ar_losses = sum(ar_losses)

        # return Q1.contiguous(), q_word_mask.contiguous(), D.contiguous(), d_word_mask.contiguous(), ar_losses, teacher_aug_Q, teacher_aug_Q_mask
        Q, q_word_mask, D, d_word_mask = collection_qd_masks([Q, q_word_mask, D, d_word_mask])
        positive_idxes = torch.tensor([_ * p_num for _ in range(Q.size(0))])

        scores = self.score(Q, D, q_mask=q_word_mask, d_mask=d_word_mask)

        cur_retriever_loss = retriever_criterion(scores=scores / SCORE_TEMPERATURE,
                                                 positive_idx_per_question=positive_idxes,
                                                 hard_negative_idx_per_question=None, dual=not is_evaluating)
        # if teacher_aug:
        #     teacher_aug_Q, teacher_aug_Q_mask = collection_qd_masks([teacher_aug_Q, teacher_aug_Q_mask])
        #     teacher_aug_scores = self.score(teacher_aug_Q, D, q_mask=teacher_aug_Q_mask, d_mask=d_word_mask)
        #     cur_teacher_aug_retriever_loss = retriever_criterion(scores=teacher_aug_scores / SCORE_TEMPERATURE,
        #                                                          positive_idx_per_question=positive_idxes,
        #                                                          hard_negative_idx_per_question=None)
        cur_q_answer_retriever_loss, q_answer_kl_loss = torch.tensor(0.0).cuda(), torch.tensor(0.0).cuda()
        # if kl_answer_re_loss:
        #     Q_Answer, q_answer_word_mask = Q_Answer.contiguous(), q_answer_word_mask.contiguous()
        #     Q_Answer, q_answer_word_mask = collection_qd_masks([Q_Answer, q_answer_word_mask])
        #
        #     q_answer_scores = self.score(Q_Answer, D, q_mask=q_answer_word_mask, d_mask=d_word_mask)
        #     cur_q_answer_retriever_loss = retriever_criterion(scores=q_answer_scores / SCORE_TEMPERATURE,
        #                                                       positive_idx_per_question=positive_idxes,
        #                                                       hard_negative_idx_per_question=None)
        #     q_answer_kl_loss = kl_loss(y_pred=scores / kl_temperature, y_true=q_answer_scores / kl_temperature)

        return cur_retriever_loss, cur_q_answer_retriever_loss, q_answer_kl_loss
        # return scores
        # return scores, D_scores

    def retriever_forward(self, Q, q_word_mask=None, labels=None):
        # with torch.no_grad():
        # Q = self.colbert.query(*Q)
        # Q_ww = self.ww_linear(Q).squeeze(-1)
        # Q = (Q.permute(2, 0, 1) * Q_ww).permute(1, 2, 0)

        D, d_word_mask, d_paras = model_helper.retrieve_for_encoded_queries(Q, q_word_mask=q_word_mask, retrieve_topk=self.ir_topk)
        # scores = self.query_wise_score(Q, D, q_mask=q_word_mask, d_mask=d_word_mask)
        # scores = self.ir_linear(scores)
        #
        # scores = scores.view(-1, 4)
        # loss_fct = CrossEntropyLoss()
        # if labels is not None:
        #     labels = labels.to(DEVICE)
        #     ir_loss = loss_fct(scores, labels)
        #     return scores, ir_loss, d_paras
        # return scores, d_paras
        return None, d_paras

    def query_wise_score(self, Q, D, q_mask=None, d_mask=None):
        D = D.to(DEVICE)
        if d_mask is not None:
            d_mask = d_mask.to(DEVICE)
            D = D * d_mask[..., None]
        scores = einsum('abh,aoch->aboc', Q, D).max(-1).values
        if q_mask is not None:
            q_mask = q_mask.to(DEVICE)
            scores = (scores * q_mask[..., None]).sum(1)

        return scores  # Q_bs * topk

    def query_wise_score_(self, Q, D, q_mask=None, d_mask=None):
        D = D.to(DEVICE)
        print(Q.size(), D.size())
        input()
        if d_mask is not None:
            d_word_mask = d_mask.to(DEVICE)
            D = (D.permute(3, 0, 1, 2) * d_word_mask).permute(1, 2, 3, 0)
        Q = Q.unsqueeze(1)
        scores = (Q @ (D.permute(0, 1, 3, 2))).max(-1).values
        if q_mask is not None:
            q_word_mask = q_mask.to(DEVICE)
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
        logger.info('*' * 20 + "saving checkpoint to " + save_dir + '*' * 20)

        torch.save(to_save, os.path.join(save_dir, "pytorch.bin"))
        args = {
            'colbert_config': self.colbert_config,
            'reader_config': self.reader_config
        }
        json.dump(args, open(os.path.join(save_dir, "training_args.bin"), 'w', encoding='utf8'), ensure_ascii=False, indent=4)

    def load(self: BertPreTrainedModel, checkpoint: str):
        # logger.info('*' * 20 + "loading checkpoint from " + checkpoint + '*' * 20)
        print('*' * 20 + "loading checkpoint from " + checkpoint + '*' * 20)
        return self.load_state_dict(torch.load(checkpoint, map_location=lambda storage, loc: storage), strict=True)


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
