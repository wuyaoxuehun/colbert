import torch

from colbert.modeling.colbert_list import ColBERT_List
from colbert.modeling.tokenization import QueryTokenizer, DocTokenizer
from colbert.utils.amp import MixedPrecisionManager
from colbert.parameters import DEVICE
from tqdm import tqdm


class ModelInference():
    def __init__(self, colbert, segmenter, amp=False):
        # assert colbert.training is False

        self.colbert = colbert
        # if torch.cuda.device_count() > 1:
        #     self.colbert = torch.nn.DataParallel(colbert)
        self.query_tokenizer = QueryTokenizer(colbert.query_maxlen, segmenter)
        self.doc_tokenizer = DocTokenizer(colbert.doc_maxlen, segmenter)

        self.amp_manager = MixedPrecisionManager(amp)

    def query(self, *args, to_cpu=False, **kw_args):
        with torch.no_grad():
            with self.amp_manager.context():
                Q = self.colbert.query(*args, **kw_args)
                # return Q.cpu() if to_cpu else Q
                return ([_.cpu() for _ in Q] if type(Q) in [list, tuple] else Q.cpu()) \
                    if to_cpu else Q

    def doc(self, *args, to_cpu=False, **kw_args):
        with torch.no_grad():
            with self.amp_manager.context():
                D = self.colbert.doc(*args, **kw_args)
                # return D.cpu() if to_cpu else D
                return ([_.cpu() for _ in D] if type(D) in [list, tuple] else D.cpu()) \
                    if to_cpu else D

    def queryFromText(self, queries, bsize=None, to_cpu=False):
        if bsize:
            batches = self.query_tokenizer.tensorize(queries, bsize=bsize)
            batches = [self.query(input_ids, attention_mask, to_cpu=to_cpu) for input_ids, attention_mask in batches]
            return torch.cat(batches)

        input_ids, attention_mask = self.query_tokenizer.tensorize(queries)
        return self.query(input_ids, attention_mask)

    def docFromText(self, docs, bsize=None, keep_dims=True, to_cpu=False):
        if bsize:
            D = []
            for offset in tqdm(range(0, len(docs), bsize)):
                input_ids, attention_mask, d_word_mask = self.doc_tokenizer.tensorize(docs[offset:offset + bsize])
                batches = self.doc(input_ids, attention_mask, to_cpu=to_cpu)
                if not keep_dims:
                    batches, d_word_mask = batches.cpu().to(dtype=torch.float16), d_word_mask.cpu().bool().squeeze(-1)
                    batches = [d[d_word_mask[idx]] for idx, d in enumerate(batches)]
                D.extend(batches)
            return D

        input_ids, attention_mask = self.doc_tokenizer.tensorize(docs)
        return self.doc(input_ids, attention_mask, keep_dims=keep_dims)

    def docFromTensorize(self, tensorizes, bsize=None, keep_dims=True, to_cpu=False, output_word_weight=False):
        if bsize:
            D = []
            D_word_weight_all = []
            iterator = range(0, len(tensorizes[0]), bsize)
            if len(tensorizes[0]) > 16:
                iterator = tqdm(iterator)

            for offset in iterator:
                input_ids, attention_mask, d_word_mask = torch.tensor(tensorizes[0][offset:offset + bsize]), \
                                                         torch.tensor(tensorizes[1][offset:offset + bsize]), \
                                                         torch.tensor(tensorizes[2][offset:offset + bsize])

                batches = self.doc(input_ids, attention_mask, to_cpu=to_cpu, output_word_weight=output_word_weight)
                D_word_weight = None
                if output_word_weight:
                    batches, D_word_weight = batches

                if not keep_dims:
                    batches, d_word_mask_bool = batches.cpu().to(dtype=torch.float16), d_word_mask.cpu().bool().squeeze(-1)
                    batches = [d[d_word_mask_bool[idx]] for idx, d in enumerate(batches)]

                    if output_word_weight:
                        D_word_weight = [dww[d_word_mask_bool[idx]] for idx, dww in enumerate(D_word_weight)]
                D.extend(batches)
                if output_word_weight:
                    D_word_weight_all.extend(D_word_weight)

            if output_word_weight:
                return D, D_word_weight_all
            return D

    def queryFromTensorize(self, tensorizes, bsize=None, keep_dims=False, to_cpu=False, output_word_weight=False):
        input_ids, attention_mask, q_word_mask = torch.tensor(tensorizes[0]), torch.tensor(tensorizes[1]), torch.tensor(tensorizes[2])
        # if bsize:
        #     batches = [self.query(input_ids, attention_mask, to_cpu=to_cpu) for input_ids, attention_mask in zip(input_ids, attention_mask)]
        #     return torch.cat(batches)

        batches = self.query(input_ids, attention_mask, output_word_weight=output_word_weight)
        Q_word_weight = None
        if output_word_weight:
            batches, Q_word_weight = batches
        if not keep_dims:
            batches, q_word_mask_bool = batches.cpu().to(dtype=torch.float16), q_word_mask.cpu().bool().squeeze(-1)
            batches = [q[q_word_mask_bool[idx]] for idx, q in enumerate(batches)]
            if output_word_weight:
                Q_word_weight = [qww[q_word_mask_bool[idx]] for idx, qww in enumerate(Q_word_weight)]
            output_batches = []
            for i, q in enumerate(batches):
                only_pos_q_word_mask = q_word_mask[i][q_word_mask_bool[i]]
                weighted_batch = q * (only_pos_q_word_mask.unsqueeze(-1))
                output_batches.append(weighted_batch)

            batches = output_batches
            # batches

        if output_word_weight:
            return batches, Q_word_weight
        return batches

    def score(self, Q, D, mask=None, lengths=None, explain=False):
        if lengths is not None:
            assert mask is None, "don't supply both mask and lengths"

            mask = torch.arange(D.size(1), device=DEVICE) + 1
            mask = mask.unsqueeze(0) <= lengths.to(DEVICE).unsqueeze(-1)

        scores = (D @ Q)
        scores = scores if mask is None else scores * mask.unsqueeze(-1)
        scores = scores.max(1)

        if explain:
            assert False, "TODO"

        return scores.values.sum(-1).cpu()

    def score_ori(self, Q, D, mask=None, lengths=None, explain=False):
        if lengths is not None:
            assert mask is None, "don't supply both mask and lengths"

            mask = torch.arange(D.size(1), device=DEVICE) + 1
            mask = mask.unsqueeze(0) <= lengths.to(DEVICE).unsqueeze(-1)

        scores = (D @ Q)
        return scores


def _stack_3D_tensors(groups):
    bsize = sum([x.size(0) for x in groups])
    maxlen = max([x.size(1) for x in groups])
    hdim = groups[0].size(2)

    output = torch.zeros(bsize, maxlen, hdim, device=groups[0].device, dtype=groups[0].dtype)

    offset = 0
    for x in groups:
        endpos = offset + x.size(0)
        output[offset:endpos, :x.size(1)] = x
        offset = endpos

    return output