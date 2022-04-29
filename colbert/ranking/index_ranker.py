# from colbert.parameters import DEVICE
# from conf import *
from itertools import accumulate

from colbert.modeling.colbert_list_qa_gen import *
from colbert.utils.utils import print_message

BSIZE = 1 << 14


class IndexRanker:
    def __init__(self, tensor, doclens, model=None):
        self.tensor = tensor
        self.doclens = doclens
        self.model = model
        self.maxsim_dtype = torch.float32
        self.doclens_pfxsum = [0] + list(accumulate(self.doclens))

        self.doclens = torch.tensor(self.doclens)
        self.doclens_pfxsum = torch.tensor(self.doclens_pfxsum)

        self.dim = self.tensor.size(-1)

        self.strides = [torch_percentile(self.doclens, p) for p in [25, 50, 75]]
        # self.strides = [torch_percentile(self.doclens, p) for p in [100]]
        # self.strides = [torch_percentile(self.doclens, p) for p in range(10, 100, 20)]
        self.strides.append(self.doclens.max().item())
        self.strides = sorted(list(set(self.strides)))

        print_message(f"#> Using strides {self.strides}..")

        self.views = self._create_views(self.tensor)
        # print([self.views[i].size() for i in range(len(self.views))])
        # input()
        self.buffers = self._create_buffers(BSIZE, self.tensor.dtype, {'cpu', 'cuda'})

    def _create_views(self, tensor):
        views = []

        for stride in self.strides:
            outdim = tensor.size(0) - stride + 1
            view = torch.as_strided(tensor, (outdim, stride, self.dim), (self.dim, self.dim, 1))
            views.append(view)

        return views

    def _create_buffers(self, max_bsize, dtype, devices):
        buffers = {}

        for device in devices:
            buffers[device] = [torch.zeros(max_bsize, stride, self.dim, dtype=dtype,
                                           device=device, pin_memory=(device == 'cpu'))
                               for stride in self.strides]

        return buffers

    # def rank_for_one(self, Q, pids):

    def rank(self, Q, pids, views=None, shift=0):
        assert len(pids) > 0
        assert Q.size(0) in [1, len(pids)]
        Q = Q.contiguous().to(DEVICE).to(dtype=self.maxsim_dtype)
        views = self.views if views is None else views
        VIEWS_DEVICE = views[0].device
        D_buffers = self.buffers[str(VIEWS_DEVICE)]
        raw_pids = pids if type(pids) is list else pids.tolist()
        pids = torch.tensor(pids) if type(pids) is list else pids

        doclens, offsets = self.doclens[pids], self.doclens_pfxsum[pids]

        assignments = (doclens.unsqueeze(1) > torch.tensor(self.strides).unsqueeze(0) + 1e-6).sum(-1)

        one_to_n = torch.arange(len(raw_pids))
        output_pids, output_scores, output_permutation = [], [], []

        for group_idx, stride in enumerate(self.strides):
            locator = (assignments == group_idx)

            if locator.sum() < 1e-5:
                continue

            group_pids, group_doclens, group_offsets = pids[locator], doclens[locator], offsets[locator]
            group_Q = Q if Q.size(0) == 1 else Q[locator]

            group_offsets = group_offsets.to(VIEWS_DEVICE) - shift
            group_offsets_uniq, group_offsets_expand = torch.unique_consecutive(group_offsets, return_inverse=True)

            D_size = group_offsets_uniq.size(0)
            D = torch.index_select(views[group_idx], 0, group_offsets_uniq, out=D_buffers[group_idx][:D_size])  # 从每个pid第一个embedding开始的位置截取stride长度个数向量
            D = D.to(DEVICE)
            D = D[group_offsets_expand.to(DEVICE)].to(dtype=self.maxsim_dtype)
            mask = torch.arange(stride, device=DEVICE) + 1
            mask = mask.unsqueeze(0) <= group_doclens.to(DEVICE).unsqueeze(-1)  # 对超过对应pid长度的部分进行mask

            scores = (D @ group_Q) * mask.unsqueeze(-1)
            scores = scores.max(1).values.sum(-1).cpu()
            # scores = scores.sum(-1).cpu()

            output_pids.append(group_pids)
            output_scores.append(scores)
            output_permutation.append(one_to_n[locator])

            # print(scores.size())
            # print(D.size())
            # print(group_Q.size())
            # print(one_to_n, locator)
            # input()

        output_permutation = torch.cat(output_permutation).sort().indices
        output_pids = torch.cat(output_pids)[output_permutation].tolist()
        output_scores = torch.cat(output_scores)[output_permutation].tolist()

        assert len(raw_pids) == len(output_pids)
        assert len(raw_pids) == len(output_scores)
        assert raw_pids == output_pids

        return output_scores

    def get_doc_embeddings_by_pids(self, pids):
        doclens, offsets = self.doclens[pids], self.doclens_pfxsum[pids]
        output_tensors = torch.zeros(size=(len(pids), max(doclens), self.dim))
        output_mask = torch.zeros(size=(len(pids), max(doclens)))
        for idx, (doclen, offset) in enumerate(zip(doclens, offsets)):
            output_tensors[idx, :doclen, :] = self.tensor[offset:offset + doclen]
            output_mask[idx, :doclen] = 1

        return output_tensors, output_mask

    def rank_forward(self, Q, pids, views=None, shift=0, depth=10, output_D_embedding=False):
        assert len(pids) > 0
        assert Q.size(0) in [1, len(pids)]
        Q = Q.contiguous().to(DEVICE).to(dtype=self.maxsim_dtype)

        views = self.views if views is None else views
        VIEWS_DEVICE = views[0].device

        D_buffers = self.buffers[str(VIEWS_DEVICE)]

        raw_pids = pids if type(pids) is list else pids.tolist()
        pids = torch.tensor(pids) if type(pids) is list else pids

        doclens, offsets = self.doclens[pids], self.doclens_pfxsum[pids]

        assignments = (doclens.unsqueeze(1) > torch.tensor(self.strides).unsqueeze(0) + 1e-6).sum(-1)

        output_pids, output_scores, output_permutation = [], [], []
        one_to_n = torch.arange(len(raw_pids))
        output_D = []
        output_D_mask = []
        for group_idx, stride in enumerate(self.strides):
            locator = (assignments == group_idx)

            if locator.sum() < 1e-5:
                continue

            group_pids, group_doclens, group_offsets = pids[locator], doclens[locator], offsets[locator]
            group_Q = Q if Q.size(0) == 1 else Q[locator]

            # group_offsets = group_offsets.to(VIEWS_DEVICE) - shift
            # group_offsets_uniq, group_offsets_expand = torch.unique_consecutive(group_offsets, return_inverse=True)
            # D_size = group_offsets_uniq.size(0)
            # assert D_size == group_offsets.size(0)
            # D = torch.index_select(views[group_idx], 0, group_offsets_uniq, out=D_buffers[group_idx][:D_size])

            D = torch.index_select(views[group_idx], 0, group_offsets, out=D_buffers[group_idx][:group_offsets.size(0)])
            D = D.to(DEVICE)
            D = D.to(dtype=self.maxsim_dtype)
            # D = D[group_offsets_expand.to(DEVICE)].to(dtype=self.maxsim_dtype)
            mask = torch.arange(stride, device=DEVICE) + 1
            mask = mask.unsqueeze(0) <= group_doclens.to(DEVICE).unsqueeze(-1)

            # scores = (D @ group_Q) * mask.unsqueeze(-1)
            # scores = scores.max(1).values.sum(-1).cpu()
            # print(scores.size())
            # print(group_Q.size(), D.size(), torch.ones(group_Q.size()[:2]).size(), mask.size())
            # input()
            # scores = ColBERT_List_qa.score(Q=group_Q.permute(0, 2, 1), D=D, q_mask=torch.ones((1, group_Q.size(2))).cuda(), d_mask=mask)[0].cpu()
            scores = self.model.score(Q=group_Q.permute(0, 2, 1), D=D,
                                      q_mask=torch.ones((1, group_Q.size(2)), dtype=torch.long).cuda(), d_mask=mask.to(torch.long))[0].cpu()
            # print(scores.size())
            # input()
            output_pids.append(group_pids)
            output_scores.append(scores)
            output_permutation.append(one_to_n[locator])
            output_D.append(D)
            output_D_mask.append(mask)

        # output_pids = torch.cat(output_pids).tolist()
        # output_scores = torch.cat(output_scores).tolist()
        # output_embeddings =
        output_permutation = torch.cat(output_permutation).sort().indices
        output_pids = torch.cat(output_pids)[output_permutation].tolist()
        output_scores = torch.cat(output_scores)[output_permutation]

        assert len(raw_pids) == len(output_pids)
        assert len(raw_pids) == len(output_scores)
        # assert raw_pids == output_pids
        assert raw_pids == output_pids

        # assert raw_pids == output_pids
        scores_sorter = output_scores.sort(descending=True)  # output_scores...一定要注意cv以后变量名的改变呀！！！
        pids = pids[scores_sorter.indices].tolist()[:depth]
        scores = output_scores[scores_sorter.indices].tolist()[:depth]
        # doc_tensors, doc_masks = self.get_doc_embeddings_by_pids(pids)
        # return pids, doc_tensors, doc_masks
        if output_D_embedding:
            output_D = torch.cat(output_D)[output_permutation]
            output_D_mask = torch.cat(output_D_mask)[output_permutation]
            output_D = output_D[scores_sorter.indices][:depth]
            output_D_mask = output_D_mask[scores_sorter.indices][:depth]
            return pids, output_D, output_D_mask
        return pids, scores

    def batch_rank(self, all_query_embeddings, all_query_indexes, all_pids, sorted_pids):
        assert sorted_pids is True

        ######

        scores = []
        range_start, range_end = 0, 0

        for pid_offset in range(0, len(self.doclens), 50_000):
            pid_endpos = min(pid_offset + 50_000, len(self.doclens))

            range_start = range_start + (all_pids[range_start:] < pid_offset).sum()
            range_end = range_end + (all_pids[range_end:] < pid_endpos).sum()

            pids = all_pids[range_start:range_end]
            query_indexes = all_query_indexes[range_start:range_end]

            print_message(f"###--> Got {len(pids)} query--passage pairs in this sub-range {(pid_offset, pid_endpos)}.")

            if len(pids) == 0:
                continue

            print_message(f"###--> Ranking in batches the pairs #{range_start} through #{range_end} in this sub-range.")

            tensor_offset = self.doclens_pfxsum[pid_offset].item()
            tensor_endpos = self.doclens_pfxsum[pid_endpos].item() + 512

            collection = self.tensor[tensor_offset:tensor_endpos].to(DEVICE)
            views = self._create_views(collection)

            print_message(f"#> Ranking in batches of {BSIZE} query--passage pairs...")

            for batch_idx, offset in enumerate(range(0, len(pids), BSIZE)):
                if batch_idx % 100 == 0:
                    print_message("#> Processing batch #{}..".format(batch_idx))

                endpos = offset + BSIZE
                batch_query_index, batch_pids = query_indexes[offset:endpos], pids[offset:endpos]

                Q = all_query_embeddings[batch_query_index]

                scores.extend(self.rank(Q, batch_pids, views, shift=tensor_offset))

        return scores


def torch_percentile(tensor, p):
    assert p in range(1, 100 + 1)
    assert tensor.dim() == 1

    return tensor.kthvalue(int(p * tensor.size(0) / 100.0)).values.item()
