from itertools import accumulate
from multiprocessing import Pool

import faiss
import torch

from colbert.indexing.index_manager import load_index_part
from colbert.indexing.loaders import get_parts, load_doclens
from colbert.utils.utils import print_message, flatten

BSIZE = 1 << 14
DEVICE = "cuda"


class ColbertRanker:
    def __init__(self, index_path, model=None, dim=None):
        # Load parts metadata
        all_parts, all_parts_paths, _ = get_parts(index_path)
        self.parts_paths = all_parts_paths
        self.maxsim_dtype = torch.float32
        # Load doclens metadata
        all_doclens = load_doclens(index_path, flatten=False)
        self.parts_doclens = all_doclens
        self.doclens = flatten(self.parts_doclens)
        self.num_embeddings = sum(self.doclens)
        self.dim = dim
        self.tensor: torch.Tensor = self._load_parts(dim)
        self.model = model
        self.init_ranker()

    def init_ranker(self):
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
        self.buffers = self._create_buffers(BSIZE, self.tensor.dtype, {'cpu', 'cuda'})

    def _create_views(self, tensor: torch.Tensor):
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

    def _load_parts(self, dim, verbose=True):
        tensor = torch.zeros(self.num_embeddings + 512, dim, dtype=torch.float16)
        if verbose:
            print_message("tensor.size() = ", tensor.size())
        offset = 0
        for idx, filename in enumerate(self.parts_paths):
            print_message("|> Loading", filename, "...", condition=verbose)

            endpos = offset + sum(self.parts_doclens[idx])
            part = load_index_part(filename, verbose=verbose)
            tensor[offset:endpos] = part
            offset = endpos
        return tensor

    def rank_forward(self, Q, pids, views: torch.Tensor = None, depth=10, output_D_embedding=False):
        assert len(pids) > 0
        assert Q.size(0) in [1, len(pids)]
        Q = Q.contiguous().to(DEVICE).to(dtype=self.maxsim_dtype)

        views = self.views if views is None else views
        VIEWS_DEVICE = views[0].device

        D_buffers = self.buffers[str(VIEWS_DEVICE)]

        raw_pids = pids if type(pids) is list else pids.tolist()
        pids = torch.tensor(pids) if type(pids) is list else pids

        doclens, offsets = self.doclens[pids], self.doclens_pfxsum[pids]
        # input((doclens.size(), self.strides))
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

            D = torch.index_select(views[group_idx], 0, group_offsets, out=D_buffers[group_idx][:group_offsets.size(0)])
            D = D.to(DEVICE)
            D = D.to(dtype=self.maxsim_dtype)
            mask = torch.arange(stride, device=DEVICE) + 1
            mask = mask.unsqueeze(0) <= group_doclens.to(DEVICE).unsqueeze(-1)

            scores = self.model.score(Q=group_Q.permute(0, 2, 1), D=D,
                                      q_mask=torch.ones((1, group_Q.size(2)), dtype=torch.long).cuda(), d_mask=mask.to(torch.long))[0].cpu()

            output_pids.append(group_pids)
            output_scores.append(scores)
            output_permutation.append(one_to_n[locator])
            output_D.append(D)
            output_D_mask.append(mask)

        output_permutation = torch.cat(output_permutation).sort().indices
        output_pids = torch.cat(output_pids)[output_permutation].tolist()
        output_scores = torch.cat(output_scores)[output_permutation]

        assert len(raw_pids) == len(output_pids)
        assert len(raw_pids) == len(output_scores)
        assert raw_pids == output_pids

        scores_sorter = output_scores.sort(descending=True)
        pids = pids[scores_sorter.indices].tolist()[:depth]
        scores = output_scores[scores_sorter.indices].tolist()[:depth]
        if output_D_embedding:
            output_D = torch.cat(output_D)[output_permutation]
            output_D_mask = torch.cat(output_D_mask)[output_permutation]
            output_D = output_D[scores_sorter.indices][:depth]
            output_D_mask = output_D_mask[scores_sorter.indices][:depth]
            return pids, output_D, output_D_mask
        return pids, scores


class ColbertIndex:
    def __init__(self, index_path, faiss_index_path, nprobe, rank=None):
        self.index_path = index_path
        self.faiss_index_path = faiss_index_path
        print_message("#> Loading the FAISS index from", faiss_index_path, "..")
        self.faiss_index = faiss.read_index(faiss_index_path)
        res = faiss.StandardGpuResources()
        # self.faiss_index = faiss.index_cpu_to_gpu(res, rank, self.faiss_index)
        # self.faiss_index = faiss.index_cpu_to_all_gpus(self.faiss_index)
        self.faiss_index.nprobe = nprobe
        self.emb2pid = None
        self.build_emb2pid()
        # self.faiss_index.cudaMallocWarning_ = False
        self.parallel_pool = Pool(16)

    def build_emb2pid(self):
        print_message("#> Building the emb2pid mapping..")
        all_doclens = load_doclens(self.index_path, flatten=False)
        pid_offset = 0
        all_doclens = flatten(all_doclens)
        total_num_embeddings = sum(all_doclens)
        self.emb2pid = torch.zeros(total_num_embeddings, dtype=torch.int)
        offset_doclens = 0
        for pid, dlength in enumerate(all_doclens):
            self.emb2pid[offset_doclens: offset_doclens + dlength] = pid_offset + pid
            offset_doclens += dlength
        print_message("len(self.emb2pid) =", len(self.emb2pid))

    def retrieve(self, faiss_depth, Q, verbose=False, output_embedding_ids=False):
        embedding_ids, all_scores = self.queries_to_embedding_ids(faiss_depth, Q, verbose=verbose)
        pids = self.embedding_ids_to_pids(embedding_ids.view(Q.size(0), Q.size(1) * embedding_ids.size(1)), verbose=verbose)
        if output_embedding_ids:
            return pids, embedding_ids, all_scores
        return pids

    def queries_to_embedding_ids(self, faiss_depth, Q, verbose=True):
        # Flatten into a matrix for the faiss search.
        num_queries, embeddings_per_query, dim = Q.size()
        Q_faiss = Q.view(num_queries * embeddings_per_query, dim).cpu().contiguous()

        # Search in large batches with faiss.
        print_message("#> Search in batches with faiss. \t\t",
                      f"Q.size() = {Q.size()}, Q_faiss.size() = {Q_faiss.size()}",
                      condition=verbose)

        embeddings_ids = []
        all_scores = []
        faiss_bsize = embeddings_per_query * 5000
        for offset in range(0, Q_faiss.size(0), faiss_bsize):
            endpos = min(offset + faiss_bsize, Q_faiss.size(0))

            print_message("#> Searching from {} to {}...".format(offset, endpos), condition=verbose)

            some_Q_faiss = Q_faiss[offset:endpos].float().numpy()
            scores, some_embedding_ids = self.faiss_index.search(some_Q_faiss, faiss_depth)

            embeddings_ids.append(torch.from_numpy(some_embedding_ids))
            all_scores.append(torch.from_numpy(scores))

        embedding_ids = torch.cat(embeddings_ids)
        all_scores = torch.cat(all_scores)

        return embedding_ids, all_scores

    def embedding_ids_to_pids(self, embedding_ids, verbose=True):
        # Find unique PIDs per query.
        print_message("#> Lookup the PIDs..", condition=verbose)
        all_pids = self.emb2pid[embedding_ids]

        print_message(f"#> Converting to a list [shape = {all_pids.size()}]..", condition=verbose)
        all_pids = all_pids.tolist()

        print_message("#> Removing duplicates (in parallel if large enough)..", condition=verbose)

        if len(all_pids) > 5000:
            all_pids = list(self.parallel_pool.map(uniq, all_pids))
        else:
            all_pids = list(map(uniq, all_pids))

        print_message("#> Done with embedding_ids_to_pids().", condition=verbose)

        return all_pids




def uniq(l):
    return list(set(l))


def torch_percentile(tensor: torch.Tensor, p):
    assert p in range(1, 100 + 1)
    assert tensor.dim() == 1
    return tensor.kthvalue(int(p * tensor.size(0) / 100.0)).values.item()
