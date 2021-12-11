import os
import time
import myfaiss
import random
import torch
import itertools

from colbert.utils.runs import Run
from multiprocessing import Pool
from colbert.modeling.inference import ModelInference
from colbert.evaluation.ranking_logger import RankingLogger

from colbert.utils.utils import print_message, batch
from colbert.ranking.rankers import Ranker
from tqdm import tqdm
from colbert.base_config import segmenter
from file_utils import dump_json

all_paras = {}


def merge_retrieval_output_as_example(q, ranks):
    global all_paras
    for i in range(4):
        paras = []
        for rank, (fake_p_id, score) in enumerate(ranks[i]):
            real_p = all_paras[int(fake_p_id)]
            # print(real_p)
            t = {
                'p_id': real_p['p_id'],
                "paragraph": ''.join(real_p['paragraph_cut']['tok'].split()),
                "paragraph_cut": real_p['paragraph_cut'],
                "score": score,
                "rank": rank,
            }
            paras.append(t)
        q['paragraph_' + 'abcd'[i]] = paras
    return q


from corpus.retrieval_utils import load_all_paras


def retrieve(args):
    inference = ModelInference(args.colbert, segmenter, amp=args.amp)
    ranker = Ranker(args, inference, faiss_depth=args.faiss_depth)

    ranking_logger = RankingLogger(Run.path, qrels=None)
    milliseconds = 0
    global all_paras
    all_paras = load_all_paras()

    def pre_encode_q(qs):
        output = []
        ts = []
        for q in qs:
            for opt in 'ABCD':
                t = {
                    "question_cut": q['question_cut'],
                    "background_cut": q['background_cut'],
                    "option_cut": q[f'{opt}_cut'],
                }
                ts.append(t)
        encoded = ranker.encode(ts, bsize=64)
        for i in range(len(qs)):
            output.append(encoded[i * 4:(i + 1) * 4])
        return output

    with ranking_logger.context(args.retrieval_output_file, also_save_annotations=False) as rlogger:
        queries = args.queries
        # qids_in_order = list(queries.keys())
        bsize = 1000
        rankings = []
        encoded_qs = pre_encode_q([queries[i] for i in range(len(queries))])
        # for qoffset, qbatch in batch(encoded_qs, bsize, provide_offset=True):
        for qoffset, qbatch in batch(encoded_qs, bsize, provide_offset=True):
            # qbatch_text = [queries[qid] for qid in qbatch]
            for query_idx, q in tqdm(enumerate(qbatch), total=len(qbatch)):
                # torch.cuda.synchronize('cuda:0')
                # s = time.time()
                ranks = []
                for i in range(4):
                    Q = q[i]
                    # Q = list(Q)
                    # Q[0] = Q[0][:, Q[1].squeeze(0) > 0, ...] #这里将mask的部分直接过滤掉，后面不用这些mask的token做检索，提升速度
                    Q = Q.unsqueeze(0)
                    torch.save(Q, "temp.pt")
                    # print(Q.size())
                    pids, scores = ranker.rank(Q)
                    # torch.cuda.synchronize()
                    # milliseconds += (time.time() - s) * 1000.0
                    ranks.append(list(zip(pids, scores))[:args.depth])

                if len(pids) and query_idx % bsize == 0:
                    print(qoffset + query_idx, len(scores), len(pids), scores[0], pids[0],
                          milliseconds / (qoffset + query_idx + 1), 'ms')

                rankings.append(merge_retrieval_output_as_example(queries[qoffset + query_idx], ranks))

            for query_idx, (qid, ranking) in enumerate(zip(qbatch, rankings)):
                query_idx = qoffset + query_idx

                if query_idx % bsize == 0:
                    print_message(f"#> Logging query #{query_idx} now...")

                # ranking = [(score, pid, None) for pid, score in itertools.islice(ranking, args.depth)]
                # rlogger.log(qid, ranking, is_ranked=True)
        dump_json(rankings, args.retrieval_output_file)

    print('\n\n')
    print(ranking_logger.filename)
    print("#> Done.")
    print('\n\n')
