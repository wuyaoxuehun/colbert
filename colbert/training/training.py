import os
import random
import time
import torch
import torch.nn as nn
import numpy as np

from transformers import AdamW
from colbert.utils.runs import Run
from colbert.utils.amp import MixedPrecisionManager

from colbert.training.lazy_batcher import LazyBatcher
from colbert.training.eager_batcher import EagerBatcher
from colbert.parameters import DEVICE

# from colbert.modeling.colbert import ColBERT
from colbert.modeling.colbert_list import ColBERT_List
from colbert.utils.utils import print_message
from colbert.training.utils import print_progress, manage_checkpoints
from colbert.base_config import pos_num, neg_num, pn_num, pretrain, ColBert, pre_batch_num_base, pre_batch_warm_up_rate
from colbert.training.losses import binary_listnet, listnet_loss
from transformers import BertTokenizerFast, get_linear_schedule_with_warmup


def train(args):
    random.seed(12345)
    np.random.seed(12345)
    torch.manual_seed(12345)
    if args.distributed:
        torch.cuda.manual_seed_all(12345)

    if args.distributed:
        assert args.bsize % args.nranks == 0, (args.bsize, args.nranks)
        assert args.accumsteps == 1
        args.bsize = args.bsize // args.nranks

        print("Using args.bsize =", args.bsize, "(per process) and args.accumsteps =", args.accumsteps)

    if args.lazy:
        reader = LazyBatcher(args, (0 if args.rank == -1 else args.rank), args.nranks)
    else:
        reader = EagerBatcher(args, (0 if args.rank == -1 else args.rank), args.nranks)

    if args.rank not in [-1, 0]:
        torch.distributed.barrier()

    # tokenizer = BertTokenizerFast.from_pretrained(pretrain)
    colbert = ColBert.from_pretrained(pretrain,
                                      query_maxlen=args.query_maxlen,
                                      doc_maxlen=args.doc_maxlen,
                                      dim=args.dim,
                                      similarity_metric=args.similarity, )

    if args.checkpoint is not None:
        assert args.resume_optimizer is False, "TODO: This would mean reload optimizer too."
        print_message(f"#> Starting from checkpoint {args.checkpoint} -- but NOT the optimizer!")

        checkpoint = torch.load(args.checkpoint, map_location='cpu')

        try:
            colbert.load_state_dict(checkpoint['model_state_dict'])
        except:
            print_message("[WARNING] Loading checkpoint with strict=False")
            colbert.load_state_dict(checkpoint['model_state_dict'], strict=False)

    if args.rank == 0:
        torch.distributed.barrier()

    colbert = colbert.to(DEVICE)
    colbert.train()

    optimizer = AdamW(filter(lambda p: p.requires_grad, colbert.parameters()), lr=args.lr, eps=1e-8)
    optimizer.zero_grad()
    t_total = len(reader) // args.accumsteps * args.epoch
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1 * t_total), num_training_steps=t_total)
    amp = MixedPrecisionManager(args.amp)

    if args.distributed:
        colbert = torch.nn.parallel.DistributedDataParallel(colbert, device_ids=[args.rank],
                                                            output_device=args.rank,
                                                            find_unused_parameters=True)
    # criterion = nn.CrossEntropyLoss()
    # criterion = binary_listnet
    criterion = listnet_loss
    # labels = torch.zeros(args.bsize, dtype=torch.long, device=DEVICE)
    labels = torch.zeros((args.bsize, args.bsize * pn_num * (1 + pre_batch_num_base)), dtype=torch.float, device=DEVICE)
    # for i in range(args.bsize):
    #     labels[i, i * pn_num:i * pn_num + pos_num] = torch.ones(pos_num)

    start_time = time.time()

    start_batch_idx = 0
    if args.resume:
        assert args.checkpoint is not None
        start_batch_idx = checkpoint['batch']

        reader.skip_to_batch(start_batch_idx, checkpoint['arguments']['bsize'])

    args.max_grad_norm = 1.0

    from tqdm import tqdm
    from collections import deque
    pre_batch_que = deque(maxlen=pre_batch_num_base)
    pre_batch_num = 0
    for epoch in range(args.epoch):
        train_loss = 0.0
        reader.skip_to_batch(0, args.bsize)
        if args.rank < 1:
            batch_iterator = tqdm(zip(range(start_batch_idx, args.maxsteps), reader), total=len(reader), )
        else:
            batch_iterator = zip(range(start_batch_idx, args.maxsteps), reader)
        labels = labels.to(DEVICE)
        if pre_batch_num_base > 0 and epoch >= int(pre_batch_warm_up_rate * args.epoch):
            pre_batch_num = pre_batch_num_base
            if args.rank == 0:
                print_message('activated pre batch ' + str(pre_batch_num))
        Temperature = 0.2
        for batch_idx, BatchSteps in batch_iterator:
            this_batch_loss = 0.0

            for queries, passages in BatchSteps:
                q_ids, q_mask, q_word_mask = queries
                d_ids, d_mask, d_word_mask, score = passages
                for i in range(q_ids.size(0)):
                    # labels[i, i * pn_num:i * pn_num + pos_num] = score[i]
                    labels[i, i * pn_num:(i + 1) * pn_num] = score[i * pn_num:(i + 1) * pn_num] / Temperature

                with amp.context():
                    q_word_mask_input = None
                    d_word_mask_input = None
                    if args.mask_q == 1:
                        q_word_mask_input = q_word_mask
                        d_word_mask_input = d_word_mask

                    pre_batch_input = None
                    if pre_batch_num and pre_batch_que:
                        pre_batch_input = pre_batch_que

                    scores = colbert((q_ids, q_mask), (d_ids, d_mask), q_mask=q_word_mask_input,
                                     d_mask=d_word_mask_input, pre_batch_input=pre_batch_input, output_prebatch=pre_batch_num > 0)

                    if pre_batch_num:
                        scores, D = scores
                        pre_batch_que.append((D.clone().detach(), d_word_mask.clone().detach()))
                    loss = criterion(y_pred=scores, y_true=labels[:scores.size(0), :scores.size(1)])
                    if args.accumsteps > 1:
                        loss = loss / args.accumsteps
                amp.backward(loss)

                train_loss += loss.item()
                this_batch_loss += loss.item()

            if (batch_idx + 1) % args.accumsteps == 0:
                amp.step(colbert, optimizer)
                scheduler.step()
                # if args.amp:
                #     torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                # else:
                #     torch.nn.utils.clip_grad_norm_(colbert.parameters(), args.max_grad_norm)
                # optimizer.step()
                # scheduler.step()
                # colbert.zero_grad()

            avg_loss = train_loss / (batch_idx + 1)

            elapsed = float(time.time() - start_time)
            if args.rank < 1:
                batch_iterator.set_postfix(avg_loss='%.4f' % avg_loss, epoch='%4d' % epoch,  # elapsed='%.4d' % elapsed,
                                           batch_loss='%.4f' % this_batch_loss, lr='{:.4E}'.format(scheduler.get_last_lr()[0]))

        if args.rank < 1 and (epoch + 1) % 5 == 0:
            # num_examples_seen = (batch_idx - start_batch_idx) * args.bsize * args.nranks
            num_examples_seen = len(reader)
            # log_to_mlflow = (batch_idx % 20 == 0)
            log_to_mlflow = True
            elapsed = float(time.time() - start_time)
            Run.log_metric('train/avg_loss', avg_loss, step=epoch, log_to_mlflow=log_to_mlflow)
            Run.log_metric('train/batch_loss', this_batch_loss, step=epoch, log_to_mlflow=log_to_mlflow)
            Run.log_metric('train/examples', num_examples_seen, step=epoch, log_to_mlflow=log_to_mlflow)
            Run.log_metric('train/throughput', num_examples_seen / elapsed, step=epoch, log_to_mlflow=log_to_mlflow)

            manage_checkpoints(args, colbert, optimizer, epoch + 1)
